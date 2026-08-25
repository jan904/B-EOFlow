import json
import os
import numpy as np
import torch
import argparse

from src.model.data_utils import (
    load_data,
    prepare_train_test_data,
    get_condition_vocab,
    load_metacells,
    split_holdout_combinations,
)
from src.model.training import train_model
from src.model.build_model import get_model, ModelConfig
from src.model.naming import checkpoint_name, run_dir_name, holdout_suffix


def parse_args():
    parser = argparse.ArgumentParser(description="Train INN model")

    parser.add_argument("--dataset", type=str, default="kang")
    parser.add_argument("--top_genes", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--sigma_noise", type=float, default=0.5)
    parser.add_argument("--lam_MTC", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--model_path", type=str, default="/home/jhoefer/sandbox/models/EOFlow")
    parser.add_argument("--log_root", type=str, default="/home/jhoefer/sandbox/results/logs")
    parser.add_argument("--model_prefix", type=str, default=None)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--use_counts", action="store_true")
    parser.add_argument("--use_metacells", action="store_true")
    parser.add_argument(
        "--metacell_group_keys",
        nargs="+",
        default=None,
        help="obs columns to group cells by when building metacells (e.g. cytokine cell_type). "
        "Defaults to --conditions when unset, matching the prior coupled behavior.",
    )
    parser.add_argument(
        "--holdout_combo",
        nargs="+",
        default=None,
        help="obs column=value pairs defining one combination to hold out of training "
        "for leave-one-combo-out OOD evaluation (e.g. cytokine=IL-2 'cell_type=CD8 T cells'). "
        "Requires --use_metacells.",
    )
    parser.add_argument(
        "--holdout_file",
        type=str,
        default=None,
        help="JSON file mapping a name to a combo, e.g. {\"combo_1\": {\"cytokine\": \"IL-2\", "
        "\"cell_type\": \"CD8 Memory\"}, ...}. Every combo in it is held out of training, so "
        "evaluation gets several OOD combos instead of one - see configs/holdout_combos.json. "
        "Mutually exclusive with --holdout_combo. Requires --use_metacells.",
    )
    parser.add_argument(
        "--holdout_name",
        type=str,
        default=None,
        help="short name for the holdout SET, used in the checkpoint and results paths "
        "(the per-combo names from the file would be far too long). Defaults to the file's "
        "basename, e.g. holdout_combos.json -> '_holdout_combos'.",
    )
    parser.add_argument(
        "--probe_combo",
        nargs="+",
        default=None,
        help="obs column=value pairs for a SECOND combo, held out of training and scored "
        "every --ood_probe_every epochs (MMD / R2 / dispersion / nearest-mean accuracy) "
        "so checkpoints can be selected on OOD quality instead of likelihood. Keep it "
        "distinct from --holdout_combo: selecting on a combo makes it a validation set. "
        "Requires --use_metacells.",
    )
    parser.add_argument(
        "--probe_holdout",
        action="store_true",
        help="probe the combos already held out by --holdout_file/--holdout_combo instead "
        "of a separate --probe_combo, reporting median and IQR across them. Ten combos "
        "give a spread where one gives a bare number, but these are the combos the "
        "results report, so the curve is LOGGED ONLY: no *_best_ood.pt is written unless "
        "--select_on_ood is passed as well. Mutually exclusive with --probe_combo.",
    )
    parser.add_argument(
        "--select_on_ood",
        action="store_true",
        help="with --probe_holdout, also write *_best_ood.pt selected on the held-out "
        "combos' median MMD. This makes them a validation set - whatever the run reports "
        "as its test combos is then no longer untouched. Off by default for that reason.",
    )
    parser.add_argument("--ood_probe_every", type=int, default=50)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--test_size", type=float, default=0.0)
    parser.add_argument("--condition_type", type=str, default=None, choices=["mixture", "normal"])
    parser.add_argument("--train_means", action="store_true")
    parser.add_argument("--lr_means", type=float, default=5e-4)
    parser.add_argument(
        "--supervise_latent_meaning",
        type=str,
        default=None,
        choices=["counterfactual", "partition", "both", None],
    )
    parser.add_argument("--lam_supervise", type=float, default=1.0)
    parser.add_argument("--partition_divisor", type=int, default=8)
    # None -> 2*sqrt(2 ln K), see INNs_model.default_means_seperation
    parser.add_argument("--means_seperation", type=float, default=None)
    # Factorized mixture prior: mu(combo) = sum of per-condition level embeddings
    parser.add_argument("--factorize_means", action="store_true")
    # mu = a_cell_type + w_cell_type * b_cytokine instead of a + b: one learned scalar per
    # cell type on the treatment embedding. Needs --factorize_means and --control_condition.
    parser.add_argument("--treatment_gain", action="store_true")
    parser.add_argument("--means_dim", type=int, default=None)
    # which condition is the treatment axis (its control level is the dataset's own
    # control_label); required for --factorize_means to expose a treatment shift
    parser.add_argument("--control_condition", type=str, default=None)
    parser.add_argument("--latent_per_condition", type=int, default=None)
    parser.add_argument("--train_sigma", action="store_true")
    parser.add_argument("--lr_sigma", type=float, default=5e-4)
    parser.add_argument("--balance_classes", action="store_true")

    return parser.parse_args()


def _load_holdout_file(path, conditions):
    """{name: {condition: value}} from JSON, validated against `conditions`.

    Names are kept (combo_1, combo_2, ...) rather than derived from the combo itself: they
    are short enough for paths and figure labels, and they stay stable if the underlying
    combo is ever swapped, so results from different runs remain comparable by name.
    """
    with open(path) as handle:
        combos = json.load(handle)
    if not isinstance(combos, dict) or not combos:
        raise ValueError(f"{path} must be a non-empty object mapping name -> combo.")
    for name, combo in combos.items():
        if set(combo) != set(conditions):
            raise ValueError(
                f"{path}: '{name}' has keys {sorted(combo)}, expected {sorted(conditions)}."
            )
    return combos


def _parse_holdout_combo(tokens):
    combo = {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep:
            raise ValueError(f"Invalid --holdout_combo token '{token}', expected 'key=value'.")
        combo[key] = value
    return combo


def _name_knobs(args):
    """The subset of `args` that the checkpoint name is built from - see src.model.naming,
    which owns the scheme so train.py, eval_checkpoint.py and the notebooks agree on it."""
    return {
        "lam_MTC": args.lam_MTC,
        "sigma_noise": args.sigma_noise,
        "lr_means": args.lr_means,
        "latent_per_condition": args.latent_per_condition,
        "means_dim": args.means_dim,
    }


def main():
    args = parse_args()

    if args.holdout_combo is not None and not args.use_metacells:
        raise ValueError("--holdout_combo requires --use_metacells.")

    # Set device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Data loading and preprocessing
    holdout_combo = None
    if args.use_metacells:
        metacell_group_keys = args.metacell_group_keys or args.conditions
        # donors=["Donor1"]: no --donors CLI flag exists yet, matching load_data's own
        # default - keep in sync if that ever becomes configurable, so a cache built
        # under one donor set is never mistaken for another.
        adata, _, control_label = load_metacells(
            args.dataset, args.top_genes, metacell_group_keys, donors=["Donor1"]
        )
    else:
        adata, _, control_label = load_data(
            args.dataset, args.top_genes, log_transform=True, cell_types=["CD4 Memory"]
        )

    # Fix the combo/condition vocabulary from the full (pre-holdout) data, so a held-out
    # combo still gets a reserved slot in the one-hot encoding and the mixture-prior means,
    # instead of vanishing along with its rows.
    condition_categories = None
    combo_categories = None
    if args.conditions is not None:
        condition_categories, combo_categories = get_condition_vocab(adata, args.conditions)

    # The OOD probe's combo has to leave the training set as well, otherwise it is not
    # out-of-distribution and the curve it produces means nothing. It is kept separate
    # from --holdout_combo on purpose: selecting a checkpoint on a combo turns that combo
    # into a validation set, so the reported test combo must be one nothing selects on.
    if args.probe_combo and args.probe_holdout:
        raise ValueError("Pass either --probe_combo or --probe_holdout, not both.")
    if args.select_on_ood and not args.probe_holdout:
        raise ValueError("--select_on_ood only applies to --probe_holdout.")

    probe_combo = _parse_holdout_combo(args.probe_combo) if args.probe_combo else None
    if probe_combo is not None and not args.use_metacells:
        raise ValueError("--probe_combo requires --use_metacells.")
    if args.probe_holdout and not args.use_metacells:
        raise ValueError("--probe_holdout requires --use_metacells.")
    # captured before the holdout split below, because the probe scores the combos whose
    # cells that split removes - it needs the frame that still contains them
    probe_adata = adata if (probe_combo is not None or args.probe_holdout) else None

    if args.holdout_file is not None:
        if args.holdout_combo is not None:
            raise ValueError("Pass either --holdout_file or --holdout_combo, not both.")
        if not args.use_metacells:
            raise ValueError("--holdout_file requires --use_metacells.")

    holdout_combos = {}
    if args.holdout_file is not None:
        holdout_combos = _load_holdout_file(args.holdout_file, args.conditions)
    elif args.use_metacells and args.holdout_combo is not None:
        holdout_combos = {"combo_1": _parse_holdout_combo(args.holdout_combo)}

    if holdout_combos:
        # every named combo leaves training, plus the probe's if there is one. Holding out
        # several rather than one is what makes the OOD numbers interpretable: a single
        # combo gives one value with no error bar, and the ones we have used carry ~34 real
        # metacells, where a noise floor is nearly as large as the effect being measured.
        holdout_combo = next(iter(holdout_combos.values()))   # kept for downstream config
        to_hold = list(holdout_combos.values())
        if probe_combo is not None:
            to_hold.append(probe_combo)
        adata, _ = split_holdout_combinations(adata, to_hold, group_keys=metacell_group_keys)
        listed = ", ".join(f"{k}={v}" for k, v in holdout_combos.items())
        print(f"Held out {len(holdout_combos)} combo(s): {listed}")
        if probe_combo is not None:
            print(f"  plus the probe combo {probe_combo}")
        print(f"Training on the remaining {adata.n_obs} metacells.")
    elif probe_combo is not None:
        adata, _ = split_holdout_combinations(
            adata, [probe_combo], group_keys=metacell_group_keys
        )
        print(f"Held out probe combo {probe_combo}; {adata.n_obs} metacells remain.")

    dataset, dataloader, test_dataset, test_dataloader = prepare_train_test_data(
        adata,
        args.batch_size,
        device,
        dtype,
        counts=args.use_counts,
        label_key=args.conditions,
        test_size=args.test_size,
        combo_categories=combo_categories,
        condition_categories=condition_categories,
    )
    ctrl_idx = 0  # dataset.cats[0].categories.tolist().index(control_label)

    D_dim = dataset.X.shape[1]
    N_dim = D_dim

    # Get condition shapes for model initialization
    condition_shapes = None
    if args.conditions is not None:
        condition_shapes = [len(condition_categories[cond]) for cond in args.conditions]

    top_genes = min(args.top_genes, D_dim)

    # Paths
    # Define default name for model and output dir based on hyperparams
    name_knobs = _name_knobs(args)

    # Model will be saved to group share
    model_path = os.path.join(
        "/g/stegle/jhoefer/models/EOFlow", args.dataset, "top_" + str(top_genes)
    )

    # Output (logs, plots) will be saved to local sandbox
    output_dir_path = os.path.join(
        "/home/jhoefer/sandbox/results", args.dataset, "top_" + str(top_genes)
    )

    # Built by src.model.naming, not concatenated here, so an empty --model_prefix means
    # "no prefix" in exactly the way the notebooks' resolve_checkpoint reads it. Spelled
    # out locally this used to test `is not None`, which turned --model_prefix "" into a
    # leading underscore that no reader would look for.
    model_name = checkpoint_name(args.model_prefix, **name_knobs)
    legacy_model_name = checkpoint_name(args.model_prefix, legacy=True, **name_knobs)
    output_dir_name = run_dir_name(args.model_prefix, **name_knobs)

    if args.train_sigma:
        model_path += "_train_sigma"
        output_dir_path += "_train_sigma"
    elif args.use_metacells:
        model_path += "_metacells"
        output_dir_path += "_metacells"

        if holdout_combo is not None:
            # built by src.model.naming so the notebooks derive the identical component
            suffix = holdout_suffix(
                holdout_file=args.holdout_file,
                combo=holdout_combo,
                holdout_name=args.holdout_name,
            )
            model_path += suffix
            output_dir_path += suffix
    else:
        # If use_counts is True, append "_counts" to model name and output dir name
        if args.use_counts:
            model_path += "_counts"
            output_dir_path += "_counts"

        if args.supervise_latent_meaning is not None:
            model_path += f"_{args.supervise_latent_meaning}"
            output_dir_path += f"_{args.supervise_latent_meaning}"

        if args.conditions is not None and args.supervise_latent_meaning is None:
            model_path += "_cond"
            output_dir_path += "_cond"

            if args.condition_type is not None:
                model_path += f"_{args.condition_type}"
                output_dir_path += f"_{args.condition_type}"

                if args.condition_type == "mixture":
                    if args.train_means == True:
                        model_path += f"_train_means"
                        output_dir_path += f"_train_means"
                    else:
                        model_path += "_empirical"
                        output_dir_path += "_empirical"

    # The prior parameterization has to be part of the path: factorized and free-means
    # checkpoints are structurally incompatible but every other path component is
    # identical, so without this a factorized run resolves to an existing free-means
    # checkpoint and the resume-if-exists branch below would try to load it.
    if args.factorize_means:
        model_path += "_factorized"
        output_dir_path += "_factorized"

    # Same reasoning one level down: a gain checkpoint and an additive one differ by a
    # parameter, so they must not share a path (loading one into the other is refused in
    # remap_keys, which would abort a resume rather than silently continue).
    if args.treatment_gain:
        model_path += "_gain"
        output_dir_path += "_gain"

    log_dir = os.path.join(output_dir_path, "logs", output_dir_name)

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Training setup
    kwargs_data = {
        "device": device,
        "dtype": dtype,
        "N_dim": N_dim,
        "D_dim": D_dim,
        "dataset": dataset,
        "train_dataloader": dataloader,
        "test_dataloader": test_dataloader,
        "data_mean": dataset.X.mean(dim=0),
        "data_std": torch.ones(D_dim),
        "sigma_noise": args.sigma_noise,
        "sigma_inflate": args.sigma_noise,
    }

    kwargs_loss = {
        "use_NLL": True,
        "use_MER": True,
        "mode_MER": "unbiased",  #'full', #'unbiased'
        "lam_MTC": args.lam_MTC,
        "lam_ME_i": list(np.zeros(D_dim)),
        # 'use_align': False,
        # 'dims_align': D_dim - 1,
        # 'lam_align': 1, #0.1,
        # 'lam_disent': 0, #0.1,
        "use_rec": False,
        "dims_rec": D_dim - 1,
        "lam_rec": 100,  #
        "sigma_noise": args.sigma_noise,
    }

    metrics_loss = {
        "epoch": [],
        "loss": [],
        "z2": [],
        "NLL": [],
        "MTC": [],
        "H_i": [],
        "H_core": [],
        "H_detail": [],
        "MI_core_detail": [],
        "L2_rec": [],
    }
    val_metrics_loss = metrics_loss.copy()

    # Checkpoints written before lr_means/lpc entered the name sit at the old path and no
    # longer resolve. Say so rather than starting from scratch under a name that looks like
    # a resume - and refuse outright if a resume was actually asked for, since silently
    # training a fresh model for 14 hours is the expensive failure here.
    legacy_full = os.path.join(model_path, legacy_model_name)
    if not os.path.exists(os.path.join(model_path, model_name)) and os.path.exists(legacy_full):
        message = (
            f"A checkpoint exists under the pre-lr_means naming scheme:\n"
            f"  {legacy_full}\n"
            f"and this run resolves to the new name:\n"
            f"  {os.path.join(model_path, model_name)}\n"
            f"They are the same configuration only if that run used lr_means="
            f"{args.lr_means} and latent_per_condition={args.latent_per_condition}. "
            f"If so, rename it:\n  mv {legacy_full} {os.path.join(model_path, model_name)}"
        )
        if args.checkpoint:
            raise SystemExit(f"{message}\n\n--checkpoint was requested; refusing to start fresh.")
        print(f"NOTE: {message}\n")

    start_epoch = 0
    log_path = None
    if os.path.exists(os.path.join(model_path, model_name)) and args.checkpoint:
        checkpoint = torch.load(
            os.path.join(model_path, model_name),
            map_location=device,
            weights_only=False,
        )

        model_config = checkpoint["model_config"]
        model, optimizer = get_model(config=model_config)
        model = model.to(device)

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        saved_state = checkpoint["model_state_dict"]

        remapped_state = {}
        for k, v in saved_state.items():
            if not k.startswith("flow.") and not k.startswith("means"):
                remapped_state[f"flow.{k}"] = v
            else:
                remapped_state[k] = v

        missing, unexpected = model.load_state_dict(remapped_state, strict=False)

        non_prior_missing = [k for k in missing if not k.startswith("means")]
        if non_prior_missing:
            print(f"WARNING: unexpected missing keys: {non_prior_missing}")
        else:
            print(f"Flow weights loaded. Prior initialised fresh ({len(missing)} new params).")

        metrics_loss = checkpoint["metrics_loss"]
        if "val_metrics_loss" in checkpoint:
            val_metrics_loss = checkpoint["val_metrics_loss"]
        log_path = checkpoint["log_path"]
        start_epoch = checkpoint["epoch"] + 1

    else:
        assert not (
            args.supervise_latent_meaning in ["partition", "both"]
            and args.latent_per_condition is None
        ), "latent_per_condition must be specified when supervise_latent_meaning is 'partition' or 'both'."

        if args.treatment_gain and not (args.factorize_means and args.control_condition):
            raise SystemExit(
                "--treatment_gain needs --factorize_means and --control_condition: it scales "
                "the factorized prior's treatment embedding, relative to the pinned control."
            )

        model_config = ModelConfig(
            N_dim=N_dim,
            condition_shapes=condition_shapes,
            ch_hidden=args.width,
            N_blocks=args.depth,
            lr=args.lr,
            warmup_steps=2 * len(dataloader),
            pre_normalize=True,
            normalize=True,
            condition_type=args.condition_type,
            trainable_means=args.train_means,
            lr_means=args.lr_means,
            means_seperation=args.means_seperation,
            n_clusters=len(dataset.cats[0].categories),
            supervise_latent_meaning=args.supervise_latent_meaning,
            ctrl_idx=ctrl_idx,
            lam_supervise=args.lam_supervise,
            latent_per_condition=args.latent_per_condition,
            partition_divisor=args.partition_divisor,
            trainable_sigma=args.train_sigma,
            lr_sigma=args.lr_sigma,
            balance_classes=args.balance_classes,
            holdout_combos=list(holdout_combos.values()) or None,
            combo_categories=combo_categories,
            condition_categories=condition_categories,
            conditions=args.conditions,
            factorize_means=args.factorize_means,
            treatment_gain=args.treatment_gain,
            means_dim=args.means_dim,
            control_condition=args.control_condition,
            # control_label comes from load_data/load_metacells, so the pinned level is
            # always the dataset's own control rather than a separately-typed string
            control_label=control_label if args.factorize_means else None,
        )

        # Model initialization
        model, optimizer = get_model(config=model_config)
        model = model.to(device)
        print(model_config)

    NUM_EPOCHS = max(1, np.ceil(args.epochs * args.batch_size / adata.X.shape[0]).astype(int))
    ood_probe = None
    ood_select = True
    if probe_combo is not None:
        from src.analysis.ood_probe import OODProbe

        ood_probe = OODProbe(
            probe_adata,            # pre-holdout: it still contains the probe combo's cells
            probe_combo,
            combo_categories,
            args.conditions,
            control_label,
            device=device,
        )
        print(
            f"OOD probe on {ood_probe.label} ({ood_probe.n_real} real metacells), "
            f"every {args.ood_probe_every} epochs."
        )
    elif args.probe_holdout:
        from src.analysis.ood_probe import MultiOODProbe

        if not holdout_combos:
            raise ValueError(
                "--probe_holdout needs combos to probe: pass --holdout_file or "
                "--holdout_combo as well."
            )
        ood_probe = MultiOODProbe(
            probe_adata,            # pre-holdout: it still contains the held-out cells
            holdout_combos,
            combo_categories,
            args.conditions,
            control_label,
            device=device,
        )
        ood_select = args.select_on_ood
        print(
            f"OOD probe on {ood_probe.label} ({ood_probe.n_real} real metacells total), "
            f"every {args.ood_probe_every} epochs."
        )
        for name, reason in ood_probe.skipped.items():
            print(f"  skipped {name}: {reason}")
        print(
            "  writing *_best_ood.pt selected on these combos - they are a validation "
            "set now, not a test set."
            if ood_select
            else "  logging only; no checkpoint is selected on these combos."
        )

    model, optimizer, metrics_loss, val_metrics_loss, log_path = train_model(
        model,
        optimizer,
        metrics_loss,
        kwargs_data,
        kwargs_loss,
        model_config=model_config,
        val_metrics_loss=val_metrics_loss,
        num_epochs=NUM_EPOCHS,
        start_epoch=start_epoch,
        print_info=True,
        log_dir=log_dir,
        continue_log_path=log_path,
        save_model_path=os.path.join(model_path, model_name),
        conditions=args.conditions,
        use_counts=args.use_counts,
        save_on_validation=args.validation,
        ood_probe=ood_probe,
        ood_probe_every=args.ood_probe_every,
        ood_select=ood_select,
    )


if __name__ == "__main__":
    main()
