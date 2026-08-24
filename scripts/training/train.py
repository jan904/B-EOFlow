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
        "--probe_combo",
        nargs="+",
        default=None,
        help="obs column=value pairs for a SECOND combo, held out of training and scored "
        "every --ood_probe_every epochs (MMD / R2 / dispersion / nearest-mean accuracy) "
        "so checkpoints can be selected on OOD quality instead of likelihood. Keep it "
        "distinct from --holdout_combo: selecting on a combo makes it a validation set. "
        "Requires --use_metacells.",
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
    parser.add_argument("--means_dim", type=int, default=None)
    # which condition is the treatment axis (its control level is the dataset's own
    # control_label); required for --factorize_means to expose a treatment shift
    parser.add_argument("--control_condition", type=str, default=None)
    parser.add_argument("--latent_per_condition", type=int, default=None)
    parser.add_argument("--train_sigma", action="store_true")
    parser.add_argument("--lr_sigma", type=float, default=5e-4)
    parser.add_argument("--balance_classes", action="store_true")

    return parser.parse_args()


def _parse_holdout_combo(tokens):
    combo = {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep:
            raise ValueError(f"Invalid --holdout_combo token '{token}', expected 'key=value'.")
        combo[key] = value
    return combo


def _sanitize(value):
    return value.replace(" ", "-").replace("/", "-")


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
    probe_combo = _parse_holdout_combo(args.probe_combo) if args.probe_combo else None
    if probe_combo is not None and not args.use_metacells:
        raise ValueError("--probe_combo requires --use_metacells.")
    probe_adata = adata if probe_combo is not None else None

    if args.use_metacells and args.holdout_combo is not None:
        holdout_combo = _parse_holdout_combo(args.holdout_combo)
        to_hold = [holdout_combo]
        if probe_combo is not None:
            to_hold.append(probe_combo)
        adata, _ = split_holdout_combinations(adata, to_hold, group_keys=metacell_group_keys)
        print(
            f"Held out combo {holdout_combo}; training on the remaining {adata.n_obs} metacells."
        )
        if probe_combo is not None:
            print(f"  plus the probe combo {probe_combo}")
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
    default_name = "MTC_" + str(args.lam_MTC) + "_sigma_" + str(args.sigma_noise)

    # Model will be saved to group share
    model_path = os.path.join(
        "/g/stegle/jhoefer/models/EOFlow", args.dataset, "top_" + str(top_genes)
    )

    # Output (logs, plots) will be saved to local sandbox
    output_dir_path = os.path.join(
        "/home/jhoefer/sandbox/results", args.dataset, "top_" + str(top_genes)
    )

    # If model_prefix is provided, append it to the model name
    if args.model_prefix is not None:
        model_name = args.model_prefix + "_" + default_name + "_model.pt"
        output_dir_name = args.model_prefix + "_" + default_name
    else:
        model_name = default_name + "_model.pt"
        output_dir_name = default_name

    if args.train_sigma:
        model_path += "_train_sigma"
        output_dir_path += "_train_sigma"
    elif args.use_metacells:
        model_path += "_metacells"
        output_dir_path += "_metacells"

        if holdout_combo is not None:
            combo_str = "_".join(f"{k}-{_sanitize(v)}" for k, v in holdout_combo.items())
            model_path += f"_holdout_{combo_str}"
            output_dir_path += f"_holdout_{combo_str}"
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
            holdout_combos=[holdout_combo] if holdout_combo is not None else None,
            combo_categories=combo_categories,
            condition_categories=condition_categories,
            conditions=args.conditions,
            factorize_means=args.factorize_means,
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
    )


if __name__ == "__main__":
    main()
