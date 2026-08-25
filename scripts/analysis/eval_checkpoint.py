"""Standard evaluation for one trained EOFlow checkpoint.

Answers, for any checkpoint (factorized or free per-combo means), the four questions we
keep re-asking by hand:

1. **prior geometry** - are the means separated, structured, and did they actually move
   from their random initialization? Rows still sitting exactly at init are combos that
   never received a gradient, which is the free-means failure mode.
2. **latent fit** - does the data land on the means the prior defines? (nearest-mean
   assignment accuracy, centroid-to-mean distance, residual scale vs the prior's 1.0)
3. **OOD** - for the held-out combo, does `encode control -> add the treatment shift ->
   decode` reproduce the real held-out metacells? Reported against three baselines and a
   split-half noise floor, because MMD and R2 are meaningless without them.
4. **support / dispersion** - do generated metacells have the variance and the
   zero-inflation of real ones? A flow on log-normalized data cannot place an atom at 0,
   so this is where it is expected to fail; the numbers say by how much.

Zero fractions are reported with a tolerance, never as `x == 0`: an exact round trip
through the flow returns zeros as +-2e-5, so a strict comparison reports 0% zeros for a
*perfect* reconstruction and is not measuring what it looks like it measures.

Usage:
    python scripts/analysis/eval_checkpoint.py CHECKPOINT [CHECKPOINT ...] [--json out.json]

Run it inside a GPU allocation (encoding all metacells takes ~5 s on a GPU, ~150 s on
CPU); it falls back to CPU automatically.
"""

import argparse
import json
import math
import os

import anndata as ad
import numpy as np
import scanpy as sc
import torch

from src.analysis.INN_OOD import observed_combos_from_adata, sample_leftout_combo_shift
from src.analysis.mmd import mmd
from src.model.build_model import get_model, init_config, remap_keys
from src.model.data_utils import get_condition_vocab

DEFAULT_CACHE = (
    "/g/stegle/jhoefer/data/metacells/parse_top2000_donor-Donor1_groupby-cytokine-cell_type.h5ad"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an EOFlow checkpoint")
    parser.add_argument("checkpoints", nargs="+", help="path(s) to *_model.pt")
    parser.add_argument("--metacell_cache", default=DEFAULT_CACHE)
    parser.add_argument(
        "--holdout_file",
        default=None,
        help="the same JSON of named combos the run was trained with (configs/"
        "holdout_combos.json). Every combo in it is scored and the results aggregated. "
        "Without it the combos are taken from the checkpoint's own config and named "
        "combo_1..N by position.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["cytokine", "cell_type"],
        help="must match the order the checkpoint's combo_categories were built in",
    )
    parser.add_argument("--control_label", default="PBS")
    parser.add_argument(
        "--reference_combo",
        nargs="+",
        default=["cytokine=IL-4", "cell_type=CD8 Memory"],
        help="a TRAINED combo pushed through the exact same counterfactual pipeline. Read "
        "it as 'what would the OOD prediction have looked like for a combo we can check', "
        "not as fit quality: the shift comes from the prior (factorized) or from the "
        "average over other cell types (free means), never from this combo's own data",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_permutations", type=int, default=50, help="MMD noise-floor splits")
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", default=None, help="write all metrics to this path")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# the filename scheme lives in src.model.naming, next to the builder that writes it.
# Re-exported here because this module's callers have always imported it from eval_checkpoint.
from src.model.naming import run_settings_from_name  # noqa: E402,F401


def _parse_combo(tokens):
    combo = {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep:
            raise ValueError(f"Invalid combo token '{token}', expected 'key=value'.")
        combo[key] = value
    return combo


def load_metacell_adata(path):
    """The cache holds raw summed counts; training consumed it log-normalized."""
    adata = ad.read_h5ad(path)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def load_checkpoint_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = init_config(dict(checkpoint["model_config"]))
    model, optimizer = get_model(config)
    model = remap_keys(model, checkpoint["model_state_dict"])

    # schedule-free keeps the parameters at the train (y) iterate while training; the
    # weights meant for evaluation are the x iterate, which only optimizer.eval() exposes.
    # Skipping this silently evaluates a different model than the one being selected on.
    if "optimizer_state_dict" in checkpoint and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer.eval()
        except Exception as err:  # optimizer/param-group mismatch on old checkpoints
            print(f"  ! could not restore the eval iterate ({err}); using saved weights")

    model.eval().to(device)
    return model, checkpoint, config


def zero_fraction(x, tol=1e-3):
    """Fraction of entries that are zero *to within tol* - see the module docstring."""
    return float((np.abs(x) < tol).mean())


def prior_geometry(model, config, combo_categories, conditions, trained_mask):
    """Norms, separation, init-stuck rows, and how additive the means are."""
    means = model.means_active.detach().cpu().numpy()
    n_components, means_dim = means.shape
    separation = 2.0 * math.sqrt(2.0 * math.log(max(n_components, 2)))
    init_norm = separation / math.sqrt(2.0)

    norms = np.linalg.norm(means, axis=1)
    stuck = np.abs(norms - init_norm) < 1e-3

    dist = np.linalg.norm(means[:, None] - means[None, :], axis=-1)
    idx = np.where(trained_mask)[0]
    sub = dist[np.ix_(idx, idx)]
    off = ~np.eye(len(idx), dtype=bool)

    levels = {c: list(config.condition_categories[c]) for c in conditions}
    level_idx = np.array(
        [[levels[c].index(part) for c, part in zip(conditions, combo.split("__"))]
         for combo in combo_categories]
    )

    # least-squares additive fit mu ~ sum of per-factor level effects, on trained combos
    n_levels = [len(levels[c]) for c in conditions]
    design = np.zeros((len(idx), sum(n_levels)))
    offset = 0
    for f, n in enumerate(n_levels):
        design[np.arange(len(idx)), offset + level_idx[idx, f]] = 1
        offset += n
    coef, *_ = np.linalg.lstsq(design, means[idx], rcond=None)
    resid = means[idx] - design @ coef
    ss_tot = ((means[idx] - means[idx].mean(0)) ** 2).sum()
    additive_r2 = float(1 - (resid**2).sum() / ss_tot) if ss_tot > 0 else float("nan")

    # Learned per-cell-type gains, when the prior has them. Reported per level (not just
    # summarized) because the interesting question is *which* cell types the model decided
    # respond weakly - a gain near 0 says "this cell type barely moves", and one far above
    # 1 says the shared shift is too small for it.
    gain = getattr(model, "treatment_gain", None)
    gains = None
    if gain is not None:
        w = gain.detach().cpu().numpy()
        gain_key = conditions[model.gain_factor]
        gains = {
            str(level): float(value)
            for level, value in zip(list(config.condition_categories[gain_key]), w)
        }

    return {
        "factorized": bool(getattr(model, "factorized", False)),
        "n_components": int(n_components),
        "means_dim": int(means_dim),
        "init_row_norm": float(init_norm),
        "requested_separation": float(separation),
        "rows_at_init": int(stuck.sum()),
        "rows_at_init_untrained": int((stuck & ~trained_mask).sum()),
        "norm_median_trained": float(np.median(norms[trained_mask])),
        "pairwise_min_trained": float(sub[off].min()),
        "pairwise_median_trained": float(np.median(sub[off])),
        "additive_r2": additive_r2,
        "treatment_gain": gains,
    }


def treatment_shift_consistency(means, combo_categories, conditions, control_label,
                                condition_key, cell_type_key, trained_mask):
    """Mean pairwise cosine of a treatment's latent shift across cell types.

    1.0 means the treatment is literally one translation (what a factorized prior
    enforces); values near 0 mean the per-combo means encode no shared treatment
    direction at all, so averaging them to estimate a held-out combo is averaging noise.
    """
    index = {c: i for i, c in enumerate(combo_categories)}
    order = list(conditions)

    def label(values):
        return "__".join(str(values[c]) for c in order)

    levels = sorted({c.split("__")[order.index(condition_key)] for c in combo_categories})
    cells = sorted({c.split("__")[order.index(cell_type_key)] for c in combo_categories})

    out = {}
    for treatment in levels:
        if treatment == control_label:
            continue
        shifts = []
        for cell in cells:
            a = index.get(label({condition_key: treatment, cell_type_key: cell}))
            b = index.get(label({condition_key: control_label, cell_type_key: cell}))
            if a is None or b is None or not (trained_mask[a] and trained_mask[b]):
                continue
            shifts.append(means[a] - means[b])
        if len(shifts) < 2:
            continue
        S = np.stack(shifts)
        S = S / np.linalg.norm(S, axis=1, keepdims=True)
        C = S @ S.T
        off = ~np.eye(len(S), dtype=bool)
        out[treatment] = float(C[off].mean())
    return out


def encode(model, X, device, batch_size):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            z, _ = model(X[i : i + batch_size].to(device), c=None, rev=False)
            out.append(z.cpu())
    return torch.cat(out).numpy()


def latent_fit(Z, means, combo_id, holdout_ids, means_dim):
    """`holdout_ids`: {name: combo index} for every combo kept out of training.

    Accuracy is reported per held-out combo as well as pooled, because the combos differ
    enormously in size (27 to 177 real metacells here) and a pooled number is dominated by
    the largest one.
    """
    Zc, muc = Z[:, :means_dim], means[:, :means_dim]
    dist = (Zc**2).sum(1)[:, None] - 2 * Zc @ muc.T + (muc**2).sum(1)[None, :]
    pred = dist.argmin(1)

    held = set(holdout_ids.values())
    is_ho = np.isin(combo_id, list(held)) if held else np.zeros(len(combo_id), dtype=bool)

    centroid_gap, residual_sd = [], []
    for k in np.unique(combo_id):
        mask = combo_id == k
        if mask.sum() < 3 or k in held:
            continue
        centroid_gap.append(np.linalg.norm(Zc[mask].mean(0) - muc[k]))
        residual_sd.append((Zc[mask] - muc[k]).std())

    per_combo = {}
    for name, k in holdout_ids.items():
        mask = combo_id == k
        per_combo[name] = {
            "accuracy": float((pred[mask] == k).mean()) if mask.any() else None,
            "n": int(mask.sum()),
        }

    accs = [v["accuracy"] for v in per_combo.values() if v["accuracy"] is not None]
    return {
        "z2_per_dim": float((Z**2).mean()),
        "accuracy_trained": float((pred[~is_ho] == combo_id[~is_ho]).mean()),
        "accuracy_holdout": float(np.mean(accs)) if accs else None,
        "accuracy_holdout_sd": float(np.std(accs)) if len(accs) > 1 else None,
        "n_holdout_metacells": int(is_ho.sum()),
        "per_combo": per_combo,
        "centroid_gap_median": float(np.median(centroid_gap)),
        "residual_sd_median": float(np.median(residual_sd)),
    }


def counterfactual_metrics(model, adata, combo_categories, conditions, combo, condition_key,
                           cell_type_key, control_label, device, rng, n_permutations,
                           observed_combos=None):
    """Predict one combo from its controls and score it against the real metacells."""
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    obs_cond = adata.obs[condition_key].astype(str).to_numpy()
    obs_cell = adata.obs[cell_type_key].astype(str).to_numpy()

    target = (obs_cond == combo[condition_key]) & (obs_cell == combo[cell_type_key])
    control = (obs_cond == control_label) & (obs_cell == combo[cell_type_key])
    if target.sum() < 4 or control.sum() < 4:
        return None
    real, ctrl = X[target], X[control]

    # observed_combos matters only for a free-means model: without it the averaged
    # treatment shift silently includes mean rows of combos that never had training data
    # and are therefore still at their random initialization.
    pred = sample_leftout_combo_shift(
        model, adata, combo_categories, conditions, combo, condition_key, control_label,
        cell_type_key=cell_type_key, device=device, observed_combos=observed_combos,
    ).X

    # baseline: the same additive assumption, applied in gene space instead of latent space
    deltas = []
    for cell in sorted(set(obs_cell)):
        if cell == combo[cell_type_key]:
            continue
        a = X[(obs_cond == combo[condition_key]) & (obs_cell == cell)]
        b = X[(obs_cond == control_label) & (obs_cell == cell)]
        if len(a) >= 3 and len(b) >= 3:
            deltas.append(a.mean(0) - b.mean(0))
    gene_shift = np.clip(ctrl + np.mean(deltas, axis=0), 0, None) if deltas else None

    pairwise = np.linalg.norm(real[:, None] - real[None, :], axis=-1)
    sigma = float(np.median(pairwise[np.triu_indices_from(pairwise, 1)]))
    t_real = torch.tensor(real, dtype=torch.float32)

    def score(arr):
        return float(mmd(torch.tensor(np.asarray(arr), dtype=torch.float32), t_real,
                         sigma=sigma, unbiased=True))

    floor = []
    for _ in range(n_permutations):
        order = rng.permutation(len(real))
        half = len(real) // 2
        floor.append(float(mmd(t_real[order[:half]], t_real[order[half:]], sigma=sigma,
                               unbiased=True)))

    real_delta = real.mean(0) - ctrl.mean(0)
    top = np.argsort(-np.abs(real_delta))[:50]

    def delta_scores(arr):
        d = np.asarray(arr).mean(0) - ctrl.mean(0)
        return {
            "r_all": float(np.corrcoef(d, real_delta)[0, 1]),
            "r2_all": float(1 - ((d - real_delta) ** 2).sum() / (real_delta**2).sum()),
            "r_top50": float(np.corrcoef(d[top], real_delta[top])[0, 1]),
            "r2_top50": float(1 - ((d[top] - real_delta[top]) ** 2).sum()
                              / (real_delta[top] ** 2).sum()),
        }

    out = {
        "combo": "__".join(str(combo[c]) for c in conditions),
        "n_real": int(target.sum()),
        "n_control": int(control.sum()),
        "mmd_sigma": sigma,
        "mmd_model": score(pred),
        "mmd_do_nothing": score(ctrl),
        "mmd_noise_floor": float(np.mean(floor)),
        "mmd_noise_floor_sd": float(np.std(floor)),
        "model": delta_scores(pred),
        "variance_real": float(real.var(0).sum()),
        "variance_pred": float(np.asarray(pred).var(0).sum()),
        "variance_control": float(ctrl.var(0).sum()),
        "zero_frac_real": zero_fraction(real),
        "zero_frac_pred": zero_fraction(np.asarray(pred)),
        "zero_frac_pred_tol0.1": zero_fraction(np.asarray(pred), tol=0.1),
    }
    if gene_shift is not None:
        out["mmd_gene_space_baseline"] = score(gene_shift)
        out["gene_space_baseline"] = delta_scores(gene_shift)
    return out


def evaluate(path, args, adata, device, rng):
    print("=" * 78)
    print(os.path.basename(path))
    model, checkpoint, config = load_checkpoint_model(path, device)

    conditions = list(args.conditions)
    condition_key, cell_type_key = conditions[0], conditions[1]
    _, combo_categories = get_condition_vocab(adata, conditions)
    if list(config.combo_categories) != list(combo_categories):
        raise ValueError(
            "The checkpoint's combo vocabulary does not match the one built from this "
            "metacell cache - they must come from the same pre-holdout adata."
        )

    # named combos from the file if given, else from the checkpoint's own config
    if args.holdout_file:
        with open(args.holdout_file) as handle:
            named = json.load(handle)
    else:
        named = {f"combo_{i}": c for i, c in enumerate(config.holdout_combos or [], 1)}

    holdout_labels = {
        name: "__".join(str(combo[c]) for c in conditions) for name, combo in named.items()
    }
    holdout_ids = {n: combo_categories.index(l) for n, l in holdout_labels.items()}

    combo = (adata.obs[condition_key].astype(str) + "__"
             + adata.obs[cell_type_key].astype(str)).to_numpy()
    combo_id = np.array([combo_categories.index(c) for c in combo])
    n_metacells = np.bincount(combo_id, minlength=len(combo_categories))
    trained_mask = (n_metacells > 0) & ~np.isin(
        np.arange(len(combo_categories)), list(holdout_ids.values())
    )

    settings = run_settings_from_name(path)
    print(f"  epoch {checkpoint['epoch']} | sigma_noise {settings.get('sigma_noise')} | "
          f"lam_MTC {settings.get('lam_MTC')} | "
          f"factorize_means {getattr(config, 'factorize_means', None)} | "
          f"treatment_gain {getattr(config, 'treatment_gain', None)} | "
          f"holdouts {len(named)}: {', '.join(holdout_labels.values())}")

    # combos with training rows, as labels - free-means OOD needs this to avoid averaging
    # mean rows that never received a gradient (see counterfactual_metrics)
    observed_combos = observed_combos_from_adata(adata, conditions)
    observed_combos = {c for c in observed_combos if c not in set(holdout_labels.values())}

    geo = prior_geometry(model, config, combo_categories, conditions, trained_mask)
    means = model.means_active.detach().cpu().numpy()
    shifts = treatment_shift_consistency(means, combo_categories, conditions,
                                         args.control_label, condition_key, cell_type_key,
                                         trained_mask)
    geo["shift_consistency_mean"] = float(np.mean(list(shifts.values()))) if shifts else None

    print(f"\n  PRIOR   rows at init {geo['rows_at_init']}/{geo['n_components']} "
          f"({geo['rows_at_init_untrained']} of them untrained) | "
          f"||mu|| median {geo['norm_median_trained']:.2f} "
          f"(init {geo['init_row_norm']:.2f})")
    print(f"          pairwise trained: min {geo['pairwise_min_trained']:.2f} "
          f"median {geo['pairwise_median_trained']:.2f} | additive R2 {geo['additive_r2']:.3f} "
          f"| shift consistency {geo['shift_consistency_mean']:.3f}")

    if geo["treatment_gain"]:
        w = geo["treatment_gain"]
        ranked = sorted(w.items(), key=lambda kv: kv[1])
        weak = ", ".join(f"{k} {v:.2f}" for k, v in ranked[:3])
        strong = ", ".join(f"{k} {v:.2f}" for k, v in ranked[-3:][::-1])
        values = np.array(list(w.values()))
        # All gains staying at 1.00 means the gain bought nothing and the prior is still
        # the additive one; spread is the whole point of the parameterization.
        print(f"          gain w: median {np.median(values):.2f} sd {values.std():.2f} | "
              f"weakest {weak} | strongest {strong}")

    X = torch.tensor(adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X),
                     dtype=torch.float32)
    Z = encode(model, X, device, args.batch_size)
    fit = latent_fit(Z, model.means.detach().cpu().numpy(), combo_id, holdout_ids,
                     model.means_active.shape[1])
    print(f"\n  LATENT  E[z^2]/dim {fit['z2_per_dim']:.3f} | nearest-mean accuracy: trained "
          f"{fit['accuracy_trained']:.2%}"
          + (f", held-out {fit['accuracy_holdout']:.2%}"
             + (f" +- {fit['accuracy_holdout_sd']:.2%}" if fit["accuracy_holdout_sd"] else "")
             + f" over {len(named)} combo(s), n={fit['n_holdout_metacells']}"
             if fit["accuracy_holdout"] is not None else ""))
    print(f"          ||centroid - mean|| median {fit['centroid_gap_median']:.2f} | "
          f"residual sd {fit['residual_sd_median']:.3f} (prior assumes 1.0)")

    results = {"checkpoint": path, "epoch": int(checkpoint["epoch"]),
               "sigma_noise": settings.get("sigma_noise"), "lam_MTC": settings.get("lam_MTC"),
               "factorize_means": getattr(config, "factorize_means", None),
               "treatment_gain": getattr(config, "treatment_gain", None),
               "prior": geo, "shift_consistency": shifts, "latent": fit, "counterfactuals": []}

    targets = [(name, combo) for name, combo in named.items()]
    if args.reference_combo:
        # an in-distribution combo through the identical pipeline, as a ceiling
        targets.append(("reference", _parse_combo(args.reference_combo)))

    printed_header = False
    for kind, target in targets:
        metrics = counterfactual_metrics(model, adata, combo_categories, conditions, target,
                                         condition_key, cell_type_key, args.control_label,
                                         device, rng, args.n_permutations,
                                         observed_combos=observed_combos)
        if metrics is None:
            print(f"\n  {kind.upper():<7} {target}: too few metacells, skipped")
            continue
        metrics["kind"] = kind
        results["counterfactuals"].append(metrics)

        if not printed_header:
            print(f"\n  {'combo':<10} {'target':<26} {'n':>4} {'MMD':>8} {'gene-sp':>8} "
                  f"{'do-noth':>8} {'floor':>8} {'R2_50':>6} {'var/real':>8} {'acc':>6}")
            printed_header = True
        acc = fit["per_combo"].get(kind, {}).get("accuracy")
        print(f"  {kind:<10} {metrics['combo']:<26} {metrics['n_real']:>4} "
              f"{metrics['mmd_model']:>8.4f} "
              f"{metrics.get('mmd_gene_space_baseline', float('nan')):>8.4f} "
              f"{metrics['mmd_do_nothing']:>8.4f} {metrics['mmd_noise_floor']:>8.4f} "
              f"{metrics['model']['r2_top50']:>6.3f} "
              f"{metrics['variance_pred'] / metrics['variance_real']:>8.3f} "
              + (f"{acc:>5.0%}" if acc is not None else f"{'-':>6}"))

    held = [c for c in results["counterfactuals"] if c["kind"] != "reference"]
    if len(held) > 1:
        def agg(fn):
            return np.array([fn(c) for c in held], dtype=float)

        mmd_v = agg(lambda c: c["mmd_model"])
        base_v = agg(lambda c: c.get("mmd_gene_space_baseline", np.nan))
        r2_v = agg(lambda c: c["model"]["r2_top50"])
        var_v = agg(lambda c: c["variance_pred"] / c["variance_real"])
        # "beats" counts combos rather than averaging the ratio: the MMD scale differs per
        # combo (bandwidth is its own median distance), so a mean across combos would be
        # dominated by whichever has the largest absolute values.
        results["aggregate"] = {
            "n_combos": len(held),
            "mmd_median": float(np.median(mmd_v)),
            "mmd_iqr": [float(np.quantile(mmd_v, 0.25)), float(np.quantile(mmd_v, 0.75))],
            "beats_gene_space": float(np.mean(mmd_v < base_v)),
            "beats_do_nothing": float(np.mean(mmd_v < agg(lambda c: c["mmd_do_nothing"]))),
            "r2_top50_median": float(np.median(r2_v)),
            "r2_top50_iqr": [float(np.quantile(r2_v, 0.25)), float(np.quantile(r2_v, 0.75))],
            "var_ratio_median": float(np.median(var_v)),
        }
        a = results["aggregate"]
        print(f"\n  AGGREGATE over {a['n_combos']} held-out combos")
        print(f"          MMD median {a['mmd_median']:.4f} "
              f"[IQR {a['mmd_iqr'][0]:.4f}-{a['mmd_iqr'][1]:.4f}] | "
              f"beats gene-space in {a['beats_gene_space']:.0%}, "
              f"do-nothing in {a['beats_do_nothing']:.0%} of combos")
        print(f"          R2_top50 median {a['r2_top50_median']:.3f} "
              f"[IQR {a['r2_top50_iqr'][0]:.3f}-{a['r2_top50_iqr'][1]:.3f}] | "
              f"var/real median {a['var_ratio_median']:.3f}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    print(f"device: {device}")

    adata = load_metacell_adata(args.metacell_cache)
    print(f"metacells: {adata.shape} from {os.path.basename(args.metacell_cache)}")

    all_results = [evaluate(path, args, adata, device, rng) for path in args.checkpoints]

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(all_results, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
