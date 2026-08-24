import os

import numpy as np
import scanpy as sc
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.analysis.logistic_regression import generate_counterfactuals


def gaussian_kernel(x, y, sigma=1.0):
    dist = torch.cdist(x, y)
    return torch.exp(-(dist**2) / (2 * sigma**2))


def mmd(x, y, kernel=gaussian_kernel, sigma=1.0, unbiased=False):
    """Squared MMD between samples x (n, d) and y (m, d).

    unbiased=False (default, the "V-statistic" estimator): each within-sample term
    sums over all n^2/m^2 pairs including i=i'/j=j', where k(x_i,x_i)=1 always (for
    gaussian_kernel) - that gives each term a (1 - rho)/n-sized upward bias (rho =
    the true off-diagonal mean similarity) that shrinks but never fully vanishes as n
    grows, and differs between two MMD calls made at different n (e.g. ctrl_mmd_dists
    splitting an already-small real sample in half against mmd_dists comparing the
    full sample), which can distort comparisons across them.

    unbiased=True: the unbiased U-statistic estimator (Gretton et al., 2012,
    "A Kernel Two-Sample Test", eq. 3) - excludes the i=i'/j=j' diagonal from each
    within-sample term instead, so it isn't biased by sample size and stays
    comparable across MMD calls computed at different n. Requires n_x, n_y >= 2.
    """
    n_x = x.size(0)
    n_y = y.size(0)

    xx = kernel(x, x, sigma=sigma)
    yy = kernel(y, y, sigma=sigma)
    xy = kernel(x, y, sigma=sigma)

    if unbiased:
        if n_x < 2 or n_y < 2:
            raise ValueError("unbiased MMD requires at least 2 samples per side.")
        term_1 = (torch.sum(xx) - torch.trace(xx)) / (n_x * (n_x - 1))
        term_2 = (torch.sum(yy) - torch.trace(yy)) / (n_y * (n_y - 1))
    else:
        term_1 = torch.sum(xx) / (n_x * n_x)
        term_2 = torch.sum(yy) / (n_y * n_y)

    term_3 = 2 * torch.sum(xy) / (n_x * n_y)

    return term_1 + term_2 - term_3


def ctrl_mmd_dists(analyzer, key, iter=10, log1p_transform=False, unbiased=False):
    """log1p_transform: set True when analyzer.adata holds raw counts (e.g. scGen)
    but the counterfactuals being compared against (elsewhere) are log-space, to
    keep the two comparable. Leave False when analyzer.adata is already log-space
    (INN, scVI wrapper).

    unbiased: passed through to `mmd` - see its docstring. Matters most here, since
    this control/noise-floor estimate splits an already-small real sample in half,
    making the biased estimator's default sample-size artifact larger than in a
    same-key `mmd_dists` call comparing the full (unsplit) sample."""
    adata = analyzer.adata[analyzer.adata.obs[analyzer.labels_key] == key].copy()
    if log1p_transform:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    x = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    sigma = torch.median(
        torch.cdist(torch.tensor(x, dtype=torch.float32), torch.tensor(x, dtype=torch.float32))
    )

    ctrl_mmd = 0.0
    for _ in range(iter):
        n = x.shape[0]
        perm = np.random.permutation(n)

        half = n // 2
        idx1 = perm[:half]
        idx2 = perm[half:]

        x1 = torch.tensor(x[idx1], dtype=torch.float32)
        x2 = torch.tensor(x[idx2], dtype=torch.float32)

        ctrl_mmd += mmd(x1, x2, sigma=sigma, unbiased=unbiased)

    ctrl_mmd /= iter

    return ctrl_mmd.item()


def mmd_dists(
    analyzer, key, sigma=1.0, cf_fn=generate_counterfactuals, log1p_transform=False, unbiased=False
):
    """log1p_transform: set True when both analyzer.adata and cf_fn's counterfactuals
    are raw counts (e.g. scVI, scGen), so both sides get library-size-normalized and
    log1p'd (via scanpy, matching load_data's preprocessing) before the MMD -
    otherwise they'd be compared on a raw-count scale with much larger variance than
    the already-log-space methods (INN, Means), which biases the kernel bandwidth
    (sigma) and inflates MMD values incomparably across methods. Leave False when
    analyzer.adata is already log-space.

    unbiased: passed through to `mmd` - see its docstring. Per-group terms
    (mmd_per_group) with fewer than 2 cells are skipped when unbiased=True, since the
    unbiased estimator is undefined for n<2 (division by n*(n-1))."""
    x_cfs, z_cfs, all_source_labels, key_ = cf_fn(analyzer, key=key)
    all_source_labels = np.array(all_source_labels)

    adata_real = analyzer.adata[analyzer.adata.obs[analyzer.labels_key] == key].copy()
    if log1p_transform:
        sc.pp.normalize_total(adata_real, target_sum=1e4)
        sc.pp.log1p(adata_real)
    x_real_arr = adata_real.X.toarray() if hasattr(adata_real.X, "toarray") else adata_real.X
    x_real = torch.tensor(x_real_arr, dtype=torch.float32)

    if log1p_transform:
        adata_cfs = sc.AnnData(X=x_cfs.cpu().numpy())
        sc.pp.normalize_total(adata_cfs, target_sum=1e4)
        sc.pp.log1p(adata_cfs)
        x_cfs = torch.tensor(adata_cfs.X, dtype=torch.float32)

    x_cfs = x_cfs.to("cpu")
    x_real = x_real.to("cpu")

    sigma = torch.median(torch.cdist(x_real, x_real))

    mmd_value = mmd(x_real, x_cfs, sigma=sigma, unbiased=unbiased)

    keys = analyzer.adata.obs[analyzer.labels_key].unique()
    mmd_per_group = {}

    min_group_n = 2 if unbiased else 1
    for k in tqdm(keys):
        mask = all_source_labels == k
        x_group = x_cfs[mask]
        if len(x_group) >= min_group_n:
            mmd_group = mmd(x_real, x_group, sigma=sigma, unbiased=unbiased)
            mmd_per_group[k] = mmd_group.item()

    return mmd_value.item(), mmd_per_group


def global_mmd_by_key(models, keys=None, unbiased=False):
    """Global MMD (counterfactual vs. real, pooled over all source groups) for
    every method in `models` (the {"name", "analyzer", "cf_fn", "log1p_transform"}
    registry used in comparison.ipynb), across every held-out key.

    keys: cytokine keys to evaluate. If None, uses every non-control label found
    in each method's own analyzer (methods may be trained on different subsets).

    unbiased: passed through to `mmd_dists`/`mmd` - see their docstrings.

    Returns {method_name: {key: global_mmd}}.
    """
    results = {}
    for m in models:
        analyzer = m["analyzer"]
        method_keys = keys
        if method_keys is None:
            all_keys = analyzer.adata.obs[analyzer.labels_key].unique()
            method_keys = [k for k in all_keys if k != analyzer.control_label]

        per_key = {}
        for key in tqdm(method_keys, desc=m["name"]):
            global_mmd, _ = mmd_dists(
                analyzer,
                key=key,
                cf_fn=m["cf_fn"],
                log1p_transform=m["log1p_transform"],
                unbiased=unbiased,
            )
            per_key[key] = global_mmd
        results[m["name"]] = per_key

    return results


_LOG_MAX = 30.0  # above this a matrix cannot be log1p'd 1e4-normalized expression


def canonicalize(x, is_log=None):
    """One representation for every model: library-size normalized to 1e4, then log1p.

    `is_log=None` (the default) detects the space from the data instead of trusting a
    caller-supplied flag: log1p'd, 1e4-normalized expression tops out around 9-10, so
    anything above `_LOG_MAX` is linear. Worth having, because a flag that says "log" for
    count-scale data sends it through `expm1` and overflows to inf rather than failing
    loudly, and the per-model flags in comparison.ipynb are known to be unreliable for a
    model whose two sampling strategies decode in different spaces.

    Already-log input is sent back through `expm1` first so the renormalization means the
    same thing for it as for a raw-count or normalized-expression prediction. The point is
    that MMD values only compare across models if every matrix entering them lives in the
    same space - `mmd_dists`' median-heuristic bandwidth absorbs a pure rescaling, but not
    a change of representation: measured on real metacells, a like-for-like comparison
    scores ~0.04 within one space and ~0.60 across two.

    Note this normalizes library size away, so a prediction with the right composition but
    the wrong total scores as correct here; that is the price of putting count-space and
    log-space models on one axis.
    """
    x = x.X if hasattr(x, "X") else x
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    x = np.asarray(x, dtype=np.float32).copy()
    peak = float(np.nanmax(x))
    if is_log is None:
        is_log = peak <= _LOG_MAX
    elif is_log and peak > _LOG_MAX:
        raise ValueError(
            f"is_log=True but the matrix peaks at {peak:.1f}, which log1p'd normalized "
            f"expression cannot reach (> {_LOG_MAX}). expm1 would overflow to inf and the "
            "MMD would come back as nan - pass is_log=False, or leave it None to detect."
        )
    if not np.isfinite(x).all():
        raise ValueError(
            f"matrix contains {int((~np.isfinite(x)).sum())} non-finite entries before "
            "canonicalization - the model produced them, so fix it there rather than here."
        )

    if is_log:
        x = np.expm1(x)

    # The flow's decoder is unconstrained and emits negatives, but both spaces it is
    # trained on are non-negative, so these are decoding artefacts (INN_OOD._combo_adata
    # clamps them for the same reason). Left in, they cancel against the positives in the
    # library-size total, and a row summing to ~0 turns the normalization into a division
    # by nothing - which is where "input contains NaN" comes from downstream, not from the
    # model. Counterfactuals from `logistic_regression.generate_counterfactuals` are *not*
    # clamped by their caller, so this has to happen here.
    x = np.clip(x, 0.0, None)

    # normalize_total(target_sum=1e4) + log1p, done directly so an all-zero row stays an
    # all-zero row instead of dividing by 0.
    totals = x.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return np.log1p(x * 1e4 / totals).astype(np.float32)


_canonicalize = canonicalize  # legacy name, used across the analysis modules


def comparable_ood_mmd(real, predictions, real_is_log=None, unbiased=True, n_floor=30, seed=0):
    """MMD of each model's OOD prediction against the same real held-out cells, on one
    common footing: every matrix is canonicalized (see `_canonicalize`) and a **single**
    kernel bandwidth - the median pairwise distance among the canonical real cells - is
    shared by all of them, including the noise floor.

    `evaluate_leftout_combo_mmd` gives each model its own bandwidth from its own real
    cells in its own space, which is fine within a model but makes the bars in a
    cross-model plot incommensurable. Use this for that plot; use that one per model.

    `real`: array/AnnData of the real held-out-combo cells.
    `predictions`: {name: data} or {name: (data, is_log)}. `is_log` defaults to None,
    i.e. the space is detected per matrix (see `_canonicalize`), so a mixed set of
    log-space and count-space models needs no per-model bookkeeping; pass it explicitly
    only to override the detection.

    Returns (results, floor, sigma): {name: mmd}, the split-half MMD among the real cells
    (what a perfect prediction would score), and the shared bandwidth.
    """
    real_c = _canonicalize(real, real_is_log)
    real_t = torch.tensor(real_c, dtype=torch.float32)
    sigma = torch.median(torch.cdist(real_t, real_t))

    results = {}
    for name, item in predictions.items():
        data, is_log = item if isinstance(item, tuple) else (item, None)
        pred_t = torch.tensor(_canonicalize(data, is_log), dtype=torch.float32)
        results[name] = mmd(real_t, pred_t, sigma=sigma, unbiased=unbiased).item()

    rng = np.random.default_rng(seed)
    floor = []
    for _ in range(n_floor):
        perm = rng.permutation(len(real_t))
        half = len(real_t) // 2
        floor.append(mmd(real_t[perm[:half]], real_t[perm[half:]], sigma=sigma,
                         unbiased=unbiased).item())

    return results, float(np.mean(floor)), float(sigma)


def plot_mmd_boxplot(per_combo, normalize="do nothing", ax=None, plot_dir=None, title=None,
                     filename="ood_mmd_boxplot.png", annotate=True, yscale="linear"):
    """Distribution of MMD across several held-out combos, one box per model.

    `per_combo`: {combo_name: {series_name: mmd}}, e.g. the four models plus "gene-space"
    and "do nothing" for each combo. Missing entries are skipped rather than dropped.

    **Normalization is not optional here.** Each combo's MMD is computed at its own
    median-heuristic bandwidth, so the absolute values are on different scales - across
    ten combos they span 0.001 to 0.21, and a raw boxplot would be a plot of which combo
    has the widest cells rather than of which model predicts best. Dividing by that combo's
    own "do nothing" MMD gives a dimensionless quantity with a fixed meaning: 0 = perfect,
    1 = no better than returning the control cells untouched, >1 = actively worse. Pass
    normalize=None to see the raw values anyway.

    Points are overlaid on the boxes because ten combos is few enough that the individual
    values matter more than the quartiles, and paired across models by combo - a model can
    win on median while losing on most combos.

    `yscale="symlog"` for the raw (unnormalized) view, where values really do span orders
    of magnitude. Plain log is not an option in either view: the unbiased MMD estimator is
    centred on zero under the null and goes negative whenever a prediction sits at the
    noise floor, which is precisely the regime worth seeing. symlog is linear within
    `linthresh` of zero - set here from the data - and logarithmic beyond it.
    """
    import os

    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
    combos = list(per_combo)
    series = [s for s in dict.fromkeys(k for c in per_combo.values() for k in c)
              if s != normalize]

    values = {}
    for s in series:
        vals = []
        for combo in combos:
            v = per_combo[combo].get(s)
            denom = per_combo[combo].get(normalize) if normalize else 1.0
            if v is None or denom in (None, 0):
                continue
            vals.append(v / denom if normalize else v)
        values[s] = np.asarray(vals, dtype=float)

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(1.6 * len(series) + 3, 5))

    data = [values[s] for s in series]
    bp = ax.boxplot(data, tick_labels=series, patch_artist=True, widths=0.55,
                    showfliers=False, medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(0)
    for i, (s, color) in enumerate(zip(series, palette), start=1):
        jitter = rng.normal(0, 0.045, size=len(values[s]))
        ax.scatter(np.full(len(values[s]), i) + jitter, values[s], color=color,
                   edgecolor="black", linewidth=0.4, zorder=3, s=26)

    if normalize:
        ax.axhline(1.0, color="black", ls="--", lw=1.4, label=f"{normalize} (no prediction)")
        ax.axhline(0.0, color="grey", lw=1.0)
        ax.set_ylabel(f"MMD / {normalize} MMD  (lower is better)")
    else:
        ax.set_ylabel("MMD vs real held-out cells")

    if annotate:
        for i, s in enumerate(series, start=1):
            if len(values[s]):
                ax.annotate(f"med {np.median(values[s]):.2f}", (i, np.median(values[s])),
                            xytext=(10, 0), textcoords="offset points", fontsize=8, va="center")

    if yscale == "symlog":
        finite = np.concatenate([v for v in values.values() if len(v)])
        # linear band around zero sized to the smallest non-trivial magnitude, so the
        # at-the-floor points stay visible instead of being squashed onto the axis
        nonzero = np.abs(finite[np.abs(finite) > 0])
        linthresh = float(np.quantile(nonzero, 0.1)) if len(nonzero) else 1e-4
        ax.set_yscale("symlog", linthresh=max(linthresh, 1e-6))
    elif yscale != "linear":
        ax.set_yscale(yscale)

    ax.set_title(title or f"OOD MMD over {len(combos)} held-out combos")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=15)

    if standalone:
        plt.tight_layout()
        if plot_dir:
            plt.savefig(os.path.join(plot_dir, filename), dpi=300)
        plt.show()

    return ax, {s: values[s] for s in series}


def plot_comparable_ood_mmd(results, floor, ax=None, plot_dir=None, title=None, floor_sd=None):
    """Bar chart for `comparable_ood_mmd` - one bar per model, one shared noise floor.

    A single floor line (not one per model) is the whole point: the same real cells and
    the same bandwidth were used for every bar, so one line applies to all of them.
    """
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]

    names = list(results.keys())
    values = [results[n] for n in names]

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(7, 5))

    ax.bar(names, values, color=[palette[i % len(palette)] for i in range(len(names))])
    ax.axhline(floor, color="black", linestyle="--", linewidth=1.5,
               label=f"noise floor (random split of real) = {floor:.4f}")
    if floor_sd is not None:
        ax.axhspan(floor - floor_sd, floor + floor_sd, color="black", alpha=0.08)
    ax.axhline(0, color="grey", linewidth=0.8)

    for i, v in enumerate(values):
        ax.annotate(f"{v:.4f}", (i, v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9,
                    xytext=(0, 3 if v >= 0 else -3), textcoords="offset points")

    ax.set_ylabel("MMD vs real held-out cells (lower is better)")
    ax.set_title(title or "OOD prediction quality, common space and bandwidth")
    ax.legend()

    if standalone:
        plt.tight_layout()
        if plot_dir:
            plt.savefig(os.path.join(plot_dir, "ood_mmd_comparable.png"), dpi=300)
        plt.show()

    return ax


def plot_global_mmd_boxplot(mmd_by_method, ax=None, plot_dir=None):
    """Box plot of global MMD values (as returned by global_mmd_by_key), one box
    per method, distributed over its held-out keys, with the per-key values
    overlaid as jittered points."""
    # fixed categorical order (blue, orange, aqua, yellow) rather than a cycled
    # colormap, so a method keeps its color if the set of methods changes
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

    method_names = list(mmd_by_method.keys())
    data = [list(mmd_by_method[name].values()) for name in method_names]
    colors = [palette[i % len(palette)] for i in range(len(method_names))]

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(8, 5))

    bplot = ax.boxplot(
        data,
        tick_labels=method_names,
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(linewidth=1),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
    )

    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(0)
    for i, (values, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.normal(0, 0.04, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
            s=20,
        )

    ax.set_ylabel("Global MMD")
    ax.set_title("Global MMD (counterfactual vs. real) across held-out keys")

    if standalone:
        plt.tight_layout()
        if plot_dir:
            plt.savefig(os.path.join(plot_dir, "global_mmd_boxplot.png"), dpi=300)
        plt.show()

    return ax


def plot_mmd_dists(analyzer, ctrl_mmd, global_mmd, mmd_per_group, key, ax=None):
    # sort by MMD value
    items = sorted(mmd_per_group.items(), key=lambda x: x[1], reverse=True)

    labels = [k for k, _ in items]
    values = [v for _, v in items]

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.bar(labels, values)

    ax.axhline(
        global_mmd,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Global MMD ({key})",
    )

    ax.axhline(
        ctrl_mmd,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Control MMD (random split)",
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("MMD")
    ax.set_title(f"MMD: Counterfactual vs Real ({key})")
    ax.legend()

    if standalone:
        plt.tight_layout()
        plt.savefig(os.path.join(analyzer.plot_dir, f"mmd_counterfactuals_{key}.png"), dpi=300)
        plt.show()
