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


def mmd(x, y, kernel=gaussian_kernel, sigma=1.0):
    n_x = x.size(0)
    n_y = y.size(0)

    xx = kernel(x, x, sigma=sigma)
    yy = kernel(y, y, sigma=sigma)
    xy = kernel(x, y, sigma=sigma)

    term_1 = torch.sum(xx) / (n_x * n_x)
    term_2 = torch.sum(yy) / (n_y * n_y)
    term_3 = 2 * torch.sum(xy) / (n_x * n_y)

    return term_1 + term_2 - term_3


def ctrl_mmd_dists(analyzer, key, iter=10, log1p_transform=False):
    """log1p_transform: set True when analyzer.adata holds raw counts (e.g. scGen)
    but the counterfactuals being compared against (elsewhere) are log-space, to
    keep the two comparable. Leave False when analyzer.adata is already log-space
    (INN, scVI wrapper)."""
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

        ctrl_mmd += mmd(x1, x2, sigma=sigma)

    ctrl_mmd /= iter

    return ctrl_mmd.item()


def mmd_dists(analyzer, key, sigma=1.0, cf_fn=generate_counterfactuals, log1p_transform=False):
    """log1p_transform: set True when both analyzer.adata and cf_fn's counterfactuals
    are raw counts (e.g. scVI, scGen), so both sides get library-size-normalized and
    log1p'd (via scanpy, matching load_data's preprocessing) before the MMD -
    otherwise they'd be compared on a raw-count scale with much larger variance than
    the already-log-space methods (INN, Means), which biases the kernel bandwidth
    (sigma) and inflates MMD values incomparably across methods. Leave False when
    analyzer.adata is already log-space."""
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

    mmd_value = mmd(x_real, x_cfs, sigma=sigma)

    keys = analyzer.adata.obs[analyzer.labels_key].unique()
    mmd_per_group = {}

    for k in tqdm(keys):
        mask = all_source_labels == k
        x_group = x_cfs[mask]
        if len(x_group) > 0:
            mmd_group = mmd(x_real, x_group, sigma=sigma)
            mmd_per_group[k] = mmd_group.item()

    return mmd_value.item(), mmd_per_group


def global_mmd_by_key(models, keys=None):
    """Global MMD (counterfactual vs. real, pooled over all source groups) for
    every method in `models` (the {"name", "analyzer", "cf_fn", "log1p_transform"}
    registry used in comparison.ipynb), across every held-out key.

    keys: cytokine keys to evaluate. If None, uses every non-control label found
    in each method's own analyzer (methods may be trained on different subsets).

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
                analyzer, key=key, cf_fn=m["cf_fn"], log1p_transform=m["log1p_transform"]
            )
            per_key[key] = global_mmd
        results[m["name"]] = per_key

    return results


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
