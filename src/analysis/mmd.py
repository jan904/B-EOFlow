import os

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.analysis.logistic_regression import generate_counterfactuals
from src.utils.utils import normalize_log1p


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
    x = analyzer.adata[analyzer.adata.obs[analyzer.labels_key] == key].copy()
    x = x.X.toarray() if hasattr(x.X, "toarray") else x.X
    if log1p_transform:
        x = normalize_log1p(x)

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
    x_cfs, z_cfs, all_source_labels, key_ = cf_fn(analyzer, key=key)
    all_source_labels = np.array(all_source_labels)

    adata_real = analyzer.adata[analyzer.adata.obs[analyzer.labels_key] == key].copy()
    x_real_arr = adata_real.X.toarray() if hasattr(adata_real.X, "toarray") else adata_real.X
    if log1p_transform:
        x_real_arr = normalize_log1p(x_real_arr)
    x_real = torch.tensor(x_real_arr, dtype=torch.float32)

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


def plot_mmd_dists(analyzer, ctrl_mmd, global_mmd, mmd_per_group, key):
    # sort by MMD value
    items = sorted(mmd_per_group.items(), key=lambda x: x[1], reverse=True)

    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(10, 5))

    plt.bar(labels, values)

    plt.axhline(
        global_mmd,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Global MMD ({key})",
    )

    plt.axhline(
        ctrl_mmd,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Control MMD (random split)",
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("MMD")
    plt.title(f"MMD: Counterfactual vs Real ({key})")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(analyzer.plot_dir, f"mmd_counterfactuals_{key}.png"), dpi=300)
    plt.show()
