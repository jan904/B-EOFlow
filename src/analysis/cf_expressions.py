import os
import torch
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import t as t_dist

from src.analysis.logistic_regression import generate_counterfactuals
from src.utils.utils import extract_top_genes, normalize_log1p


def plot_mean_expression_comparison(
    analyzer,
    key,
    top_n=10,
    cfs_fn=generate_counterfactuals,
    log1p_transform=False,
    method="EOFlow",
    ax=None,
):
    top_gene_names, gene_indices = extract_top_genes(
        analyzer, key, top_n=top_n, log1p_transform=log1p_transform
    )

    x_cfs, z_cfs, all_source_labels, _ = cfs_fn(analyzer, key=key)

    x = analyzer.adata[analyzer.adata.obs[analyzer.labels_key] == key].copy()
    x = x.X.toarray() if hasattr(x.X, "toarray") else x.X
    if log1p_transform:
        x = normalize_log1p(x)
        x_cfs = normalize_log1p(x_cfs)

    mean_cfs = x_cfs.mean(dim=0).cpu().numpy()
    mean_real = x.mean(axis=0)

    ctrl = np.linspace(0, max(mean_real.max(), mean_cfs.max()), 100)

    # OLS fit of predicted (mean_cfs) on ground truth (mean_real), with a 95% CI
    # band for the regression estimate (i.e. for E[mean_cfs | mean_real], not for
    # individual new points).
    slope, intercept, _, _, _ = linregress(mean_real, mean_cfs)
    fit = slope * ctrl + intercept

    n = len(mean_real)
    x_bar = mean_real.mean()
    Sxx = np.sum((mean_real - x_bar) ** 2)
    residuals = mean_cfs - (slope * mean_real + intercept)
    mse = np.sum(residuals**2) / (n - 2)
    se_fit = np.sqrt(mse * (1 / n + (ctrl - x_bar) ** 2 / Sxx))
    ci = t_dist.ppf(0.975, df=n - 2) * se_fit

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots()

    ax.fill_between(
        ctrl,
        fit - ci,
        fit + ci,
        color="steelblue",
        alpha=0.2,
        zorder=1,
    )
    ax.plot(ctrl, fit, color="steelblue", zorder=2)
    ax.plot(ctrl, ctrl, color="gray", linestyle="--", zorder=2)

    ax.scatter(mean_real, mean_cfs, zorder=3)

    top_genes_cfs = mean_cfs[gene_indices]
    top_genes_real = mean_real[gene_indices]
    ax.scatter(
        top_genes_real, top_genes_cfs, color="red", zorder=4, label=f"top {top_n} genes for {key}"
    )

    for gene_name, gx, gy in zip(top_gene_names, top_genes_real, top_genes_cfs):
        ax.annotate(
            gene_name,
            (gx, gy),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="red",
        )

    r_squared_deg = linregress(mean_cfs[gene_indices], mean_real[gene_indices]).rvalue ** 2
    ax.plot([], [], " ", label=f"$R^2_{{DEGs}}$ = {r_squared_deg:.3f}")
    r_squared = linregress(mean_cfs, mean_real).rvalue ** 2
    ax.plot([], [], " ", label=f"$R^2_{{all}}$ = {r_squared:.3f}")
    ax.legend()

    ax.set_xlabel("ground truth")
    ax.set_ylabel("predicted")
    ax.set_title(f"Mean expression comparison for {key} ({method})")

    if standalone:
        plt.tight_layout()
        plt.savefig(
            os.path.join(analyzer.plot_dir, f"mean_expression_comparison_{key}.png"), dpi=300
        )
        plt.show()


def plot_effect_size_comparison(
    analyzer,
    key,
    top_n=10,
    cfs_fn=generate_counterfactuals,
    log1p_transform=False,
    method="EOFlow",
    ax=None,
):
    """Same as plot_mean_expression_comparison, but on effect sizes (mean(key) -
    mean(control)) instead of raw means, for both ground truth and predicted. Most
    genes barely respond to a perturbation, so raw expression is dominated by
    baseline level (highly expressed genes inflate R^2 even for a poor model);
    subtracting the same ground-truth control mean from both sides isolates the
    actual perturbation signal the model is supposed to capture.
    """
    top_gene_names, gene_indices = extract_top_genes(
        analyzer, key, top_n=top_n, log1p_transform=log1p_transform
    )

    x_cfs, z_cfs, all_source_labels, _ = cfs_fn(analyzer, key=key)

    x_key = analyzer.adata[analyzer.adata.obs[analyzer.labels_key] == key].copy()
    x_key = x_key.X.toarray() if hasattr(x_key.X, "toarray") else x_key.X
    if log1p_transform:
        x_key = normalize_log1p(x_key)
        x_cfs = normalize_log1p(x_cfs)

    x_ctrl = analyzer.adata[
        analyzer.adata.obs[analyzer.labels_key] == analyzer.control_label
    ].copy()
    x_ctrl = x_ctrl.X.toarray() if hasattr(x_ctrl.X, "toarray") else x_ctrl.X
    if log1p_transform:
        x_ctrl = normalize_log1p(x_ctrl)

    mean_ctrl = x_ctrl.mean(axis=0)
    mean_cfs = x_cfs.mean(dim=0).cpu().numpy()
    mean_real = x_key.mean(axis=0)

    # effect size = perturbed - control, ground-truth control mean used for both
    # sides so any mismatch reflects perturbation-effect quality, not baseline drift
    delta_cfs = mean_cfs - mean_ctrl
    delta_real = mean_real - mean_ctrl

    line_range = np.linspace(
        min(delta_real.min(), delta_cfs.min()), max(delta_real.max(), delta_cfs.max()), 100
    )

    # OLS fit of predicted (delta_cfs) on ground truth (delta_real), with a 95% CI
    # band for the regression estimate (i.e. for E[delta_cfs | delta_real], not for
    # individual new points).
    slope, intercept, _, _, _ = linregress(delta_real, delta_cfs)
    fit = slope * line_range + intercept

    n = len(delta_real)
    x_bar = delta_real.mean()
    Sxx = np.sum((delta_real - x_bar) ** 2)
    residuals = delta_cfs - (slope * delta_real + intercept)
    mse = np.sum(residuals**2) / (n - 2)
    se_fit = np.sqrt(mse * (1 / n + (line_range - x_bar) ** 2 / Sxx))
    ci = t_dist.ppf(0.975, df=n - 2) * se_fit

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots()

    ax.fill_between(
        line_range,
        fit - ci,
        fit + ci,
        color="steelblue",
        alpha=0.2,
        zorder=1,
    )
    ax.plot(line_range, fit, color="steelblue", zorder=2)
    ax.plot(line_range, line_range, color="gray", linestyle="--", zorder=2)

    ax.scatter(delta_real, delta_cfs, zorder=3)

    top_genes_cfs = delta_cfs[gene_indices]
    top_genes_real = delta_real[gene_indices]
    ax.scatter(
        top_genes_real, top_genes_cfs, color="red", zorder=4, label=f"top {top_n} genes for {key}"
    )

    for gene_name, gx, gy in zip(top_gene_names, top_genes_real, top_genes_cfs):
        ax.annotate(
            gene_name,
            (gx, gy),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="red",
        )

    r_squared_deg = linregress(delta_cfs[gene_indices], delta_real[gene_indices]).rvalue ** 2
    ax.plot([], [], " ", label=f"$R^2_{{DEGs}}$ = {r_squared_deg:.3f}")
    r_squared = linregress(delta_cfs, delta_real).rvalue ** 2
    ax.plot([], [], " ", label=f"$R^2_{{all}}$ = {r_squared:.3f}")
    ax.legend()

    ax.set_xlabel("ground truth effect size")
    ax.set_ylabel("predicted effect size")
    ax.set_title(f"Effect size comparison for {key} ({method})")

    if standalone:
        plt.tight_layout()
        plt.savefig(os.path.join(analyzer.plot_dir, f"effect_size_comparison_{key}.png"), dpi=300)
        plt.show()


def plot_top_deg_violin(
    analyzer, key, cfs_fn=generate_counterfactuals, log1p_transform=False, gene_idx=0, ax=None
):
    top_gene_names, gene_indices = extract_top_genes(
        analyzer, key, top_n=20, log1p_transform=log1p_transform
    )
    gene_name = top_gene_names[gene_idx]
    gene_idx = gene_indices[gene_idx]

    def gene_values(adata_subset):
        x = adata_subset.X.toarray() if hasattr(adata_subset.X, "toarray") else adata_subset.X
        if log1p_transform:
            x = normalize_log1p(x)
        return x[:, gene_idx]

    control_values = gene_values(
        analyzer.adata[analyzer.adata.obs[analyzer.labels_key] == analyzer.control_label]
    )
    key_values = gene_values(analyzer.adata[analyzer.adata.obs[analyzer.labels_key] == key])

    x_cfs, z_cfs, all_source_labels, _ = cfs_fn(analyzer, key=key)
    # counterfactual = control cells predicted forward to `key`, matching the classic
    # ctrl -> stim comparison (rather than pooling counterfactuals from every source)
    cf_mask = np.asarray(all_source_labels) == analyzer.control_label

    x_cfs_arr = x_cfs.cpu().numpy()
    if log1p_transform:
        # normalize the full per-cell profile (all genes) before slicing to gene_idx -
        # normalize_log1p's per-cell total only makes sense computed across all genes,
        # matching how gene_values() above normalizes before indexing into a single gene
        x_cfs_arr = normalize_log1p(x_cfs_arr)

    cf_values = x_cfs_arr[cf_mask, gene_idx]
    cf_values = cf_values.clip(min=0, max=np.quantile(cf_values, 0.99))

    cf_label = f"{key} (counterfactual)"
    order = [analyzer.control_label, cf_label, key]
    values = np.concatenate([control_values, cf_values, key_values]).astype(np.float32)
    groups = (
        [analyzer.control_label] * len(control_values)
        + [cf_label] * len(cf_values)
        + [key] * len(key_values)
    )

    violin_adata = ad.AnnData(
        X=values.reshape(-1, 1),
        var=pd.DataFrame(index=[str(gene_name)]),
        obs=pd.DataFrame({"group": pd.Categorical(groups, categories=order)}),
    )

    standalone = ax is None

    violin_ax = sc.pl.violin(
        violin_adata,
        keys=str(gene_name),
        groupby="group",
        order=order,
        use_raw=False,
        xlabel="",
        ylabel="expression",
        show=False,
        ax=ax,
    )
    violin_ax.set_title(f"Top DEG for {key}: {gene_name}")

    if standalone:
        plt.tight_layout()
        plt.savefig(os.path.join(analyzer.plot_dir, f"top_deg_violin_{key}.png"), dpi=300)
        plt.show()


def sample_from_means(analyzer, sigma=1.0):
    means = analyzer.model.means.detach().cpu().numpy()
    if analyzer.model.log_sigma is not None:
        sigmas = np.exp(analyzer.model.log_sigma.detach().cpu().numpy())
    else:
        sigmas = np.full_like(means, sigma)

    class_counts = np.bincount(analyzer.adata.obs[analyzer.labels_key].cat.codes)
    classes = analyzer.adata.obs[analyzer.labels_key].cat.categories.tolist()

    all_z = []
    all_labels = []
    all_cell_types = []
    for mean, sigma, class_name, count in zip(means, sigmas, classes, class_counts):
        z = np.random.normal(mean, sigma, size=(count, mean.shape[0]))
        all_z.append(z)
        all_labels.extend([class_name] * count)

        cell_type = (
            analyzer.adata.obs.loc[
                analyzer.adata.obs[analyzer.labels_key] == class_name, "cell_type"
            ]
            .mode()
            .iat[0]
        )
        all_cell_types.extend([cell_type] * count)

    z = torch.tensor(np.concatenate(all_z, axis=0), dtype=analyzer.dtype, device=analyzer.device)

    with torch.no_grad():
        x, _ = analyzer.model(z, rev=True)

    adata_sampled = ad.AnnData(
        X=x.cpu().numpy(),
        obs=pd.DataFrame(
            {
                analyzer.labels_key: pd.Categorical(all_labels, categories=classes),
                "cell_type": pd.Categorical(all_cell_types),
            }
        ),
        var=analyzer.adata.var.copy(),
    )

    return adata_sampled
