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
    analyzer, key, top_n=10, cfs_fn=generate_counterfactuals, log1p_transform=False, method="EOFlow"
):
    top_gene_names, gene_indices = extract_top_genes(
        analyzer, key, top_n=top_n, log1p_transform=log1p_transform
    )

    x_cfs, z_cfs, all_source_labels, _ = cfs_fn(analyzer, key=key)

    x = analyzer.adata[analyzer.adata.obs[analyzer.labels_key] == key].copy()
    x = x.X.toarray() if hasattr(x.X, "toarray") else x.X
    if log1p_transform:
        x = normalize_log1p(x)

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

    plt.fill_between(
        ctrl,
        fit - ci,
        fit + ci,
        color="steelblue",
        alpha=0.2,
        zorder=1,
    )
    plt.plot(ctrl, fit, color="steelblue", zorder=2)
    plt.plot(ctrl, ctrl, color="gray", linestyle="--", zorder=2)

    plt.scatter(mean_real, mean_cfs, zorder=3)

    top_genes_cfs = mean_cfs[gene_indices]
    top_genes_real = mean_real[gene_indices]
    plt.scatter(
        top_genes_real, top_genes_cfs, color="red", zorder=4, label=f"top {top_n} genes for {key}"
    )

    for gene_name, gx, gy in zip(top_gene_names, top_genes_real, top_genes_cfs):
        plt.annotate(
            gene_name,
            (gx, gy),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="red",
        )

    r_squared_deg = linregress(mean_cfs[gene_indices], mean_real[gene_indices]).rvalue ** 2
    plt.plot([], [], " ", label=f"$R^2_{{DEGs}}$ = {r_squared_deg:.3f}")
    r_squared = linregress(mean_cfs, mean_real).rvalue ** 2
    plt.plot([], [], " ", label=f"$R^2_{{all}}$ = {r_squared:.3f}")
    plt.legend()

    plt.xlabel("ground truth")
    plt.ylabel("predicted")
    plt.title(f"Mean expression comparison for {key} ({method})")
    plt.tight_layout()
    plt.savefig(os.path.join(analyzer.plot_dir, f"mean_expression_comparison_{key}.png"), dpi=300)
    plt.show()


def plot_top_deg_violin(
    analyzer, key, cfs_fn=generate_counterfactuals, log1p_transform=False, gene_idx=0
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
    cf_values = (
        x_cfs[cf_mask, gene_idx]
        .cpu()
        .numpy()
        .clip(a_min=0, a_max=np.quantile(x_cfs[cf_mask, gene_idx].cpu().numpy(), 0.99))
    )

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

    ax = sc.pl.violin(
        violin_adata,
        keys=str(gene_name),
        groupby="group",
        order=order,
        use_raw=False,
        xlabel="",
        ylabel="expression",
        show=False,
    )
    ax.set_title(f"Top DEG for {key}: {gene_name}")
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
