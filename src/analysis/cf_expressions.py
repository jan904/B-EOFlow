import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from src.analysis.logistic_regression import generate_counterfactuals
from src.utils.utils import extract_top_genes


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
        x = np.log1p(x)

    mean_cfs = x_cfs.mean(dim=0).cpu().numpy()
    mean_real = x.mean(axis=0)

    plt.scatter(mean_cfs, mean_real)

    ctrl = np.linspace(0, max(mean_real.max(), mean_cfs.max()), 100)
    plt.plot(ctrl, ctrl, color="gray", linestyle="--")

    top_genes_cfs = mean_cfs[gene_indices]
    top_genes_real = mean_real[gene_indices]
    plt.scatter(top_genes_cfs, top_genes_real, color="red", label=f"top {top_n} genes for {key}")

    r_squared = linregress(mean_cfs, mean_real).rvalue ** 2
    plt.plot([], [], " ", label=f"$R^2$ = {r_squared:.3f}")
    plt.legend()

    plt.xlabel("ground truth")
    plt.ylabel("predicted")
    plt.title(f"Mean expression comparison for {key} ({method})")
    plt.tight_layout()
    plt.savefig(os.path.join(analyzer.plot_dir, f"mean_expression_comparison_{key}.png"), dpi=300)
    plt.show()


def plot_top_deg_violin(analyzer, key, cfs_fn=generate_counterfactuals, log1p_transform=False):
    top_gene_names, gene_indices = extract_top_genes(
        analyzer, key, top_n=1, log1p_transform=log1p_transform
    )
    gene_name = top_gene_names[0]
    gene_idx = gene_indices[0]

    def gene_values(adata_subset):
        x = adata_subset.X.toarray() if hasattr(adata_subset.X, "toarray") else adata_subset.X
        if log1p_transform:
            x = np.log1p(x)
        return x[:, gene_idx]

    control_values = gene_values(
        analyzer.adata[analyzer.adata.obs[analyzer.labels_key] == analyzer.control_label]
    )
    key_values = gene_values(analyzer.adata[analyzer.adata.obs[analyzer.labels_key] == key])

    x_cfs, z_cfs, all_source_labels, _ = cfs_fn(analyzer, key=key)
    # counterfactual = control cells predicted forward to `key`, matching the classic
    # ctrl -> stim comparison (rather than pooling counterfactuals from every source)
    cf_mask = np.asarray(all_source_labels) == analyzer.control_label
    cf_values = x_cfs[cf_mask, gene_idx].cpu().numpy()

    data = [control_values, key_values, cf_values]
    labels = [analyzer.control_label, key, f"{key} (counterfactual)"]

    plt.figure()
    plt.violinplot(data, showmeans=True)
    plt.xticks([1, 2, 3], labels)
    plt.ylabel("expression")
    plt.title(f"Top DEG for {key}: {gene_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(analyzer.plot_dir, f"top_deg_violin_{key}.png"), dpi=300)
    plt.show()
