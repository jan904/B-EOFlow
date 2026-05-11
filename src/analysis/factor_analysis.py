from src.utils.utils import augment_with_noise
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scanpy as sc
import os

from src.utils.utils import filter_xdata
from src.analysis.me_metrics import get_jacobian
from src.analysis.analysis_utils import (
    calculate_pca,
)


def get_gene_importances(jac_dec, top_k=10, location=None):
    if location is None:
        gene_importances = torch.mean(torch.abs(jac_dec), dim=0)
    else:
        gene_importances = torch.abs(jac_dec)[location]
    top_genes = torch.topk(gene_importances, k=top_k, dim=0)
    return top_genes


def draw_single_factor_on_ax(ax, jac_dec, top_genes, factor_idx, gene_names, top_k=10):
    """Helper function to draw one lollipop plot on a provided axis."""
    # 1. Get indices and absolute values
    indices = top_genes.indices[:, factor_idx].cpu().numpy()
    abs_weights = top_genes.values[:, factor_idx].cpu().numpy()

    # 2. Get directionality (consensus sign across samples)
    raw_mean = jac_dec[:, indices, factor_idx].mean(dim=0).cpu().numpy()
    signs = np.sign(raw_mean)

    # 3. Get gene names
    names = [gene_names[i] for i in indices]

    # --- Plotting Logic ---
    y_pos = np.arange(top_k)[::-1]
    ax.hlines(y_pos, 0, abs_weights, color="black", linewidth=1.2)

    for i, (val, s) in enumerate(zip(abs_weights, signs)):
        marker = "+" if s >= 0 else "−"
        ax.plot(
            val,
            y_pos[i],
            "o",
            markersize=14,
            color="black",
            markerfacecolor="white",
            markeredgewidth=1.2,
        )
        ax.text(
            val,
            y_pos[i],
            marker,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title(f"Factor {factor_idx}", backgroundcolor="#D8ABAB", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="-", alpha=0.2)


def plot_factors_grid(jac_dec, top_genes, factor_indices, gene_names, top_k=10, n_cols=4):
    """Plots a grid of factor weights."""
    if isinstance(factor_indices, int):
        factor_indices = np.arange(factor_indices)
    n_factors = len(factor_indices)
    n_rows = int(np.ceil(n_factors / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = np.array(axes).flatten()  # Flatten in case of 1D or 2D array

    for i, factor_idx in enumerate(factor_indices):
        draw_single_factor_on_ax(axes[i], jac_dec, top_genes, factor_idx, gene_names, top_k)

    # Hide empty subplots if n_factors isn't a perfect multiple of n_cols
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_factors_for_covariate(
    analyzer,
    top_k=10,
    location=None,
    factor_indices=4,
):

    if analyzer.jac_dec is None:
        analyzer.compute_jacobian()

    top_genes = get_gene_importances(analyzer.jac_dec, top_k=top_k, location=location)
    plot_factors_grid(
        analyzer.jac_dec,
        top_genes,
        factor_indices=factor_indices,
        gene_names=analyzer.adata.var_names.tolist(),
        top_k=top_k,
    )


def pca_draw_single_factor_on_ax(ax, top_genes, factor_idx, gene_names, top_k=10):
    """Helper function to draw one lollipop plot on a provided axis."""
    # 1. Get indices and absolute values
    indices = top_genes.indices[factor_idx, :].cpu().numpy()
    weights = top_genes.values[factor_idx, :].cpu().numpy()

    # 2. Get directionality (consensus sign across samples)
    signs = np.sign(weights)

    weights = np.abs(weights)
    sort = np.argsort(weights)[::-1]
    weights = weights[sort]
    signs = signs[sort]
    indices = indices[sort]

    # 3. Get gene names
    names = [gene_names[i] for i in indices]

    # --- Plotting Logic ---
    y_pos = np.arange(top_k)[::-1]
    ax.hlines(y_pos, 0, weights, color="black", linewidth=1.2)

    for i, (val, s) in enumerate(zip(weights, signs)):
        marker = "+" if s >= 0 else "−"
        ax.plot(
            val,
            y_pos[i],
            "o",
            markersize=14,
            color="black",
            markerfacecolor="white",
            markeredgewidth=1.2,
        )
        ax.text(
            val,
            y_pos[i],
            marker,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title(f"Factor {factor_idx}", backgroundcolor="#D8ABAB", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="-", alpha=0.2)


def pca_plot_factors_grid(top_genes, factor_indices, gene_names, top_k=10, n_cols=4):
    """Plots a grid of factor weights."""

    if isinstance(factor_indices, int):
        factor_indices = np.arange(factor_indices)
    n_factors = len(factor_indices)
    n_rows = int(np.ceil(n_factors / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = np.array(axes).flatten()  # Flatten in case of 1D or 2D array

    for i, factor_idx in enumerate(factor_indices):
        pca_draw_single_factor_on_ax(axes[i], top_genes, factor_idx, gene_names, top_k)

    # Hide empty subplots if n_factors isn't a perfect multiple of n_cols
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def pca_get_gene_importances(analyzer, noise_level, top_k=10):

    if analyzer.pca == None or analyzer.x_pca is None:
        pca, x_pca = calculate_pca(
            analyzer.adata,
            analyzer.sigma_noise,
        )
        analyzer.set_pca(pca, x_pca)

    latent_order = np.argsort(analyzer.pca.explained_variance_)[::-1]

    pca_components_ = analyzer.pca.components_[latent_order][:10]
    pca_components_ = torch.tensor(pca_components_, dtype=torch.float32, device=analyzer.device)

    top_genes = torch.topk(pca_components_, k=top_k, dim=1)

    return top_genes
