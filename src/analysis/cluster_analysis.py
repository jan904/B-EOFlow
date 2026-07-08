import os
import numpy as np
import torch
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from matplotlib.colors import LogNorm

from src.model.data_utils import create_latent_adata


def plot_latent_means_umap(analyzer, umap_2d=None):
    means = analyzer.model.means.detach().cpu().numpy()  # (K, D)

    dataset = analyzer.kwargs_data["dataset"]
    labels = dataset.cats[0].categories.tolist()  # product state labels for means

    latent_adata = create_latent_adata(analyzer.adata, analyzer.model, analyzer.device)
    Z = latent_adata.X

    # compute UMAP once and reuse across plots
    if umap_2d is None:
        Z_combined = np.vstack([Z, means])
        combined_adata = ad.AnnData(X=Z_combined.astype("float32"))
        sc.pp.neighbors(combined_adata)
        sc.tl.umap(combined_adata, random_state=42)
        combined_2d = combined_adata.obsm["X_umap"]
    else:
        combined_2d = umap_2d

    Z_2d = combined_2d[: len(Z)]
    means_2d = combined_2d[len(Z) :]

    # one plot per categorical: cats[0] = product, cats[1:] = individual keys
    cat_names = ["product"] + analyzer.conditions

    for cat, name in zip(dataset.cats, cat_names):
        cell_labels = np.array(cat)
        unique_labels = sorted(set(cell_labels))

        cmap = plt.cm.get_cmap("tab20", len(unique_labels))
        color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}

        fig, ax = plt.subplots(figsize=(10, 8))

        for label in unique_labels:
            mask = cell_labels == label
            ax.scatter(
                *Z_2d[mask].T,
                s=5,
                alpha=0.1,
                color=color_map[label],
                label=label,
                rasterized=True,
            )

        # for product plot, color means by their product label
        # for other plots, color means by the corresponding marginal label
        if name == "product":
            mean_colors = [color_map.get(l, "black") for l in labels]
        else:
            # extract the relevant part of the product label for this key
            key_idx = analyzer.conditions.index(name)
            mean_marginal = ["__".join(l.split("__")[key_idx : key_idx + 1]) for l in labels]
            mean_colors = [color_map.get(l, "black") for l in mean_marginal]

        ax.scatter(
            *means_2d.T,
            s=150,
            c=mean_colors,
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
            marker="*",
        )
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                means_2d[i],
                fontsize=7,
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points",
            )

        ax.set_title(f"Latent space UMAP — colored by {name}")
        legend = ax.legend(
            markerscale=3,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=7,
            ncol=max(1, len(unique_labels) // 25),
        )
        # override alpha on legend markers to full opacity
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
        plt.tight_layout()
        plt.savefig(
            os.path.join(analyzer.plot_dir, f"latent_means_umap_{name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()


def plot_pairwise_distances(analyzer, metric="euclidean", log_scale=False, order=None):
    means = analyzer.model.means.detach().cpu().numpy()  # (K, D)
    dataset = analyzer.kwargs_data["dataset"]
    labels = dataset.cats[0].categories.tolist()

    if order is not None:
        order = [
            "IL-4__CD4 Memory",
            "IL-32-beta__CD4 Memory",
            "IL-1-beta__CD4 Memory",
            "IL-2__CD4 Memory",
            "CD40L__CD4 Memory",
            "CT-1__CD4 Memory",
            "IFN-epsilon__CD4 Memory",
            "IL-13__CD4 Memory",
            "CD27L__CD4 Memory",
            "FGF-beta__CD4 Memory",
            "PBS__CD4 Memory",
        ]
        idx = [labels.index(l) for l in order]
    else:
        idx = list(range(len(labels)))

    # Pairwise Euclidean distances
    dist_matrix = squareform(pdist(means, metric=metric))
    dist_matrix = dist_matrix[np.ix_(idx, idx)]

    labels = [labels[i] for i in idx]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))

    if log_scale:
        im = ax.imshow(
            dist_matrix,
            cmap="viridis",
            norm=LogNorm(vmin=dist_matrix[dist_matrix > 0].min(), vmax=dist_matrix.max()),
        )
        log_mid = np.sqrt(dist_matrix[dist_matrix > 0].min() * dist_matrix.max())
        max_val = log_mid
    else:
        im = ax.imshow(dist_matrix, cmap="viridis")
        max_val = dist_matrix.max() * 0.5

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = dist_matrix[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if val > max_val else "black",
            )

    if log_scale:
        plt.colorbar(im, ax=ax, label="Distance", format="%.1f")
    else:
        plt.colorbar(im, ax=ax, label="Distance")
    ax.set_title("Pairwise distances")
    plt.tight_layout()
    plt.savefig(
        os.path.join(analyzer.plot_dir, "pairwise_distances.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_encoded_means_pca(analyzer):
    encoded_means, _ = analyzer.model(
        analyzer.model.means.to(analyzer.device),
        rev=False,
    )

    means_np = encoded_means.detach().cpu().numpy()  # shape: (n_components, latent_dim)

    pca = PCA(n_components=2)
    means_2d = pca.fit_transform(means_np)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(means_2d[:, 0], means_2d[:, 1], s=80, alpha=0.8)

    for i, (x, y) in enumerate(means_2d):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA of encoded means")
    plt.tight_layout()
    plt.show()
