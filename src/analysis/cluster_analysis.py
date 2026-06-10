import os
import numpy as np
import torch
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

from src.model.data_utils import create_latent_adata


def plot_latent_means_umap(analyzer):
    means = analyzer.flow.means.detach().cpu().numpy()  # (K, D)

    dataset = analyzer.kwargs_data["dataset"]
    labels = dataset.cats[0].categories.tolist()

    latent_adata = create_latent_adata(analyzer.adata, analyzer.flow, analyzer.device)
    Z = latent_adata.X  # (N, D)

    # embed everything together
    Z_combined = np.vstack([Z, means])  # (N+K, D)
    combined_adata = ad.AnnData(X=Z_combined.astype("float32"))

    sc.pp.neighbors(combined_adata)
    sc.tl.umap(combined_adata, random_state=42)

    combined_2d = combined_adata.obsm["X_umap"]
    Z_2d = combined_2d[: len(Z)]
    means_2d = combined_2d[len(Z) :]

    cell_labels = analyzer.adata.obs[analyzer.labels_key].values

    fig, ax = plt.subplots(figsize=(8, 7))
    for label in labels:
        mask = cell_labels == label
        ax.scatter(*Z_2d[mask].T, s=5, alpha=0.1, label=label, rasterized=True)

    ax.scatter(
        *means_2d.T,
        s=300,
        c=range(len(labels)),
        cmap="tab10",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
        marker="*",
    )
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            means_2d[i],
            fontsize=9,
            ha="center",
            va="bottom",
            xytext=(0, 8),
            textcoords="offset points",
            fontweight="bold",
        )

    ax.set_title("Latent space UMAP with learned Gaussian means")
    ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(analyzer.plot_dir, "latent_means_umap.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_pairwise_distances(analyzer):
    means = analyzer.flow.means.detach().cpu().numpy()  # (K, D)
    dataset = analyzer.kwargs_data["dataset"]
    labels = dataset.cats[0].categories.tolist()

    order = [
        "IL-4",
        "IL-32-beta",
        "IL-1-beta",
        "IL-2",
        "CD40L",
        "CT-1",
        "IFN-epsilon",
        "IL-13",
        "CD27L",
        "FGF-beta",
        "PBS",
    ]
    idx = [labels.index(l) for l in order]

    # Pairwise Euclidean distances
    dist_matrix = squareform(pdist(means, metric="cosine"))
    dist_matrix = dist_matrix[np.ix_(idx, idx)]

    labels = [labels[i] for i in idx]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(dist_matrix, cmap="viridis")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{dist_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if dist_matrix[i, j] > dist_matrix.max() * 0.5 else "black",
            )

    plt.colorbar(im, ax=ax, label="Distance")
    ax.set_title("Pairwise distances")
    plt.tight_layout()
    plt.savefig(
        os.path.join(analyzer.plot_dir, "pairwise_distances.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_encoded_means_pca(analyzer):
    encoded_means, _ = analyzer.flow(
        analyzer.flow.means.to(analyzer.device),
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
