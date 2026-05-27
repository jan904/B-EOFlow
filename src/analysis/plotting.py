import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import scanpy as sc
import seaborn as sns
import pandas as pd
import torch
import os


def plot_ME_spectrum(
    H_i,
    latent_sort=None,
    H_i_gt=None,
    names=None,
    colors=None,
    log_scale=False,
    sigma_noise=None,
    title="Manifold Entropy Spectrum of flow",
    ax=None,
    figsize=5,
    plot_dir=None,
):

    SAVING = False
    if ax is None:
        SAVING = True
        fig, ax = plt.subplots(figsize=(2 * figsize, figsize))

    # check if H_i is list
    if isinstance(H_i, list):
        multiple = True
        N_dim = len(H_i[0])
        for H in H_i:
            assert len(H) == N_dim, "All H_i in the list must have the same length"
        H_i = torch.stack(H_i, dim=0)
        if latent_sort is not None:
            assert len(H_i) == len(
                latent_sort
            ), "Length of H_i list and latent_sort list must be the same"
    else:
        multiple = False
        N_dim = len(H_i)

    z_dims = np.arange(1, N_dim + 1)
    if latent_sort is not None:
        if multiple:
            for i in range(len(H_i)):
                H_i[i] = H_i[i][latent_sort[i]]
        else:
            H_i = H_i[latent_sort]

    ax.set_title(title)
    if multiple:
        for i in range(len(H_i)):
            label = f"H_i run {i+1}" if names is None else names[i]
            color = None if colors is None else colors[i]
            ax.plot(z_dims, H_i[i].numpy(), marker="o", label=label, color=color)
    else:
        label = "H_i" if names is None else names[0]
        color = None if colors is None else colors[0]
        ax.plot(z_dims, H_i.numpy(), marker="o", label=label, color=color)

    if H_i_gt is not None:
        ax.plot(
            z_dims[0 : H_i_gt.shape[0]],
            H_i_gt.numpy(),
            marker="x",
            label="H_i DGP",
            color="black",
            markersize=10,
        )

    if sigma_noise is not None:
        ax.axhline(
            y=np.log(sigma_noise) + 1 / 2,  # + np.log(np.sqrt(2 * np.pi)),
            color="tab:orange",
            linestyle="--",
            label="Noise level",
        )

    ax.legend()
    ax.set_xlabel(f'Latent Dimension Index {"(sorted)" if latent_sort is not None else ""}')

    if log_scale:
        ax.set_xscale("log")
    else:
        ax.set_xticks(z_dims[::50])
    ax.grid()

    if SAVING and plot_dir is not None:
        print("No axis provided, showing plot immediately.")
        plt.savefig(plot_dir)
        plt.show()

    return ax


def plot_PCA_spectrum(
    pca_explained_variance_,
    log_scale=False,
    sigma_noise=None,
    title="Manifold Entropy Spectrum of PCA",
    ax=None,
    figsize=5,
    names=None,
):

    H_i_pca = 1 / 2 + 1 / 2 * np.log(pca_explained_variance_)
    H_noise = 1 / 2 + np.log(sigma_noise)

    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize, figsize))

    z_dims = np.arange(1, len(pca_explained_variance_) + 1)
    ax.set_title(title)
    ax.plot(
        z_dims,
        H_i_pca,
        marker="o",
        label="PCA" if names is None else names[0],
    )
    ax.legend()
    ax.set_xlabel("Principal Component Index (sorted)")

    if log_scale:
        ax.set_xscale("log")
    else:
        ax.set_xticks(z_dims[::50])
    ax.grid()

    if sigma_noise is not None:
        ax.axhline(
            y=H_noise,
            color="tab:orange",
            linestyle="--",
            label="Noise level",
        )

    if ax is None:
        plt.show()

    return ax


def compare_spectra(
    analyzer,
    pca_explained_variance_,
    pca_data_entropy,
    flow_data_entropy,
    plot_name="spectra",
    title="",
    log_scale=True,
    ax=None,
):

    saving = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        saving = True

    plot_ME_spectrum(
        analyzer.H_i.detach().cpu(),
        latent_sort=analyzer.latent_sort.cpu(),
        log_scale=log_scale,
        sigma_noise=analyzer.sigma_noise,
        names=[f"EOFlow. Entropy: {flow_data_entropy:.2f}"],
        title=f"",
        figsize=5,
        ax=ax,
    )
    plot_PCA_spectrum(
        pca_explained_variance_,
        log_scale=log_scale,
        sigma_noise=analyzer.sigma_noise,
        names=[f"PCA. Entropy: {pca_data_entropy:.2f}"],
        title=f"",
        figsize=5,
        ax=ax,
    )

    if saving:
        plt.title("Manifold Entropy Spectra of EOFlow and PCA")
        plt.tight_layout()
        if analyzer.plot_dir is not None:
            plt.savefig(os.path.join(analyzer.plot_dir, plot_name))
        plt.show()
    else:
        ax.set_title(f"{title}")


def plot_umap(
    adata,
    color,
    plot_dir=None,
    plot_name="umap",
    cmap=None,
    vmin=None,
    vmax=None,
    title="UMAP",
    suptitle=None,
):

    fig = sc.pl.umap(
        adata,
        color=color,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show=False,
        return_fig=True,
    )

    # plt.suptitle(suptitle)
    if plot_dir is not None:
        fig.savefig(os.path.join(plot_dir, plot_name))
    plt.show()


def compare_latent_effects(
    adata, latent_sort, plot_dir=None, n_latent_factors=1, vmax=None, plot_name_suffix=""
):

    # Plot data colored by latent effect of PCA
    plot_umap(
        adata,
        color=[f"pca_{i}" for i in range(n_latent_factors)],
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        suptitle="UMAP colored by latent effect of PCA",
        plot_dir=plot_dir,
        plot_name=f"PCA{plot_name_suffix}",
    )

    # Plot data colored by latent effect of EOFlow
    plot_umap(
        adata,
        color=[f"latent_{ latent_sort[i]}_signed" for i in range(n_latent_factors)],
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        suptitle="UMAP colored by latent effect of EOFlow",
        plot_dir=plot_dir,
        plot_name=f"EOFlow{plot_name_suffix}",
    )


def plot_hist(latent_distribution, X_pca, plot_dir=None, plot_name="hist", plot_name_suffix=""):
    n_dims = len(latent_distribution)
    fig, axes = plt.subplots(2, n_dims, figsize=(5 * n_dims, 10))

    for k in range(n_dims):
        axes[0][k].hist(latent_distribution[k].numpy(), bins=50, color="tab:blue", alpha=0.7)
        axes[0][k].set_title(f"Histogram of latent dimension {k}")
        axes[0][k].set_xlabel("Latent value")
        axes[0][k].set_ylabel("Frequency")
        axes[0][k].grid()
        axes[1][k].hist(X_pca[:, k], bins=50, color="tab:orange", alpha=0.7)
        axes[1][k].set_title(f"Histogram of PCA component {k}")
        axes[1][k].set_xlabel("PCA value")
        axes[1][k].set_ylabel("Frequency")
        axes[1][k].grid()
    plt.tight_layout()

    if plot_dir is not None:
        plt.savefig(os.path.join(plot_dir, plot_name + plot_name_suffix))

    plt.show()


def plot_MPMI(
    MPMI_matrix, z_max, title="MPMI", dim_label="source dim", figsize=8, save_dir=None, **kwargs
):
    assert MPMI_matrix.ndim == 2, "MPMI_matrix must be 2-dimensional"
    fig, ax = plt.subplots(figsize=(figsize, figsize))  # Create a figure and axis object
    im = ax.imshow(MPMI_matrix, cmap="inferno", origin="lower", **kwargs)  # Plot the heatmap

    ax.set_title(title)
    ax.set_xlim(-0.5, z_max - 0.5)
    ax.set_ylim(-0.5, z_max - 0.5)
    ax.set_xticks(ticks=np.arange(0, z_max, step=1))
    ax.set_yticks(ticks=np.arange(0, z_max, step=1))
    ax.grid(False)
    ax.set_xlabel(dim_label)
    ax.set_ylabel(dim_label)

    plt.colorbar(im, ax=ax)  # Add a colorbar

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{title}.png"))

    plt.show()


def plot_loss(
    x,
    label,
    filter_lim,
    filter_N,
    color,
    alpha=0.4,
    width=2,
    extrapolate=False,
    plot_min_average=False,
    plot_original=True,
    N_min_average=None,
    alpha_min=None,
    alpha_smooth=1,
    linestyle="solid",
    plot_smooth=True,
    subsample=1,
    ax=None,
    x_scale=None,
    plot_dir=None,
):

    if ax is None:
        fig, ax = plt.subplots()

    x = np.array(x)
    colors = list(mcolors.TABLEAU_COLORS)
    if str(color).isnumeric():
        if int(color) == color:
            color = colors[color]
    if np.array_equal(x, np.zeros_like(x)) or np.array_equal(x, 1 / np.zeros_like(x)):
        return
    assert subsample * filter_N <= x.shape[0] // 2, "Decrease filter_N!"
    scale = np.arange(0, x.shape[0] / subsample, 1 / subsample)
    if x_scale is not None:
        scale = x_scale
    x_smooth = plot_gauss_filter(x, filter_lim, subsample * filter_N, extrapolate=extrapolate)
    if plot_smooth:
        ax.plot(
            scale,
            x_smooth,
            label=label,
            color=color,
            linewidth=width,
            linestyle=linestyle,
            alpha=alpha_smooth,
        )
        if plot_original:
            ax.plot(scale, x, color=color, alpha=alpha)
    elif plot_original:
        ax.plot(
            scale,
            x,
            color=color,
            label=label,
            linewidth=width,
            linestyle=linestyle,
            alpha=alpha,
        )

    # Plot a running average of minimum
    if plot_min_average:
        if N_min_average == None:
            N_min_average = subsample * filter_N
        x_min_smooth = []
        for i in range(x.shape[0]):
            i_min, i_max = i - N_min_average, i + N_min_average
            if i_min < 0:
                i_min = 0
            if i_max > x.shape[0]:
                i_max = x.shape[0]
            x_min = np.min(x[i_min:i_max])
            x_min_smooth.append(x_min)
        x_min_smooth = plot_gauss_filter(
            np.array(x_min_smooth),
            filter_lim,
            subsample * filter_N,
            extrapolate=extrapolate,
        )
        if alpha_min == None:
            alpha_min = alpha + 0.1
        ax.plot(
            scale,
            x_min_smooth,
            label=label + " min",
            color=color,
            linewidth=width,
            alpha=alpha_min,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.grid()
    ax.legend()

    if plot_dir is not None:
        plt.savefig(os.path.join(plot_dir, f"loss_curve.png"))

    return


def gaussian(x, mu=0, sig=1):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def plot_gauss_filter(x, filter_lim, filter_N, extrapolate=False):
    x_min_mean, x_max_mean = x[0:filter_N].mean(), x[-filter_N:].mean()
    if extrapolate:
        fit = np.polyfit(
            np.arange(-2 * filter_N, 0),
            x[-2 * filter_N :],
            2,
            rcond=None,
            full=False,
            w=None,
            cov=False,
        )
        p = np.poly1d(fit)
        x_extrapolate_r = p(np.arange(0 * filter_N, 1 * filter_N))
    else:
        x_extrapolate_r = np.repeat(x_max_mean, filter_N, axis=0)
    x_extrapolate_l = np.repeat(x_min_mean, filter_N, axis=0)
    x = np.concatenate((x_extrapolate_l, x, x_extrapolate_r), axis=None)
    min, max = 3 * filter_N // 2 - 1, -3 * filter_N // 2
    return np.convolve(
        x,
        gaussian(np.linspace(-filter_lim, filter_lim, filter_N))
        / np.sum(gaussian(np.linspace(-filter_lim, filter_lim, filter_N))),
    )[min:max]


def plot_correlation_distribution(
    correlations,
    latent_sort,
    filter_quantile=0.99,
    title="Correlation distribution per label",
):
    df = pd.DataFrame(
        {latent: dict(values) for latent, values in correlations.items()}
    ).T  # shape: (num_latents, num_labels)

    # Add sum across all latents per label
    df.loc["sum"] = (df**2).sum()

    num_labels = df.shape[1]
    ncols = 4
    nrows = int(np.ceil(num_labels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        ax = axes[i]
        vals = df[col].drop("sum").dropna()

        idx = vals.abs().idxmax()
        idx_num = int(idx.split("_")[1])
        position = (latent_sort == idx_num).nonzero(as_tuple=True)[0].item()

        # Filter by quantile around origin
        threshold = vals.abs().quantile(filter_quantile)  # adjust quantile as needed
        vals_filtered = vals[vals.abs() > threshold]
        total = df.loc["sum", col]

        sns.histplot(
            vals_filtered,
            ax=ax,
            kde=False,
            bins=30,
            color="steelblue",
            log_scale=(False, False),
        )
        ax.axvline(
            vals.mean(),
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"mean={vals.mean():.2f}",
        )

        # Highlight filtered-out region around origin
        ax.axvspan(
            -threshold,
            threshold,
            alpha=0.15,
            color="gray",
            label=f"filtered |r|<{threshold:.3f}",
        )
        ax.axvline(-threshold, color="gray", linestyle=":", linewidth=1)
        ax.axvline(threshold, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{col} | Top: latent_{position} (corr={vals[idx]:.2f})", fontsize=10)
        ax.set_xlabel("Correlation")
        ax.set_ylabel("Count")
        ax.text(
            0.97,
            0.97,
            f"sum={total:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
        )
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
