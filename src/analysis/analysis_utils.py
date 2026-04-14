import os
import gc
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.analysis.me_metrics import get_jacobian, get_MPMI, get_manifold_entropy
from src.model.data_utils import prepare_data, get_condition_shapes, AdataDataset
from src.analysis.plotting import (
    compare_spectra,
    compare_latent_effects,
    plot_umap,
    plot_hist,
    plot_MPMI,
    plot_loss,
)
from src.utils.utils import (
    compare_pca,
    compute_latent_effect,
    compute_neighbors,
    compute_correlations,
)
from src.model.build_model import get_INN
import importlib
import sys

from scipy.stats import skew, kurtosis

try:
    importlib.reload(sys.modules["src.utils.utils"])
except KeyError:
    pass


def get_losses_ablation(dir_name, device):
    losses = {}
    for model in os.listdir(dir_name):
        print(f"Evalutating model: {model}")
        loss = get_loss_from_checkpoint(dir_name, model, device)

        lambda_MTC = model.split("_")[1]
        sigma_noise = model.split("_")[3]
        if sigma_noise not in losses:
            losses[sigma_noise] = {}
        losses[sigma_noise][lambda_MTC] = loss
    return losses


def get_loss_from_checkpoint(model_path, model_name, device, loss_key="loss"):
    checkpoint = torch.load(
        os.path.join(model_path, model_name),
        map_location=device,
        weights_only=False,
    )

    gc.collect()
    torch.cuda.empty_cache()

    return checkpoint["metrics_loss"][loss_key]


def analyze_result(
    adata,
    flow,
    latent_sort,
    H_i,
    H,
    N_dim,
    metrics_loss,
    conditions=None,
    plot_dir=None,
    device="cpu",
    log_scale=True,
    sigma_inflate=None,
    vmax=6,
    n_latent_factors=3,
    use_rep="X_scVI_un",
):

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)

    # Perform PCA
    latent_sort_pca, adata, X_pca = compare_pca(
        adata,
        n_latent_factors=n_latent_factors,
        n_components=N_dim,
        noise_level=sigma_inflate,
    )

    # Compute latent effect and add to adata.obs
    adata, latent_distributions = compute_latent_effect(
        flow,
        adata,
        latent_sort,
        conditions=conditions,
        n_latent_factors=n_latent_factors,
        device=device,
    )

    pca_data_entropy, flow_data_entropy = compare_data_entropy(N_dim, latent_sort_pca, H)

    # Compare ME spectrum to PCA spectrum
    compare_spectra(
        latent_sort,
        H_i,
        latent_sort_pca,
        pca_data_entropy,
        flow_data_entropy,
        log_scale=log_scale,
        plot_dir=plot_dir,
        sigma_inflate=sigma_inflate,
    )

    # Compute neighbor graph and UMAP using the specified representation (default: "X_scVI_un")
    adata = compute_neighbors(adata, use_rep=use_rep)

    # Plot data colored by condition and cell type
    plot_umap(adata, color=["condition", "cell_type"], plot_dir=plot_dir)

    # Plot data colored by latent effect of EOFlow and PCA
    compare_latent_effects(
        adata,
        latent_sort,
        plot_dir=plot_dir,
        n_latent_factors=n_latent_factors,
        vmax=vmax,
    )

    # Plot Histogram of latent distributions
    plot_hist(latent_distributions, X_pca=X_pca, plot_dir=plot_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_loss(
        metrics_loss["loss"],
        "loss",
        filter_lim=3,
        filter_N=50,
        color="tab:blue",
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
        ax=ax,
        plot_dir=plot_dir,
    )


def analyze_model(
    adata,
    model_path,
    output_dir_path,
    device,
    dtype,
    sigma_noise=0.8,
    lam_MTC=1.0,
    N_blocks=8,
    use_counts=False,
    conditions=None,
    batch_size=512,
    umaps=True,
    MPMI=True,
    correlations=True,
    prefix=None,
    pre_normalize=False,
    N_samples=256,
):
    model_name = "MTC_" + str(lam_MTC) + "_sigma_" + str(sigma_noise) + "_model.pt"
    output_dir_name = "MTC_" + str(lam_MTC) + "_sigma_" + str(sigma_noise)

    if prefix is not None:
        model_name = prefix + "_" + model_name
        output_dir_name = prefix + "_" + output_dir_name

    if use_counts:
        model_path += "_counts"
        output_dir_path += "_counts"

    if conditions is not None:
        model_path += "_cond"
        output_dir_path += "_cond"

    # model_path += "_ablation"
    # output_dir_path += "_ablation"
    print(f"Model: {model_name}")

    plot_dir = os.path.join(output_dir_path, "plots", output_dir_name)
    log_dir = os.path.join(output_dir_path, "logs", output_dir_name)
    csv_dir = os.path.join(output_dir_path, "csv", output_dir_name)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    condition_shapes = None
    if conditions is not None:
        condition_shapes = get_condition_shapes(adata, conditions)

    dataset, dataloader = prepare_data(
        adata,
        batch_size,
        device,
        dtype,
        counts=use_counts,
        label_key=conditions,
    )
    D_dim = dataset.X.shape[1]
    N_dim = D_dim

    kwargs_data = {
        "device": device,
        "dtype": dtype,
        "N_dim": N_dim,
        "D_dim": D_dim,
        "dataloader": dataloader,
        "data_mean": dataset.X.mean(dim=0),
        "data_std": dataset.X.std(dim=0),
        "sigma_noise": sigma_noise,
        "sigma_inflate": sigma_noise,
    }

    kwargs_loss = {
        "use_NLL": True,
        "use_MER": True,
        "mode_MER": "unbiased",
        "lam_MTC": lam_MTC,
        "lam_ME_i": list(np.zeros(D_dim)),
        "use_rec": False,
        "dims_rec": D_dim - 1,
        "lam_rec": 100,  #
        "sigma_noise": sigma_noise,
    }

    metrics_loss = {
        "epoch": [],
        "loss": [],
        "z2": [],
        "NLL": [],
        "MTC": [],
        "H_i": [],
        "H_core": [],
        "H_detail": [],
        "MI_core_detail": [],
        "L2_rec": [],
    }

    flow, optimizer_flow, metrics_loss = get_INN_from_checkpoint(
        model_dir=model_path,
        model_name=model_name,
        device=device,
        N_dim=N_dim,
        condition_shapes=condition_shapes,
        N_blocks=N_blocks,
        pre_normalize=pre_normalize,
    )

    x_input = torch.tensor(adata.X.toarray(), dtype=torch.float32, device=device)[:N_samples]

    jac_dec, ljd, z, x = get_jacobian(
        kwargs_data,
        flow,
        condition_shapes=condition_shapes,
        N_samples=256,
        print_info=False,
        x_input=x_input,
    )

    with torch.no_grad():
        H_i, H, latent_sort = get_manifold_entropy(
            kwargs_data, jac_dec.detach(), ljd.detach(), z=z.detach(), print_info=True
        )

    if umaps:
        analyze_result(
            adata,
            flow,
            latent_sort,
            H_i,
            H,
            N_dim,
            metrics_loss,
            conditions=conditions,
            plot_dir=plot_dir,
            device=device,
            log_scale=False,
            sigma_inflate=kwargs_data["sigma_inflate"],
            n_latent_factors=8,
            vmax=2,
        )

    if correlations:
        correlations = compute_correlations(
            adata,
            flow,
            latent_sort=latent_sort[:9],
            device=device,
            condition=conditions if conditions is not None else None,
            label_keys=["condition", "cell_type"],
        )

        df = pd.DataFrame(
            {
                latent: dict(values)  # convert list of tuples to dict for each latent
                for latent, values in correlations.items()
            }
        )
        df.to_csv(os.path.join(csv_dir, "correlations.csv"))
        df.style.background_gradient(cmap="coolwarm", axis=None, vmin=-1, vmax=1).format("{:.3f}")

    if MPMI:
        max_dim = min(50, kwargs_data["N_dim"])
        MPMI_flow = get_MPMI(
            kwargs_data,
            jac_dec[:, latent_sort],
            jac_dec[:, latent_sort],
            max_dim=max_dim,
            dtype=torch.float64,
            device="cpu",
            take_mean=False,
        )

        MPMI_flow = MPMI_flow + MPMI_flow.transpose(1, 2)

        plot_MPMI(
            MPMI_flow.mean(0).numpy(),
            z_max=max_dim,
            title=f"MPMI of flow",
            dim_label="latent dim",
            figsize=8,
            save_dir=plot_dir,
        )

    return H_i.detach().cpu(), latent_sort.cpu()


def get_INN_from_checkpoint(
    model_dir,
    model_name,
    device,
    N_dim=2000,
    N_blocks=12,
    condition_shapes=None,
    pre_normalize=False,
):

    checkpoint_path = os.path.join(model_dir, model_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    flow, optimizer_flow = get_INN(
        N_dim=N_dim,
        condition_shapes=condition_shapes,
        N_blocks=N_blocks,
        ch_hidden=2048,
        coupling_block_type="GLOW",
        RQS_bins=10,
        lr=1e-3,
        optimizer_type="schedulefree",
        warmup_steps=100,
        pre_normalize=pre_normalize,
    )
    flow = flow.to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    flow.load_state_dict(checkpoint["model_state_dict"])
    optimizer_flow.load_state_dict(checkpoint["optimizer_state_dict"])
    metrics_loss = checkpoint["metrics_loss"]

    return flow, optimizer_flow, metrics_loss


def get_ME_spectra_ablation(adata, dir_name, device, dtype, use_counts=False):
    H_is = {}
    latent_sorts = {}

    for model in os.listdir(dir_name):
        print(f"Evalutating model: {model}")
        model_path = os.path.join(dir_name, model)

        lambda_MTC = model.split("_")[1]
        sigma_noise = model.split("_")[3]

        if sigma_noise not in H_is:
            H_is[sigma_noise] = {}
            latent_sorts[sigma_noise] = {}

        H_i, latent_sort = analyze_model(
            adata,
            dir_name,
            "/home/jhoefer/sandbox/results",
            device,
            dtype,
            sigma_noise=sigma_noise,
            lam_MTC=lambda_MTC,
            use_counts=use_counts,
            umaps=False,
            MPMI=False,
            correlations=False,
        )
        H_is[sigma_noise][lambda_MTC] = H_i
        latent_sorts[sigma_noise][lambda_MTC] = latent_sort

    return H_is, latent_sorts


def return_bottleneck_representation(
    adata,
    model_dir,
    device,
    dtype,
    sigma_noise,
    lam_MTC,
    N_dim=2000,
    conditions=None,
    bottleneck_dim=100,
    pre_normalize=False,
):
    if conditions is not None:
        model_name = (
            conditions[0] + "_MTC_" + str(lam_MTC) + "_sigma_" + str(sigma_noise) + "_model.pt"
        )
    else:
        model_name = "MTC_" + str(lam_MTC) + "_sigma_" + str(sigma_noise) + "_model.pt"

    D_dim = N_dim

    dataset = AdataDataset(adata, label_key=conditions)
    dataloader = DataLoader(dataset, batch_size=adata.X.shape[0], shuffle=False)

    condition_shapes = get_condition_shapes(adata, conditions) if conditions is not None else None
    flow, _, _ = get_INN_from_checkpoint(
        model_dir,
        model_name,
        condition_shapes=condition_shapes,
        device=device,
        pre_normalize=pre_normalize,
    )

    kwargs_data = {
        "device": device,
        "dtype": dtype,
        "N_dim": N_dim,
        "D_dim": D_dim,
        "dataloader": dataloader,
        "data_mean": dataset.X.mean(dim=0),
        "data_std": dataset.X.std(dim=0),
        "sigma_noise": sigma_noise,
        "sigma_inflate": sigma_noise,
    }

    latent_representation = []
    for x, c in dataloader:
        if conditions is not None and c != -1:
            c = [cond.to(device=device, dtype=dtype) for cond in c]
        else:
            c = None
        x = torch.tensor(x, device=device, dtype=dtype)

        z, _ = flow(x, c=c, rev=False)
        latent_representation.append(z.cpu().detach().numpy())
    latent_representation = np.concatenate(latent_representation, axis=0)

    condition_shapes = None
    if conditions is not None:
        condition_shapes = get_condition_shapes(adata, conditions)

    jac_dec, ljd, z, x = get_jacobian(
        kwargs_data, flow, condition_shapes=condition_shapes, N_samples=128, print_info=False
    )

    with torch.no_grad():
        H_i, H, latent_sort = get_manifold_entropy(
            kwargs_data, jac_dec.detach(), ljd.detach(), z=z.detach(), print_info=True
        )

    latent_bottleneck_representation = latent_representation[:, latent_sort[:bottleneck_dim]]
    return latent_bottleneck_representation


def compare_data_entropy(N_dim, pca_variance, H):
    H_pca = 1 / 2 * (N_dim + np.log(pca_variance).sum())
    H_flow = H.mean()
    return H_pca, H_flow


def test_latent_shape(
    flow,
    adata,
    latent_sort,
    device,
    n_latent_factors=10,
    conditions=None,
    axes=None,
    plot_dir=None,
):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    adata, latent_distributions = compute_latent_effect(
        flow,
        adata,
        latent_sort,
        conditions=conditions,
        n_latent_factors=n_latent_factors,
        device=device,
        batch_size=1024,
    )

    skews = []
    kurtoses = []
    for k in range(n_latent_factors):
        skews.append(skew(latent_distributions[k].numpy()))
        kurtoses.append(kurtosis(latent_distributions[k].numpy()))

    axes[0].plot(range(n_latent_factors), skews, marker="o")
    axes[0].set_title("Skewness of latent distributions")
    axes[0].set_xlabel("Latent factor")
    axes[0].set_ylabel("Skewness")

    axes[1].plot(range(n_latent_factors), kurtoses, marker="o")
    axes[1].set_title("Kurtosis of latent distributions")
    axes[1].set_xlabel("Latent factor")
    axes[1].set_ylabel("Kurtosis")

    plt.tight_layout()
    if plot_dir is not None:
        plt.savefig(os.path.join(plot_dir, "latent_distribution_shapes.png"))
    plt.show()
