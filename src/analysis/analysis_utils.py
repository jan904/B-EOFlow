import os
import gc
import sys
import torch
import scvi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.analysis.me_metrics import get_jacobian, get_MPMI, get_manifold_entropy
from src.model.data_utils import (
    prepare_data,
    get_condition_shapes,
    AdataDataset,
    prepare_train_test_data,
)
from src.analysis.plotting import (
    compare_spectra,
    compare_latent_effects,
    plot_umap,
    plot_hist,
    plot_MPMI,
    plot_loss,
)
from src.utils.utils import (
    compute_latent_effect,
    compute_neighbors,
    compute_correlations,
    filter_xdata,
    augment_with_noise,
    load_scvi,
)
from src.model.build_model import get_INN
from sklearn.decomposition import PCA
import sys

from scipy.stats import skew, kurtosis


class FlowAnalyser:
    def __init__(
        self,
        adata,
        flow,
        kwargs_data,
        labels_key,
        control_label,
        plot_dir,
        csv_dir,
        checkpoint_path,
        metrics_loss=None,
        val_metrics_loss=None,
        conditions=None,
        subset=False,
        latent_sort=None,
        H_i=None,
        H=None,
        test_size=0.1,
    ):
        self.adata = adata
        self.flow = flow
        self.kwargs_data = kwargs_data
        self.device = kwargs_data["device"]
        self.dtype = kwargs_data["dtype"]
        self.sigma_noise = kwargs_data["sigma_noise"]
        self.labels_key = labels_key
        self.control_label = control_label
        self.plot_dir = plot_dir
        self.csv_dir = csv_dir
        self.checkpoint_path = checkpoint_path

        self.metrics_loss = metrics_loss
        self.val_metrics_loss = val_metrics_loss

        condition_shapes = None
        if conditions is not None:
            condition_shapes = get_condition_shapes(adata, conditions)
        self.condition_shapes = condition_shapes
        self.conditions = conditions

        self.jac_dec = None
        self.latent_sort = latent_sort
        self.H_i = H_i
        self.H = H

        self.pca = None
        self.x_pca = None

        self.test_size = test_size

        self.subset = subset

    @classmethod
    def from_checkpoint(
        cls,
        adata,
        dataset_name,
        labels_key,
        control_label,
        lam_MTC,
        sigma_noise,
        prefix="",
        folder_postfix="",
        conditions=None,
        device=None,
        dtype=torch.float32,
        batch_size=1024,
        test_size=0.1,
    ):
        if device == None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dataset, train_dataloader, test_dataset, test_dataloader = prepare_train_test_data(
            adata,
            batch_size,
            device,
            dtype,
            label_key=conditions,
            test_size=test_size,
        )
        D_dim = dataset.X.shape[1]
        N_dim = D_dim

        model_path, plot_dir, log_dir, csv_dir = FlowAnalyser._build_dirs(
            dataset_name, lam_MTC, sigma_noise, N_dim, prefix=prefix, folder_postfix=folder_postfix
        )
        kwargs_data, kwargs_loss, metrics_loss = FlowAnalyser._build_kwargs(
            device,
            dtype,
            N_dim,
            D_dim,
            train_dataloader,
            test_dataloader,
            dataset,
            sigma_noise,
            lam_MTC,
        )

        if os.path.exists(model_path):
            flow, _, metrics_loss, val_metrics_loss = get_INN_from_checkpoint(
                model_path, device=device
            )
        else:
            raise FileNotFoundError(f"Checkpoint not found at {model_path}")

        if adata.obsm.get("X_scVI_un") is None:
            load_scvi(adata, dataset_name, N_dim)

        return FlowAnalyser(
            adata,
            flow,
            kwargs_data,
            labels_key,
            control_label,
            plot_dir,
            csv_dir,
            metrics_loss=metrics_loss,
            val_metrics_loss=val_metrics_loss,
            conditions=conditions,
            checkpoint_path=model_path,
        )

    @staticmethod
    def _build_dirs(dataset_name, lam_MTC, sigma_noise, N_dim, prefix="", folder_postfix=""):
        model_name = f"{prefix}MTC_{lam_MTC}_sigma_{sigma_noise}_model.pt"
        output_dir_name = f"{prefix}MTC_{lam_MTC}_sigma_{sigma_noise}"

        base_model_path = os.path.join(
            "/g/stegle/jhoefer/models/EOFlow",
            dataset_name,
            "top_" + str(N_dim) + folder_postfix,
        )
        base_output_dir_path = os.path.join(
            "/home/jhoefer/sandbox/results",
            dataset_name,
            "top_" + str(N_dim) + folder_postfix,
        )

        plot_dir = os.path.join(base_output_dir_path, "plots", output_dir_name)
        log_dir = os.path.join(base_output_dir_path, "logs", output_dir_name)
        csv_dir = os.path.join(base_output_dir_path, "csv", output_dir_name)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        return os.path.join(base_model_path, model_name), plot_dir, log_dir, csv_dir

    @staticmethod
    def _build_kwargs(
        device,
        dtype,
        N_dim,
        D_dim,
        train_dataloader,
        test_dataloader,
        dataset,
        sigma_noise,
        lam_MTC,
    ):
        kwargs_data = {
            "device": device,
            "dtype": dtype,
            "N_dim": N_dim,
            "D_dim": D_dim,
            "train_dataloader": train_dataloader,
            "test_dataloader": test_dataloader,
            "data_mean": dataset.X.mean(dim=0),
            "data_std": torch.ones(D_dim),
            "sigma_noise": sigma_noise,
            "sigma_inflate": sigma_noise,
        }

        kwargs_loss = {
            "use_NLL": True,
            "use_MER": True,
            "mode_MER": "unbiased",  #'full', #'unbiased'
            "lam_MTC": lam_MTC,
            "lam_ME_i": list(np.zeros(D_dim)),
            # 'use_align': False,
            # 'dims_align': D_dim - 1,
            # 'lam_align': 1, #0.1,
            # 'lam_disent': 0, #0.1,
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

        return kwargs_data, kwargs_loss, metrics_loss

    def compute_jacobian(self, x_input=None, calculate_entropy=True, N_samples=256, use_noise=True):
        if x_input is None:
            idx = torch.randperm(self.adata.X.shape[0])[:N_samples]
            x = self.adata.X.toarray() if hasattr(self.adata.X, "toarray") else self.adata.X
            x_input = torch.tensor(x, dtype=torch.float32, device=self.device)[idx]
            if use_noise:
                x_input = x_input + torch.randn_like(x_input) * self.sigma_noise

        jac_dec, ljd, z, x = get_jacobian(
            self.kwargs_data,
            self.flow,
            condition_shapes=self.condition_shapes,
            N_samples=N_samples,
            print_info=False,
            x_input=x_input,
            subtract_mean=False,
        )

        if calculate_entropy and not self.subset:
            with torch.no_grad():
                H_i, H, latent_sort = get_manifold_entropy(
                    self.kwargs_data,
                    jac_dec.detach(),
                    ljd.detach(),
                    z=z.detach(),
                    print_info=True,
                )
            self.latent_sort = latent_sort
            self.H_i = H_i
            self.H = H

        self.jac_dec = jac_dec

    def subset(self, covariates, keys):
        _, mask = filter_xdata(self.adata, covariates=covariates, keys=keys, device=self.device)

        if self.latent_sort is None:
            self.compute_jacobian()

        return FlowAnalyser(
            self.adata[mask],
            self.flow,
            self.kwargs_data,
            self.labels_key,
            self.control_label,
            self.plot_dir,
            self.csv_dir,
            self.checkpoint_path,
            conditions=self.conditions,
            subset=True,
            latent_sort=self.latent_sort,
            H_i=self.H_i,
            H=self.H,
        )

    def set_enrichment_scores(self, hm_names, hm_acts):
        self.hm_names = hm_names
        self.hm_acts = hm_acts

    def set_pca(self, pca, x_pca):
        self.pca = pca
        self.x_pca = x_pca


def get_losses_ablation(dir_name, key_position, device=None):

    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    losses = {}
    for model in os.listdir(dir_name):
        print(f"Evalutating model: {model}")
        checkpoint_path = os.path.join(dir_name, model)
        loss = get_loss_from_checkpoint(checkpoint_path, device)

        key = ""
        for pos in key_position:
            key += model.split("_")[pos] + "_"
        key = key[:-1]  # Remove the trailing underscore
        losses[key] = loss

    return losses


def get_loss_from_checkpoint(checkpoint_path, device, loss_key="loss"):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    gc.collect()
    torch.cuda.empty_cache()

    return checkpoint["metrics_loss"][loss_key]


def analyze_result(
    analyzer,
    log_scale=True,
    vmax=6,
    n_latent_factors=4,
    use_rep="X_scVI_un",
    computation_batch_size=1024,
):

    if analyzer.plot_dir is not None:
        os.makedirs(analyzer.plot_dir, exist_ok=True)

    # Perform PCA
    latent_sort_pca, adata, X_pca = compare_pca(
        analyzer,
        n_latent_factors=n_latent_factors,
    )

    # Compute latent effect and add to adata.obs
    adata, latent_distributions = compute_latent_effect(
        analyzer,
        n_latent_factors=n_latent_factors,
        batch_size=computation_batch_size,
    )

    pca_data_entropy, flow_data_entropy = compare_data_entropy(analyzer, latent_sort_pca)

    # Compare ME spectrum to PCA spectrum
    compare_spectra(
        analyzer,
        latent_sort_pca,
        pca_data_entropy,
        flow_data_entropy,
        log_scale=log_scale,
    )

    # Compute neighbor graph and UMAP using the specified representation (default: "X_scVI_un")
    adata = compute_neighbors(analyzer.adata, use_rep=use_rep)
    analyzer.adata = adata

    # Plot data colored by condition and cell type
    plot_umap(
        analyzer.adata, color=[f"{analyzer.labels_key}", "cell_type"], plot_dir=analyzer.plot_dir
    )

    # Plot data colored by latent effect of EOFlow and PCA
    compare_latent_effects(
        analyzer.adata,
        analyzer.latent_sort,
        plot_dir=analyzer.plot_dir,
        n_latent_factors=n_latent_factors,
        vmax=vmax,
    )

    # Plot Histogram of latent distributions
    plot_hist(latent_distributions, X_pca=X_pca, plot_dir=analyzer.plot_dir)

    if analyzer.metrics_loss is None:
        losses = get_loss_from_checkpoint(analyzer.checkpoint_path, device=analyzer.device)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    shared_kwargs = dict(
        filter_lim=3,
        filter_N=50,
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
        plot_dir=analyzer.plot_dir,
    )

    val_x = np.linspace(
        0, len(analyzer.metrics_loss["loss"]), len(analyzer.val_metrics_loss["loss"])
    )
    if analyzer.val_metrics_loss is None:
        plots = [
            (analyzer.metrics_loss["loss"], "training loss", "tab:blue", ax[0], None),
            (analyzer.metrics_loss["NLL"], "NLL loss", "tab:blue", ax[1], None),
            (analyzer.metrics_loss["MTC"], "MTC loss", "tab:orange", ax[1], None),
        ]
    else:
        plots = [
            (analyzer.metrics_loss["loss"], "training loss", "tab:blue", ax[0], None),
            (analyzer.val_metrics_loss["loss"], "validation loss", "tab:orange", ax[0], val_x),
            (analyzer.metrics_loss["NLL"], "NLL loss", "tab:blue", ax[1], None),
            (analyzer.metrics_loss["MTC"], "MTC loss", "tab:orange", ax[1], None),
        ]

    for data, label, color, axis, x_scale in plots:
        plot_loss(data, label, color=color, ax=axis, x_scale=x_scale, **shared_kwargs)


def analyze_model(
    adata,
    dataset_name,
    labels_key,
    control_label,
    lam_MTC,
    sigma_noise,
    device=None,
    dtype=None,
    prefix="",
    folder_postfix="",
    conditions=None,
    batch_size=512,
    umaps=True,
    MPMI=True,
    correlations=True,
):

    analyzer = FlowAnalyser.from_checkpoint(
        adata,
        dataset_name,
        labels_key,
        control_label,
        lam_MTC,
        sigma_noise,
        prefix=prefix,
        folder_postfix=folder_postfix,
        conditions=conditions,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    analyzer.compute_jacobian()

    if umaps:
        analyze_result(
            analyzer,
            log_scale=False,
            vmax=2,
            n_latent_factors=8,
            computation_batch_size=1024,
        )

    if correlations:
        correlations = compute_correlations(
            analyzer,
            label_keys=[f"{labels_key}", "cell_type"],
        )

        df = pd.DataFrame(
            {
                latent: dict(values)  # convert list of tuples to dict for each latent
                for latent, values in correlations.items()
            }
        )
        df.to_csv(os.path.join(analyzer.csv_dir, "correlations.csv"))
        df.style.background_gradient(cmap="coolwarm", axis=None, vmin=-1, vmax=1).format("{:.3f}")

    if MPMI:
        max_dim = min(50, analyzer.kwargs_data["N_dim"])
        MPMI_flow = get_MPMI(
            analyzer,
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
            save_dir=analyzer.plot_dir,
        )

    return analyzer.H_i, analyzer.latent_sort


def get_INN_from_checkpoint(
    checkpoint_path,
    device,
):

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model_config = checkpoint["model_config"]
    flow, optimizer_flow = get_INN(model_config)
    flow = flow.to(device)
    flow.load_state_dict(checkpoint["model_state_dict"])
    optimizer_flow.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint found at epoch {checkpoint['epoch']}.")
    metrics_loss = checkpoint["metrics_loss"]
    val_metrics_loss = checkpoint["val_metrics_loss"] if "val_metrics_loss" in checkpoint else None

    return flow, optimizer_flow, metrics_loss, val_metrics_loss


def return_bottleneck_representation(
    analyzer=None,
    adata=None,
    dataset_name=None,
    labels_key=None,
    control_label=None,
    device=None,
    dtype=None,
    sigma_noise=None,
    lam_MTC=None,
    conditions=None,
    prefix="",
    folder_postfix="",
    bottleneck_dim=100,
):

    if analyzer is None:
        required = [adata, dataset_name, labels_key, control_label, lam_MTC, sigma_noise]
        if any(arg is None for arg in required):
            raise ValueError(
                "Must provide either analyzer or all of: adata, dataset_name, labels_key, control_label, lam_MTC, sigma_noise"
            )
        analyzer = FlowAnalyser.from_checkpoint(
            adata,
            dataset_name,
            labels_key,
            control_label,
            lam_MTC,
            sigma_noise,
            prefix=prefix,
            folder_postfix=folder_postfix,
            conditions=conditions,
            device=device,
            dtype=dtype,
            batch_size=1024,
        )

    latent_representation = []
    dataloader = analyzer.kwargs_data["dataloader"]
    for x, c in dataloader:
        if conditions is not None and c != -1:
            c = [cond.to(device=analyzer.device, dtype=analyzer.dtype) for cond in c]
        else:
            c = None
        x = torch.tensor(x, device=analyzer.device, dtype=analyzer.dtype)

        z, _ = analyzer.flow(x, c=c, rev=False)
        latent_representation.append(z.cpu().detach().numpy())
    latent_representation = np.concatenate(latent_representation, axis=0)

    if analyzer.jac_dec == None:
        analyzer.compute_jacobian()

    latent_bottleneck_representation = latent_representation[
        :, analyzer.latent_sort[:bottleneck_dim]
    ]
    return latent_bottleneck_representation, analyzer.latent_sort


def compare_data_entropy(analyzer, pca_variance):

    if analyzer.H is None:
        analyzer.calculate_jacobian()

    H_pca = 1 / 2 * (analyzer.adata.n_vars + np.log(pca_variance).sum())
    H_flow = analyzer.H.mean()
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


def calculate_pca(adata, sigma_noise, n_noise_levels=1, n_components=None):
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()

    noise_levels = (
        [sigma_noise / x for x in range(1, n_noise_levels + 1)] if sigma_noise > 0 else []
    )
    X_augmented = augment_with_noise(X, noise_levels=noise_levels)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_augmented)

    return pca, X_pca


def compare_pca(analyzer, n_latent_factors=1, n_components=None, n_noise_levels=1):
    if analyzer.pca is None or analyzer.x_pca is None:
        pca, x_pca = calculate_pca(
            analyzer.adata,
            analyzer.sigma_noise,
            n_noise_levels=n_noise_levels,
            n_components=n_components,
        )
        analyzer.set_pca(pca, x_pca)

    # Store each component in adata.obs
    for k in range(n_latent_factors):
        analyzer.adata.obs[f"pca_{k}"] = analyzer.x_pca[: analyzer.adata.n_obs, k]

    return analyzer.pca.explained_variance_, analyzer.adata, analyzer.x_pca


def check_analyzer_params(
    analyzer, dataset_name, lam_MTC, sigma_noise, N_dim, prefix="", folder_postfix=""
):
    """Check if analyzer was initialized with these parameters."""
    if analyzer is None:
        return False

    expected_path, _, _, _ = FlowAnalyser._build_dirs(
        dataset_name, lam_MTC, sigma_noise, N_dim, prefix=prefix, folder_postfix=folder_postfix
    )

    return analyzer.checkpoint_path == expected_path
