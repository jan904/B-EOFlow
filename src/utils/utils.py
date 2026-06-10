import torch
import tqdm
from sklearn.decomposition import PCA
import scanpy as sc
import numpy as np
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
import torch
import pertpy as pt
from scipy.stats import pointbiserialr
import pandas as pd
from collections import defaultdict
from IPython.display import display

import scvi

# importlib.reload(sys.modules["src.utils.plotting"])
from src.analysis.plotting import (
    compare_spectra,
    compare_latent_effects,
    plot_umap,
    plot_hist,
)
from src.model.data_utils import AdataDataset


def augment_with_noise(X, noise_levels):
    """Augment X with multiple noise levels, appending to original data."""
    X_augmented = []  # start with original
    for noise_level in noise_levels:
        X_noisy = X + np.random.randn(*X.shape) * noise_level
        X_augmented.append(X_noisy)
    return np.concatenate(X_augmented, axis=0)


def compute_latent_effect(analyzer, batch_size=256, n_latent_factors=1):

    if analyzer.latent_sort is None or analyzer.conditions is not None:
        analyzer.compute_jacobian()

    latent_distributions = []
    for k in tqdm.tqdm(range(n_latent_factors)):
        effects = []
        for i in range(0, analyzer.adata.X.shape[0], batch_size):
            X_batch = analyzer.adata.X[i : i + batch_size]
            if hasattr(X_batch, "toarray"):
                X_batch = X_batch.toarray()
            X_batch = torch.tensor(X_batch, dtype=torch.float32).to(analyzer.device)
            if analyzer.conditions is not None and analyzer.condition_type == "normal":
                cat = pd.Categorical(analyzer.adata.obs[analyzer.conditions[0]][i : i + batch_size])
                codes = torch.tensor(cat.codes, dtype=torch.long)
                cond_batch = [F.one_hot(codes, num_classes=len(cat.categories)).to(analyzer.device)]
            else:
                cond_batch = None

            with torch.no_grad():
                z, _ = analyzer.flow(X_batch, c=cond_batch, rev=False)
                z_effect = z[:, analyzer.latent_sort[k]].clone()

            effects.append(z_effect.cpu())

        latent_effect = torch.cat(effects, dim=0)
        if n_latent_factors < 16:
            analyzer.adata.obs[f"latent_{ analyzer.latent_sort[k]}_signed"] = latent_effect.numpy()
        latent_distributions.append(latent_effect)
    return analyzer.adata, latent_distributions


def compute_TC(H_i, H, latent_sort, N_dim):
    H_i_sort = H_i[latent_sort].detach().cpu()  # compute Total Correlation
    H = H.mean().detach().cpu()
    TC = (H_i.sum() - H) / N_dim
    return TC


def compute_neighbors(adata, use_rep="X_scVI_un"):
    needs_neighbors = (
        "neighbors" not in adata.uns
        or "params" not in adata.uns["neighbors"]
        or adata.uns["neighbors"]["params"].get("use_rep") != use_rep
    )
    if needs_neighbors:
        sc.pp.neighbors(adata, use_rep=use_rep)
        sc.tl.umap(adata)
    return adata


def print_mem(msg):
    print(f"{msg}: {torch.cuda.memory_allocated()/1024**2:.1f} MiB")


def compute_counterfactuals(flow, adata, device, label=None, dim_to_flip=0):
    with torch.no_grad():
        if label is None:
            X = torch.tensor(adata.X.toarray(), dtype=torch.float32).to(device)
        else:
            X = torch.tensor(
                adata.X.toarray()[adata.obs["condition"] == label], dtype=torch.float32
            ).to(device)
        z, _ = flow(X, rev=False)
        z_cf = z.clone()
        z_cf[:, dim_to_flip] = -z_cf[:, dim_to_flip]  # flip the specified latent dimension
        X_cf, _ = flow(z_cf, rev=True)

    return X_cf.cpu().numpy()


def build_counterfactual_adata(flow, adata, device, labels=["stim", "ctrl"], dim_to_flip=0):
    adatas = []

    # original data
    adata_orig = adata.copy()
    adata_orig.obs["source"] = "real"
    adatas.append(adata_orig)

    for label in labels:
        X_cf = compute_counterfactuals(flow, adata, device, label=label, dim_to_flip=dim_to_flip)
        adata_subset = adata[adata.obs["condition"] == label].copy()
        adata_subset.X = X_cf

        # label as counterfactual of that condition
        adata_subset.obs["condition"] = f"{label}_cf"
        adata_subset.obs["source"] = "cf"

        adatas.append(adata_subset)

    # combine all
    adata_combined = adatas[0].concatenate(
        *adatas[1:],
        batch_key="batch",
        batch_categories=[f"part_{i}" for i in range(len(adatas))],
    )

    return adata_combined


def compute_counterfactual_distances(flow, adata, device, rep_key="X_pca", dim_to_flip=0):

    adata_combined = build_counterfactual_adata(
        flow, adata, device, labels=["stim", "ctrl"], dim_to_flip=dim_to_flip
    )
    sc.pp.pca(adata_combined)

    distance = pt.tl.Distance("edistance", obsm_key=rep_key)

    df = distance.pairwise(adata_combined, groupby="condition")

    return df


def compute_nearest_neighbor_distances(data, device, k=5, batch_size=1024, get_indices=False):
    if hasattr(data, "toarray"):
        data = data.toarray()
    data = torch.tensor(data, dtype=torch.float32).to(device)

    n_samples = data.shape[0]
    nearest_distances = torch.zeros((n_samples, k), dtype=torch.float32).to(device=device)
    nearest_indices = torch.zeros((n_samples, k), dtype=torch.long).to(device=device)

    for i in range(0, n_samples, batch_size):
        end_i = min(i + batch_size, n_samples)
        batch = data[i:end_i]
        # compute distances from batch to all data points
        dists = torch.cdist(batch.reshape(batch.shape[0], -1), data.reshape(data.shape[0], -1))
        # set diagonal to large value to ignore self-distance
        for j in range(end_i - i):
            dists[j, i + j] = float("inf")
        # #find nearest neighbors
        temp = torch.topk(dists, k=k, largest=False)
        nearest_distances[i:end_i] = temp.values
        nearest_indices[i:end_i] = temp.indices
    if get_indices:
        return nearest_distances.detach().cpu(), nearest_indices.detach().cpu()
    return nearest_distances.detach().cpu()


def latent_correlation_with_label(z_dim, adata, label_key):
    correlations = []
    label_values = adata.obs[label_key].values
    if len(np.unique(label_values)) == 2:
        label_values = (label_values == np.unique(label_values)[0]).astype(int)
        corr, p_value = pointbiserialr(label_values, z_dim.cpu().numpy())
        correlations.append((label_key, corr.item()))
    else:
        for label in np.unique(label_values):
            label_values_ = (label_values == label).astype(int)
            corr, p_value = pointbiserialr(label_values_, z_dim.cpu().numpy())
            correlations.append((f"{label}", corr.item()))
    return correlations


def compute_correlations(
    analyzer,
    n_latents=None,
    label_keys=["condition", "cell_type"],
):

    if analyzer.latent_sort is None:
        analyzer.compute_jacobian()

    if n_latents is None:
        n_latents = len(analyzer.latent_sort)

    dataset = AdataDataset(analyzer.adata, label_key=analyzer.conditions, device=analyzer.device)
    dataloader = DataLoader(dataset, batch_size=analyzer.adata.X.shape[0], shuffle=False)

    with torch.no_grad():
        for x, c in dataloader:
            x = x.clone().detach().to(device=analyzer.device, dtype=analyzer.dtype)
            if analyzer.conditions is not None and c != -1:
                c = [cond.to(device=analyzer.device, dtype=analyzer.dtype) for cond in c]
            else:
                c = None
            z, _ = analyzer.flow(x, c=c, rev=False)
            correlations = {}
            for k in range(n_latents):
                z_dim = z[:, analyzer.latent_sort[k]]
                correlations[f"latent_{analyzer.latent_sort[k]}"] = []
                for label_key in label_keys:
                    corr = latent_correlation_with_label(z_dim.cpu(), analyzer.adata, label_key)
                    correlations[f"latent_{analyzer.latent_sort[k]}"].extend(corr)
    return correlations


def reverse_dict(d):
    d2 = defaultdict(dict)
    for k1 in d.keys():
        for k2 in d[k1].keys():
            d2[k2][k1] = d[k1][k2]
    return d2


def display_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    display(
        df.style.background_gradient(cmap="coolwarm", axis=None, vmin=-1, vmax=1).format("{:.3f}")
    )


def filter_xdata(adata, covariates, keys, device):

    mask = np.ones(adata.shape[0], dtype=np.bool)
    if covariates is not None and keys is not None:
        for covariate, key in zip(covariates, keys):
            if key not in adata.obs[covariate].unique():
                raise ValueError(f"Key {key} not found in covariate {covariate}")
            mask &= adata.obs[covariate] == key

    x_data = adata[mask, :].X
    x_data = x_data.toarray()
    x_data = torch.tensor(x_data, dtype=torch.float32, device=device)

    return x_data[:256, :], mask


def load_scvi(adata, dataset_name, N_dim):

    if adata.obsm.get("X_scVI_un") is not None:
        return

    model_dir_unsupervised = os.path.join("/home/jhoefer/sandbox/models", "scvi_model_unsupervised")
    model_name = f"{dataset_name}_{N_dim}_model.pt"

    checkpoint_exists_unsupervised = os.path.exists(
        os.path.join(model_dir_unsupervised, model_name)
    )

    if not checkpoint_exists_unsupervised:
        raise FileNotFoundError(f"Checkpoint not found at {model_dir_unsupervised}")
    else:
        scvi_model_unsupervised = scvi.model.SCVI.load(
            model_dir_unsupervised,
            adata=adata,
            prefix=f"{dataset_name}_{N_dim}_",
        )

    SCVI_LATENT_KEY_UNSUPERVISED = "X_scVI_un"

    Z_scvi_unsupervised = scvi_model_unsupervised.get_latent_representation()
    adata.obsm[SCVI_LATENT_KEY_UNSUPERVISED] = Z_scvi_unsupervised


def check_adata_params(adata, dataset_name, top_genes, log_transform=True, cell_types=None):
    """Check if adata was loaded with these parameters."""
    if "load_params" not in adata.uns:
        return False

    params = adata.uns["load_params"]
    return (
        params["dataset_name"] == dataset_name
        and params["top_genes"] == top_genes
        and params["log_transform"] == log_transform
        and params["cell_types"] == cell_types
    )
