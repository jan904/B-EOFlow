import scanpy as sc
import numpy as np
import torch

import requests
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.cluster import KMeans


class AdataDataset(Dataset):
    def __init__(self, adata, label_key=None, device=None, counts=False, dtype=torch.float32):
        super().__init__()

        if counts:
            assert "counts" in adata.layers, "Counts layer not found in adata.layers"
            X = adata.layers["counts"]
        else:
            X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        self.X = torch.tensor(X, dtype=dtype)

        self.y = None
        if label_key is not None:
            cats = []
            onehots = []
            for key in label_key:
                cat = pd.Categorical(adata.obs[key])
                cats.append(cat)
                codes = torch.tensor(cat.codes, dtype=torch.long)

                # Handle possible -1 (NaN categories)
                if (codes < 0).any():
                    raise ValueError(f"Found NaNs in {key}, cannot one-hot encode safely.")
                num_classes = len(cat.categories)

                onehot = F.one_hot(codes, num_classes=num_classes)  # (N, num_classes)
                onehots.append(onehot)

            self.y = onehots
            self.cats = cats

        self.device = device

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]

        if self.y is not None:
            y = [onehot[idx] for onehot in self.y]
            return x, y
        y = torch.ones(x.shape[0], dtype=torch.float32) * -1
        return x, y


# https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/
def load_parse_data(
    donors=["Donor1"],
    subset=[
        "CD40L",
        "FGF-beta",
        "CT-1",
        "IFN-epsilon",
        "IL-32-beta",
        "IL-1-beta",
        "IL-4",
        "CD27L",
        "IL-2",
        "IL-13",
    ],
    top_genes=2000,
    log_transform=True,
    cell_types=None,
):
    """
    Process the Parse dataset with options for subsetting and data filtering.

    Parameters:
        donor (list): Donors to subset to. Defaults to ['Donor1'].
        subset (bool): Whether to subset the data to specific cytokine labels. Defaults to True.
        top_genes (int): Number of highly variable genes to select. Defaults to 2000.
        log_transform (bool): Whether to apply log transformation. Defaults to True.
        cell_types (list): List of cell types to include. Defaults to None.

    Returns:
        tuple: (anndata.AnnData, labels_key, control_label) - The processed AnnData object,
               the key for perturbation labels, and the control label.
    """
    # Load the dataset
    adata = sc.read(
        "/g/stegle/jhoefer/data/parse.h5ad",
        backup_url="https://figshare.com/ndownloader/files/53372768",
        backed="r",
    )

    # Keys and labels
    labels_key = "cytokine"
    control_label = "PBS"

    # Subsample to one donor
    if donors is not None:
        adata = adata[adata.obs["donor"].isin(donors)]
    adata = adata.to_memory()

    # Optionally subset to specific cell types
    if cell_types is not None:
        adata = adata[adata.obs["cell_type"].isin(cell_types)].copy()

    # Store counts in layers
    adata.layers["counts"] = adata.X.copy()

    # Optionally subset to specific cytokines
    if subset:
        to_select = subset + [control_label]
        adata = adata[adata.obs[labels_key].isin(to_select)].copy()

    # Filter cells and genes
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_counts=10)

    # Find highly variable genes
    sc.pp.highly_variable_genes(
        adata, subset=True, n_top_genes=top_genes, flavor="seurat_v3", layer="counts"
    )

    if log_transform:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    return adata, labels_key, control_label


def load_kang_data(top_genes, log_transform=True):
    """
    Load and preprocess the {dataset_name} dataset.

    Returns:
    tuple: A tuple containing (adata, labels_key, control_label)
            - adata: AnnData object with processed data
            - labels_key: string indicating the column name for condition labels
            - control_label: string indicating the control condition label
    """

    # Load the data
    save_dir = os.path.join("/g/stegle/jhoefer/data", "kang_counts_25k.h5ad")
    adata = sc.read(save_dir, backup_url="https://figshare.com/ndownloader/files/34464122")

    # Set metadata
    labels_key = "condition"
    control_label = "ctrl"

    # Filter cells and genes
    sc.pp.filter_cells(adata, min_counts=1000)
    sc.pp.filter_genes(adata, min_cells=250)

    # Store the counts for later use
    adata.layers["counts"] = adata.X.copy()

    # Rename label to condition, replicate to patient
    adata.obs = adata.obs.rename({"label": "condition", "replicate": "patient"}, axis=1)

    print(adata)

    # assign sample
    adata.obs["sample"] = (
        adata.obs["condition"].astype("str") + "&" + adata.obs["patient"].str.slice(8, 13)
    )

    # set cell_types abbreviations (recommended given MOFA appends names)
    abbreviations = {
        "CD4 T cells": "CD4T",
        "B cells": "B",
        "NK cells": "NK",
        "CD8 T cells": "CD8T",
        "FCGR3A+ Monocytes": "FGR3",
        "CD14+ Monocytes": "CD14",
        "Dendritic cells": "DCs",
        "Megakaryocytes": "Mega",
    }
    adata.obs["cell_abbr"] = adata.obs["cell_type"].replace(abbreviations)

    # Find highly variable genes
    sc.pp.highly_variable_genes(
        adata, subset=True, n_top_genes=top_genes, flavor="seurat_v3", layer="counts"
    )

    if log_transform:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    return adata, labels_key, control_label


def load_data(
    dataset_name,
    top_genes,
    log_transform=True,
    cell_types=None,
    subset=[
        "CD40L",
        "FGF-beta",
        "CT-1",
        "IFN-epsilon",
        "IL-32-beta",
        "IL-1-beta",
        "IL-4",
        "CD27L",
        "IL-2",
        "IL-13",
    ],
):
    if dataset_name == "kang":
        adata, labels_key, control_label = load_kang_data(
            top_genes=top_genes, log_transform=log_transform
        )
    elif dataset_name == "parse":
        adata, labels_key, control_label = load_parse_data(
            top_genes=top_genes, log_transform=log_transform, cell_types=cell_types, subset=subset
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    adata.uns["load_params"] = {
        "dataset_name": dataset_name,
        "top_genes": top_genes,
        "log_transform": log_transform,
        "cell_types": cell_types,
        "subset": subset,
    }
    return adata, labels_key, control_label


def prepare_data(adata, batchsize, device, dtype, label_key=None, counts=False, shuffle=True):
    dataset = AdataDataset(adata, label_key=label_key, device=device, dtype=dtype, counts=counts)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
    return dataset, dataloader


def prepare_train_test_data(
    adata,
    batchsize,
    device,
    dtype,
    label_key=None,
    counts=False,
    test_size=0.2,
    seed=0,
):
    if not 0 < test_size < 1:
        if test_size == 0:
            dataset, dataloader = prepare_data(
                adata, batchsize, device, dtype, label_key=label_key, counts=counts
            )
            return (
                dataset,
                dataloader,
                None,
                None,
            )
        raise ValueError("test_size must be between 0 and 1.")

    n_obs = adata.n_obs
    if n_obs < 2:
        raise ValueError("Need at least two observations to create a train/test split.")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_obs)
    rng.shuffle(indices)

    split_idx = int(np.floor((1 - test_size) * n_obs))
    split_idx = min(max(split_idx, 1), n_obs - 1)

    train_adata = adata[indices[:split_idx]].copy()
    test_adata = adata[indices[split_idx:]].copy()

    train_dataset = AdataDataset(
        train_adata, label_key=label_key, device=device, dtype=dtype, counts=counts
    )
    test_dataset = AdataDataset(
        test_adata, label_key=label_key, device=device, dtype=dtype, counts=counts
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    return train_dataset, train_dataloader, test_dataset, test_dataloader


def get_condition_shapes(adata, conditions):
    condition_shapes = []
    for cond in conditions:
        if cond not in adata.obs.columns:
            raise ValueError(f"Condition '{cond}' not found in adata.obs columns.")
        else:
            n_unique = adata.obs[cond].nunique()
            condition_shapes.append(n_unique)
    return condition_shapes


def create_latent_adata(adata, flow, device, dtype=torch.float32):
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()

    dataset, dataloader = prepare_data(
        adata, batchsize=1024, device=device, dtype=dtype, shuffle=False
    )

    z_latent = []
    for X_batch, _ in dataloader:
        X_batch = X_batch.to(device)
        with torch.no_grad():
            z_batch, _ = flow(X_batch, rev=False)
        z_latent.append(z_batch.cpu().numpy())

    z_latent = np.concatenate(z_latent, axis=0)
    latent_adata = adata.copy()
    latent_adata.X = z_latent.astype("float32")

    return latent_adata


def build_metacells(
    adata,
    group_keys=["cell_type", "cytokine"],
    use_rep="X_pca",
    metacells_per_group=200,
):

    sc.pp.pca(adata, n_comps=50)
    metacells = []
    metacell_obs = []

    for group_vals, group_adata in adata.obs.groupby(group_keys):
        idx = group_adata.index
        sub = adata[idx]

        if len(sub) < 10:
            continue

        num_metacells = min(metacells_per_group, len(sub) // 4)
        X_rep = sub.obsm[use_rep]
        labels = KMeans(n_clusters=num_metacells, random_state=0).fit_predict(X_rep)

        for k in range(num_metacells):
            mask = labels == k
            if mask.sum() < 2:
                continue

            metacell_x = np.asarray(sub.layers["counts"][mask].sum(axis=0))
            metacells.append(metacell_x)

            obs_dict = {
                key: group_vals[i] if isinstance(group_vals, tuple) else group_vals
                for i, key in enumerate(group_keys)
            }
            obs_dict["n_cells"] = mask.sum()
            metacell_obs.append(obs_dict)

    X = np.vstack(metacells)
    meta_adata = sc.AnnData(X=X, var=adata.var, obs=pd.DataFrame(metacell_obs))

    meta_adata.layers["counts"] = meta_adata.X.copy()
    sc.pp.normalize_total(meta_adata, target_sum=1e4)
    sc.pp.log1p(meta_adata)

    return meta_adata
