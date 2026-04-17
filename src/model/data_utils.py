import scanpy as sc
import numpy as np
import torch

import requests
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


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
):
    """
    Process the Parse dataset with options for subsetting and data filtering.

    Parameters:
        donor (list): Donors to subset to. Defaults to ['Donor1'].
        subset (bool): Whether to subset the data to specific cytokine labels. Defaults to True.
        top_genes (int): Number of highly variable genes to select. Defaults to 2000.
        log_transform (bool): Whether to apply log transformation. Defaults to True.

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


def load_data(dataset_name, top_genes, log_transform=True):
    if dataset_name == "kang":
        return load_kang_data(top_genes=top_genes, log_transform=log_transform)
    elif dataset_name == "parse":
        return load_parse_data(top_genes=top_genes, log_transform=log_transform)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def prepare_data(adata, batchsize, device, dtype, label_key=None, counts=False):
    dataset = AdataDataset(adata, label_key=label_key, device=device, dtype=dtype, counts=counts)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    return dataset, dataloader


def get_condition_shapes(adata, conditions):
    condition_shapes = []
    for cond in conditions:
        if cond not in adata.obs.columns:
            raise ValueError(f"Condition '{cond}' not found in adata.obs columns.")
        else:
            n_unique = adata.obs[cond].nunique()
            condition_shapes.append(n_unique)
    return condition_shapes
