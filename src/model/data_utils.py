import scanpy as sc
import anndata as ad
import numpy as np
import torch
from types import SimpleNamespace

import requests
import os
import itertools
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.cluster import KMeans


class AdataDataset(Dataset):
    def __init__(
        self,
        adata,
        label_key=None,
        device=None,
        counts=False,
        dtype=torch.float32,
        combo_categories=None,
        condition_categories=None,
    ):
        """
        combo_categories / condition_categories: explicit category vocabularies (see
        get_combo_categories / get_condition_categories) to one-hot encode against,
        instead of inferring categories from whichever rows are present in `adata`.
        Pass these (built from the full, pre-holdout data) so a combo with zero rows in
        this particular `adata` — e.g. a leave-one-combo-out holdout — still gets a
        reserved column/index, matching the model's mixture-prior `means` layout.
        """
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

            combined = adata.obs[label_key].astype(str).agg("__".join, axis=1)
            cat = pd.Categorical(combined, categories=combo_categories)
            cats.append(cat)
            codes = torch.tensor(cat.codes, dtype=torch.long)

            # Handle possible -1 (NaN categories)
            if (codes < 0).any():
                raise ValueError(
                    f"Found NaNs in combined {label_key}, cannot one-hot encode safely."
                )
            num_classes = len(cat.categories)

            onehot = F.one_hot(codes, num_classes=num_classes)  # (N, num_classes)
            onehots.append(onehot)

            for key in label_key:
                cond_categories = (
                    condition_categories.get(key) if condition_categories is not None else None
                )
                cat = pd.Categorical(adata.obs[key].astype(str), categories=cond_categories)
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

    def label_to_index(self, label, level=0):
        return self.cats[level].categories.get_loc(label)

    def index_to_label(self, index, level=0):
        return self.cats[level].categories[index]


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
    donors=["Donor1"],
):
    if dataset_name == "kang":
        adata, labels_key, control_label = load_kang_data(
            top_genes=top_genes, log_transform=log_transform
        )
    elif dataset_name == "parse":
        adata, labels_key, control_label = load_parse_data(
            top_genes=top_genes,
            log_transform=log_transform,
            cell_types=cell_types,
            subset=subset,
            donors=donors,
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


def prepare_data(
    adata,
    batchsize,
    device,
    dtype,
    label_key=None,
    counts=False,
    shuffle=True,
    combo_categories=None,
    condition_categories=None,
):
    dataset = AdataDataset(
        adata,
        label_key=label_key,
        device=device,
        dtype=dtype,
        counts=counts,
        combo_categories=combo_categories,
        condition_categories=condition_categories,
    )
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
    combo_categories=None,
    condition_categories=None,
):
    if not 0 < test_size < 1:
        if test_size == 0:
            dataset, dataloader = prepare_data(
                adata,
                batchsize,
                device,
                dtype,
                label_key=label_key,
                counts=counts,
                combo_categories=combo_categories,
                condition_categories=condition_categories,
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
        train_adata,
        label_key=label_key,
        device=device,
        dtype=dtype,
        counts=counts,
        combo_categories=combo_categories,
        condition_categories=condition_categories,
    )
    test_dataset = AdataDataset(
        test_adata,
        label_key=label_key,
        device=device,
        dtype=dtype,
        counts=counts,
        combo_categories=combo_categories,
        condition_categories=condition_categories,
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


def get_condition_vocab(adata, conditions):
    """Fixed category vocabulary for `conditions`: per-condition value lists, and the
    combo labels spanning their full Cartesian product.

    Compute this once from the full adata, before any holdout filtering, and pass both
    outputs into every AdataDataset/prepare_*_data call (train, holdout, and any later
    eval set) as `condition_categories`/`combo_categories`. That keeps the one-hot
    widths — and therefore the mixture-prior `means` size (n_clusters=len(combo_categories))
    — fixed regardless of which rows later get dropped for a leave-combo-out split, so a
    held-out combo still gets a reserved slot instead of vanishing from the vocabulary.

    Returns:
        condition_categories: dict {condition: sorted list of its unique string values}.
        combo_categories: list of '__'-joined combo labels, one per element of the
            Cartesian product of the per-condition value lists (same join order/format
            AdataDataset uses for its combined label column) — including combos with
            zero rows in `adata`, e.g. because the design isn't a full factorial.
    """
    condition_categories = {}
    for cond in conditions:
        if cond not in adata.obs.columns:
            raise ValueError(f"Condition '{cond}' not found in adata.obs columns.")
        condition_categories[cond] = sorted(adata.obs[cond].astype(str).unique().tolist())

    value_lists = [condition_categories[cond] for cond in conditions]
    combo_categories = ["__".join(combo) for combo in itertools.product(*value_lists)]

    return condition_categories, combo_categories


def create_latent_adata(
    adata,
    flow,
    device,
    conditions=None,
    condition_type=None,
    dtype=torch.float32,
    combo_categories=None,
    condition_categories=None,
    hard_condition=None,
    control_condition=None,
):
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()

    # label_key=conditions, not None: "normal" and "hybrid" feed a condition to the flow,
    # and without it `y_batch` is empty so the flow silently encodes unconditioned - a
    # different function from the one that was trained, with no error raised.
    needs_condition = conditions is not None and condition_type in ("normal", "hybrid")
    dataset, dataloader = prepare_data(
        adata,
        batchsize=1024,
        device=device,
        dtype=dtype,
        shuffle=False,
        label_key=list(conditions) if needs_condition else None,
        combo_categories=combo_categories,
        condition_categories=condition_categories,
    )

    hard_factor = 0
    if condition_type == "hybrid":
        from src.model.INN.INNs_model import _resolve_hard_factor, hybrid_condition_split

        hard_factor = _resolve_hard_factor(
            SimpleNamespace(
                condition_type="hybrid",
                conditions=list(conditions),
                hard_condition=hard_condition,
                control_condition=control_condition,
            )
        )

    z_latent = []
    for X_batch, y_batch in dataloader:
        c = None
        if needs_condition:
            c = [cond.to(device=device, dtype=dtype) for cond in y_batch]
            if condition_type == "hybrid":
                c, _ = hybrid_condition_split(c, hard_factor)
        X_batch = X_batch.to(device)
        with torch.no_grad():
            z_batch, _ = flow(X_batch, c=c, rev=False)
        z_latent.append(z_batch.cpu().numpy())

    z_latent = np.concatenate(z_latent, axis=0)
    latent_adata = adata.copy()
    latent_adata.X = z_latent.astype("float32")

    return latent_adata


def get_metacell_cache_path(
    dataset_name,
    top_genes,
    group_keys,
    donors=None,
    cache_dir="/g/stegle/jhoefer/data/metacells",
):
    """Deterministic `cache_path` for `build_metacells`, so every caller building
    metacells from the same (dataset, top_genes, donors, group_keys) combination -
    regardless of which model's own `log_transform` it wants back - resolves to the
    same file and shares one on-disk cache instead of re-running the clustering per notebook.
    """
    donor_part = "-".join(donors) if donors else "all"
    group_part = "-".join(group_keys)
    fname = f"{dataset_name}_top{top_genes}_donor-{donor_part}_groupby-{group_part}.h5ad"
    return os.path.join(cache_dir, fname)


def build_metacells(
    adata,
    group_keys=["cell_type", "cytokine"],
    use_rep="X_pca",
    cells_per_metacell=20,
    log_transform=True,
    cache_path=None,
    labels_key=None,
    control_label=None,
):
    """Groups by the exact `group_keys` combination first, then builds metacells
    within each group by k-means on `use_rep`, summing raw counts within each
    cluster - grouping before clustering, rather than clustering globally, is what
    guarantees every metacell is "pure": built only from cells sharing the same
    `group_keys` values, never mixing e.g. two different (cytokine, cell_type) combos.

    The clustering embedding is always computed from a log-normalized copy of
    `adata.layers["counts"]`, regardless of what `adata.X` holds on entry - grouping
    cells into metacells is a data-prep step, not part of any model, so it shouldn't
    depend on whichever `log_transform` a caller's own `load_data(...)` call happened
    to use (that used to make different callers cluster the same cells differently and
    end up with different metacell counts for the same held-out combo). `log_transform`
    here instead controls only the *returned* `.X` - `True` for models that read
    `.X` directly and want log-space (e.g. EOFlow, Means); `False` for models that
    need raw counts in `.X` because they have no separate `layer=` to point at instead
    (e.g. scGen - scVI is unaffected either way since it reads `.layers["counts"]`).

    `cache_path`: optional path to an `.h5ad` file caching the metacells *before* the
    `log_transform` step - since metacell construction never depends on `log_transform`
    (see above), one cache file is valid for every caller regardless of their own
    `log_transform` choice. If the file exists, it's loaded directly and clustering is
    skipped entirely (only the cheap `log_transform` step below still runs, and `adata`
    is never touched - it may be `None`); if not, metacells are built as usual from
    `adata` (required in this case) and the pre-`log_transform` result is written to
    `cache_path` for next time. Build the path with `get_metacell_cache_path` so calls
    for the same (dataset, top_genes, donors, group_keys) from different notebooks/
    models agree on the filename and actually share the cache. No caching if left as
    `None`.

    `labels_key`/`control_label`: optional passthrough (from `load_data`'s own return
    values), stashed into the cached file's `.uns` when a fresh cache is written, so a
    later cache hit can hand them straight back too. Prefer calling `load_metacells`
    instead of this function directly - it wires this all together, including
    skipping `load_data` itself (not just the clustering) on a cache hit.
    """

    if cache_path is not None and os.path.exists(cache_path):
        meta_adata = ad.read_h5ad(cache_path)
    else:
        if adata is None:
            raise ValueError("adata is required when cache_path is None or doesn't exist yet.")

        counts_adata = ad.AnnData(X=adata.layers["counts"].copy())
        sc.pp.normalize_total(counts_adata, target_sum=1e4)
        sc.pp.log1p(counts_adata)
        sc.pp.pca(counts_adata, n_comps=50)
        adata.obsm[use_rep] = counts_adata.obsm[use_rep]

        metacells = []
        metacell_obs = []

        for group_vals, group_adata in adata.obs.groupby(group_keys):
            idx = group_adata.index
            sub = adata[idx]

            if len(sub) < cells_per_metacell:
                continue

            num_metacells = len(sub) // cells_per_metacell
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
        meta_obs = pd.DataFrame(metacell_obs)
        meta_obs.index = meta_obs.index.astype(str)  # h5ad needs string obs_names
        meta_adata = ad.AnnData(X=X, var=adata.var.copy(), obs=meta_obs)
        meta_adata.layers["counts"] = meta_adata.X.copy()

        if cache_path is not None:
            if labels_key is not None:
                meta_adata.uns["labels_key"] = labels_key
            if control_label is not None:
                meta_adata.uns["control_label"] = control_label
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            meta_adata.write_h5ad(cache_path)

    if log_transform:
        sc.pp.normalize_total(meta_adata, target_sum=1e4)
        sc.pp.log1p(meta_adata)

    return meta_adata


def copy_with_log_transform(adata, log_transform=True, target_sum=1e4):
    """Returns a `.copy()` of `adata` (as returned by `build_metacells`/
    `load_metacells` with `log_transform=False`, i.e. `.X` holds raw counts) with
    `normalize_total`+`log1p` applied if `log_transform`, or an untouched copy
    otherwise.

    For a notebook like `comparison.ipynb` that needs the same metacells in both
    raw-count space (scGen, scVI) and log-space (EOFlow, Means), call
    `load_metacells(..., log_transform=False)` once for the shared raw-count base,
    then this per model - one on-disk cache read instead of four, and the cheap
    normalize+log1p is the only thing that reruns per model.
    """
    adata = adata.copy()
    if log_transform:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
    return adata


def load_metacells(
    dataset_name,
    top_genes,
    group_keys,
    donors=None,
    log_transform=True,
    cache_dir="/g/stegle/jhoefer/data/metacells",
):
    """High-level metacell loader for the USE_METACELLS branch in each model
    notebook/train.py, in place of calling `load_data` + `build_metacells` directly.

    If a cache for this (dataset_name, top_genes, donors, group_keys) combination
    already exists, `load_data` is skipped entirely - so the multi-GB source h5ad
    read, cell/gene filtering, and HVG selection never run, not just the clustering - and
    the metacells are returned straight from the cache, with labels_key/control_label
    (load_data's other two return values) recovered from the cache's own `.uns`. On a
    cache miss, this calls `load_data(dataset_name, top_genes, cell_types=None,
    donors=donors)` itself, builds the metacells, and writes the cache for next time.

    Returns `(adata, labels_key, control_label)`, matching `load_data`'s shape.
    """
    cache_path = get_metacell_cache_path(
        dataset_name, top_genes, group_keys, donors=donors, cache_dir=cache_dir
    )

    if os.path.exists(cache_path):
        meta_adata = build_metacells(
            None, group_keys=group_keys, cache_path=cache_path, log_transform=log_transform
        )
        return meta_adata, meta_adata.uns["labels_key"], meta_adata.uns["control_label"]

    # log_transform=False: build_metacells only ever reads adata.layers["counts"]
    # (always raw), never adata.X, so log-transforming it here would be pure wasted
    # compute on the full single-cell matrix - the actual log_transform choice below
    # is what controls the returned metacells' .X.
    adata, labels_key, control_label = load_data(
        dataset_name, top_genes, log_transform=False, cell_types=None, donors=donors
    )
    meta_adata = build_metacells(
        adata,
        group_keys=group_keys,
        cache_path=cache_path,
        log_transform=log_transform,
        labels_key=labels_key,
        control_label=control_label,
    )
    return meta_adata, labels_key, control_label


def split_holdout_combinations(adata, holdout_combos, group_keys=None):
    """Split off specific obs-column-value combinations for a leave-one-combo-out
    (LOCO) OOD evaluation, e.g. holdout_combos=[{"cytokine": "IL-2", "cell_type": "CD8 T cells"}]
    removes exactly the cells/metacells matching that combo from the training set.

    This only filters rows — it deliberately does not touch the combo vocabulary.
    Compute `condition_categories`/`combo_categories` (get_condition_categories /
    get_combo_categories) from the full `adata` *before* calling this, and pass them
    into every AdataDataset/prepare_*_data call (train, holdout, and any later eval
    set). That reserves a fixed slot for every combo in the Cartesian product of the
    conditions — including this held-out one — in both the one-hot encoding and the
    mixture-prior `means` (size it as `n_clusters=len(combo_categories)`). The held-out
    combo's mean row then simply never receives an EMA update from update_means_epoch
    (src/model/INN/INN_training.py), which already skips zero-count rows, instead of the
    row not existing at all and `mu = c @ means` breaking at eval time.
    """
    for combo in holdout_combos:
        for key in combo:
            if key not in adata.obs.columns:
                raise ValueError(f"Holdout key '{key}' not found in adata.obs columns.")
        if group_keys is not None and not set(combo).issubset(group_keys):
            raise ValueError(
                f"Holdout combo keys {list(combo)} must be a subset of group_keys {group_keys}."
            )

    mask = np.zeros(adata.n_obs, dtype=bool)
    for combo in holdout_combos:
        combo_mask = np.ones(adata.n_obs, dtype=bool)
        for key, value in combo.items():
            combo_mask &= (adata.obs[key] == value).to_numpy()
        if not combo_mask.any():
            raise ValueError(f"Holdout combo {combo} matched zero rows in adata.")
        mask |= combo_mask

    train_adata = adata[~mask].copy()
    holdout_adata = adata[mask].copy()

    return train_adata, holdout_adata
