import scanpy as sc
import numpy as np
from scipy.stats import spearmanr
from scipy.sparse import issparse
import torch
import scvi
import os

from src.model.data_utils import prepare_data
from src.analysis.analysis_utils import calculate_pca

# ── DE evaluation helpers ────────────────────────────────────────────────────


def evaluate_de_results(de_dfs, perturbations, n):
    rhos, direction_agreements, top_overlaps, lfc_shifts = [], [], [], []
    for pert in perturbations:
        base = de_dfs[f"{pert}_original"]
        drop = de_dfs[f"{pert}_drop_{n}"]

        top_n = min(len(base), len(drop), 100)
        if top_n == 0:
            rhos.append(0)
            top_overlaps.append(0)
            direction_agreements.append(0)
            lfc_shifts.append(1)
            continue

        merged = base.merge(drop, on="names", suffixes=("_base", "_drop"))
        rho, _ = spearmanr(merged["logfoldchanges_base"], merged["logfoldchanges_drop"])
        dir_agreement = np.mean(
            np.sign(merged["logfoldchanges_base"]) == np.sign(merged["logfoldchanges_drop"])
        )
        top_base = set(base.sort_values("pvals_adj", kind="stable").head(top_n)["names"])
        top_drop = set(drop.sort_values("pvals_adj", kind="stable").head(top_n)["names"])
        top_overlap = len(top_base & top_drop) / top_n
        lfc_shift = np.mean(np.abs(merged["logfoldchanges_base"] - merged["logfoldchanges_drop"]))

        rhos.append(rho)
        direction_agreements.append(dir_agreement)
        top_overlaps.append(top_overlap)
        lfc_shifts.append(lfc_shift)

    return rhos, direction_agreements, top_overlaps, lfc_shifts


def _sig_filter(de_df):
    return de_df[(de_df["pvals_adj"] < 0.05) & (de_df["logfoldchanges"].abs() > 0.25)]


def _run_rank_genes(adata, labels_key, reference):
    sc.tl.rank_genes_groups(adata, groupby=labels_key, reference=reference, method="wilcoxon")


def _collect_de_results(adata, perturbations, key, de_dfs, de_sig_dfs):
    for pert in perturbations:
        de_df = sc.get.rank_genes_groups_df(adata, group=pert)
        de_dfs[f"{pert}_{key}"] = de_df
        de_sig_dfs[f"{pert}_{key}"] = _sig_filter(de_df)


# ── Ground truth DE ──────────────────────────────────────────────────────────


def evaluate_de_effects_gt(analyzer, perturbations, de_dfs, de_sig_dfs):
    _run_rank_genes(analyzer.adata, analyzer.labels_key, analyzer.control_label)
    _collect_de_results(analyzer.adata, perturbations, "original", de_dfs, de_sig_dfs)
    return de_dfs, de_sig_dfs


# ── Reconstruction helpers ───────────────────────────────────────────────────


def _reconstruct_flow(analyzer, dataloader, keep_dim, use_noise):
    x_recon = []
    for x_batch, _ in dataloader:
        x_batch = x_batch.to(device=analyzer.device, dtype=analyzer.dtype)
        if use_noise:
            x_batch = x_batch + torch.randn_like(x_batch) * analyzer.sigma_noise
        with torch.no_grad():
            z, _ = analyzer.flow(x_batch, rev=False)
            z[:, analyzer.latent_sort[keep_dim:]] = 0
            x_rec, _ = analyzer.flow(z, rev=True)
        x_recon.append(x_rec)
    return torch.cat(x_recon, dim=0).cpu().numpy().clip(min=0)


def _reconstruct_pca(analyzer, dataloader, keep_dim):
    x_recon = []
    for x_batch, _ in dataloader:
        z = analyzer.pca.transform(x_batch.cpu().numpy())
        z[:, analyzer.latent_sort[keep_dim:]] = 0
        x_recon.append(analyzer.pca.inverse_transform(z))
    return np.concatenate(x_recon, axis=0).clip(min=0)


def _reconstruct_vae(analyzer, keep_dim, batch_size=1024):

    base_path = "/g/stegle/jhoefer/models/scvi"
    model_dict = analyzer.plot_dir.split("/")[-2]

    dir = os.path.join(base_path, model_dict)
    path = os.path.join(dir, f"{keep_dim}_model.pt")
    checkpoint_exists = os.path.exists(path)

    adata_tmp = analyzer.adata.copy()

    if not checkpoint_exists:
        scvi.model.SCVI.setup_anndata(adata_tmp, layer="counts")
        model = scvi.model.SCVI(adata_tmp, n_latent=keep_dim)
        model.train(
            max_epochs=200,
            batch_size=batch_size,
            early_stopping=True,
            early_stopping_patience=10,
            plan_kwargs={"lr": 1e-3},
            check_val_every_n_epoch=5,
            enable_progress_bar=False,  # keep training quiet inside loops
        )
        model.save(dir)
        os.rename(os.path.join(dir, "model.pt"), path)  # rename to keep_dim_model.pt

    else:
        model = scvi.model.SCVI.load(path, adata=adata_tmp)

    x_recon = model.get_normalized_expression(
        adata=adata_tmp,
        library_size="latent",  # use model's learned library size
        n_samples=25,  # average over posterior samples for stability
        return_mean=True,
        batch_size=batch_size,
    ).values  # returns a DataFrame

    return x_recon.clip(min=0)


def _make_drop_adata(analyzer, X_recon):
    drop = analyzer.adata.copy()
    drop.X = X_recon
    return drop


# ── Main evaluation ──────────────────────────────────────────────────────────


def evaluate_de_effects(
    analyzer, perturbations, keep_dims, vae_bottlenecks, use_noise=False, batch_size=1024
):

    if analyzer.pca is None:
        pca, x_pca = calculate_pca(analyzer.adata, analyzer.sigma_noise)
        analyzer.set_pca(pca, x_pca)

    de_dfs, de_sig_dfs = {}, {}

    de_dfs, de_sig_dfs = evaluate_de_effects_gt(analyzer, perturbations, de_dfs, de_sig_dfs)
    de_dfs_pca, de_sig_dfs_pca = de_dfs.copy(), de_sig_dfs.copy()
    de_dfs_vae, de_sig_dfs_vae = de_dfs.copy(), de_sig_dfs.copy()

    dataset, dataloader = prepare_data(
        analyzer.adata,
        batchsize=batch_size,
        device=analyzer.device,
        dtype=analyzer.dtype,
        label_key=None,
        shuffle=False,
    )

    metrics = {k: [] for k in ("rhos", "dir_agreements", "top_overlaps", "lfc_shifts")}
    metrics_pca = {k: [] for k in ("rhos", "dir_agreements", "top_overlaps", "lfc_shifts")}
    metrics_vae = {k: [] for k in ("rhos", "dir_agreements", "top_overlaps", "lfc_shifts")}

    for keep_dim in keep_dims:
        # Reconstruct
        x_flow = _reconstruct_flow(analyzer, dataloader, keep_dim, use_noise)
        x_pca = _reconstruct_pca(analyzer, dataloader, keep_dim)

        # Run DE
        for x_recon, dfs, sig_dfs, tag in [
            (x_flow, de_dfs, de_sig_dfs, "flow"),
            (x_pca, de_dfs_pca, de_sig_dfs_pca, "pca"),
            (x_vae, de_dfs_vae, de_sig_dfs_vae, "vae"),
        ]:
            drop = _make_drop_adata(analyzer, x_recon)
            _run_rank_genes(drop, analyzer.labels_key, analyzer.control_label)
            _collect_de_results(drop, perturbations, f"drop_{keep_dim}", dfs, sig_dfs)

        # Collect metrics
        for m, val in zip(
            metrics.values(), evaluate_de_results(de_sig_dfs, perturbations, keep_dim)
        ):
            m.append(val)
        for m, val in zip(
            metrics_pca.values(),
            evaluate_de_results(de_sig_dfs_pca, perturbations, keep_dim),
        ):
            m.append(val)

    for keep_dim in vae_bottlenecks:
        x_vae = _reconstruct_vae(analyzer, keep_dim, batch_size)

        for x_recon, dfs, sig_dfs, tag in [
            (x_vae, de_dfs_vae, de_sig_dfs_vae, "vae"),
        ]:
            drop = _make_drop_adata(analyzer, x_recon)
            _run_rank_genes(drop, analyzer.labels_key, analyzer.control_label)
            _collect_de_results(drop, perturbations, f"drop_{keep_dim}", dfs, sig_dfs)

        for m, val in zip(
            metrics_vae.values(),
            evaluate_de_results(de_sig_dfs_vae, perturbations, keep_dim),
        ):
            m.append(val)

    return tuple(
        np.array(v) for v in [*metrics.values(), *metrics_pca.values(), *metrics_vae.values()]
    )
