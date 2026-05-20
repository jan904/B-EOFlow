import scanpy as sc
import numpy as np
from scipy.stats import spearmanr
from scipy.sparse import issparse
import torch

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


def _make_drop_adata(analyzer, X_recon):
    drop = analyzer.adata.copy()
    drop.X = X_recon
    return drop


# ── Main evaluation ──────────────────────────────────────────────────────────


def evaluate_de_effects(analyzer, perturbations, keep_dims, use_noise=False, batch_size=1024):

    if analyzer.pca is None:
        pca, x_pca = calculate_pca(analyzer.adata, analyzer.sigma_noise)
        analyzer.set_pca(pca, x_pca)

    de_dfs, de_sig_dfs = {}, {}

    de_dfs, de_sig_dfs = evaluate_de_effects_gt(analyzer, perturbations, de_dfs, de_sig_dfs)
    de_dfs_pca, de_sig_dfs_pca = de_dfs.copy(), de_sig_dfs.copy()

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

    for keep_dim in keep_dims:
        # Reconstruct
        x_flow = _reconstruct_flow(analyzer, dataloader, keep_dim, use_noise)
        x_pca = _reconstruct_pca(analyzer, dataloader, keep_dim)

        # Run DE
        for x_recon, dfs, sig_dfs, tag in [
            (x_flow, de_dfs, de_sig_dfs, "flow"),
            (x_pca, de_dfs_pca, de_sig_dfs_pca, "pca"),
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

    return tuple(np.array(v) for v in [*metrics.values(), *metrics_pca.values()])
