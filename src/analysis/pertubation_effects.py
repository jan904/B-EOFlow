import scanpy as sc
import numpy as np
from scipy.stats import spearmanr
import torch

from src.model.data_utils import prepare_data


def evaluate_de_results(de_dfs, pertubations, n):
    rhos = []
    direction_agreements = []
    top_overlaps = []
    lfc_shifts = []

    for pert in pertubations:
        base = de_dfs[f"{pert}_original"]
        drop = de_dfs[f"{pert}_drop_{n}"]

        merged = base.merge(drop, on="names", suffixes=("_base", "_drop"))

        rho, _ = spearmanr(merged["logfoldchanges_base"], merged["logfoldchanges_drop"])

        direction_agreement = np.mean(
            np.sign(merged["logfoldchanges_base"]) == np.sign(merged["logfoldchanges_drop"])
        )

        top_n = 100

        top_base = set(base.sort_values("pvals_adj").head(top_n)["names"])
        top_drop = set(drop.sort_values("pvals_adj").head(top_n)["names"])

        top_overlap = len(top_base & top_drop) / top_n

        lfc_shift = np.mean(np.abs(merged["logfoldchanges_base"] - merged["logfoldchanges_drop"]))

        rhos.append(rho)
        direction_agreements.append(direction_agreement)
        top_overlaps.append(top_overlap)
        lfc_shifts.append(lfc_shift)

    return rhos, direction_agreements, top_overlaps, lfc_shifts


def evaluate_de_effects_gt(analyzer, pertubations, de_dfs, de_sig_dfs):

    sc.tl.rank_genes_groups(
        analyzer.adata,
        groupby=f"{analyzer.labels_key}",
        reference=analyzer.control_label,
        method="wilcoxon",
    )
    for pert in pertubations:
        de_df = sc.get.rank_genes_groups_df(analyzer.adata, group=pert)
        de_sig = de_df[(de_df["pvals_adj"] < 0.05) & (abs(de_df["logfoldchanges"]) > 0.25)]

        de_dfs[f"{pert}_original"] = de_df
        de_sig_dfs[f"{pert}_original"] = de_sig

    return de_dfs, de_sig_dfs


def evaluate_de_effects(analyzer, pertubations, keep_dims, batch_size=1024):

    de_dfs = {}
    de_sig_dfs = {}

    de_dfs, de_sig_dfs = evaluate_de_effects_gt(analyzer, pertubations, de_dfs, de_sig_dfs)

    rhos = []
    direction_agreements = []
    top_overlaps = []
    lfc_shifts = []

    z = []

    dataset, dataloader = prepare_data(
        analyzer.adata,
        batchsize=batch_size,
        device=analyzer.device,
        dtype=analyzer.dtype,
        label_key=None,
        shuffle=False,
    )

    for keep_dim in keep_dims:
        x = []
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device=analyzer.device, dtype=analyzer.dtype)
            x_batch = x_batch + torch.randn_like(x_batch) * analyzer.sigma_noise

            with torch.no_grad():
                z_batch, _ = analyzer.flow(x_batch, rev=False)
                z_batch[:, analyzer.latent_sort[keep_dim:]] = 0
                x_recon, _ = analyzer.flow(z_batch, rev=True)

            x.append(x_recon)
        x = torch.cat(x, dim=0)

        drop_adata = analyzer.adata.copy()
        drop_adata.X = x.cpu().numpy()

        sc.tl.rank_genes_groups(
            drop_adata,
            groupby=analyzer.labels_key,
            reference=analyzer.control_label,
            method="wilcoxon",
        )

        for pert in pertubations:
            de_df_drop = sc.get.rank_genes_groups_df(drop_adata, group=pert)
            de_sig_drop = de_df_drop[
                (de_df_drop["pvals_adj"] < 0.05) & (abs(de_df_drop["logfoldchanges"]) > 0.25)
            ]

            de_dfs[f"{pert}_drop_{keep_dim}"] = de_df_drop
            de_sig_dfs[f"{pert}_drop_{keep_dim}"] = de_sig_drop

        rho, direction_agreement, top_overlap, lfc_shift = evaluate_de_results(
            de_sig_dfs, pertubations, keep_dim
        )
        rhos.append(rho)
        direction_agreements.append(direction_agreement)
        top_overlaps.append(top_overlap)
        lfc_shifts.append(lfc_shift)

    return (
        np.array(rhos),
        np.array(direction_agreements),
        np.array(top_overlaps),
        np.array(lfc_shifts),
    )
