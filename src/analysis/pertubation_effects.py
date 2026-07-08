import scanpy as sc
import numpy as np
from scipy.stats import spearmanr
from scipy.sparse import issparse
import torch
from torch.utils.checkpoint import checkpoint
from functools import partial
import scvi
import os
import pandas as pd

from src.model.data_utils import prepare_data
from src.analysis.analysis_utils import calculate_pca

# ── DE evaluation helpers ────────────────────────────────────────────────────


def _get_logfold_changes(adata, labels_key, reference, sig_genes=None):
    perturbations = adata.obs[f"{labels_key}"].unique()
    data = adata.layers["counts"]
    if issparse(data):
        data = data.toarray()

    ctrl_mask = adata.obs[labels_key] == reference
    ctrl_mean = data[ctrl_mask].mean(axis=0)

    lfc_results = {}
    for pert in perturbations:
        if pert == reference:
            continue
        pert_mask = adata.obs[labels_key] == pert
        pert_mean = data[pert_mask].mean(axis=0)

        lfc = np.log2(pert_mean + 1e-8) - np.log2(ctrl_mean + 1e-8)

        df = pd.DataFrame(
            {
                "names": adata.var_names,
                "logfoldchanges": lfc,
                "pvals_adj": np.zeros_like(lfc),
            }
        )

        if sig_genes is not None and pert in sig_genes:
            df = df[df["names"].isin(sig_genes[pert])]

        lfc_results[pert] = df

    return lfc_results


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


def _collect_lfc_results(perturbations, key, lfc_results, de_dfs, de_sig_dfs):
    for pert in perturbations:
        de_df = lfc_results[pert]
        de_dfs[f"{pert}_{key}"] = de_df
        de_sig_dfs[f"{pert}_{key}"] = _sig_filter(de_df)


def _get_scvi_arch(keep_dim):
    if keep_dim <= 10:
        return dict(n_layers=1, n_hidden=64)
    elif keep_dim <= 50:
        return dict(n_layers=2, n_hidden=128)
    else:
        return dict(n_layers=3, n_hidden=256)


# ── Ground truth DE ──────────────────────────────────────────────────────────


def evaluate_de_effects_gt(analyzer, perturbations, de_dfs, de_sig_dfs):
    _run_rank_genes(analyzer.adata, analyzer.labels_key, analyzer.control_label)
    _collect_de_results(analyzer.adata, perturbations, "original", de_dfs, de_sig_dfs)
    return de_dfs, de_sig_dfs


# ── Reconstruction helpers ───────────────────────────────────────────────────


def _reconstruct_flow(
    analyzer, dataloader, keep_dim, use_noise, perm_module=None, select_module=None
):
    x_recon = []
    for x_batch, _ in dataloader:
        x_batch = x_batch.to(device=analyzer.device, dtype=analyzer.dtype)
        if use_noise:
            x_batch = x_batch + torch.randn_like(x_batch) * analyzer.sigma_noise
        with torch.no_grad():
            z, _ = analyzer.model(x_batch, rev=False)
            if perm_module is not None:
                P = perm_module.get_permutation_matrix()
                hard = torch.zeros_like(P)
                hard[torch.arange(len(P)), P.argmax(dim=1)] = 1.0
                z = z @ hard
                z[:, analyzer.latent_sort[keep_dim:]] = 0
            if select_module is not None:
                ranking = select_module.get_hard_ranking()
                z[:, ranking[keep_dim:]] = 0
            else:
                z[:, analyzer.latent_sort[keep_dim:]] = 0

            x_rec, _ = analyzer.model(z, rev=True)
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
    model_dict = analyzer.plot_dir.split("/")[-1]
    print(model_dict)

    dir = os.path.join(base_path, model_dict)
    path = os.path.join(dir, f"{keep_dim}_model.pt")
    checkpoint_exists = os.path.exists(path)

    adata_tmp = analyzer.adata.copy()

    if not checkpoint_exists:
        scvi.model.SCVI.setup_anndata(adata_tmp, layer="counts")
        arch = _get_scvi_arch(keep_dim)
        model = scvi.model.SCVI(adata_tmp, n_latent=keep_dim, **arch)
        model.train(
            max_epochs=200,
            batch_size=batch_size,
            early_stopping=True,
            check_val_every_n_epoch=5,
            enable_progress_bar=True,
        )
        model.save(dir, overwrite=True)
        os.rename(os.path.join(dir, "model.pt"), path)

    else:
        model = scvi.model.SCVI.load(dir, prefix=f"{keep_dim}_", adata=adata_tmp)

    latent = model.get_latent_representation()
    activity = np.var(latent, axis=0)
    print(f"Active Dimensions for VAE with {keep_dim} latent dims: {np.sum(activity > 1e-3)}")
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
    drop.layers["counts"] = X_recon
    return drop


def _get_gt_sig_genes(de_sig_dfs, perturbations):
    """Extract significant gene sets from ground truth DE results."""
    sig_genes = {}
    for pert in perturbations:
        sig_df = de_sig_dfs[f"{pert}_original"]
        sig_genes[pert] = set(sig_df["names"].values)
    return sig_genes


# ── Main evaluation ──────────────────────────────────────────────────────────


def evaluate_de_effects(
    analyzer,
    perturbations,
    keep_dims,
    vae_bottlenecks,
    use_noise=False,
    batch_size=1024,
    FULL_DE=False,
):

    if analyzer.pca is None:
        pca, x_pca = calculate_pca(analyzer.adata, analyzer.sigma_noise)
        analyzer.set_pca(pca, x_pca)

    de_dfs, de_sig_dfs = {}, {}

    de_dfs, de_sig_dfs = evaluate_de_effects_gt(analyzer, perturbations, de_dfs, de_sig_dfs)
    sig_genes = _get_gt_sig_genes(de_sig_dfs, perturbations)

    if not FULL_DE:
        de_dfs, de_sig_dfs = {}, {}
        gt_adata = analyzer.adata.copy()
        gt_adata.layers["counts"] = gt_adata.X.copy()
        lfc_results = _get_logfold_changes(
            gt_adata, analyzer.labels_key, analyzer.control_label, sig_genes=sig_genes
        )
        _collect_lfc_results(perturbations, f"original", lfc_results, de_dfs, de_sig_dfs)

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

    for keep_dim in vae_bottlenecks:
        print(f"Evaluating DE effects for VAE bottleneck {keep_dim}...")
        x_vae = _reconstruct_vae(analyzer, keep_dim, batch_size)

        for x_recon, dfs, sig_dfs, tag in [
            (x_vae, de_dfs_vae, de_sig_dfs_vae, "vae"),
        ]:
            drop = _make_drop_adata(analyzer, x_recon)
            if FULL_DE:
                _run_rank_genes(drop, analyzer.labels_key, analyzer.control_label)
                _collect_de_results(drop, perturbations, f"drop_{keep_dim}", dfs, sig_dfs)
            else:
                lfc_results = _get_logfold_changes(
                    drop, analyzer.labels_key, analyzer.control_label, sig_genes=sig_genes
                )
                _collect_lfc_results(perturbations, f"drop_{keep_dim}", lfc_results, dfs, sig_dfs)

        for m, val in zip(
            metrics_vae.values(),
            evaluate_de_results(de_sig_dfs_vae, perturbations, keep_dim),
        ):
            m.append(val)

    for keep_dim in keep_dims:
        print(f"Evaluating DE effects for top {keep_dim} dimensions...")
        # Reconstruct
        x_flow = _reconstruct_flow(analyzer, dataloader, keep_dim, use_noise)
        x_pca = _reconstruct_pca(analyzer, dataloader, keep_dim)

        # Run DE
        for x_recon, dfs, sig_dfs, tag in [
            (x_flow, de_dfs, de_sig_dfs, "flow"),
            (x_pca, de_dfs_pca, de_sig_dfs_pca, "pca"),
        ]:
            drop = _make_drop_adata(analyzer, x_recon)
            if FULL_DE:
                _run_rank_genes(drop, analyzer.labels_key, analyzer.control_label)
                _collect_de_results(drop, perturbations, f"drop_{keep_dim}", dfs, sig_dfs)
            else:
                lfc_results = _get_logfold_changes(
                    drop, analyzer.labels_key, analyzer.control_label, sig_genes=sig_genes
                )
                _collect_lfc_results(perturbations, f"drop_{keep_dim}", lfc_results, dfs, sig_dfs)

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

    return tuple(
        np.array(v) for v in [*metrics.values(), *metrics_pca.values(), *metrics_vae.values()]
    )


def evaluate_permuted_results(
    analyzer,
    perturbations,
    keep_dims,
    perm_module=None,
    select_module=None,
    use_noise=False,
    batch_size=1024,
    FULL_DE=True,
):

    if perm_module is None and select_module is None:
        raise ValueError("At least one of perm_module or select_module must be provided.")

    if analyzer.pca is None:
        pca, x_pca = calculate_pca(analyzer.adata, analyzer.sigma_noise)
        analyzer.set_pca(pca, x_pca)

    de_dfs, de_sig_dfs = {}, {}

    de_dfs, de_sig_dfs = evaluate_de_effects_gt(analyzer, perturbations, de_dfs, de_sig_dfs)
    sig_genes = _get_gt_sig_genes(de_sig_dfs, perturbations)
    if not FULL_DE:
        de_dfs, de_sig_dfs = {}, {}
        gt_adata = analyzer.adata.copy()
        gt_adata.layers["counts"] = gt_adata.X.copy()
        lfc_results = _get_logfold_changes(
            gt_adata, analyzer.labels_key, analyzer.control_label, sig_genes=sig_genes
        )
        _collect_lfc_results(perturbations, f"original", lfc_results, de_dfs, de_sig_dfs)

    de_dfs_permuted, de_sig_dfs_permuted = de_dfs.copy(), de_sig_dfs.copy()

    dataset, dataloader = prepare_data(
        analyzer.adata,
        batchsize=batch_size,
        device=analyzer.device,
        dtype=analyzer.dtype,
        label_key=None,
        shuffle=False,
    )

    metrics = {k: [] for k in ("rhos", "dir_agreements", "top_overlaps", "lfc_shifts")}
    metrics_permuted = {k: [] for k in ("rhos", "dir_agreements", "top_overlaps", "lfc_shifts")}

    for keep_dim in keep_dims:
        print(f"Evaluating DE effects for top {keep_dim} dimensions...")
        # Reconstruct
        x_flow = _reconstruct_flow(analyzer, dataloader, keep_dim, use_noise)
        x_flow_permuted = _reconstruct_flow(
            analyzer,
            dataloader,
            keep_dim,
            use_noise,
            perm_module=perm_module,
            select_module=select_module,
        )

        # Run DE
        for x_recon, dfs, sig_dfs, tag in [
            (x_flow, de_dfs, de_sig_dfs, "flow"),
            (x_flow_permuted, de_dfs_permuted, de_sig_dfs_permuted, "flow_permuted"),
        ]:
            drop = _make_drop_adata(analyzer, x_recon)
            if FULL_DE:
                _run_rank_genes(drop, analyzer.labels_key, analyzer.control_label)
                _collect_de_results(drop, perturbations, f"drop_{keep_dim}", dfs, sig_dfs)
            else:
                lfc_results = _get_logfold_changes(
                    drop, analyzer.labels_key, analyzer.control_label, sig_genes=sig_genes
                )
                _collect_lfc_results(perturbations, f"drop_{keep_dim}", lfc_results, dfs, sig_dfs)

        # Collect metrics
        for m, val in zip(
            metrics.values(), evaluate_de_results(de_sig_dfs, perturbations, keep_dim)
        ):
            m.append(val)
        for m, val in zip(
            metrics_permuted.values(),
            evaluate_de_results(de_sig_dfs_permuted, perturbations, keep_dim),
        ):
            m.append(val)

    return tuple(np.array(v) for v in [*metrics.values(), *metrics_permuted.values()])


def differentiable_lfc(x_recon_tensor, adata, labels_key, reference, perturbations, sig_genes=None):
    results = {"var_names": list(adata.var_names)}  # shared gene index
    ctrl_mask = torch.tensor((adata.obs[labels_key] == reference).values, dtype=torch.bool)
    ctrl_mean = x_recon_tensor[ctrl_mask].mean(dim=0).clamp(min=1e-8)

    for pert in perturbations:
        pert_mask = torch.tensor((adata.obs[labels_key] == pert).values, dtype=torch.bool)
        pert_mean = x_recon_tensor[pert_mask].mean(dim=0).clamp(min=1e-8)
        lfc = torch.log2(pert_mean) - torch.log2(ctrl_mean)

        if sig_genes is not None and pert in sig_genes:
            idx = [i for i, g in enumerate(results["var_names"]) if g in sig_genes[pert]]
            idx = torch.tensor(idx, device=x_recon_tensor.device)
            lfc = lfc[idx]
            results[f"var_names_{pert}"] = [results["var_names"][i] for i in idx.cpu().tolist()]

        results[pert] = lfc

    return results


def differentiable_lfc_results(de_dfs, lfc_results, perturbations):
    lfc_shifts = []
    for pert in perturbations:
        base_df = de_dfs[f"{pert}_original"]

        lfc_pred = lfc_results[pert]

        gene_order = base_df["names"].values
        var_names = lfc_results.get(f"var_names_{pert}", lfc_results["var_names"])

        valid_genes = [g for g in gene_order if g in var_names]
        pred_idx = [var_names.index(g) for g in valid_genes]
        base_idx = [i for i, g in enumerate(gene_order) if g in var_names]

        lfc_pred = lfc_pred[pred_idx]
        lfc_base = torch.tensor(base_df["logfoldchanges"].values[base_idx], dtype=torch.float32).to(
            lfc_pred.device
        )

        valid = ~torch.isnan(lfc_base) & ~torch.isinf(lfc_base)

        lfc_shifts.append((lfc_pred[valid] - lfc_base[valid]).abs().mean())

    return lfc_shifts


class LearnedOrthogonalRotation(torch.nn.Module):
    def __init__(self, N_dim):
        super().__init__()
        self.N_dim = N_dim
        self.B = torch.nn.Parameter(torch.zeros(N_dim, N_dim))

    def get_rotation_matrix(self):
        # Compute A = B - B^T to ensure skew-symmetry
        A = self.B - self.B.T
        # Compute the matrix exponential to get an orthogonal matrix
        I = torch.eye(self.N_dim, device=self.B.device)
        R = torch.linalg.solve(I - A, I + A)  # Cayley transform for orthogonal matrix
        return R

    def forward(self, z):
        R = self.get_rotation_matrix()
        return z @ R


class LearnedSoftPermutation(torch.nn.Module):
    def __init__(self, N_dim, temp=1.0, n_iters=20):
        super().__init__()
        self.N_dim = N_dim
        self.temp = temp
        self.n_iters = n_iters
        self.S = torch.nn.Parameter(torch.zeros(N_dim, N_dim))

    def get_permutation_matrix(self):
        # Sinkhorn normalization → doubly stochastic matrix
        P = self.S / self.temp
        for _ in range(self.n_iters):
            P = P - torch.logsumexp(P, dim=1, keepdim=True)  # row normalise
            P = P - torch.logsumexp(P, dim=0, keepdim=True)  # col normalise
        return torch.exp(P)

    def forward(self, z):
        P = self.get_permutation_matrix()
        return z @ P


class LearnedDimSelector(torch.nn.Module):
    def __init__(self, N_dim):
        super().__init__()
        # one score per dimension — higher = more important
        self.scores = torch.nn.Parameter(torch.zeros(N_dim))
        self.temp = 0.1

    def get_mask(self, keep_dim):
        # find the threshold: the keep_dim-th largest score
        topk_vals, _ = torch.topk(self.scores, keep_dim)
        threshold = topk_vals[-1]
        # soft step function centered at threshold
        soft_mask = torch.sigmoid((self.scores - threshold) / self.temp)
        return soft_mask  # (N_dim,) values in (0, 1)

    def forward(self, z, keep_dim):
        mask = self.get_mask(keep_dim)
        return z * mask  # (N, N_dim) — soft zeroing of unselected dims

    def get_hard_ranking(self):
        # at test time: just argsort scores descending
        return self.scores.argsort(descending=True).cpu().numpy()


def exponential_weight(x, N_dim, alpha):
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.exp(-alpha * x / N_dim)


def decay_loss(lfc_shifts, keep_dims, N_dim):
    loss = 0.0
    weight = exponential_weight(keep_dims, N_dim, alpha=5)
    for shift in lfc_shifts:
        loss += weight * shift
    return loss
