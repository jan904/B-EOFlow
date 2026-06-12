import torch
import argparse
from functools import partial
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint

from src.analysis.pertubation_effects import (
    evaluate_de_effects_gt,
    _get_gt_sig_genes,
    LearnedSoftPermutation,
    _get_logfold_changes,
    _collect_lfc_results,
    differentiable_lfc_results,
    decay_loss,
)
from src.model.data_utils import (
    load_data,
    prepare_data,
    create_latent_adata,
)
from src.analysis.analysis_utils import (
    FlowAnalyser,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train INN model")

    parser.add_argument("--dataset", type=str, default="parse")
    parser.add_argument("--top_genes", type=int, default=2000)
    parser.add_argument("--lam_MTC", type=float, default=0.1)
    parser.add_argument("--sigma_noise", type=float, default=0.3)
    parser.add_argument("--model_prefix", type=str, default="")
    parser.add_argument("--folder_postfix", type=str, default="")
    parser.add_argument("--perturbations", nargs="+", default=["IL-4"])
    parser.add_argument("--FULL_DE", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    dataset_name = args.dataset
    top_genes = args.top_genes
    cell_types = ["CD4 Memory"]

    adata, labels_key, control_label = load_data(
        dataset_name,
        top_genes,
        log_transform=True,
        cell_types=cell_types,
    )

    N_dim = adata.X.shape[1]
    D_dim = N_dim

    lam_MTC = args.lam_MTC
    sigma_noise = args.sigma_noise
    prefix = args.model_prefix
    folder_postfix = args.folder_postfix

    analyzer = FlowAnalyser.from_checkpoint(
        adata,
        dataset_name,
        labels_key,
        control_label,
        lam_MTC,
        sigma_noise,
        prefix=prefix,
        folder_postfix=folder_postfix,
    )

    if analyzer.jac_dec is None:
        analyzer.compute_jacobian()

    perturbations = args.perturbations
    keep_dims = [1000, 500, 250, 100, 50, 25, 10, 5, 3, 1]

    perm_module = LearnedSoftPermutation(N_dim).to(device)
    perm_optimizer = torch.optim.Adam(perm_module.parameters(), lr=5e-3)

    de_dfs, de_sig_dfs = {}, {}
    de_dfs_gt, de_sig_dfs_gt = evaluate_de_effects_gt(analyzer, perturbations, de_dfs, de_sig_dfs)
    sig_genes = _get_gt_sig_genes(de_sig_dfs_gt, perturbations)
    if not args.FULL_DE:
        de_dfs, de_sig_dfs = {}, {}
        gt_adata = analyzer.adata.copy()
        gt_adata.layers["counts"] = gt_adata.X.copy()
        lfc_results = _get_logfold_changes(
            gt_adata, analyzer.labels_key, analyzer.control_label, sig_genes=sig_genes
        )
        _collect_lfc_results(perturbations, f"original", lfc_results, de_dfs, de_sig_dfs)

    latent_adata = create_latent_adata(analyzer.adata, analyzer.flow, analyzer.device)
    dataset, dataloader = prepare_data(
        latent_adata,
        batchsize=256,
        device=analyzer.device,
        dtype=analyzer.dtype,
        label_key=None,
        shuffle=False,
    )

    flow_rev = partial(analyzer.flow, rev=True)

    for epoch in tqdm(range(args.epochs)):
        for keep_dim in keep_dims:
            ctrl_sum = None
            pert_sums = {pert: None for pert in perturbations}

            ctrl_count = 0
            pert_counts = {pert: 0 for pert in perturbations}

            ctrl_mask_full = torch.tensor(
                (analyzer.adata.obs[analyzer.labels_key].values == analyzer.control_label),
                dtype=torch.bool,
                device=analyzer.device,
            )
            pert_masks_full = {
                pert: torch.tensor(
                    (analyzer.adata.obs[analyzer.labels_key].values == pert),
                    dtype=torch.bool,
                    device=analyzer.device,
                )
                for pert in perturbations
            }

            offset = 0

            for i_batch, (z_batch, _) in enumerate(dataloader):
                z_batch = z_batch.to(device=analyzer.device, dtype=analyzer.dtype)
                mask = torch.zeros_like(z_batch)
                mask[:, analyzer.latent_sort[:keep_dim]] = 1

                z_perm = perm_module(z_batch)
                z_masked = z_perm * mask

                x_recon, _ = checkpoint(flow_rev, z_masked, use_reentrant=False)

                bsz = z_batch.shape[0]
                ctrl_mask_batch = ctrl_mask_full[offset : offset + bsz]
                if ctrl_mask_batch.any():
                    sum = x_recon[ctrl_mask_batch].sum(dim=0)
                    ctrl_sum = sum if ctrl_sum is None else ctrl_sum + sum
                    ctrl_count += ctrl_mask_batch.sum().item()

                for pert in perturbations:
                    pert_mask_batch = pert_masks_full[pert][offset : offset + bsz]
                    if pert_mask_batch.any():
                        sum = x_recon[pert_mask_batch].sum(dim=0)
                        pert_sums[pert] = sum if pert_sums[pert] is None else pert_sums[pert] + sum
                        pert_counts[pert] += pert_mask_batch.sum().item()

                offset += bsz

                if (i_batch % 20 == 0 or i_batch == len(dataloader) - 1) and i_batch > 0:
                    ctrl_mean = (ctrl_sum / ctrl_count).clamp(min=1e-8)
                    results = {"var_names": list(analyzer.adata.var_names)}

                    for pert in perturbations:
                        pert_mean = (pert_sums[pert] / pert_counts[pert]).clamp(min=1e-8)
                        lfc = torch.log2(pert_mean) - torch.log2(ctrl_mean)

                        if sig_genes is not None and pert in sig_genes:
                            idx = [
                                i
                                for i, g in enumerate(results["var_names"])
                                if g in sig_genes[pert]
                            ]
                            idx = torch.tensor(idx, device=analyzer.device)
                            lfc = lfc[idx]
                            results[f"var_names_{pert}"] = [
                                results["var_names"][i] for i in idx.cpu().tolist()
                            ]

                        results[pert] = lfc

                    lfc_shift = differentiable_lfc_results(de_dfs, results, perturbations)
                    loss = decay_loss(lfc_shift, keep_dim, N_dim)
                    print(f"Keep dim {keep_dim} - Loss: {loss.item():.4f}")
                    loss.backward()
                    perm_optimizer.step()
                    perm_optimizer.zero_grad()

                    ctrl_sum = None
                    pert_sums = {pert: None for pert in perturbations}

                    ctrl_count = 0
                    pert_counts = {pert: 0 for pert in perturbations}

        perm_module.temp = max(0.01, 1.0 * 0.95**epoch)

        torch.save(
            {
                "permutation_module": perm_module.state_dict(),
                "optimizer": perm_optimizer.state_dict(),
                "epoch": epoch,
            },
            "/home/jhoefer/sandbox/models/permutations/permutation_checkpoint.pt",
        )


if __name__ == "__main__":
    main()
