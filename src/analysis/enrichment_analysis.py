import torch
import pandas as pd
import numpy as np
from IPython.display import display
import gc
import decoupler as dc
import matplotlib.pyplot as plt
import os

DIR_SEP = " | "


def jaccard(set1, set2):

    a = set(int(x) for x in set1)
    b = set(int(x) for x in set2)

    intersection = len(a & b)
    union = len(a | b)

    return intersection / union if union != 0 else 0.0


def evaluate_gene_contribution(
    adata,
    analyzer1,
    analyzer2=None,
    gene_set=0,
    top_latents=4,
    top_genes=15,
    location1=None,
    location2=None,
):

    if analyzer2 is None:
        analyzer2 = analyzer1

    col = analyzer1.hm_names[gene_set]
    values_1 = torch.tensor(analyzer1.hm_acts[col].values)
    values_2 = torch.tensor(analyzer2.hm_acts[col].values)

    _, idx_1 = torch.topk(values_1, k=top_latents)
    _, idx_2 = torch.topk(values_2, k=top_latents)

    latents_1 = idx_1.cpu().numpy()
    latents_2 = idx_2.cpu().numpy()

    if location1 is not None and location2 is not None:
        jac_dec_1 = torch.abs(analyzer1.jac_dec)[location1]
        jac_dec_2 = torch.abs(analyzer2.jac_dec)[location2]
    else:
        jac_dec_1 = torch.mean(torch.abs(analyzer1.jac_dec), dim=0)
        jac_dec_2 = torch.mean(torch.abs(analyzer2.jac_dec), dim=0)

    genes_set_1 = []
    genes_set_2 = []

    gene_names_1 = []
    gene_names_2 = []

    for latent_1, latent_2 in zip(latents_1, latents_2):
        set_1 = torch.abs(jac_dec_1[:, latent_1]).cpu().numpy()
        set_2 = torch.abs(jac_dec_2[:, latent_2]).cpu().numpy()
        _, top_genes_1 = torch.topk(torch.tensor(set_1), k=top_genes)
        _, top_genes_2 = torch.topk(torch.tensor(set_2), k=top_genes)

        gene_names_1.append([adata.var_names[int(i)] for i in np.array(top_genes_1)])
        gene_names_2.append([adata.var_names[int(i)] for i in np.array(top_genes_2)])
        genes_set_1.append(top_genes_1)
        genes_set_2.append(top_genes_2)

    similarities = np.zeros((top_latents, top_latents))
    for i in range(top_latents):
        for j in range(i + 1, top_latents):
            sim = jaccard(genes_set_1[i], genes_set_2[j])
            similarities[i, j] = np.round(sim, 3)
            similarities[j, i] = np.round(sim, 3)

    df_sim = pd.DataFrame(similarities, index=latents_1, columns=latents_2)
    df_sim.columns.name = "Latents"
    styled = (
        df_sim.style.background_gradient(cmap="viridis", axis=None, vmin=0, vmax=1)
        .format("{:.3f}")
        .set_caption(f"Jaccard Similarity between top {top_latents} latents of {col}")
    )
    display(styled)

    for i in range(top_latents):
        print(f"Latent {latents_1[i]} (set 1) top genes: {', '.join(gene_names_1[i])}")
        print(f"Latent {latents_2[i]} (set 2) top genes: {', '.join(gene_names_2[i])}")
        print("-" * 80)

    return similarities, gene_names_1, gene_names_2, latents_1, latents_2


def compare_intra_gene_contributions(gene_names_1, gene_names_2, latents_1, latents_2):

    all_genes = sorted(set.union(*[set(g) for g in gene_names_1 + gene_names_2]))

    df = pd.DataFrame(
        {
            **{
                f"dim_{latents_1[i]}_1": [g in set(dim) for g in all_genes]
                for i, dim in enumerate(gene_names_1)
            },
        },
        index=all_genes,
    )
    df = df.reindex(sorted(df.columns), axis=1)
    styled = df.astype(int).style.background_gradient(cmap="Blues")
    display(styled)

    equal = all(a == b for a, b in zip(gene_names_1, gene_names_2)) and all(
        a == b for a, b in zip(latents_1, latents_2)
    )
    if not equal:
        df = pd.DataFrame(
            {
                **{
                    f"dim_{latents_2[i]}_2": [g in set(dim) for g in all_genes]
                    for i, dim in enumerate(gene_names_2)
                },
            },
            index=all_genes,
        )
        df = df.reindex(sorted(df.columns), axis=1)
        styled = df.astype(int).style.background_gradient(cmap="Blues")
        display(styled)


def compare_inter_gene_contributions(gene_names_1, gene_names_2, latents_1, latents_2):

    equal = all(a == b for a, b in zip(gene_names_1, gene_names_2)) and all(
        a == b for a, b in zip(latents_1, latents_2)
    )
    if equal:
        print("The same latents and genes are contributing in both cases.")
        return

    all_genes = sorted(set.union(*[set(g) for g in gene_names_1 + gene_names_2]))

    df = pd.DataFrame(
        {
            **{
                f"Top_{i}_1": [g in set(dim) for g in all_genes]
                for i, dim in enumerate(gene_names_1)
            },
            **{
                f"Top_{i}_2": [g in set(dim) for g in all_genes]
                for i, dim in enumerate(gene_names_2)
            },
        },
        index=all_genes,
    )
    df = df.reindex(sorted(df.columns), axis=1)
    styled = df.astype(int).style.background_gradient(cmap="Blues")
    display(styled)


def bulk_enrichment_analysis(
    analyzer,
    top_k=None,
    location=None,
    use_noise=True,
    reorder=True,
    return_results=False,
):

    gc.collect()
    torch.cuda.empty_cache()

    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    N_dim = analyzer.adata.X.shape[1]
    if analyzer.jac_dec is None:
        analyzer.compute_jacobian(use_noise=use_noise)

    if location is not None:
        gene_importances = analyzer.jac_dec[location]  # torch.mean(torch.abs(jac_dec), dim=0)
    else:
        gene_importances = torch.mean(analyzer.jac_dec, dim=0)

    if reorder:
        gene_importances = gene_importances[
            :, analyzer.latent_sort
        ]  # Reorder genes according to latent importance
    gene_scores_df = pd.DataFrame(
        gene_importances.cpu().numpy(),  # Transponse so rows correspond to latent factors and columns to genes
        columns=(
            analyzer.adata.var_names[analyzer.latent_sort] if reorder else analyzer.adata.var_names
        ),
        index=[f"dim_{i}" for i in range(N_dim)],
    )

    hallmark = dc.op.hallmark(organism="human")
    progeny = dc.op.progeny(organism="human")

    hm_acts, hm_padj = dc.mt.ulm(data=gene_scores_df, net=hallmark)
    msk = (hm_padj < 0.05).any(axis=0)  # pathway significant in at least one dim
    hm_acts = hm_acts.loc[:, msk]

    pw_acts, pw_padj = dc.mt.ulm(data=gene_scores_df, net=progeny)
    msk_pw = (pw_padj < 0.05).any(axis=0)  # pathway significant in at least one dim
    pw_acts = pw_acts.loc[:, msk_pw]

    hm_names = hm_acts.T.index
    pw_names = pw_acts.T.index

    if return_results:
        analyzer.set_enrichment_scores(hm_names, hm_acts)
        return hm_names, hm_acts

    else:
        fig, ax = plt.subplots(
            len(hm_names) // 6 + 1, 6, figsize=(20, (len(hm_names) // 6 + 1) * 5)
        )
        for i, name in enumerate(hm_names):
            dc.pl.barplot(data=hm_acts.T, name=name, top=25, figsize=(3, 12), ax=ax[i // 6, i % 6])
            ax[i // 6, i % 6].set_title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(analyzer.plot_dir, "hallmark_pathway_activities.png"))
        plt.show()

        # fig, ax = plt.subplots(
        #     len(pw_names) // 6 + 1, 6, figsize=(20, (len(pw_names) // 6 + 1) * 5)
        # )
        # for i, name in enumerate(pw_names):
        #     dc.pl.barplot(
        #         data=pw_acts.T, name=name, top=25, figsize=(3, 12), ax=ax[i // 6, i % 6]
        #     )
        #     ax[i // 6, i % 6].set_title(name)
        # plt.tight_layout()
        # plt.savefig(os.path.join(analyzer.plot_dir, "progeny_pathway_activities.png"))
        # plt.show()

        analyzer.set_enrichment_scores(hm_names, hm_acts)


# ---------------------------------------------------------------------------
# Direction enrichment: do the *learned vectors* encode biological programmes?
#
# `bulk_enrichment_analysis` above scores latent *axes* - row k of the mean decoder
# Jacobian is the gene loading of dim k, which is the right unit when the prior is
# axis-aligned. Under a factorized mixture prior it is not: with latent_per_condition=4
# and 198 components the condition block is means_dim=792 dims wide, but the learned
# means block has effective rank ~28 and every treatment shift is dense across all 792
# dims. No single axis carries a cytokine, and the flow may rotate freely within the
# block, so an individual axis is not identifiable.
#
# What is identifiable is a *direction*. These helpers carry a latent direction `v` to
# gene space through the decoder's local linear map,
#
#     dx = J_dec(z0) . v
#
# with a forward-mode JVP - one pass per direction, no 1937x1937 Jacobian materialized
# (verified against central finite differences at cosine 0.999999). The resulting
# (directions x genes) frame goes straight into the same `dc.mt.ulm` call
# `bulk_enrichment_analysis` uses.
#
# Scope the base points the same way the axis analysis does - `analyzer.subset(...)`
# then read `.adata` from the subset - because J is a *local* map: the same direction
# reads differently at CD14 Mono control cells than at CD4 Memory ones (measured cosine
# 0.90 between the two, against a 0.995 sampling-noise floor).
# ---------------------------------------------------------------------------


def learned_subspace(model, rank_tol=1e-3):
    """`(basis, singular_values, rank)` of the mixture-prior means block.

    The means are (n_components, N_dim) but only the first `means_dim` columns are
    learnable, and a factorized prior composes every component from `sum(condition_shapes)`
    level vectors - so the block collapses to a subspace of about that size, less the one
    degree of freedom the control gauge pins. `basis` rows are right singular vectors,
    zero-padded back to N_dim so they can be used as latent directions directly.
    """
    means = model.means.detach().float()
    block = means[:, : model.means_dim]
    _, S, Vh = torch.linalg.svd(block, full_matrices=False)
    rank = int((S > S.max() * rank_tol).sum())
    basis = torch.zeros(rank, model.N_dim, device=means.device, dtype=means.dtype)
    basis[:, : model.means_dim] = Vh[:rank]
    return basis, S[:rank], rank


def latent_directions(
    model,
    combo_categories,
    conditions,
    condition_key,
    control_label,
    cell_type_key=None,
    cell_type=None,
    cytokines=None,
    cell_types=None,
    n_components=None,
):
    """The model's own learned vectors, as `{name: (family, latent vector)}`.

    Three families:
      `cytokine`  - `mu(cy, ct) - mu(PBS, ct)`, the learned treatment shift.
      `cell_type` - each cell type's control mean minus the average one, i.e. identity.
                    Centred because `a_ct` is only identified up to a global offset.
      `component` - right singular vectors of the means block: the axes the prior
                    actually uses, and the honest replacement for enriching all 792 dims.

    `cell_type` picks which cell type the treatment shifts are read at. Irrelevant for a
    purely additive prior - the shift is the same everywhere by construction - but not
    under `treatment_gain`, where it is `w_ct * b_cy`. It is recorded in the returned
    info dict so a figure cannot hide which one was used.
    """
    if cell_type_key is None:
        cell_type_key = next(k for k in conditions if k != condition_key)

    means = model.means.detach()
    order = list(conditions)
    levels = {c: sorted({lbl.split("__")[i] for lbl in combo_categories})
              for i, c in enumerate(order)}
    cytokines = cytokines or [c for c in levels[condition_key] if c != control_label]
    cell_types = cell_types or levels[cell_type_key]
    if cell_type is None:
        cell_type = cell_types[0]

    def index(cy, ct):
        label = "__".join(str({condition_key: cy, cell_type_key: ct}[k]) for k in order)
        return combo_categories.index(label) if label in combo_categories else None

    out = {}

    ctrl_at = index(control_label, cell_type)
    for cy in cytokines:
        i = index(cy, cell_type)
        if i is not None and ctrl_at is not None:
            out[f"cy{DIR_SEP}{cy}"] = ("cytokine", means[i] - means[ctrl_at])

    ct_vecs = {ct: means[index(control_label, ct)]
               for ct in cell_types if index(control_label, ct) is not None}
    if ct_vecs:
        centre = torch.stack(list(ct_vecs.values())).mean(0)
        for ct, v in ct_vecs.items():
            out[f"ct{DIR_SEP}{ct}"] = ("cell_type", v - centre)

    basis, svals, rank = learned_subspace(model)
    for i in range(rank if n_components is None else min(rank, n_components)):
        out[f"pc{DIR_SEP}{i}"] = ("component", basis[i])

    info = {"cell_type_read_at": cell_type, "subspace_rank": rank,
            "singular_values": svals.cpu().numpy()}
    return out, info


def null_directions(model, reference_norm, n=50, seed=0):
    """Matched random directions scaled to `reference_norm`, the control the existing
    `padj < 0.05` filter cannot provide.

    Two families, because they answer different questions:
      `null_shared`    - random directions in the shared N(0,1) block (dims >= means_dim),
                         which no condition ever moves. The floor for "a direction the
                         prior never learned".
      `null_condition` - random directions inside the condition block but projected
                         orthogonal to the learned subspace. The sharper control: same
                         block, same scale, simply not a direction the model uses.

    Pass `reference_norm = median ||b_cy||` so the nulls are the same length as the real
    directions - an unscaled random vector would fail on magnitude alone.
    """
    g = torch.Generator().manual_seed(seed)
    N, M = model.N_dim, model.means_dim
    basis, _, _ = learned_subspace(model)
    basis = basis.cpu().float()

    out = {}
    for i in range(n):
        v = torch.zeros(N)
        v[M:] = torch.randn(N - M, generator=g)
        out[f"null_shared{DIR_SEP}{i}"] = ("null_shared", v / v.norm() * reference_norm)

        u = torch.zeros(N)
        u[:M] = torch.randn(M, generator=g)
        u = u - basis.T @ (basis @ u)  # project out the learned subspace
        if u.norm() > 1e-8:
            out[f"null_condition{DIR_SEP}{i}"] = (
                "null_condition", u / u.norm() * reference_norm)
    return out


def direction_gene_loadings(analyzer, directions, n_cells=64, batch_size=64, seed=0):
    """`(loadings, meta)` - `J_dec(z0) . v` averaged over control base cells, one row per
    direction, columns genes.

    Base points are the control cells of `analyzer.adata`, so scope the analyzer first
    (`analyzer.subset(covariates=["cell_type"], keys=["CD14 Mono"])`) to read the
    directions at one cell type. `loadings` drops straight into the same call
    `bulk_enrichment_analysis` makes:

        acts, padj = dc.mt.ulm(data=loadings, net=dc.op.hallmark(organism="human"))
    """
    adata = analyzer.adata
    mask = (adata.obs[analyzer.labels_key].astype(str) == analyzer.control_label).to_numpy()
    if not mask.any():
        raise ValueError(
            f"No '{analyzer.control_label}' cells in this analyzer's adata - the "
            "directions are displacements away from control, so there is no base point."
        )

    idx = np.flatnonzero(mask)
    if len(idx) > n_cells:
        idx = np.random.default_rng(seed).choice(idx, n_cells, replace=False)

    X = adata.X[idx]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    x0 = torch.as_tensor(X, device=analyzer.device, dtype=analyzer.dtype)
    with torch.no_grad():
        z0, _ = analyzer.model(x0, rev=False)

    decode = lambda z: analyzer.model(z, rev=True)[0]
    rows, names, families = {}, [], []
    for name, (family, v) in directions.items():
        v = v.to(device=z0.device, dtype=z0.dtype)
        acc, seen = None, 0
        for s in range(0, z0.shape[0], batch_size):
            chunk = z0[s : s + batch_size]
            _, jv = torch.func.jvp(decode, (chunk,), (v.expand_as(chunk),))
            acc = jv.sum(0) if acc is None else acc + jv.sum(0)
            seen += chunk.shape[0]
        rows[name] = (acc / seen).detach().cpu().numpy()
        names.append(name)
        families.append(family)

    del z0, x0
    gc.collect()
    torch.cuda.empty_cache()

    loadings = pd.DataFrame(rows, index=list(map(str, adata.var_names))).T
    meta = pd.DataFrame({"family": families}, index=names)
    meta["label"] = [n.split(DIR_SEP, 1)[1] for n in names]
    meta["n_base_cells"] = len(idx)
    return loadings, meta
