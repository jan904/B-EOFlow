import torch
import pandas as pd
import numpy as np
from IPython.display import display
import gc
import decoupler as dc
import matplotlib.pyplot as plt
import os


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
):

    gc.collect()
    torch.cuda.empty_cache()

    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    N_dim = analyzer.adata.X.shape[1]
    if analyzer.jac_dec is None:
        analyzer.compute_jacobian()

    if location is not None:
        gene_importances = analyzer.jac_dec[location]  # torch.mean(torch.abs(jac_dec), dim=0)
    else:
        gene_importances = torch.mean(analyzer.jac_dec, dim=0)

    gene_importances = gene_importances[
        :, analyzer.latent_sort
    ]  # Reorder genes according to latent importance
    gene_scores_df = pd.DataFrame(
        gene_importances.T.cpu().numpy(),  # Transponse so rows correspond to latent factors and columns to genes
        columns=analyzer.adata.var_names,
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

    fig, ax = plt.subplots(len(hm_names) // 6 + 1, 6, figsize=(20, (len(hm_names) // 6 + 1) * 5))
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
