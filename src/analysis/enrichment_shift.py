"""Does the learned cytokine shift correspond to known biology?

`enrichment_analysis.bulk_enrichment_analysis` answers a different question: it runs ULM
on the *decoder Jacobian*, rows = latent dims, so it reports what each latent axis means.
That was the right unit when the prior held a free mean per combo.

Under the factorized prior the cytokine effect is not an axis, it is a vector -
`mu(cy, ct) - mu(PBS, ct)`, one shared treatment direction (rescaled by `w_ct` under
`treatment_gain`). So the object to enrich is the **gene-space image of that shift**, one
signed signature per (cytokine, cell type):

    pred[cy, ct] = mean(decode(encode(x_PBS,ct) + shift)) - mean(x_PBS,ct)
    real[cy, ct] = mean(x_cy,ct)                          - mean(x_PBS,ct)

`.X` is log1p-normalized, so a difference of means is a log fold change - exactly what
ULM wants. `pred` is the model's own counterfactual (the same encode/shift/decode as
`INN_OOD.sample_leftout_combo_shift`, generalized off the holdout path); `real` is the
ground truth to score it against.

Three claims, kept separate on purpose:

1. **Positive controls** (`EXPECTED_HALLMARK`) - does the model's shift put the textbook
   pathway near the top? IL-2 -> IL2_STAT5_SIGNALING, IFN-epsilon -> INTERFERON_ALPHA,
   the TNF-superfamily ligands -> TNFA_SIGNALING_VIA_NFKB. Read `rank_pred`.
2. **Agreement with the data** (`pred_real_agreement`) - Spearman rho between the
   predicted and real pathway-activity vectors. A model can recover exactly what the data
   shows even where the data is not textbook; this separates the two.
3. **Specificity** (`null_scores`) - gene labels are permuted to give the scores a
   threshold to beat. Without it a large ULM t-value means very little, because the gene
   universe here is 1937 HVGs already selected on this dataset, not the genome.

The OOD arm (`analyze_holdout_enrichment`) is the payoff: the model never trained on
(IL-2, CD8 Memory), so whether its predicted shift still ranks IL2_STAT5 first *for that
cell type* is the claim the factorized prior was built to support.

Note on the background: ULM scores are relative among the HVGs present, not genome-wide,
and only ~39 of the 50 hallmark sets clear decoupler's `tmin=5` at this gene count. The
per-set coverage is reported so a striking score on a 6-gene set is not read as a 200-gene
one.
"""

import os
import ssl

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Textbook expectation per cytokine in the Parse PBMC panel. Only sets that survive
# decoupler's tmin on an HVG background are useful here, so these are deliberately the
# broad, well-covered ones rather than the most specific set that exists.
EXPECTED_HALLMARK = {
    "IL-2": ["IL2_STAT5_SIGNALING"],
    "IFN-epsilon": ["INTERFERON_ALPHA_RESPONSE", "INTERFERON_GAMMA_RESPONSE"],
    "IL-1-beta": ["TNFA_SIGNALING_VIA_NFKB", "INFLAMMATORY_RESPONSE"],
    "IL-32-beta": ["TNFA_SIGNALING_VIA_NFKB", "INFLAMMATORY_RESPONSE"],
    "CD40L": ["TNFA_SIGNALING_VIA_NFKB"],
    "CD27L": ["TNFA_SIGNALING_VIA_NFKB"],
    # cardiotrophin-1 signals through gp130, the IL-6 family receptor
    "CT-1": ["IL6_JAK_STAT3_SIGNALING"],
    # IL-4/IL-13 act through STAT6 and have no hallmark set of their own; PROGENy's
    # JAK-STAT is the readout for them, so they carry no hallmark expectation
    "IL-4": [],
    "IL-13": [],
    # bFGF has no described receptor programme in resting PBMC - the negative control
    "FGF-beta": [],
}

EXPECTED_PROGENY = {
    "IL-2": ["JAK-STAT"],
    "IL-4": ["JAK-STAT"],
    "IL-13": ["JAK-STAT"],
    "CT-1": ["JAK-STAT"],
    "IFN-epsilon": ["JAK-STAT"],
    "IL-1-beta": ["NFkB", "TNFa"],
    "IL-32-beta": ["NFkB", "TNFa"],
    "CD40L": ["NFkB", "TNFa"],
    "CD27L": ["NFkB", "TNFa"],
    "FGF-beta": [],
}

# Cytokines with no expected programme; their scores are what an "effect" looks like when
# there is nothing to find, and they belong on every plot as the empirical floor.
NEGATIVE_CONTROLS = ("FGF-beta",)

ROW_SEP = " | "


def _row_key(cytokine, cell_type):
    return f"{cytokine}{ROW_SEP}{cell_type}"


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def load_nets(organism="human", cache_dir=None, verbose=True):
    """`{"hallmark": df, "progeny": df}` of decoupler gene-set networks.

    `dc.op.*` downloads from omnipathdb, which needs the same unverified-SSL workaround
    `enrichment_analysis.bulk_enrichment_analysis` uses on this cluster. Pass `cache_dir`
    to keep parquet copies on disk, so a re-run of the notebook does not depend on the
    compute node having outbound network at all.
    """
    import decoupler as dc

    ssl._create_default_https_context = ssl._create_unverified_context

    getters = {"hallmark": dc.op.hallmark, "progeny": dc.op.progeny}
    nets = {}
    for name, getter in getters.items():
        path = os.path.join(cache_dir, f"{name}_{organism}.parquet") if cache_dir else None
        if path is not None and os.path.exists(path):
            nets[name] = pd.read_parquet(path)
            if verbose:
                print(f"[nets] {name}: {nets[name]['source'].nunique()} sets (cached)")
            continue
        net = getter(organism=organism)
        if path is not None:
            os.makedirs(cache_dir, exist_ok=True)
            net.to_parquet(path)
        nets[name] = net
        if verbose:
            print(f"[nets] {name}: {net['source'].nunique()} sets")
    return nets


def net_coverage(net, var_names):
    """Targets per gene set that are actually in `var_names` - the number that decides
    whether a set's score is worth reading. `dc.mt.ulm` silently drops sets below `tmin`.
    """
    present = set(map(str, var_names))
    hits = net[net["target"].astype(str).isin(present)]
    return pd.DataFrame(
        {
            "n_total": net.groupby("source").size(),
            "n_covered": hits.groupby("source").size(),
        }
    ).fillna({"n_covered": 0}).astype(int).sort_values("n_covered", ascending=False)


def _encode_decode_shift(model, x_ctrl, shift, batch_size=512, clamp_min=0.0, cell_type=None):
    """Encode control cells once, add the latent shift, decode. Batched so a large
    control pool cannot blow the 16 GB allocations this repo sometimes runs on.

    `cell_type` is only read by a hybrid model, whose flow was conditioned on it;
    `flow_condition_for` returns None for a mixture model so this is unchanged there.
    """
    from src.analysis.INN_OOD import flow_condition_for

    outs = []
    with torch.no_grad():
        for start in range(0, x_ctrl.shape[0], batch_size):
            chunk = x_ctrl[start : start + batch_size]
            c = flow_condition_for(model, cell_type, chunk.shape[0], chunk.device, chunk.dtype)
            z, _ = model(chunk, c=c, rev=False)
            x_cf, _ = model(z + shift, c=c, rev=True)
            outs.append(x_cf)
    x_cf = torch.cat(outs, dim=0)
    if clamp_min is not None:
        # the decoder is unconstrained and emits negatives; log1p space is not. Same
        # reasoning (and same floor) as INN_OOD._combo_adata.
        x_cf = x_cf.clamp(min=clamp_min)
    return x_cf


def compute_signatures(
    analyzer,
    cytokines=None,
    cell_types=None,
    conditions=None,
    condition_key=None,
    cell_type_key="cell_type",
    combo_categories=None,
    control_label=None,
    min_real=5,
    min_control=5,
    batch_size=512,
    clamp_min=0.0,
    verbose=True,
):
    """Predicted and real log-fold-change signatures per (cytokine, cell type).

    Returns `(signatures, meta)` where `signatures` is
    `{"pred": DataFrame, "real": DataFrame}` - rows `"<cytokine> | <cell type>"`, columns
    genes - and `meta` carries the cell counts and the latent shift norm per row.

    `real` only gets a row where at least `min_real` cells of that combo exist in
    `analyzer.adata`; held-out combos qualify, since only the *training* split dropped
    them. Rows appear in `pred` regardless, so a combo can be predicted and left
    unscored - the agreement helpers intersect the two indices.
    """
    model = analyzer.model
    adata = analyzer.adata
    conditions = list(conditions or analyzer.conditions)
    condition_key = condition_key or analyzer.labels_key
    control_label = control_label or analyzer.control_label
    combo_categories = list(combo_categories or analyzer.combo_categories)
    device, dtype = analyzer.device, analyzer.dtype

    obs_cond = adata.obs[condition_key].astype(str).to_numpy()
    obs_cell = adata.obs[cell_type_key].astype(str).to_numpy()

    if cytokines is None:
        cytokines = [c for c in sorted(set(obs_cond)) if c != control_label]
    if cell_types is None:
        cell_types = sorted(set(obs_cell))

    means = model.means.detach()

    def combo_idx(cy, ct):
        values = {condition_key: cy, cell_type_key: ct}
        label = "__".join(str(values[k]) for k in conditions)
        if label not in combo_categories:
            return None
        return combo_categories.index(label)

    pred_rows, real_rows, meta_rows = {}, {}, []
    var_names = list(map(str, adata.var_names))

    for ct in cell_types:
        ctrl_mask = (obs_cell == ct) & (obs_cond == control_label)
        if ctrl_mask.sum() < min_control:
            if verbose:
                print(f"[signatures] skipping {ct}: {ctrl_mask.sum()} control cells (< {min_control})")
            continue

        x_ctrl_np = _dense(adata.X[ctrl_mask])
        x_ctrl = torch.as_tensor(x_ctrl_np, device=device, dtype=dtype)
        ctrl_mean = x_ctrl_np.mean(axis=0)

        ctrl_idx = combo_idx(control_label, ct)
        if ctrl_idx is None:
            if verbose:
                print(f"[signatures] skipping {ct}: no control combo in the vocabulary")
            continue

        for cy in cytokines:
            idx = combo_idx(cy, ct)
            if idx is None:
                continue

            # difference of prior means, not model.treatment_shift: it needs no gauge to
            # be pinned and stays correct under treatment_gain, where the shift is
            # w_ct * b_cy and no single treatment vector serves every cell type.
            #
            # A hybrid prior is indexed by treatment level (11), not by combo (198) - the
            # cell type is the flow's condition, not part of the mean - so it needs the
            # same difference taken over its own vocabulary.
            if getattr(model, "condition_type", None) == "hybrid":
                from src.analysis.INN_OOD import hybrid_treatment_shift

                shift = hybrid_treatment_shift(model, cy, control_label).to(
                    device=device, dtype=dtype
                )
            else:
                shift = (means[idx] - means[ctrl_idx]).to(device=device, dtype=dtype)

            x_cf = _encode_decode_shift(
                model, x_ctrl, shift, batch_size=batch_size, clamp_min=clamp_min,
                cell_type=ct,
            )
            key = _row_key(cy, ct)
            pred_rows[key] = x_cf.mean(dim=0).cpu().numpy() - ctrl_mean

            real_mask = (obs_cell == ct) & (obs_cond == cy)
            n_real = int(real_mask.sum())
            if n_real >= min_real:
                real_rows[key] = _dense(adata.X[real_mask]).mean(axis=0) - ctrl_mean

            meta_rows.append(
                {
                    "row": key,
                    "cytokine": cy,
                    "cell_type": ct,
                    "n_control": int(ctrl_mask.sum()),
                    "n_real": n_real,
                    "has_real": n_real >= min_real,
                    "shift_norm": float(torch.linalg.norm(shift).cpu()),
                }
            )

        del x_ctrl
        torch.cuda.empty_cache()

    signatures = {
        "pred": pd.DataFrame(pred_rows, index=var_names).T,
        "real": pd.DataFrame(real_rows, index=var_names).T,
    }
    meta = pd.DataFrame(meta_rows).set_index("row")

    if verbose:
        print(
            f"[signatures] {len(signatures['pred'])} predicted, "
            f"{len(signatures['real'])} with real ground truth, "
            f"over {meta['cell_type'].nunique()} cell types"
        )
    return signatures, meta


def run_enrichment(signature, net, method="ulm", **kwargs):
    """`(acts, padj)` from decoupler, rows aligned to `signature`'s rows.

    Sets falling below decoupler's `tmin` on this gene background are dropped, which is
    why the returned column count is smaller than the network's set count.
    """
    import decoupler as dc

    fn = {"ulm": dc.mt.ulm, "gsea": dc.mt.gsea}[method]
    acts, padj = fn(data=signature, net=net, **kwargs)
    return acts, padj


def expected_table(acts, padj, meta, expected=None, coverage=None):
    """One row per (cytokine, cell type, expected pathway): its score, adjusted p, and
    **rank** among all tested pathways for that signature (1 = strongest).

    The rank is the number to read. A raw ULM t-value is not comparable across cell types
    with different control-pool sizes, but "the textbook pathway came first out of 39" is.
    """
    expected = EXPECTED_HALLMARK if expected is None else expected
    ranks = acts.rank(axis=1, ascending=False, method="min")

    rows = []
    for key, info in meta.iterrows():
        if key not in acts.index:
            continue
        for pathway in expected.get(info["cytokine"], []):
            if pathway not in acts.columns:
                rows.append(
                    {
                        "row": key,
                        "cytokine": info["cytokine"],
                        "cell_type": info["cell_type"],
                        "pathway": pathway,
                        "score": np.nan,
                        "padj": np.nan,
                        "rank": np.nan,
                        "n_tested": acts.shape[1],
                        "note": "set dropped below tmin on this gene background",
                    }
                )
                continue
            rows.append(
                {
                    "row": key,
                    "cytokine": info["cytokine"],
                    "cell_type": info["cell_type"],
                    "pathway": pathway,
                    "score": float(acts.loc[key, pathway]),
                    "padj": float(padj.loc[key, pathway]),
                    "rank": int(ranks.loc[key, pathway]),
                    "n_tested": acts.shape[1],
                    "note": "",
                }
            )

    table = pd.DataFrame(rows)
    if coverage is not None and len(table):
        table["n_covered"] = table["pathway"].map(coverage["n_covered"])
    return table


def combine_expected(table_pred, table_real):
    """Side-by-side `pred` vs `real` expected-pathway table, on the rows both scored."""
    keys = ["row", "cytokine", "cell_type", "pathway"]
    merged = table_pred.merge(
        table_real, on=keys, suffixes=("_pred", "_real"), how="left"
    )
    cols = keys + [
        c
        for c in ["score_pred", "score_real", "rank_pred", "rank_real", "padj_pred",
                  "padj_real", "n_covered_pred", "n_tested_pred"]
        if c in merged.columns
    ]
    return merged[cols].sort_values(["cytokine", "cell_type"]).reset_index(drop=True)


def pred_real_agreement(acts_pred, acts_real, meta=None):
    """Spearman rho between the predicted and real pathway-activity vectors, per row.

    This is claim (2) and it is independent of claim (1): a model can reproduce exactly
    the profile the data shows in a cell type where that profile is not the textbook one.
    """
    shared_rows = acts_pred.index.intersection(acts_real.index)
    shared_cols = acts_pred.columns.intersection(acts_real.columns)

    rows = []
    for key in shared_rows:
        a = acts_pred.loc[key, shared_cols].to_numpy(dtype=float)
        b = acts_real.loc[key, shared_cols].to_numpy(dtype=float)
        rho, p = spearmanr(a, b)
        entry = {"row": key, "spearman_rho": float(rho), "p": float(p),
                 "n_pathways": len(shared_cols)}
        if meta is not None and key in meta.index:
            entry["cytokine"] = meta.loc[key, "cytokine"]
            entry["cell_type"] = meta.loc[key, "cell_type"]
        rows.append(entry)

    frame = pd.DataFrame(rows)
    return frame.sort_values("spearman_rho", ascending=False).reset_index(drop=True)


def null_scores(signature, net, n_perm=200, seed=0, method="ulm", verbose=True):
    """Score distribution under permuted gene labels - claim (3).

    Each permutation shuffles the signature's gene labels (the same shuffle for every
    row, so the correlation structure across rows survives) and re-runs the enrichment.
    Returns the stacked null scores; `null_threshold` turns them into a cutoff.

    Needed because the background here is 1937 dataset-selected HVGs, not the genome, so
    the analytic ULM p-value is optimistic about what a "significant" pathway is.
    """
    rng = np.random.default_rng(seed)
    genes = np.asarray(signature.columns)

    nulls = []
    for i in range(n_perm):
        permuted = signature.copy()
        permuted.columns = genes[rng.permutation(len(genes))]
        acts, _ = run_enrichment(permuted, net, method=method)
        nulls.append(acts.stack().rename("score").reset_index().assign(perm=i))
        if verbose and (i + 1) % 50 == 0:
            print(f"[null] {i + 1}/{n_perm} permutations")

    out = pd.concat(nulls, ignore_index=True)
    out.columns = ["row", "pathway", "score", "perm"]
    return out


def null_threshold(nulls, quantile=0.95):
    """Two-sided |score| cutoff from the permutation null: global, and per pathway."""
    absolute = nulls.assign(abs_score=nulls["score"].abs())
    return {
        "global": float(absolute["abs_score"].quantile(quantile)),
        "per_pathway": absolute.groupby("pathway")["abs_score"].quantile(quantile),
        "quantile": quantile,
    }


def empirical_p(table, nulls, score_col="score"):
    """Two-sided empirical p per expected-pathway row, from that pathway's own null."""
    if not len(table):
        return table
    grouped = {name: g["score"].abs().to_numpy() for name, g in nulls.groupby("pathway")}

    def _p(row):
        draws = grouped.get(row["pathway"])
        if draws is None or not np.isfinite(row[score_col]):
            return np.nan
        return float((np.sum(draws >= abs(row[score_col])) + 1) / (len(draws) + 1))

    return table.assign(p_perm=table.apply(_p, axis=1))


def plot_pathway_heatmap(acts_pred, acts_real, meta, cell_type, expected=None,
                         top_n=15, path=None, title=None):
    """Cytokines x pathways, predicted next to real, on a shared colour scale.

    Pathways are the union of the strongest `top_n` in each panel, so a pathway the model
    invents is as visible as one it misses. Expected cells are outlined.
    """
    expected = EXPECTED_HALLMARK if expected is None else expected
    keys = meta.index[meta["cell_type"] == cell_type]
    keys = [k for k in keys if k in acts_pred.index and k in acts_real.index]
    if not keys:
        print(f"[heatmap] nothing to plot for {cell_type}")
        return None

    shared = acts_pred.columns.intersection(acts_real.columns)
    pred = acts_pred.loc[keys, shared]
    real = acts_real.loc[keys, shared]
    picks = (
        pred.abs().max(axis=0).nlargest(top_n).index.union(
            real.abs().max(axis=0).nlargest(top_n).index
        )
    )
    order = real[picks].abs().max(axis=0).sort_values(ascending=False).index
    pred, real = pred[order], real[order]
    labels = [k.split(ROW_SEP)[0] for k in keys]

    vmax = float(np.nanmax(np.abs(np.concatenate([pred.to_numpy(), real.to_numpy()]))))
    fig, axes = plt.subplots(1, 2, figsize=(1 + 0.42 * len(order) * 2, 1.5 + 0.34 * len(keys)),
                             sharey=True)
    for ax, frame, name in zip(axes, [pred, real], ["predicted shift", "real data"]):
        im = ax.imshow(frame.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(name, fontsize=10)
        for i, key in enumerate(keys):
            cy = meta.loc[key, "cytokine"]
            for pathway in expected.get(cy, []):
                if pathway in order:
                    j = list(order).index(pathway)
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="black", lw=1.6))
    fig.colorbar(im, ax=axes, shrink=0.7, label="ULM activity")
    fig.suptitle(title or f"Hallmark activity - {cell_type} (outlined = expected)", fontsize=11)
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def plot_expected_ranks(table, threshold=None, path=None, title=None):
    """Left: score of the expected pathway, predicted vs real, against the permutation
    threshold. Right: its rank among all tested pathways (1 is best, lower is better).
    """
    if not len(table):
        print("[ranks] no expected pathways to plot")
        return None

    frame = table.copy()
    frame["label"] = frame["cytokine"] + ROW_SEP + frame["cell_type"] + ROW_SEP + frame["pathway"]
    frame = frame.sort_values("rank_pred", na_position="last")
    y = np.arange(len(frame))

    fig, axes = plt.subplots(1, 2, figsize=(13, 1.2 + 0.32 * len(frame)), sharey=True)
    axes[0].scatter(frame["score_pred"], y, color="tab:blue", label="predicted shift", zorder=3)
    if "score_real" in frame:
        axes[0].scatter(frame["score_real"], y, color="tab:orange", marker="x",
                        label="real data", zorder=3)
    axes[0].axvline(0, color="grey", lw=0.8)
    if threshold is not None:
        for sign in (-1, 1):
            axes[0].axvline(sign * threshold, color="black", ls="--", lw=0.9,
                            label="permutation 95%" if sign == 1 else None)
    axes[0].set_xlabel("ULM activity of the expected pathway")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].scatter(frame["rank_pred"], y, color="tab:blue", zorder=3)
    if "rank_real" in frame:
        axes[1].scatter(frame["rank_real"], y, color="tab:orange", marker="x", zorder=3)
    n_tested = frame["n_tested_pred"].dropna()
    if len(n_tested):
        axes[1].axvline(float(n_tested.iloc[0]) / 2, color="black", ls="--", lw=0.9,
                        label="chance (median rank)")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("rank among tested pathways (1 = strongest)")
    axes[1].grid(axis="x", alpha=0.3)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(frame["label"], fontsize=8)
    fig.suptitle(title or "Expected pathway per cytokine", fontsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def plot_agreement(agreement, path=None, title=None):
    """Spearman rho between predicted and real pathway profiles, one bar per combo."""
    if not len(agreement):
        return None
    frame = agreement.sort_values("spearman_rho")
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.28 * len(frame)))
    colors = ["tab:red" if r < 0 else "tab:green" for r in frame["spearman_rho"]]
    ax.barh(frame["row"], frame["spearman_rho"], color=colors)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Spearman rho (predicted vs real pathway activity)")
    ax.set_xlim(-1, 1)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(title or "Predicted vs real pathway profile", fontsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def _summarize(expected_combined, agreement, negative_controls=NEGATIVE_CONTROLS):
    """The three headline numbers, printed rather than buried in a frame."""
    scored = expected_combined.dropna(subset=["rank_pred"])
    if len(scored):
        n = len(scored)
        top1 = int((scored["rank_pred"] == 1).sum())
        top5 = int((scored["rank_pred"] <= 5).sum())
        print(f"\nExpected pathway rank (predicted shift): top-1 {top1}/{n}, top-5 {top5}/{n}, "
              f"median rank {scored['rank_pred'].median():.0f} of {int(scored['n_tested_pred'].iloc[0])}")
        if "rank_real" in scored:
            real = scored.dropna(subset=["rank_real"])
            if len(real):
                print(f"  same statistic on the real data:  top-1 "
                      f"{int((real['rank_real'] == 1).sum())}/{len(real)}, top-5 "
                      f"{int((real['rank_real'] <= 5).sum())}/{len(real)}, "
                      f"median rank {real['rank_real'].median():.0f}")
    if len(agreement):
        print(f"Predicted vs real pathway profile: median rho "
              f"{agreement['spearman_rho'].median():.3f} "
              f"({int((agreement['spearman_rho'] > 0).sum())}/{len(agreement)} positive)")
    if negative_controls and len(agreement) and "cytokine" in agreement:
        neg = agreement[agreement["cytokine"].isin(negative_controls)]
        if len(neg):
            print(f"Negative controls {list(negative_controls)}: median rho "
                  f"{neg['spearman_rho'].median():.3f} over {len(neg)} combo(s)")


def analyze_shift_enrichment(
    analyzer,
    cell_types=None,
    cytokines=None,
    net=None,
    nets=None,
    net_name="hallmark",
    expected=None,
    condition_key=None,
    cell_type_key="cell_type",
    n_perm=200,
    heatmap_cell_types=None,
    plot_dir=None,
    csv_dir=None,
    cache_dir=None,
    verbose=True,

    clamp_min=0.0,):
    """Full in-distribution analysis: signatures -> enrichment -> the three claims.

    Returns a dict with `signatures`, `meta`, `acts_pred`/`acts_real`, `expected`
    (pred vs real side by side, with permutation p), `agreement`, `nulls`, `threshold`
    and `coverage`. Set `n_perm=0` to skip the permutation null.
    """
    expected = EXPECTED_HALLMARK if expected is None else expected
    plot_dir = plot_dir or getattr(analyzer, "plot_dir", None)
    csv_dir = csv_dir or getattr(analyzer, "csv_dir", None)

    if net is None:
        nets = nets or load_nets(cache_dir=cache_dir, verbose=verbose)
        net = nets[net_name]

    signatures, meta = compute_signatures(
        analyzer,
        cytokines=cytokines,
        cell_types=cell_types,
        condition_key=condition_key,
        cell_type_key=cell_type_key,
        clamp_min=clamp_min,
        verbose=verbose,
    )

    coverage = net_coverage(net, analyzer.adata.var_names)
    acts_pred, padj_pred = run_enrichment(signatures["pred"], net)
    acts_real, padj_real = run_enrichment(signatures["real"], net)
    if verbose:
        print(f"[enrichment] {acts_pred.shape[1]} of {net['source'].nunique()} {net_name} "
              f"sets cleared decoupler's tmin on {analyzer.adata.n_vars} genes")

    table_pred = expected_table(acts_pred, padj_pred, meta, expected, coverage)
    table_real = expected_table(acts_real, padj_real, meta, expected, coverage)
    combined = combine_expected(table_pred, table_real)
    agreement = pred_real_agreement(acts_pred, acts_real, meta)

    nulls, threshold = None, None
    if n_perm:
        nulls = null_scores(signatures["pred"], net, n_perm=n_perm, verbose=verbose)
        threshold = null_threshold(nulls)
        combined = empirical_p(combined, nulls, score_col="score_pred")
        if verbose:
            print(f"[null] |score| 95th percentile under permuted gene labels: "
                  f"{threshold['global']:.2f}")

    _summarize(combined, agreement)

    if plot_dir:
        for ct in (heatmap_cell_types or meta["cell_type"].unique()[:3]):
            plot_pathway_heatmap(
                acts_pred, acts_real, meta, ct, expected,
                path=os.path.join(plot_dir, f"enrichment_{net_name}_{ct.replace('/', '-')}.png"),
            )
        plot_expected_ranks(
            combined,
            threshold=threshold["global"] if threshold else None,
            path=os.path.join(plot_dir, f"enrichment_{net_name}_expected_ranks.png"),
        )
        plot_agreement(
            agreement,
            path=os.path.join(plot_dir, f"enrichment_{net_name}_agreement.png"),
        )

    if csv_dir:
        combined.to_csv(os.path.join(csv_dir, f"enrichment_{net_name}_expected.csv"), index=False)
        agreement.to_csv(os.path.join(csv_dir, f"enrichment_{net_name}_agreement.csv"), index=False)
        acts_pred.to_csv(os.path.join(csv_dir, f"enrichment_{net_name}_acts_pred.csv"))
        acts_real.to_csv(os.path.join(csv_dir, f"enrichment_{net_name}_acts_real.csv"))

    return {
        "signatures": signatures,
        "meta": meta,
        "acts_pred": acts_pred,
        "acts_real": acts_real,
        "padj_pred": padj_pred,
        "padj_real": padj_real,
        "expected": combined,
        "agreement": agreement,
        "nulls": nulls,
        "threshold": threshold,
        "coverage": coverage,
        "net_name": net_name,
    }


def analyze_holdout_enrichment(
    analyzer,
    holdout_combos,
    baseline=None,
    net=None,
    nets=None,
    net_name="hallmark",
    expected=None,
    condition_key=None,
    cell_type_key="cell_type",
    n_perm=0,
    plot_dir=None,
    csv_dir=None,
    cache_dir=None,
    verbose=True,

    clamp_min=0.0,):
    """The OOD arm: the same analysis restricted to the combos held out of training.

    `holdout_combos`: list of dicts as in `configs/holdout_combos.json` - take it from
    `model_config.holdout_combos`, which is what the checkpoint actually held out, rather
    than the notebook's file, which can have been edited since.

    `baseline`: the result dict from `analyze_shift_enrichment`. When given, each held-out
    combo's expected-pathway rank is reported next to the in-distribution rank the same
    cytokine achieved in the cell types the model *did* train on - the comparison that
    says whether prediction degrades off the training design, rather than whether the
    pathway was findable at all.
    """
    expected = EXPECTED_HALLMARK if expected is None else expected
    plot_dir = plot_dir or getattr(analyzer, "plot_dir", None)
    csv_dir = csv_dir or getattr(analyzer, "csv_dir", None)
    condition_key = condition_key or analyzer.labels_key

    if net is None:
        nets = nets or load_nets(cache_dir=cache_dir, verbose=verbose)
        net = nets[net_name]

    combos = [dict(c) for c in holdout_combos]
    pairs = {(c[condition_key], c[cell_type_key]) for c in combos}
    cytokines = sorted({cy for cy, _ in pairs})
    cell_types = sorted({ct for _, ct in pairs})

    signatures, meta = compute_signatures(
        analyzer,
        cytokines=cytokines,
        cell_types=cell_types,
        condition_key=condition_key,
        cell_type_key=cell_type_key,
        clamp_min=clamp_min,
        verbose=False,
    )

    # compute_signatures sweeps the cytokine x cell type grid; keep only the held-out
    # cells of it, so an in-distribution combo cannot be reported as an OOD success
    keep = [k for k in meta.index if (meta.loc[k, "cytokine"], meta.loc[k, "cell_type"]) in pairs]
    meta = meta.loc[keep]
    signatures = {name: frame.loc[frame.index.intersection(keep)]
                  for name, frame in signatures.items()}

    missing = pairs - {(meta.loc[k, "cytokine"], meta.loc[k, "cell_type"]) for k in keep}
    if missing and verbose:
        print(f"[ood] no control pool for {sorted(missing)} - not evaluated")
    unscored = meta.index[~meta["has_real"]]
    if len(unscored) and verbose:
        print(f"[ood] no real held-out cells for {list(unscored)} - predicted only")

    coverage = net_coverage(net, analyzer.adata.var_names)
    acts_pred, padj_pred = run_enrichment(signatures["pred"], net)
    acts_real, padj_real = run_enrichment(signatures["real"], net)

    table_pred = expected_table(acts_pred, padj_pred, meta, expected, coverage)
    table_real = expected_table(acts_real, padj_real, meta, expected, coverage)
    combined = combine_expected(table_pred, table_real)
    agreement = pred_real_agreement(acts_pred, acts_real, meta)

    threshold, nulls = None, None
    if n_perm:
        nulls = null_scores(signatures["pred"], net, n_perm=n_perm, verbose=verbose)
        threshold = null_threshold(nulls)
        combined = empirical_p(combined, nulls, score_col="score_pred")
    elif baseline is not None and baseline.get("threshold"):
        # reuse the in-distribution null rather than re-running it on 10 rows, where the
        # permutation distribution would be far noisier
        threshold = baseline["threshold"]
        combined = empirical_p(combined, baseline["nulls"], score_col="score_pred")

    if baseline is not None and len(combined):
        base = baseline["expected"]
        base = base[~base.set_index(["cytokine", "cell_type"]).index.isin(pairs)]
        in_dist = (
            base.dropna(subset=["rank_pred"])
            .groupby(["cytokine", "pathway"])["rank_pred"]
            .median()
            .rename("rank_pred_in_dist")
        )
        combined = combined.merge(in_dist, on=["cytokine", "pathway"], how="left")

    print("\n=== OOD: combos held out of training ===")
    _summarize(combined, agreement)
    if "rank_pred_in_dist" in combined:
        paired = combined.dropna(subset=["rank_pred", "rank_pred_in_dist"])
        if len(paired):
            print(f"Median expected-pathway rank: {paired['rank_pred'].median():.0f} on held-out "
                  f"combos vs {paired['rank_pred_in_dist'].median():.0f} on trained ones")

    if plot_dir:
        plot_expected_ranks(
            combined,
            threshold=threshold["global"] if threshold else None,
            path=os.path.join(plot_dir, f"enrichment_{net_name}_ood_expected_ranks.png"),
            title="Expected pathway - combos held out of training",
        )
        plot_agreement(
            agreement,
            path=os.path.join(plot_dir, f"enrichment_{net_name}_ood_agreement.png"),
            title="Predicted vs real pathway profile - held-out combos",
        )

    if csv_dir:
        combined.to_csv(os.path.join(csv_dir, f"enrichment_{net_name}_ood_expected.csv"), index=False)
        agreement.to_csv(os.path.join(csv_dir, f"enrichment_{net_name}_ood_agreement.csv"), index=False)

    return {
        "signatures": signatures,
        "meta": meta,
        "acts_pred": acts_pred,
        "acts_real": acts_real,
        "expected": combined,
        "agreement": agreement,
        "threshold": threshold,
        "nulls": nulls,
        "coverage": coverage,
        "net_name": net_name,
    }
