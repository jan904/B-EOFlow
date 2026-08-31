import os
import torch
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import t as t_dist

from src.analysis.logistic_regression import generate_counterfactuals
from src.utils.utils import extract_top_genes


def _get_real_x(analyzer, condition, log1p_transform=False):
    """Dense expression matrix for cells labeled `condition`, library-size
    normalized + log1p'd (via scanpy) when log1p_transform is set, to bring
    raw-count data onto the same scale as an already-log-space method."""
    adata = analyzer.adata[analyzer.adata.obs[analyzer.labels_key] == condition].copy()
    if log1p_transform:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X


def plot_metric_boxplot(per_combo, value, model_key="model", combo_key="combo", ax=None,
                        order=None, color_order=None, hline=None, hline_label=None,
                        ylabel=None, title=None, plot_dir=None, filename=None,
                        annotate=True, ylim=None):
    """Distribution of one per-combo metric across held-out combos, one box per model.

    `per_combo` is the tidy frame the 4.x sections build - one row per (combo, model) -
    and `value` the column to plot. The counterpart of `mmd.plot_mmd_boxplot` for metrics
    that are already dimensionless (R^2, cos2), which is why nothing is normalized here:
    an R^2 means the same thing in every combo, so the raw values are directly comparable
    and a single median hides the spread rather than summarizing it.

    Points are overlaid because ten combos is few enough that the individual values carry
    more than the quartiles do - the same reason as in `plot_mmd_boxplot`, and the reason
    a boxplot is worth more here than the median column of the summary table.

    Model order defaults to first appearance in the frame, so passing the models in the
    same order as elsewhere keeps a model's colour stable across sections; `order` pins it
    explicitly. `hline` draws a reference line, e.g. 0 for a score whose null is "no effect".

    `color_order` decouples colour from position. Without it a model's colour is its index
    among the boxes actually drawn, so two panels of the same figure that show different
    subsets - a variance panel that drops "real" next to a zero-fraction panel that keeps
    it - would give the same model different colours. Pass the full model list to both and
    each keeps one colour throughout.
    """
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]

    models = list(order) if order is not None else list(dict.fromkeys(per_combo[model_key]))
    values = {
        m: per_combo.loc[per_combo[model_key] == m, value].to_numpy(dtype=float)
        for m in models
    }
    values = {m: v[np.isfinite(v)] for m, v in values.items()}
    models = [m for m in models if len(values[m])]

    keys = list(color_order) if color_order is not None else models
    colors = [palette[(keys.index(m) if m in keys else i) % len(palette)]
              for i, m in enumerate(models)]

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 5))

    bp = ax.boxplot([values[m] for m in models], tick_labels=models, patch_artist=True,
                    widths=0.55, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(0)
    for i, (m, color) in enumerate(zip(models, colors), start=1):
        jitter = rng.normal(0, 0.045, size=len(values[m]))
        ax.scatter(np.full(len(values[m]), i) + jitter, values[m], color=color,
                   edgecolor="black", linewidth=0.4, zorder=3, s=26)

    if hline is not None:
        ax.axhline(hline, color="black", ls="--", lw=1.4, label=hline_label)
        if hline_label:
            ax.legend(fontsize=8)

    if annotate:
        for i, m in enumerate(models, start=1):
            ax.annotate(f"med {np.median(values[m]):.2f}", (i, np.median(values[m])),
                        xytext=(10, 0), textcoords="offset points", fontsize=8, va="center")

    ax.set_ylabel(ylabel or value)
    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)

    if standalone and plot_dir and filename:
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    return ax


def effect_size_scores(predicted, truth):
    """Direction and magnitude scores for a predicted effect size, as a dict.

    `cos2` - squared uncentered correlation. Pure *direction*: the residual R^2 the
    prediction would reach after being optimally rescaled, so it is unaffected by a
    systematically too-small or too-large effect.

    `r2` - 1 - SS_res/SS_tot with SS_tot taken about **zero**, not about the mean.
    The natural null for an effect size is "no effect", so r2=0 means no better than
    predicting no change (i.e. than doing nothing) and negative values mean worse than
    that. Unlike the squared Pearson correlation this one does see magnitude errors:
    a prediction with perfect direction at half the true amplitude scores 0.75.

    `scale` - the factor the prediction would have to be multiplied by to minimise the
    residual. >1 means the model under-predicts the effect. `cos2 - r2` is exactly what
    is being lost to that mis-scaling, so the pair separates "wrong direction" from
    "right direction, wrong size" - two failures the squared Pearson correlation
    reported in the same number.
    """
    predicted = np.asarray(predicted, dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).ravel()

    ss_truth = float(truth @ truth)
    ss_pred = float(predicted @ predicted)
    if ss_truth == 0:
        return {"cos2": np.nan, "r2": np.nan, "scale": np.nan}
    if ss_pred == 0:
        # an all-zero prediction is the "do nothing" baseline, which scores exactly 0 by
        # construction rather than being undefined; only its rescaling is meaningless
        return {"cos2": 0.0, "r2": 0.0, "scale": np.nan}

    dot = float(predicted @ truth)
    return {
        "cos2": dot**2 / (ss_pred * ss_truth),
        "r2": 1 - float(((predicted - truth) ** 2).sum()) / ss_truth,
        "scale": dot / ss_pred,
    }


def gene_separability(real, generated, var_names, detect_threshold=0.1, canonicalize=True):
    """Per-gene realism: how separable is each gene on its own, and how does it differ?

    `auc` is the one-dimensional ROC AUC for telling real from generated cells using that
    gene alone (0.5 = indistinguishable, folded so it is always >= 0.5). Reading it next to
    the whole-transcriptome classifier of section 4.8 is the point: individually the genes
    are nearly fine (median ~0.56 among expressed ones) while the 1937-gene classifier
    reaches 0.99, so its power comes from summing many weakly-wrong marginals rather than
    from any one gene being wrong. That is why patching individual defects - clamping,
    snapping the forbidden zone, erasing the silent-gene leak - each moved the AUC so little.

    `canonicalize`: put both sides in one space first (see `mmd._canonicalize`, which
    detects it per matrix). Required, not cosmetic: scVI's predictions are count-scale while
    EOFlow's are log-normalized, so a fixed `detect_threshold` of 0.1 means "a tenth of a
    count" for one model and "detectably expressed" for another, and the AUCs would not be
    comparable across models either.

    `detect_*` is the fraction of cells in which the gene exceeds `detect_threshold`, i.e.
    the on/off structure. This is where the systematic error lives: the model turns genes on
    too often (~0.79 vs 0.71 over expressed genes, higher in 80% of them) while keeping the
    mean roughly right, so it compensates by lowering the expressed level.
    """
    def _dense(x):
        x = x.X if hasattr(x, "X") else x
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        return np.asarray(x, dtype=np.float32)

    from sklearn.metrics import roc_auc_score

    if canonicalize:
        from src.analysis.mmd import _canonicalize
        real_x, gen_x = _canonicalize(real), _canonicalize(generated)
    else:
        real_x, gen_x = _dense(real), _dense(generated)
    labels = np.r_[np.zeros(len(real_x)), np.ones(len(gen_x))]
    stacked = np.vstack([real_x, gen_x])

    auc = np.empty(stacked.shape[1])
    for g in range(stacked.shape[1]):
        score = roc_auc_score(labels, stacked[:, g])
        auc[g] = max(score, 1 - score)

    return pd.DataFrame({
        "gene": np.asarray(var_names),
        "auc": auc,
        "detect_real": (real_x > detect_threshold).mean(0),
        "detect_gen": (gen_x > detect_threshold).mean(0),
        "mean_real": real_x.mean(0),
        "mean_gen": gen_x.mean(0),
        "sd_real": real_x.std(0),
        "sd_gen": gen_x.std(0),
    }).set_index("gene")


def plot_gene_violins(real, predictions, var_names, genes=None, n_genes=6, min_detection=0.30,
                      detect_threshold=0.1, axes=None, plot_dir=None,
                      filename="gene_violins.png", canonicalize=True):
    """Real vs each model's cells, gene by gene.

    Unlike `plot_top_deg_violin`, which picks genes for *responding to the perturbation*,
    genes here are picked for being **badly modelled** - the least realistic among those
    actually expressed. Restricting to genes detected in at least `min_detection` of real
    cells matters: without it the selection fills up with genes that are structurally silent
    in real cells and leak in the predictions (see `silent_gene_leakage`), which is a
    different and much smaller defect.

    Look for the zero spike: a real gene is typically bimodal - a spike at zero plus an
    expressed mode - and the failure mode is the model replacing that with one broad
    unimodal smear covering both, i.e. losing "this gene is off in *this* cell".

    `canonicalize` (default True) puts every model in one space before plotting. All the
    violins for a gene share a y-axis, so without it a count-space model (scVI) sets the
    range in the hundreds and flattens every log-space model against the axis.
    """
    def _dense(x):
        x = x.X if hasattr(x, "X") else x
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        return np.asarray(x, dtype=np.float32)

    if canonicalize:
        from src.analysis.mmd import _canonicalize
        prep = _canonicalize
    else:
        prep = _dense

    real_x = prep(real)
    predictions = {name: prep(data) for name, data in predictions.items()}
    var_names = np.asarray(var_names)
    names = list(predictions)

    if genes is None:
        tables = [gene_separability(real_x, predictions[n], var_names, detect_threshold,
                                    canonicalize=False)
                  for n in names]
        auc = np.mean([t["auc"].to_numpy() for t in tables], axis=0)
        expressed = tables[0]["detect_real"].to_numpy() >= min_detection
        if not expressed.any():
            raise ValueError(f"No gene is detected in >= {min_detection:.0%} of the real cells.")
        ranked = np.where(expressed)[0][np.argsort(-auc[expressed])]
        chosen = ranked[:n_genes]
    else:
        chosen = np.array([int(np.where(var_names == g)[0][0]) for g in genes])
        auc = None

    standalone = axes is None
    if standalone:
        ncols = min(len(chosen), 3)
        nrows = int(np.ceil(len(chosen) / ncols))
        _, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows), squeeze=False)
        axes = axes.flat

    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    for ax, g in zip(axes, chosen):
        data = [real_x[:, g]] + [predictions[n][:, g] for n in names]
        parts = ax.violinplot(data, showextrema=False, widths=0.85)
        for body, color in zip(parts["bodies"], palette):
            body.set_facecolor(color)
            body.set_alpha(0.7)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels(["real"] + names, rotation=20, fontsize=7)
        detected = [(d > detect_threshold).mean() for d in data]
        title = f"{var_names[g]}"
        if auc is not None:
            title += f"  (mean AUC {auc[g]:.2f})"
        ax.set_title(title + "\ndetected: " + " / ".join(f"{d:.0%}" for d in detected), fontsize=8)
        ax.set_ylabel("log-normalized expression", fontsize=8)

    axes = list(axes)
    for ax in axes[len(chosen):]:
        ax.axis("off")

    if standalone:
        plt.tight_layout()
        if plot_dir:
            plt.savefig(os.path.join(plot_dir, filename), dpi=300)
        plt.show()

    return axes


def plot_expression_histograms(panels, bins=200, xlim=(-1.5, 8), zoom=(-0.6, 1.2),
                               axes=None, plot_dir=None, filename="expression_histograms.png"):
    """Distribution of *all* gene-expression entries, real against each model's output.

    The single most direct check of generative realism, and it shows three things at once
    that the summary metrics each see only part of:

    * the **atom at zero** (~66-73% of real entries) that the model replaces with a smooth
      hump of roughly the width of the dequantization noise it was trained with - a
      continuous density cannot put mass on a point, so this is structural, not tuning;
    * the **negative tail**, which real expression never has (~32% of raw decoder entries);
    * the **forbidden zone** between 0 and one count (~0.94 in log space at median depth,
      narrower for deeper metacells) where real data cannot land and the model puts mass.

    Above ~1.0 the two distributions typically overlap closely, which is why MMD and the
    effect sizes look reasonable while the classifier separates them perfectly: everything
    that distinguishes them lives in the low-expression region this plot zooms into.

    Pass predictions *unclamped* to see the negative tail - `sample_leftout_combo_shift`
    and the other samplers take `clamp_min=None` for exactly this. With the clamped output
    the negatives instead pile up at exactly 0, which is worth knowing when reading it.

    `panels`: {name: (real, generated)} or {name: (real, generated, log1p_transform)}. The
    real cells are given **per panel**, not once for all of them, because each model's
    analyzer holds its own space - scVI's are raw counts while EOFlow's are log-normalized,
    and one shared reference would compare a log-space histogram against a count-space one.
    Set `log1p_transform=True` for a count-space model to bring both of its sides onto the
    log axis. The panels are then each internally consistent, though their x-axes are only
    comparable among models sharing a space.

    The transform is sign-preserving - `sign(x) * log1p(|x|)`, with the library size taken
    from the non-negative part - so negatives survive it instead of turning into NaN. For
    x >= 0 it is exactly the usual normalize_total + log1p.
    """
    def _dense(x):
        x = x.X if hasattr(x, "X") else x
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        return np.asarray(x, dtype=np.float32)

    def _prepare(x, do_log):
        x = _dense(x)
        if do_log:
            totals = np.clip(x, 0, None).sum(axis=1, keepdims=True)
            totals[totals == 0] = 1.0
            x = x * 1e4 / totals
            x = np.sign(x) * np.log1p(np.abs(x))
        return x.ravel()

    names = list(panels)

    standalone = axes is None
    if standalone:
        ncols = min(len(names), 2)
        nrows = int(np.ceil(len(names) / ncols))
        _, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows), squeeze=False)
        axes = axes.flat

    edges = np.linspace(*xlim, bins)
    for ax, name in zip(axes, names):
        entry = panels[name]
        real_x, gen_x = entry[0], entry[1]
        do_log = entry[2] if len(entry) > 2 else False
        real_v = _prepare(real_x, do_log)
        gen = _prepare(gen_x, do_log)
        ax.hist(real_v, bins=edges, alpha=0.55, density=True, color="#2a78d6", label="real")
        ax.hist(gen, bins=edges, alpha=0.55, density=True, color="#eb6834", label=name)
        ax.axvline(0, color="black", lw=1, ls="--")
        if zoom is not None:
            inset = ax.inset_axes([0.55, 0.45, 0.42, 0.5])
            inset.hist(real_v, bins=edges, alpha=0.55, density=True, color="#2a78d6")
            inset.hist(gen, bins=edges, alpha=0.55, density=True, color="#eb6834")
            inset.set_xlim(*zoom)
            inset.tick_params(labelsize=6)
            inset.set_title("zoom on zero", fontsize=7)
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_xlabel("log-normalized expression")
        ax.set_ylabel("density (log scale)")
        ax.set_title(
            f"{name}\nreal zeros {(real_v == 0).mean():.1%} | "
            f"generated at 0 {(gen == 0).mean():.1%} | negative {(gen < 0).mean():.1%}",
            fontsize=9,
        )
        ax.legend(fontsize=8)

    axes = list(axes)
    for ax in axes[len(names):]:  # a 2-column grid leaves one empty for an odd model count
        ax.axis("off")

    if standalone:
        plt.tight_layout()
        if plot_dir:
            plt.savefig(os.path.join(plot_dir, filename), dpi=300)
        plt.show()

    return axes


def negativity_table(predictions, real=None):
    """How much of each model's decoded output falls below zero.

    Expression cannot be negative in either space these models work in, so every negative
    entry is a decoding artefact: the flow's decoder ends in an unconstrained linear layer,
    and scGen's does too. It matters because *every* other metric silently hides it - the
    OOD samplers clamp (`INN_OOD._combo_adata`), the violins clamp, the classifier now
    clamps, and `mmd._canonicalize` clips before normalizing - so a model can be a third
    negative and still score well everywhere.

    Measured on the RAW predictions, before any clamping or canonicalization; pass the
    untouched output of the counterfactual function.

    On whether the classifier exploits it: a *linear* discriminator cannot, since it has no
    way to form a "this value is negative" feature - negatives merely shift its score, and
    on a matched test they lowered AUC (0.971 raw vs 0.997 clamped) by blurring the signal
    it does use. A classifier with a hidden layer can, and does. So negativity is a real
    defect that the linear C2ST under-reports rather than reports.
    """
    rows = []
    items = dict(predictions)
    if real is not None:
        items = {"real (reference)": real, **items}

    for name, data in items.items():
        x = data.X if hasattr(data, "X") else data
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        x = np.asarray(x, dtype=np.float64)
        neg = x < 0
        rows.append({
            "model": name,
            "% entries < 0": float(neg.mean()),
            "% cells with any": float(neg.any(axis=1).mean()),
            "min": float(x.min()),
            "mean where < 0": float(x[neg].mean()) if neg.any() else 0.0,
            "negative mass %": float(-x[neg].sum() / np.abs(x).sum()) if neg.any() else 0.0,
        })
    return pd.DataFrame(rows).set_index("model")


def silent_gene_leakage(real, predictions, var_names, top_n=8, tol=1e-6, reference=None):
    """How much expression each model puts into genes that are *never* expressed in the
    real cells of this combo.

    A different gene selection from `extract_top_genes`, and it answers a different
    question: those genes are picked for responding to the perturbation, these for being
    structurally off - immunoglobulin V genes in monocytes, lineage markers of the wrong
    lineage, pseudogenes. A flow with a Gaussian latent has full support in gene space and
    no way to hold a gene at exactly zero, so it leaks small values into them.

    This is what a discriminator trained on real-vs-generated cells actually keys on: for
    IL-4 x CD14 Mono the top coefficients were all IGLV/IGHV/IGKV genes, zero in 100% of
    real cells and nonzero in up to 32% of predictions. It is invisible to MMD - the leak
    contributes ~2% of the kernel bandwidth - and to any DE-based score, because these
    genes are never in a DE list. Reported as a table plus violins because the magnitude
    (~0.02 in log space, against ~1.6 for a typically expressed gene) matters as much as
    the presence: it is a real defect, and a small one.

    `reference`: extra real cells (e.g. the control cells of the same cell type) that must
    *also* be zero for a gene to count as silent. Worth passing: with only ~160 cells on the
    treated side alone, some genes are zero by sampling rather than by biology, and those
    are not evidence of anything when a model puts expression into them.

    Returns (table, gene_names): a per-model DataFrame and the worst `top_n` genes by mean
    leaked value, for plotting.
    """
    from src.analysis.mmd import _canonicalize

    real_c = _canonicalize(real)
    silent = (np.abs(real_c) < tol).all(axis=0)
    if reference is not None:
        silent &= (np.abs(_canonicalize(reference)) < tol).all(axis=0)
    var_names = np.asarray(var_names)

    rows, leaked = [], {}
    for name, data in predictions.items():
        pred = _canonicalize(data)[:, silent]
        leaked[name] = pred
        nonzero = pred > 0.1
        rows.append({
            "model": name,
            "silent genes": int(silent.sum()),
            "% entries nonzero": float(nonzero.mean()),
            "mean value": float(pred.mean()),
            "mean where nonzero": float(pred[nonzero].mean()) if nonzero.any() else 0.0,
            "max value": float(pred.max()),
        })

    table = pd.DataFrame(rows).set_index("model")
    table.attrs["leaked"] = {k: v.ravel() for k, v in leaked.items()}
    pooled = np.mean([p.mean(axis=0) for p in leaked.values()], axis=0)
    worst = np.argsort(-pooled)[:top_n]
    return table, list(var_names[silent][worst])


def plot_silent_gene_leakage(real, predictions, var_names, top_n=8, ax=None, plot_dir=None,
                             reference=None):
    """Violins of what each model puts into the structurally silent genes.

    The real cells are a point mass at zero by construction - that is the definition of the
    gene set - so they are drawn as a reference line rather than a violin.
    """
    from src.analysis.mmd import _canonicalize

    table, genes = silent_gene_leakage(
        real, predictions, var_names, top_n=top_n, reference=reference
    )
    var_names = np.asarray(var_names)
    n_silent = int(table["silent genes"].iloc[0])
    idx = [int(np.where(var_names == g)[0][0]) for g in genes]

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    names = list(predictions)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    leaked_values = table.attrs["leaked"]

    # Only the entries that actually leak, on a log axis: a violin over all entries is
    # ~96% exact zeros and shows nothing but a spike. The share that leaks is in the title.
    pooled, labels = [], []
    for name in names:
        vals = leaked_values[name]
        pooled.append(np.log10(vals[vals > 1e-3]) if (vals > 1e-3).any() else np.array([-3.0]))
        labels.append(f"{name}\n{(vals > 0.1).mean():.1%} nonzero")
    parts = ax[0].violinplot(pooled, showextrema=False, widths=0.8)
    for body, color in zip(parts["bodies"], palette):
        body.set_facecolor(color)
        body.set_alpha(0.7)
    ax[0].axhline(np.log10(0.1), color="black", ls="--", lw=1.2, label="0.1 (detection-ish)")
    ax[0].set_xticks(range(1, len(names) + 1))
    ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("log10 predicted expression, nonzero entries")
    ax[0].set_title(f"Leak into {n_silent} silent genes (real = 0 in all of them)")
    ax[0].legend(fontsize=8)

    width = 0.8 / len(names)
    x = np.arange(len(genes))
    for i, (name, color) in enumerate(zip(names, palette)):
        pred = _canonicalize(predictions[name])[:, idx]
        ax[1].bar(x + i * width - 0.4, pred.mean(axis=0), width, label=name, color=color)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(genes, rotation=60, ha="right", fontsize=8)
    ax[1].set_ylabel("mean predicted expression")
    ax[1].set_title(f"Worst {top_n} silent genes (real = 0 in every cell)")
    ax[1].legend(fontsize=8)

    if standalone:
        plt.tight_layout()
        if plot_dir:
            plt.savefig(os.path.join(plot_dir, "ood_silent_genes.png"), dpi=300)
        plt.show()

    return ax, table


def comparable_support_table(real, control, predictions, top_n=50, extra=None):
    """Per-model table of how well each prediction reproduces the real cells' *support*,
    alongside the effect-size scores - all in one canonical space.

    Every matrix is put through `mmd._canonicalize` first (library-size normalized,
    log1p'd, space detected per matrix), because none of these quantities are comparable
    across representations: a variance computed on raw counts and one computed on
    log-normalized data differ by orders of magnitude for reasons that have nothing to do
    with model quality.

    Columns:
      `var ratio`    - total variance of the prediction over that of the real cells.
                       Counterfactuals tend to come out *under*-dispersed, and this is the
                       part of the gap MMD reacts to most strongly.
      `zero frac`    - fraction of entries within 0.1 of zero, against the real cells'
                       own value. These metacells are ~73% zeros; a continuous flow cannot
                       place an atom at zero, so this quantifies how close it gets.
                       Measured with a tolerance because an exact `== 0` test reports 0%
                       even for a perfect round trip (it returns zeros as +-2e-5).
      `cos2`/`R2_res`- direction and magnitude of the predicted mean effect on the top
                       `top_n` genes by |effect|, one shared gene set across models so the
                       numbers are comparable (see `effect_size_scores`).

    `extra`: optional {model: {column: value}} merged in, e.g. the MMD values from
    `mmd.comparable_ood_mmd`, so one table carries the whole comparison.
    """
    from src.analysis.mmd import _canonicalize

    real_c = _canonicalize(real)
    control_c = _canonicalize(control)
    real_delta = real_c.mean(0) - control_c.mean(0)
    top = np.argsort(-np.abs(real_delta))[:top_n]

    real_var = float(real_c.var(0).sum())
    zero_frac = lambda x: float((np.abs(x) < 0.1).mean())

    rows = []
    for name, data in predictions.items():
        pred = _canonicalize(data)
        scores = effect_size_scores(pred.mean(0)[top] - control_c.mean(0)[top], real_delta[top])
        row = {
            "model": name,
            "var ratio": float(pred.var(0).sum()) / real_var,
            "zero frac": zero_frac(pred),
            "cos2": scores["cos2"],
            "R2_res": scores["r2"],
            "best rescale": scores["scale"],
        }
        if extra and name in extra:
            row.update(extra[name])
        rows.append(row)

    table = pd.DataFrame(rows).set_index("model")
    table.attrs["real_var"] = real_var
    table.attrs["real_zero_frac"] = zero_frac(real_c)
    return table


def plot_support_table(table, ax=None, plot_dir=None):
    """Variance ratio and zero fraction per model, against the real cells' own values."""
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
    names = list(table.index)
    colors = [palette[i % len(palette)] for i in range(len(names))]

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].bar(names, table["var ratio"], color=colors)
    ax[0].axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="real")
    ax[0].set_ylabel("variance / real variance")
    ax[0].set_title("Dispersion (1.0 = matches real)")
    ax[0].legend()

    ax[1].bar(names, table["zero frac"], color=colors)
    ax[1].axhline(
        table.attrs["real_zero_frac"], color="black", linestyle="--", linewidth=1.5,
        label=f"real = {table.attrs['real_zero_frac']:.3f}",
    )
    ax[1].set_ylabel("fraction of entries |x| < 0.1")
    ax[1].set_title("Zero inflation")
    ax[1].legend()

    for axis in ax:
        axis.tick_params(axis="x", rotation=20)

    if standalone:
        plt.tight_layout()
        if plot_dir:
            plt.savefig(os.path.join(plot_dir, "ood_support.png"), dpi=300)
        plt.show()

    return ax


def _log1p_transform_cfs(x_cfs):
    """Same transform as _get_real_x, for a counterfactual tensor with no
    accompanying AnnData - wraps it in one just for the scanpy call."""
    adata_cfs = ad.AnnData(X=x_cfs.cpu().numpy())
    sc.pp.normalize_total(adata_cfs, target_sum=1e4)
    sc.pp.log1p(adata_cfs)
    return torch.tensor(adata_cfs.X, dtype=torch.float32)


def plot_mean_expression_comparison(
    analyzer,
    key,
    top_n=10,
    cfs_fn=generate_counterfactuals,
    log1p_transform=False,
    method="EOFlow",
    ax=None,
):
    top_gene_names, gene_indices = extract_top_genes(
        analyzer, key, top_n=top_n, log1p_transform=log1p_transform
    )

    x_cfs, z_cfs, all_source_labels, _ = cfs_fn(analyzer, key=key)
    if log1p_transform:
        x_cfs = _log1p_transform_cfs(x_cfs)

    x = _get_real_x(analyzer, key, log1p_transform=log1p_transform)

    mean_cfs = x_cfs.mean(dim=0).cpu().numpy()
    mean_real = x.mean(axis=0)

    ctrl = np.linspace(0, max(mean_real.max(), mean_cfs.max()), 100)

    # OLS fit of predicted (mean_cfs) on ground truth (mean_real), with a 95% CI
    # band for the regression estimate (i.e. for E[mean_cfs | mean_real], not for
    # individual new points).
    slope, intercept, _, _, _ = linregress(mean_real, mean_cfs)
    fit = slope * ctrl + intercept

    n = len(mean_real)
    x_bar = mean_real.mean()
    Sxx = np.sum((mean_real - x_bar) ** 2)
    residuals = mean_cfs - (slope * mean_real + intercept)
    mse = np.sum(residuals**2) / (n - 2)
    se_fit = np.sqrt(mse * (1 / n + (ctrl - x_bar) ** 2 / Sxx))
    ci = t_dist.ppf(0.975, df=n - 2) * se_fit

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots()

    ax.fill_between(
        ctrl,
        fit - ci,
        fit + ci,
        color="steelblue",
        alpha=0.2,
        zorder=1,
    )
    ax.plot(ctrl, fit, color="steelblue", zorder=2)
    ax.plot(ctrl, ctrl, color="gray", linestyle="--", zorder=2)

    ax.scatter(mean_real, mean_cfs, zorder=3)

    top_genes_cfs = mean_cfs[gene_indices]
    top_genes_real = mean_real[gene_indices]
    ax.scatter(
        top_genes_real, top_genes_cfs, color="red", zorder=4, label=f"top {top_n} genes for {key}"
    )

    for gene_name, gx, gy in zip(top_gene_names, top_genes_real, top_genes_cfs):
        ax.annotate(
            gene_name,
            (gx, gy),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="red",
        )

    r_squared_deg = linregress(mean_cfs[gene_indices], mean_real[gene_indices]).rvalue ** 2
    ax.plot([], [], " ", label=f"$R^2_{{DEGs}}$ = {r_squared_deg:.3f}")
    r_squared = linregress(mean_cfs, mean_real).rvalue ** 2
    ax.plot([], [], " ", label=f"$R^2_{{all}}$ = {r_squared:.3f}")
    ax.legend()

    ax.set_xlabel("ground truth")
    ax.set_ylabel("predicted")
    ax.set_title(f"Mean expression comparison for {key} ({method})")

    if standalone:
        plt.tight_layout()
        plt.savefig(
            os.path.join(analyzer.plot_dir, f"mean_expression_comparison_{key}.png"), dpi=300
        )
        plt.show()


def plot_effect_size_comparison(
    analyzer,
    key,
    top_n=10,
    cfs_fn=generate_counterfactuals,
    log1p_transform=False,
    method="EOFlow",
    ax=None,
):
    """Same as plot_mean_expression_comparison, but on effect sizes (mean(key) -
    mean(control)) instead of raw means, for both ground truth and predicted. Most
    genes barely respond to a perturbation, so raw expression is dominated by
    baseline level (highly expressed genes inflate R^2 even for a poor model);
    subtracting the same ground-truth control mean from both sides isolates the
    actual perturbation signal the model is supposed to capture.
    """
    top_gene_names, gene_indices = extract_top_genes(
        analyzer, key, top_n=top_n, log1p_transform=log1p_transform
    )

    x_cfs, z_cfs, all_source_labels, _ = cfs_fn(analyzer, key=key)
    if log1p_transform:
        x_cfs = _log1p_transform_cfs(x_cfs)

    x_key = _get_real_x(analyzer, key, log1p_transform=log1p_transform)
    x_ctrl = _get_real_x(analyzer, analyzer.control_label, log1p_transform=log1p_transform)

    mean_ctrl = x_ctrl.mean(axis=0)
    mean_cfs = x_cfs.mean(dim=0).cpu().numpy()
    mean_real = x_key.mean(axis=0)

    # effect size = perturbed - control, ground-truth control mean used for both
    # sides so any mismatch reflects perturbation-effect quality, not baseline drift
    delta_cfs = mean_cfs - mean_ctrl
    delta_real = mean_real - mean_ctrl

    line_range = np.linspace(
        min(delta_real.min(), delta_cfs.min()), max(delta_real.max(), delta_cfs.max()), 100
    )

    # OLS fit of predicted (delta_cfs) on ground truth (delta_real), with a 95% CI
    # band for the regression estimate (i.e. for E[delta_cfs | delta_real], not for
    # individual new points).
    slope, intercept, _, _, _ = linregress(delta_real, delta_cfs)
    fit = slope * line_range + intercept

    n = len(delta_real)
    x_bar = delta_real.mean()
    Sxx = np.sum((delta_real - x_bar) ** 2)
    residuals = delta_cfs - (slope * delta_real + intercept)
    mse = np.sum(residuals**2) / (n - 2)
    se_fit = np.sqrt(mse * (1 / n + (line_range - x_bar) ** 2 / Sxx))
    ci = t_dist.ppf(0.975, df=n - 2) * se_fit

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots()

    ax.fill_between(
        line_range,
        fit - ci,
        fit + ci,
        color="steelblue",
        alpha=0.2,
        zorder=1,
    )
    ax.plot(line_range, fit, color="steelblue", zorder=2)
    ax.plot(line_range, line_range, color="gray", linestyle="--", zorder=2)

    ax.scatter(delta_real, delta_cfs, zorder=3)

    top_genes_cfs = delta_cfs[gene_indices]
    top_genes_real = delta_real[gene_indices]
    ax.scatter(
        top_genes_real, top_genes_cfs, color="red", zorder=4, label=f"top {top_n} genes for {key}"
    )

    for gene_name, gx, gy in zip(top_gene_names, top_genes_real, top_genes_cfs):
        ax.annotate(
            gene_name,
            (gx, gy),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="red",
        )

    # squared Pearson (kept for continuity with older figures) plus the two scores that
    # separate direction from magnitude - see `effect_size_scores`. Worth reading the
    # three together: a large gap between cos^2 and R^2 means the effect is pointing the
    # right way but at the wrong amplitude, which the Pearson value cannot show at all.
    r_squared_deg = linregress(delta_cfs[gene_indices], delta_real[gene_indices]).rvalue ** 2
    ax.plot([], [], " ", label=f"$R^2_{{DEGs}}$ = {r_squared_deg:.3f}")
    r_squared = linregress(delta_cfs, delta_real).rvalue ** 2
    ax.plot([], [], " ", label=f"$R^2_{{all}}$ = {r_squared:.3f}")

    deg_scores = effect_size_scores(delta_cfs[gene_indices], delta_real[gene_indices])
    all_scores = effect_size_scores(delta_cfs, delta_real)
    ax.plot([], [], " ", label=f"$\\cos^2_{{DEGs}}$ = {deg_scores['cos2']:.3f}   "
                               f"$R^2_{{res, DEGs}}$ = {deg_scores['r2']:.3f}")
    ax.plot([], [], " ", label=f"$\\cos^2_{{all}}$ = {all_scores['cos2']:.3f}   "
                               f"$R^2_{{res, all}}$ = {all_scores['r2']:.3f}")
    ax.plot([], [], " ", label=f"best rescale = {all_scores['scale']:.2f}x")
    ax.legend(fontsize=8)

    ax.set_xlabel("ground truth effect size")
    ax.set_ylabel("predicted effect size")
    ax.set_title(f"Effect size comparison for {key} ({method})")

    if standalone:
        plt.tight_layout()
        plt.savefig(os.path.join(analyzer.plot_dir, f"effect_size_comparison_{key}.png"), dpi=300)
        plt.show()


def plot_top_deg_violin(
    analyzer, key, cfs_fn=generate_counterfactuals, log1p_transform=False, gene_idx=0, ax=None
):
    top_gene_names, gene_indices = extract_top_genes(
        analyzer, key, top_n=20, log1p_transform=log1p_transform
    )
    gene_name = top_gene_names[gene_idx]
    gene_idx = gene_indices[gene_idx]

    control_values = _get_real_x(analyzer, analyzer.control_label, log1p_transform)[:, gene_idx]
    key_values = _get_real_x(analyzer, key, log1p_transform)[:, gene_idx]

    x_cfs, z_cfs, all_source_labels, _ = cfs_fn(analyzer, key=key)
    # counterfactual = control cells predicted forward to `key`, matching the classic
    # ctrl -> stim comparison (rather than pooling counterfactuals from every source)
    cf_mask = np.asarray(all_source_labels) == analyzer.control_label

    if not cf_mask.any():
        # cfs_fn doesn't tag rows by source condition - e.g. INN_OOD.cfs_fn_from_adata,
        # which wraps an already-generated OOD set under a single label ("Shift"). Those
        # predictions are for one combo only, so there is no per-source split to make and
        # every row is the comparison of interest. Without this the mask selects nothing
        # and the quantile below raises on an empty array.
        cf_mask = np.ones(len(all_source_labels), dtype=bool)

    if log1p_transform:
        # normalize the full per-cell profile (all genes) before slicing to gene_idx -
        # library-size normalization only makes sense computed across all genes,
        # matching how _get_real_x normalizes before indexing into a single gene
        x_cfs = _log1p_transform_cfs(x_cfs)
    x_cfs_arr = x_cfs.cpu().numpy()

    cf_values = x_cfs_arr[cf_mask, gene_idx]
    cf_values = cf_values.clip(min=0, max=np.quantile(cf_values, 0.99))

    cf_label = f"{key} (counterfactual)"
    order = [analyzer.control_label, cf_label, key]
    values = np.concatenate([control_values, cf_values, key_values]).astype(np.float32)
    groups = (
        [analyzer.control_label] * len(control_values)
        + [cf_label] * len(cf_values)
        + [key] * len(key_values)
    )

    violin_adata = ad.AnnData(
        X=values.reshape(-1, 1),
        var=pd.DataFrame(index=[str(gene_name)]),
        obs=pd.DataFrame({"group": pd.Categorical(groups, categories=order)}),
    )

    standalone = ax is None

    violin_ax = sc.pl.violin(
        violin_adata,
        keys=str(gene_name),
        groupby="group",
        order=order,
        use_raw=False,
        xlabel="",
        ylabel="expression",
        show=False,
        ax=ax,
    )
    violin_ax.set_title(f"Top DEG for {key}: {gene_name}")

    if standalone:
        plt.tight_layout()
        plt.savefig(os.path.join(analyzer.plot_dir, f"top_deg_violin_{key}.png"), dpi=300)
        plt.show()


def sample_from_means(analyzer, sigma=1.0):
    means = analyzer.model.means.detach().cpu().numpy()
    if analyzer.model.log_sigma is not None:
        sigmas = np.exp(analyzer.model.log_sigma.detach().cpu().numpy())
    else:
        sigmas = np.full_like(means, sigma)

    class_counts = np.bincount(analyzer.adata.obs[analyzer.labels_key].cat.codes)
    classes = analyzer.adata.obs[analyzer.labels_key].cat.categories.tolist()

    all_z = []
    all_labels = []
    all_cell_types = []
    for mean, sigma, class_name, count in zip(means, sigmas, classes, class_counts):
        z = np.random.normal(mean, sigma, size=(count, mean.shape[0]))
        all_z.append(z)
        all_labels.extend([class_name] * count)

        cell_type = (
            analyzer.adata.obs.loc[
                analyzer.adata.obs[analyzer.labels_key] == class_name, "cell_type"
            ]
            .mode()
            .iat[0]
        )
        all_cell_types.extend([cell_type] * count)

    z = torch.tensor(np.concatenate(all_z, axis=0), dtype=analyzer.dtype, device=analyzer.device)

    with torch.no_grad():
        # a hybrid flow decodes per cell type; `all_cell_types` already holds one entry per
        # sampled row, in the same order as `z`. None for a mixture model.
        c = None
        if getattr(analyzer.model, "condition_type", None) == "hybrid":
            cats = list(getattr(analyzer.model, "hard_categories", None) or [])
            missing = sorted(set(all_cell_types) - set(cats))
            if missing:
                raise ValueError(f"cell types not in the model's hard_categories: {missing}")
            c = analyzer.model.flow_condition(
                [cats.index(t) for t in all_cell_types], device=z.device, dtype=z.dtype
            )
        x, _ = analyzer.model(z, c=c, rev=True)

    adata_sampled = ad.AnnData(
        X=x.cpu().numpy(),
        obs=pd.DataFrame(
            {
                analyzer.labels_key: pd.Categorical(all_labels, categories=classes),
                "cell_type": pd.Categorical(all_cell_types),
            }
        ),
        var=analyzer.adata.var.copy(),
    )

    return adata_sampled
