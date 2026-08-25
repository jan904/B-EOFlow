"""Read the per-cell-type gain scalars out of a `--treatment_gain` model.

With `treatment_gain` the factorized prior relaxes strict additivity to

    mu(cell_type, cytokine) = a_cell_type + w_cell_type * (b_cytokine - b_control)

so `w` is one learned scalar per cell type saying how strongly that cell type responds
to the single shared treatment direction. It is a rank-1 interaction: it rescales the
shift, it cannot rotate it - two cell types still respond along the same axis.

**Only ratios of `w` mean anything.** `w * b` has a global scale redundancy - multiply
every `w` by c and every `b` by 1/c and every `mu` is unchanged - so the absolute value
of a single gain is not interpretable, and it is reported relative to the median. The
weights initialise at 1.0, so a run that never moved them looks like a flat column of
1.0 rather than anything meaningful.

`compare_to_data` is the check worth doing: if `w` is picking up real biology, cell types
the model gives a large gain should be the ones that actually move furthest from control
in the data.
"""

import numpy as np
import pandas as pd


def gain_weights(model, config):
    """`(DataFrame, gain_condition)` of per-level gains, or `(None, None)`.

    None when the model has no gain parameter - a plain additive factorized prior, or a
    free-means one. `w_rel` is `w / median(w)`, the only comparable form.
    """
    weights = getattr(model, "treatment_gain", None)
    if weights is None:
        return None, None

    gain_factor = getattr(model, "gain_factor", None)
    conditions = list(getattr(config, "conditions", None) or [])
    categories = getattr(config, "condition_categories", None) or {}

    values = weights.detach().cpu().numpy().astype(float)
    gain_condition = conditions[gain_factor] if conditions and gain_factor is not None else None
    levels = list(categories.get(gain_condition, [])) if gain_condition else []
    if len(levels) != len(values):
        # fall back to positions rather than mislabelling: a wrong level name here would
        # silently attribute one cell type's amplitude to another
        levels = [f"level_{i}" for i in range(len(values))]

    median = float(np.median(values))
    frame = pd.DataFrame({"w": values, "w_rel": values / median if median else np.nan},
                         index=pd.Index(levels, name=gain_condition or "level"))
    return frame.sort_values("w_rel", ascending=False), gain_condition


def real_response_amplitude(adata, condition_key, cell_type_key, control_label):
    """Per cell type, the mean distance from control across treatments, from the data.

    The empirical counterpart of `w`: how far that cell type actually moves when treated,
    averaged over every cytokine present. Combos held out of training are included - the
    model never saw them, so they do not make this circular.
    """
    obs_cond = adata.obs[condition_key].astype(str).to_numpy()
    obs_cell = adata.obs[cell_type_key].astype(str).to_numpy()
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

    rows = {}
    for cell in np.unique(obs_cell):
        control = (obs_cell == cell) & (obs_cond == control_label)
        if control.sum() < 1:
            continue
        base = X[control].mean(0)
        shifts = [
            float(np.linalg.norm(X[(obs_cell == cell) & (obs_cond == cy)].mean(0) - base))
            for cy in np.unique(obs_cond)
            if cy != control_label and ((obs_cell == cell) & (obs_cond == cy)).sum() > 0
        ]
        if shifts:
            rows[cell] = float(np.mean(shifts))
    return pd.Series(rows, name="real_amplitude")


def analyze_gain(model, config, adata=None, control_label=None, plot_path=None, verbose=True):
    """Entry point. Returns the frame (or None), printing a summary when `verbose`."""
    frame, gain_condition = gain_weights(model, config)
    if frame is None:
        if verbose:
            print("This model has no treatment_gain parameter "
                  "(train with --treatment_gain / GAIN=1 to get one).")
        return None

    conditions = list(getattr(config, "conditions", None) or [])
    if adata is not None and gain_condition and len(conditions) == 2:
        condition_key = conditions[1 - conditions.index(gain_condition)]
        label = control_label or getattr(config, "control_label", None)
        if label is not None:
            real = real_response_amplitude(adata, condition_key, gain_condition, label)
            frame = frame.join(real)
            if frame["real_amplitude"].notna().sum() > 2:
                frame["real_rel"] = frame["real_amplitude"] / frame["real_amplitude"].median()

    if verbose:
        spread = frame["w_rel"].max() / frame["w_rel"].min() if frame["w_rel"].min() > 0 else np.nan
        print(f"treatment_gain over {len(frame)} {gain_condition or 'level'}(s)\n")
        print(frame.round(3).to_string())
        print(f"\nspread (max/min of w_rel): {spread:.2f}x" if np.isfinite(spread)
              else "\nspread: undefined (a gain is <= 0)")
        if (frame["w"] <= 0).any():
            flipped = frame.index[frame["w"] <= 0].tolist()
            print(f"NEGATIVE gain on {flipped}: these respond OPPOSITE to the shared "
                  "treatment direction, which is usually a fit artifact rather than biology.")
        if np.allclose(frame["w"].to_numpy(), 1.0, atol=1e-3):
            print("All gains are still ~1.0 - the parameter never moved, so this run is "
                  "effectively additive.")
        if "real_rel" in frame:
            both = frame[["w_rel", "real_rel"]].dropna()
            if len(both) > 2:
                print(f"\nSpearman(w_rel, real_rel) = {both.corr(method='spearman').iloc[0,1]:.3f} "
                      f"over {len(both)} cell types "
                      "(positive = the learned amplitude tracks the real one)")

    if plot_path:
        plot_gain(frame, gain_condition, plot_path)
    return frame


def plot_gain(frame, gain_condition=None, path=None):
    import matplotlib.pyplot as plt

    has_real = "real_rel" in frame and frame["real_rel"].notna().any()
    fig, axes = plt.subplots(1, 2 if has_real else 1, figsize=(12 if has_real else 7, 5))
    axes = np.atleast_1d(axes)

    order = frame.sort_values("w_rel")
    colors = ["crimson" if v <= 0 else "steelblue" for v in order["w"]]
    axes[0].barh(range(len(order)), order["w_rel"], color=colors)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order.index, fontsize=8)
    axes[0].axvline(1.0, color="gray", ls=":", lw=1)
    axes[0].set_xlabel("gain / median gain")
    axes[0].set_title(f"Learned response amplitude per {gain_condition or 'level'}")
    axes[0].grid(alpha=0.3, axis="x")

    if has_real:
        both = frame[["w_rel", "real_rel"]].dropna()
        axes[1].scatter(both["real_rel"], both["w_rel"], color="steelblue")
        for name, row in both.iterrows():
            axes[1].annotate(str(name), (row["real_rel"], row["w_rel"]), fontsize=7, alpha=0.8)
        lim = [0, max(both.max()) * 1.1]
        axes[1].plot(lim, lim, color="gray", ls=":", lw=1)
        axes[1].set_xlabel("real amplitude / median")
        axes[1].set_ylabel("learned gain / median")
        axes[1].set_title("Learned vs real response amplitude")
        axes[1].grid(alpha=0.3)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
