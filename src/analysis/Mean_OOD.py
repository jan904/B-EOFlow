import numpy as np
import pandas as pd
import anndata as ad


def _resolve_cell_type_key(holdout_combo, condition_key, cell_type_key):
    if cell_type_key is None:
        cell_type_key = next(k for k in holdout_combo if k != condition_key)
    if cell_type_key not in holdout_combo:
        raise ValueError(f"Cell-type key '{cell_type_key}' not in holdout_combo {holdout_combo}.")
    return cell_type_key


def _control_mean(adata, holdout_combo, condition_key, control_label, cell_type_key):
    control_mask = (adata.obs[cell_type_key] == holdout_combo[cell_type_key]) & (
        adata.obs[condition_key] == control_label
    )
    if not control_mask.any():
        raise ValueError(
            f"No control cells found for cell type '{holdout_combo[cell_type_key]}' at "
            f"'{condition_key}'='{control_label}'."
        )

    x_control = adata.X[control_mask.to_numpy()]
    x_control = x_control.toarray() if hasattr(x_control, "toarray") else np.asarray(x_control)
    return x_control, x_control.mean(axis=0)


def _combo_adata(x, holdout_combo, var=None, clamp_min=0.0):
    """`clamp_min`: floor applied to the predicted expression before it becomes `.X`,
    matching `INN_OOD._combo_adata` (see its docstring) and the `.clamp(min=0)` the scGen
    path already applies.

    Shifting control cells by a mean delta in log1p space can push genes below zero, which
    is outside the non-negative space the real cells live in. Keeping this convention
    identical across every method is what makes the MMD comparison apples-to-apples -
    clamping one method's predictions but not another's changes the ranking on its own.
    Pass None to keep the unclamped values.
    """
    x = np.asarray(x, dtype=np.float32)
    if clamp_min is not None:
        x = np.clip(x, clamp_min, None)
    obs = pd.DataFrame({key: [value] * x.shape[0] for key, value in holdout_combo.items()})
    return ad.AnnData(
        X=x,
        obs=obs,
        var=var.copy() if var is not None else None,
    )


def estimate_leftout_combo_mean(
    mean_shifter, adata, holdout_combo, condition_key, control_label, cell_type_key=None
):
    """Estimate the mean for a held-out (condition, cell_type) combo - the MeanShifter
    counterpart to INN_OOD.estimate_leftout_combo_mean's `mu(cell_type=target, control) +
    shift`, but simpler: MeanShifter has no per-cell-type resolution at all (`.deltas_`
    pools every cell type into one global shift per condition, see
    logistic_regression.generate_counterfactuals_mean), so holding the combo's rows out
    before `MeanShifter.train()` already makes `deltas_[condition]` exactly "the average
    shift across the other cell types" - the same quantity INN_OOD's
    `_average_treatment_shift` computes explicitly for the INN. There's no `.means`-style
    structure to cache the result into, so this just returns it.
    """
    cell_type_key = _resolve_cell_type_key(holdout_combo, condition_key, cell_type_key)
    target_treatment = holdout_combo[condition_key]
    if target_treatment == control_label:
        raise ValueError(
            "The provided holdout combo is already at the control label; no shift is needed."
        )
    if target_treatment not in mean_shifter.deltas_:
        raise ValueError(
            f"'{target_treatment}' not in mean_shifter.deltas_ - make sure MeanShifter was "
            "trained on data that still contains this treatment for other cell types."
        )

    _, control_mean = _control_mean(adata, holdout_combo, condition_key, control_label, cell_type_key)
    shift = np.asarray(mean_shifter.deltas_[target_treatment]).reshape(-1)

    return control_mean + shift


def sample_leftout_combo_normal(
    mean_shifter,
    adata,
    holdout_combo,
    condition_key,
    control_label,
    cell_type_key=None,
    n_samples=500,
    sigma=1.0,
    var=None,
):
    """OOD sampling strategy 1: draw the held-out combo's cells from an isotropic Normal
    centered at its estimated mean (estimate_leftout_combo_mean) - directly in
    gene-expression space, since MeanShifter has no separate latent space to decode from
    (generate_counterfactuals_mean's "z_cf mirrors x_cf" convention applies here too).
    """
    mean = estimate_leftout_combo_mean(
        mean_shifter, adata, holdout_combo, condition_key, control_label, cell_type_key
    )
    x = np.random.normal(mean, sigma, size=(n_samples, mean.shape[0]))
    return _combo_adata(x, holdout_combo, var=var)


def sample_leftout_combo_shift(
    mean_shifter, adata, holdout_combo, condition_key, control_label, cell_type_key=None
):
    """OOD sampling strategy 2: shift real control cells of the held-out combo's cell
    type directly by `mean_shifter.deltas_[target_treatment]` - no encode/decode, matching
    generate_counterfactuals_mean's shift-in-expression-space convention.
    """
    cell_type_key = _resolve_cell_type_key(holdout_combo, condition_key, cell_type_key)
    target_treatment = holdout_combo[condition_key]
    if target_treatment == control_label:
        raise ValueError(
            "The provided holdout combo is already at the control label; no shift is needed."
        )

    x_control, _ = _control_mean(adata, holdout_combo, condition_key, control_label, cell_type_key)
    shift = np.asarray(mean_shifter.deltas_[target_treatment]).reshape(-1)
    x_cf = x_control + shift

    return _combo_adata(x_cf, holdout_combo, var=adata.var)
