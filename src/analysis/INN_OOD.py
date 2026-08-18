import copy
import os

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import torch

from src.analysis.mmd import ctrl_mmd_dists, mmd_dists


def _combo_label(values, combo_order):
    return "__".join(str(values[k]) for k in combo_order)


def _resolve_cell_type_key(conditions, condition_key, cell_type_key):
    if cell_type_key is None:
        cell_type_key = next(k for k in conditions if k != condition_key)
    if cell_type_key not in conditions:
        raise ValueError(f"Cell-type key '{cell_type_key}' not in conditions {conditions}.")
    return cell_type_key


def _other_cell_types(combo_categories, combo_order, cell_type_key, exclude):
    cell_values = []
    for combo in combo_categories:
        values = dict(zip(combo_order, combo.split("__")))
        cell_values.append(values[cell_type_key])
    cell_values = sorted(set(cell_values))
    return [v for v in cell_values if v != exclude]


def _average_treatment_shift(
    model,
    combo_categories,
    combo_order,
    holdout_combo,
    condition_key,
    control_label,
    cell_type_key,
    target_treatment,
):
    """Average, across every other cell type, of the learned latent-mean shift between
    `target_treatment` and `control_label` - the same estimate
    `estimate_leftout_combo_mean` adds to the held-out combo's control mean."""
    cell_values = _other_cell_types(
        combo_categories, combo_order, cell_type_key, holdout_combo[cell_type_key]
    )

    shifts = []
    for other_cell in cell_values:
        other_control = {**holdout_combo, cell_type_key: other_cell, condition_key: control_label}
        other_treatment = {
            **holdout_combo,
            cell_type_key: other_cell,
            condition_key: target_treatment,
        }

        other_control_label = _combo_label(other_control, combo_order)
        other_treatment_label = _combo_label(other_treatment, combo_order)

        if (
            other_control_label not in combo_categories
            or other_treatment_label not in combo_categories
        ):
            continue

        other_control_idx = combo_categories.index(other_control_label)
        other_treatment_idx = combo_categories.index(other_treatment_label)
        shifts.append(model.means[other_treatment_idx] - model.means[other_control_idx])

    if not shifts:
        raise ValueError(
            "No valid shifts were found for the other cell types. "
            "Check that your `combo_categories` and holdout configuration match the training design."
        )

    return torch.stack(shifts).mean(dim=0)


def _combo_adata(x, holdout_combo, var=None):
    obs = pd.DataFrame({key: [value] * x.shape[0] for key, value in holdout_combo.items()})
    return ad.AnnData(
        X=x.detach().cpu().numpy(),
        obs=obs,
        var=var.copy() if var is not None else None,
    )


def estimate_leftout_combo_mean(
    model,
    combo_categories,
    conditions,
    holdout_combo,
    condition_key,
    control_label,
    cell_type_key=None,
):
    """Estimate and cache the mean for a held-out combo.

    For a held-out combo such as {cell_type: "CD8 T cells", cytokine: "IL-2"}, we use the
    control mean for the same cell type and add the average treatment-control shift across
    all other cell types:

        mu_target ≈ mu(cell_type=target, condition=control)
                    + mean_{other cell types}[ mu(other, treatment) - mu(other, control) ]

    This is useful when the model reserves a slot for the held-out combo but the empirical
    mean was never learned because that combo was left out.

    The function writes the estimated row directly into the underlying mixture-prior means
    tensor so the model can sample from the held-out combo without requiring a real latent
    mean from training data.
    """
    if model is None or getattr(model, "means", None) is None:
        raise ValueError("A model with a `.means` attribute is required.")
    if not isinstance(conditions, (list, tuple)):
        raise TypeError("`conditions` must be a list/tuple of condition column names.")
    if condition_key not in conditions:
        raise ValueError(f"Condition key '{condition_key}' not in conditions {conditions}.")
    cell_type_key = _resolve_cell_type_key(conditions, condition_key, cell_type_key)

    combo_order = list(conditions)

    target_treatment = holdout_combo[condition_key]
    if target_treatment == control_label:
        raise ValueError(
            "The provided holdout combo is already at the control label; no shift is needed."
        )

    target_label = _combo_label(holdout_combo, combo_order)
    if target_label not in combo_categories:
        raise ValueError(
            f"Held-out combo '{target_label}' is not in combo_categories; "
            "make sure the vocabulary was built from the full pre-holdout adata."
        )

    control_target = {**holdout_combo, condition_key: control_label}
    control_target_label = _combo_label(control_target, combo_order)
    if control_target_label not in combo_categories:
        raise ValueError(
            f"Control-side combo '{control_target_label}' is not in combo_categories; "
            "a reserved slot for this cell-type/control row is missing."
        )

    shift = _average_treatment_shift(
        model,
        combo_categories,
        combo_order,
        holdout_combo,
        condition_key,
        control_label,
        cell_type_key,
        target_treatment,
    )

    control_target_idx = combo_categories.index(control_target_label)
    target_idx = combo_categories.index(target_label)

    estimated_mean = model.means[control_target_idx] + shift

    if hasattr(model, "means_active") and model.means_active is not None:
        model.means_active.data[target_idx] = estimated_mean[: model.means_active.shape[1]]
    else:
        model.means.data[target_idx] = estimated_mean

    return estimated_mean


def sample_leftout_combo_normal(
    model,
    combo_categories,
    conditions,
    holdout_combo,
    condition_key,
    control_label,
    cell_type_key=None,
    n_samples=500,
    sigma=1.0,
    var=None,
    dtype=torch.float32,
):
    """OOD sampling strategy 1: draw the held-out combo's cells from an isotropic Normal
    distribution centered at its estimated latent mean (`estimate_leftout_combo_mean`),
    with std taken from the model's learned per-component `log_sigma` when available
    (same convention as `cf_expressions.sample_from_means`), else the constant `sigma`.
    Decodes the samples back through the flow to expression space.

    Returns an AnnData of the sampled cells (`.X` = decoded expression), with one obs
    column per entry of `holdout_combo` set to its value.
    """
    mean = estimate_leftout_combo_mean(
        model=model,
        combo_categories=combo_categories,
        conditions=conditions,
        holdout_combo=holdout_combo,
        condition_key=condition_key,
        control_label=control_label,
        cell_type_key=cell_type_key,
    ).detach()

    combo_order = list(conditions)
    target_idx = combo_categories.index(_combo_label(holdout_combo, combo_order))

    if getattr(model, "log_sigma", None) is not None:
        std = torch.exp(model.log_sigma[target_idx].detach())
    else:
        std = torch.as_tensor(sigma, device=mean.device, dtype=mean.dtype)

    z = mean.unsqueeze(0) + std * torch.randn(
        n_samples, mean.shape[0], device=mean.device, dtype=mean.dtype
    )

    with torch.no_grad():
        x, _ = model(z.to(dtype=dtype), rev=True)

    return _combo_adata(x, holdout_combo, var=var)


def sample_leftout_combo_shift(
    model,
    adata,
    combo_categories,
    conditions,
    holdout_combo,
    condition_key,
    control_label,
    cell_type_key=None,
    device=None,
    dtype=torch.float32,
):
    """OOD sampling strategy 2: encode real control cells of the held-out combo's cell
    type, shift each one in latent space by the average treatment-control mean shift
    across all other cell types (the same shift `estimate_leftout_combo_mean` adds to the
    control mean), then decode. Unlike `sample_leftout_combo_normal`, this carries over
    each control cell's own latent structure (e.g. cell state, library size) instead of
    drawing fresh isotropic noise around a single point.

    Returns an AnnData of the shifted cells (`.X` = decoded expression), one row per real
    control cell of the held-out combo's cell type.
    """
    cell_type_key = _resolve_cell_type_key(conditions, condition_key, cell_type_key)
    combo_order = list(conditions)

    target_treatment = holdout_combo[condition_key]
    if target_treatment == control_label:
        raise ValueError(
            "The provided holdout combo is already at the control label; no shift is needed."
        )

    shift = _average_treatment_shift(
        model,
        combo_categories,
        combo_order,
        holdout_combo,
        condition_key,
        control_label,
        cell_type_key,
        target_treatment,
    ).detach()

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

    if device is None:
        device = shift.device

    x_control = torch.as_tensor(x_control, device=device, dtype=dtype)
    shift = shift.to(device=device, dtype=dtype)

    with torch.no_grad():
        z_control, _ = model(x_control, rev=False)
        z_cf = z_control + shift
        x_cf, _ = model(z_cf, rev=True)

    return _combo_adata(x_cf, holdout_combo, var=adata.var)


def scoped_analyzer_for_combo(analyzer, holdout_combo, conditions, condition_key, cell_type_key=None):
    """A shallow copy of `analyzer` with `.adata` restricted to the held-out combo's cell
    type. `analyzer.labels_key` (e.g. "cytokine") only encodes one of the two conditions,
    so filtering by it alone would pool every cell type sharing that treatment label -
    scoping to the held-out cell type first makes that single-key filtering resolve to
    just this combo, so the existing single-condition evaluation helpers
    (mmd.ctrl_mmd_dists/mmd_dists, cf_expressions.plot_mean_expression_comparison/
    plot_effect_size_comparison) can be reused unchanged. A shallow copy is enough since
    only `.adata` is reassigned (to a new object), never mutated in place.
    """
    cell_type_key = _resolve_cell_type_key(conditions, condition_key, cell_type_key)
    scoped = copy.copy(analyzer)
    scoped.adata = analyzer.adata[
        analyzer.adata.obs[cell_type_key] == holdout_combo[cell_type_key]
    ].copy()
    return scoped


def cfs_fn_from_adata(adata_pred, source_label=None):
    """Wraps a precomputed AnnData of already-generated cells (e.g. from
    sample_leftout_combo_normal/sample_leftout_combo_shift) into the
    `cfs_fn(analyzer, key) -> (x_cfs, z_cfs, source_labels, key)` interface expected by
    cf_expressions.plot_mean_expression_comparison/plot_effect_size_comparison and
    mmd.mmd_dists, so those can be reused for a one-shot set of OOD samples that doesn't
    vary per source condition the way a real counterfactual sweep does."""
    x_cfs = torch.tensor(
        adata_pred.X.toarray() if hasattr(adata_pred.X, "toarray") else adata_pred.X,
        dtype=torch.float32,
    )
    labels = [source_label or "OOD"] * adata_pred.n_obs

    def cfs_fn(analyzer, key):
        return x_cfs, x_cfs, labels, key

    return cfs_fn


def evaluate_leftout_combo_mmd(scoped_analyzer, predictions, key, iter=10, ax=None, plot_dir=None):
    """Bar chart of MMD between the real held-out-combo cells (found via
    `scoped_analyzer`, see `scoped_analyzer_for_combo`) and each named prediction AnnData
    in `predictions` (e.g. {"Normal": adata_ood_normal, "Shift": adata_ood_shift}), plus
    the random-split MMD among the real cells themselves (mmd.ctrl_mmd_dists) as a noise
    floor - how much MMD to expect even between two halves of real data.

    Returns {name: global_mmd} and the control MMD.
    """
    ctrl_mmd = ctrl_mmd_dists(scoped_analyzer, key=key, iter=iter)

    results = {}
    for name, adata_pred in predictions.items():
        global_mmd, _ = mmd_dists(
            scoped_analyzer, key=key, cf_fn=cfs_fn_from_adata(adata_pred, source_label=name)
        )
        results[name] = global_mmd

    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(6, 5))

    names = list(results.keys())
    values = list(results.values())
    ax.bar(names, values, color="steelblue")
    ax.axhline(
        ctrl_mmd,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Control MMD (random split of real)",
    )
    ax.set_ylabel("MMD (vs. real)")
    ax.set_title(f"MMD: OOD sampling strategies vs real ({key})")
    ax.legend()

    if standalone:
        plt.tight_layout()
        if plot_dir:
            plt.savefig(os.path.join(plot_dir, f"leftout_combo_mmd_{key}.png"), dpi=300)
        plt.show()

    return results, ctrl_mmd
