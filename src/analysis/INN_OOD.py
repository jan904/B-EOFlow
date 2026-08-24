import copy
import os

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import roc_curve, auc

from src.analysis.mmd import ctrl_mmd_dists, mmd_dists
from src.analysis.logistic_regression import init_classifier, train_classifier


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


def observed_combos_from_adata(adata, conditions):
    """Set of '__'-joined combo labels that actually have rows in `adata` - pass the
    *training* adata (post-holdout) so this reflects which combos the model's mixture
    means were ever fitted on.

    Needed because `combo_categories` is the full Cartesian product of the conditions
    (see `data_utils.get_condition_vocab`), so membership in it says nothing about
    whether a combo had training data: combos that were dropped upstream - e.g. by
    `build_metacells`, which skips any group with fewer than `cells_per_metacell`
    cells - still hold a reserved `means` row
    that never receives a gradient (or an `update_means_epoch` update) and therefore
    stays at its initialization. Averaging such rows into a treatment shift mixes pure
    initialization noise into the estimate at full weight.
    """
    combined = adata.obs[list(conditions)].astype(str).agg("__".join, axis=1)
    return set(combined.unique().tolist())


def _average_treatment_shift(
    model,
    combo_categories,
    combo_order,
    holdout_combo,
    condition_key,
    control_label,
    cell_type_key,
    target_treatment,
    observed_combos=None,
):
    """Average, across every other cell type, of the learned latent-mean shift between
    `target_treatment` and `control_label` - the same estimate
    `estimate_leftout_combo_mean` adds to the held-out combo's control mean.

    `observed_combos`: set of combo labels that had training data (see
    `observed_combos_from_adata`). Cell types whose control- or treatment-side combo is
    missing from it are skipped, so untrained `means` rows - still at their random
    initialization - can't contribute to the average. Leaving this as None keeps every
    combo in `combo_categories`, which is only safe if the design really is full
    factorial in the *training* data.
    """
    cell_values = _other_cell_types(
        combo_categories, combo_order, cell_type_key, holdout_combo[cell_type_key]
    )

    shifts = []
    skipped_unobserved = []
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

        if observed_combos is not None and (
            other_control_label not in observed_combos
            or other_treatment_label not in observed_combos
        ):
            skipped_unobserved.append(other_cell)
            continue

        other_control_idx = combo_categories.index(other_control_label)
        other_treatment_idx = combo_categories.index(other_treatment_label)
        shifts.append(model.means[other_treatment_idx] - model.means[other_control_idx])

    if skipped_unobserved:
        print(
            f"[_average_treatment_shift] skipped {len(skipped_unobserved)} cell type(s) with no "
            f"training data for '{target_treatment}' and/or '{control_label}': "
            f"{sorted(skipped_unobserved)}"
        )

    if not shifts:
        raise ValueError(
            "No valid shifts were found for the other cell types. "
            "Check that your `combo_categories` and holdout configuration match the training "
            "design, and that `observed_combos` isn't excluding every cell type."
        )

    return torch.stack(shifts).mean(dim=0)


def _combo_adata(x, holdout_combo, var=None, clamp_min=0.0):
    """`clamp_min`: floor applied to the decoded expression before it becomes `.X`.

    The flow's decoder is unconstrained and happily emits negative values, but both the
    log1p-normalized and raw-count spaces it is trained on are non-negative, so those
    negatives are decoding artefacts rather than predictions. Left in, they inflate every
    downstream distance against the (non-negative) real cells - and the scGen comparison
    path in `comparison.ipynb` already clamps its own samples, so leaving these unclamped
    handicapped the flow in the benchmark rather than measuring it. Pass None to keep the
    raw decoder output.
    """
    x = x.detach().cpu().numpy()
    if clamp_min is not None:
        x = np.clip(x, clamp_min, None)
    obs = pd.DataFrame({key: [value] * x.shape[0] for key, value in holdout_combo.items()})
    return ad.AnnData(
        X=x,
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
    observed_combos=None,
):
    """Estimate and cache the mean for a held-out combo.

    For a held-out combo such as {cell_type: "CD8 T cells", cytokine: "IL-2"}, we use the
    control mean for the same cell type and add the average treatment-control shift across
    all other cell types:

        mu_target ≈ mu(cell_type=target, condition=control)
                    + mean_{other cell types}[ mu(other, treatment) - mu(other, control) ]

    This is useful when the model reserves a slot for the held-out combo but the empirical
    mean was never learned because that combo was left out.

    `observed_combos`: set of combo labels with training data (see
    `observed_combos_from_adata`); cell types missing either side of the shift are then
    excluded from the average instead of contributing an untrained, still-at-init `means`
    row. Strongly recommended - `combo_categories` alone cannot tell the two apart.

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

    if getattr(model, "factorized", False):
        # Nothing to estimate: mu(held-out combo) = a_cell_type + b_cytokine, and both
        # factors were fitted from other combos, so the model already holds the right
        # mean for a combo it never saw. Returning it (rather than overwriting a row)
        # also keeps this a read-only call for factorized models.
        return model.means[combo_categories.index(target_label)].detach()

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
        observed_combos=observed_combos,
    )

    control_target_idx = combo_categories.index(control_target_label)
    target_idx = combo_categories.index(target_label)

    estimated_mean = model.means[control_target_idx] + shift

    # `means_free` and not `means_active`: the latter is a property, so under a
    # factorized prior it returns a freshly composed tensor and this write would land on
    # a temporary and vanish (the same trap `update_means_epoch` used to fall into).
    # Factorized models return above and never reach here.
    if getattr(model, "means_free", None) is not None:
        model.means_free.data[target_idx] = estimated_mean[: model.means_free.shape[1]]
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
    observed_combos=None,
):
    """OOD sampling strategy 1: draw the held-out combo's cells from an isotropic Normal
    distribution centered at its estimated latent mean (`estimate_leftout_combo_mean`),
    with std taken from the model's learned per-component `log_sigma` when available
    (same convention as `cf_expressions.sample_from_means`), else the constant `sigma`.
    Decodes the samples back through the flow to expression space.

    Returns an AnnData of the sampled cells (`.X` = decoded expression, floored at 0 -
    see `_combo_adata`), with one obs column per entry of `holdout_combo` set to its value.

    `observed_combos`: forwarded to `estimate_leftout_combo_mean` - see there.
    """
    mean = estimate_leftout_combo_mean(
        model=model,
        combo_categories=combo_categories,
        conditions=conditions,
        holdout_combo=holdout_combo,
        condition_key=condition_key,
        control_label=control_label,
        cell_type_key=cell_type_key,
        observed_combos=observed_combos,
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
    observed_combos=None,
    clamp_min=0.0,
):
    """OOD sampling strategy 2: encode real control cells of the held-out combo's cell
    type, shift each one in latent space by the average treatment-control mean shift
    across all other cell types (the same shift `estimate_leftout_combo_mean` adds to the
    control mean), then decode. Unlike `sample_leftout_combo_normal`, this carries over
    each control cell's own latent structure (e.g. cell state, library size) instead of
    drawing fresh isotropic noise around a single point.

    Returns an AnnData of the shifted cells (`.X` = decoded expression, floored at 0 -
    see `_combo_adata`), one row per real control cell of the held-out combo's cell type.

    `observed_combos`: forwarded to `_average_treatment_shift` - see there.
    """
    cell_type_key = _resolve_cell_type_key(conditions, condition_key, cell_type_key)
    combo_order = list(conditions)

    target_treatment = holdout_combo[condition_key]
    if target_treatment == control_label:
        raise ValueError(
            "The provided holdout combo is already at the control label; no shift is needed."
        )

    if getattr(model, "factorized", False):
        # The shift is just the difference between two prior means the model already
        # holds - no 17-way average of per-cell-type differences to estimate, because
        # the treatment factor is shared across cell types by construction.
        #
        # Written as a difference of means rather than by reading the treatment
        # embedding directly: it needs no gauge to be pinned, and it stays correct
        # unchanged if the prior later gains a cell-type-specific interaction term
        # (where a single treatment vector would no longer be well defined).
        target_label = _combo_label(holdout_combo, combo_order)
        control_label_combo = _combo_label(
            {**holdout_combo, condition_key: control_label}, combo_order
        )
        for lbl in (target_label, control_label_combo):
            if lbl not in combo_categories:
                raise ValueError(
                    f"Combo '{lbl}' is not in combo_categories; make sure the vocabulary "
                    "was built from the full pre-holdout adata."
                )
        means = model.means.detach()
        shift = (
            means[combo_categories.index(target_label)]
            - means[combo_categories.index(control_label_combo)]
        )
    else:
        shift = _average_treatment_shift(
            model,
            combo_categories,
            combo_order,
            holdout_combo,
            condition_key,
            control_label,
            cell_type_key,
            target_treatment,
            observed_combos=observed_combos,
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

    return _combo_adata(x_cf, holdout_combo, var=adata.var, clamp_min=clamp_min)


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


def _origin_auc(scoped_analyzer, x_real, x_other, num_epochs=10, n_repeats=3):
    """Mean ROC AUC over `n_repeats` splits for telling `x_real` from `x_other`.

    Same origin-classifier setup as `logistic_regression.evaluate_cf_roc` - label the two
    sets "original"/"cfs", train on a split, score the held-out one - but on two arrays
    handed in directly, so it works for a held-out combo where only the control cells were
    transformed and the prediction has no row-for-row correspondence with `analyzer.adata`.

    Repeated because the held-out combo has ~34 real cells: with `init_classifier`'s
    test_size=0.2 a single split scores ~7 of them, and one draw of 7 is not an estimate.
    """
    import copy as _copy

    n_real, n_other = len(x_real), len(x_other)
    obs = pd.DataFrame(
        {"origin": ["original"] * n_real + ["cfs"] * n_other},
        index=[str(i) for i in range(n_real + n_other)],
    )
    combined = ad.AnnData(X=np.vstack([x_real, x_other]).astype(np.float32), obs=obs)

    analyzer_cfs = _copy.copy(scoped_analyzer)
    analyzer_cfs.adata = combined

    weights = torch.tensor(
        [1.0 / max(n_other, 1), 1.0 / max(n_real, 1)],
        dtype=torch.float32,
        device=scoped_analyzer.device,
    )
    weights = weights / weights.sum() * len(weights)

    aucs = []
    for _ in range(n_repeats):
        classifier, train_loader, test_loader, train_dataset, _, _ = init_classifier(
            analyzer_cfs, key="original", latent=False, hidden_dim=None, labels_key="origin"
        )
        train_classifier(
            classifier, train_loader, test_loader, weights, scoped_analyzer.device,
            num_epochs=num_epochs,
        )

        target_idx = train_dataset.cats[0].categories.tolist().index("original")
        classifier.eval()
        probs, labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                probs.append(
                    torch.softmax(classifier(x.to(scoped_analyzer.device)), dim=1).cpu().numpy()
                )
                labels.append(y[0].argmax(dim=1).cpu().numpy())
        probs = np.concatenate(probs)[:, target_idx]
        labels = np.concatenate(labels) == target_idx
        if labels.min() == labels.max():  # a split with only one class scores nothing
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        aucs.append(auc(fpr, tpr))

    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate_leftout_combo_roc(
    scoped_analyzer, predictions, key, control_label=None, num_epochs=10, n_repeats=3,
    canonicalize=True,
):
    """Classifier two-sample test for the held-out combo: can a classifier tell each
    model's counterfactuals from the real cells?

    Complements `evaluate_leftout_combo_mmd`. MMD compares the two samples through a fixed
    Gaussian kernel at a median bandwidth, which in 1937 dimensions is a smooth, isotropic
    test; a learned discriminator instead seizes on whatever is structurally wrong - and
    these counterfactuals are known to carry too little variance and too few exact zeros,
    which is precisely the kind of difference a kernel smooths over and a classifier does
    not.

    AUC 0.5 = indistinguishable from real, 1.0 = trivially separable. Meaningless without
    the two references returned alongside:
      `floor`   - two halves of the real cells against each other. Above ~0.5 means the
                  test is overfitting at this sample size and every model AUC is inflated.
      `control` - what "do nothing" scores (real vs the untransformed control cells), so a
                  model has to beat this, not the floor, to have done anything at all.

    `canonicalize`: put every matrix in one space first (see `mmd._canonicalize`), so a
    count-space model is not separated from log-space real cells on units alone.

    Returns (results, floor, control_auc).
    """
    from src.analysis.mmd import _canonicalize

    prep = _canonicalize if canonicalize else (lambda x: _as_array(x))

    adata = scoped_analyzer.adata
    x_real = prep(adata[adata.obs[scoped_analyzer.labels_key] == key])

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x_real))
    half = len(x_real) // 2
    floor = _origin_auc(
        scoped_analyzer, x_real[perm[:half]], x_real[perm[half:]], num_epochs, n_repeats
    )

    control_auc = None
    control_label = control_label or getattr(scoped_analyzer, "control_label", None)
    if control_label is not None:
        x_ctrl = prep(adata[adata.obs[scoped_analyzer.labels_key] == control_label])
        control_auc = _origin_auc(scoped_analyzer, x_real, x_ctrl, num_epochs, n_repeats)

    results = {
        name: _origin_auc(scoped_analyzer, x_real, prep(pred), num_epochs, n_repeats)
        for name, pred in predictions.items()
    }
    return results, floor, control_auc


def _as_array(x):
    x = x.X if hasattr(x, "X") else x
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


def evaluate_leftout_combo_mmd(
    scoped_analyzer, predictions, key, iter=10, ax=None, plot_dir=None, unbiased=False,
    log1p_transform=False,
):
    """Bar chart of MMD between the real held-out-combo cells (found via
    `scoped_analyzer`, see `scoped_analyzer_for_combo`) and each named prediction AnnData
    in `predictions` (e.g. {"Normal": adata_ood_normal, "Shift": adata_ood_shift}), plus
    the random-split MMD among the real cells themselves (mmd.ctrl_mmd_dists) as a noise
    floor - how much MMD to expect even between two halves of real data.

    unbiased: passed through to `mmd.ctrl_mmd_dists`/`mmd.mmd_dists` (see their
    docstrings) - worth setting True here specifically, since the control MMD splits
    an already-small real held-out-combo sample in half while each prediction's MMD
    compares against the full sample, and the biased estimator's default sample-size
    artifact is largest exactly at small n.

    log1p_transform: normalize+log1p both the real cells and the predictions before the
    MMD, i.e. the same flag `mmd_dists`/`ctrl_mmd_dists` already take - set it whenever
    the model's `.adata` is not already log-space (scVI, scGen). It used not to be
    forwarded at all, so a model whose real cells are raw counts while its predictions
    come out normalized had the two sides compared *across* representations. That is not
    a mild bias: measured on real metacells, a like-for-like comparison scores ~0.04
    within one space and ~0.60 across spaces - which is precisely the range scVI's bar
    was reaching, and it swamped any real difference in prediction quality. The kernel
    bandwidth itself is not the problem (both helpers take a median heuristic from the
    real side, so a pure rescaling of the counts barely moves the result); having the
    two point clouds in different representations is.

    Returns {name: global_mmd} and the control MMD.
    """
    ctrl_mmd = ctrl_mmd_dists(
        scoped_analyzer, key=key, iter=iter, unbiased=unbiased,
        log1p_transform=log1p_transform,
    )

    results = {}
    for name, adata_pred in predictions.items():
        global_mmd, _ = mmd_dists(
            scoped_analyzer,
            key=key,
            cf_fn=cfs_fn_from_adata(adata_pred, source_label=name),
            unbiased=unbiased,
            log1p_transform=log1p_transform,
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
