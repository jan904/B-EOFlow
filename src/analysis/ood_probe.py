"""Cheap OOD probe to run *during* training, so checkpoint selection can see the thing
we actually care about instead of only the likelihood.

The training NLL keeps improving long after OOD prediction stops improving - it is
dominated by how sharply the flow resolves the count lattice, not by whether a combo it
never saw lands in the right place. This measures the latter every N epochs, for a
**validation** combo held out alongside the reported test combo.

Four numbers, because the two axes move independently (across one lam_MTC sweep the
mean-level score was flat at 0.72-0.78 while MMD varied 4x):

  `mmd`        - unbiased MMD of the decoded counterfactual against the real cells, at a
                 bandwidth fixed once at setup so the curve is comparable across epochs.
  `r2_top50`   - residual R^2 of the predicted mean effect on the 50 strongest DE genes
                 (see cf_expressions.effect_size_scores: 0 = no better than predicting no
                 change, and unlike a correlation it does see magnitude errors).
  `var_ratio`  - variance of the prediction over variance of the real cells; the
                 dispersion collapse behind most of the MMD gap.
  `accuracy`   - fraction of the real validation metacells whose nearest prior mean is
                 their own combo (198-way, chance 0.5%). Nearly free - it only needs an
                 encode - and across one sweep it ranked configurations identically to
                 MMD, so it is the one to watch if only one number is wanted.

Selecting on this combo makes it a validation set, not a test set: keep it distinct from
whatever combo the paper reports.
"""

import numpy as np
import torch

from src.analysis.cf_expressions import effect_size_scores
from src.analysis.mmd import mmd


class OODProbe:
    """Precomputes everything that does not depend on the model, then scores a model.

    Everything expensive - selecting the real/control cells, the DE gene set, the kernel
    bandwidth - happens once here; `__call__` only runs a few hundred rows through the
    flow, which is well under a second on a GPU against ~5 s for a training epoch.
    """

    def __init__(self, adata, combo, combo_categories, conditions, control_label,
                 condition_key=None, cell_type_key=None, top_n=50, device="cpu"):
        condition_key = condition_key or conditions[0]
        cell_type_key = cell_type_key or conditions[1]

        obs_cond = adata.obs[condition_key].astype(str).to_numpy()
        obs_cell = adata.obs[cell_type_key].astype(str).to_numpy()
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

        target = (obs_cond == combo[condition_key]) & (obs_cell == combo[cell_type_key])
        control = (obs_cond == control_label) & (obs_cell == combo[cell_type_key])
        if target.sum() < 2 or control.sum() < 2:
            raise ValueError(
                f"OOD probe combo {combo} has {int(target.sum())} real and "
                f"{int(control.sum())} control cells; need at least 2 of each."
            )

        self.device = device
        self.real = torch.tensor(X[target], dtype=torch.float32)
        self.control = torch.tensor(X[control], dtype=torch.float32)
        self.n_real = int(target.sum())

        # combo indices into the prior means; the shift is the difference of the two means,
        # which is well defined for a factorized prior even though this combo never trained
        label = lambda values: "__".join(str(values[c]) for c in conditions)
        self.target_idx = combo_categories.index(label(combo))
        self.control_idx = combo_categories.index(
            label({**combo, condition_key: control_label})
        )
        self.combo_id = self.target_idx
        self.label = label(combo)

        real_np = self.real.numpy()
        self.real_delta = real_np.mean(0) - self.control.numpy().mean(0)
        self.top = np.argsort(-np.abs(self.real_delta))[:top_n]
        self.real_var = float(real_np.var(0).sum())

        # bandwidth from the real cells only - they never change, so fixing it here keeps
        # the metric comparable from epoch to epoch
        with torch.no_grad():
            self.sigma = float(torch.median(torch.cdist(self.real, self.real)))

    @torch.no_grad()
    def __call__(self, model):
        """Score `model`. Caller is responsible for the schedule-free eval iterate."""
        was_training = model.training
        model.eval()

        device = next(model.parameters()).device
        real = self.real.to(device)
        control = self.control.to(device)

        means = model.means.detach()
        shift = means[self.target_idx] - means[self.control_idx]

        z_control, _ = model(control, c=None, rev=False)
        x_pred, _ = model(z_control + shift, rev=True)
        pred = torch.clamp(x_pred, min=0.0)

        # nearest-mean assignment for the real cells (198-way, chance 1/K)
        z_real, _ = model(real, c=None, rev=False)
        md = model.means_active.shape[1]
        dist = torch.cdist(z_real[:, :md], means[:, :md])
        accuracy = float((dist.argmin(1) == self.combo_id).float().mean())

        value = float(mmd(real, pred, sigma=self.sigma, unbiased=True))

        pred_np = pred.cpu().numpy()
        delta = pred_np.mean(0) - control.cpu().numpy().mean(0)
        scores = effect_size_scores(delta[self.top], self.real_delta[self.top])

        if was_training:
            model.train()

        return {
            "mmd": value,
            "r2_top50": scores["r2"],
            "cos2_top50": scores["cos2"],
            "var_ratio": float(pred_np.var(0).sum()) / self.real_var,
            "accuracy": accuracy,
        }


class MultiOODProbe:
    """One `OODProbe` per named combo, scored together and aggregated.

    For probing the **held-out** combos rather than a separate validation combo: ten combos
    give a spread where one gives a single number with no error bar, which matters here
    because a held-out combo carries only ~34 real metacells and the per-combo noise floor
    is nearly as large as the effect being measured.

    Aggregation is median + IQR, matching `eval_checkpoint.py`'s: the MMD scale differs per
    combo (each bandwidth is its own median pairwise distance), so a mean across combos is
    dominated by whichever combo has the largest absolute values.

    Reading this curve is safe; *selecting* a checkpoint on it is not, if these are the
    combos the paper reports - that turns the test set into a validation set. See
    `ood_select` in INN_training.train_INN_model.
    """

    def __init__(self, adata, combos, combo_categories, conditions, control_label, **kwargs):
        self.probes = {}
        self.skipped = {}
        for name, combo in combos.items():
            try:
                self.probes[name] = OODProbe(
                    adata, combo, combo_categories, conditions, control_label, **kwargs
                )
            except ValueError as exc:
                # a combo with too few real or control cells is dropped rather than fatal:
                # the remaining combos still give a usable curve
                self.skipped[name] = str(exc)
        if not self.probes:
            raise ValueError(
                f"No probe combo had enough cells; skipped {len(self.skipped)}: {self.skipped}"
            )
        self.label = f"{len(self.probes)} holdout combos"
        self.n_real = sum(probe.n_real for probe in self.probes.values())

    @torch.no_grad()
    def __call__(self, model):
        per_combo = {name: probe(model) for name, probe in self.probes.items()}

        def values(key):
            return np.array([m[key] for m in per_combo.values()], dtype=float)

        mmd_v, r2_v = values("mmd"), values("r2_top50")
        metrics = {
            # medians keep the same key names a single OODProbe returns, so the training
            # loop and `format_probe` need no special case for the multi-combo variant
            "mmd": float(np.median(mmd_v)),
            "r2_top50": float(np.median(r2_v)),
            "cos2_top50": float(np.median(values("cos2_top50"))),
            "var_ratio": float(np.median(values("var_ratio"))),
            "accuracy": float(np.mean(values("accuracy"))),
            "mmd_iqr": [float(np.quantile(mmd_v, 0.25)), float(np.quantile(mmd_v, 0.75))],
            "r2_top50_iqr": [float(np.quantile(r2_v, 0.25)), float(np.quantile(r2_v, 0.75))],
            "n_combos": len(per_combo),
            "per_combo": per_combo,
        }
        return metrics


def format_probe(metrics):
    spread = ""
    if "mmd_iqr" in metrics:
        spread = (
            f" (n={metrics['n_combos']}, MMD IQR "
            f"{metrics['mmd_iqr'][0]:.4f}-{metrics['mmd_iqr'][1]:.4f})"
        )
    return (
        f"OOD[{metrics.get('combo', '')}] MMD: {metrics['mmd']:.4f} | "
        f"R2_top50: {metrics['r2_top50']:.3f} | var_ratio: {metrics['var_ratio']:.3f} | "
        f"acc: {metrics['accuracy']:.1%}{spread}"
    )
