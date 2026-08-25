"""How a run's hyperparameters become a checkpoint filename, and back again.

One home for the scheme, because it has three consumers that used to spell it out
separately - `scripts/training/train.py` writes it, `scripts/analysis/eval_checkpoint.py`
reads settings back out of it, and the notebooks build it to find a checkpoint. When
those drifted, the result was nine checkpoints labelled `5e-3` that had every one of them
trained at `lr_means=5e-4`: the prefix was a hand-maintained string that no code checked.

The name carries the knobs that change *what is trained* without changing the
checkpoint's structure, since two such runs are different results that would otherwise
share a path and overwrite each other. Structure-changing knobs (`factorize_means`,
`treatment_gain`) live in the directory instead - there a collision is a failed load
rather than a bad label, so it announces itself.

Two spellings exist:

    legacy   {prefix}_MTC_0.6_sigma_0.5_model.pt
    current  {prefix}_MTC_0.6_sigma_0.5_lrm_0.0005_lpc_2_model.pt

Checkpoints written before `lrm_`/`lpc_` existed keep the legacy spelling, so
`resolve_checkpoint` accepts either - preferring the current one when both are present,
since that is the newer run. A missing `lrm_`/`lpc_`/`mdim_` therefore means "not
recorded", not "not set": every legacy checkpoint trained at lr_means 5e-4 with
latent_per_condition 2.
"""

import os

# (key in the parsed dict, token in the filename, how to read the value back).
# `run_name` writes these and `run_settings_from_name` reads them; keeping the two
# built from one table is the point of this module.
_FIELDS = (
    ("lam_MTC", "MTC_", float),
    ("sigma_noise", "sigma_", float),
    ("lr_means", "lrm_", float),
    ("latent_per_condition", "lpc_", int),
    ("means_dim", "mdim_", int),
)


def run_name(lam_MTC, sigma_noise, lr_means, latent_per_condition=None, means_dim=None):
    """`MTC_0.6_sigma_0.5_lrm_0.0005_lpc_2`.

    A knob whose value is None is left out rather than spelled `None`: absence reads as
    "the model's own default", which is what it is. `means_dim` overrides
    `latent_per_condition` in INNs_model, so both are recorded when both are set - which
    one is set is itself information, since the same means_dim can arrive either way.
    """
    parts = [
        "MTC_" + str(lam_MTC),
        "sigma_" + str(sigma_noise),
        "lrm_" + str(lr_means),
    ]
    if latent_per_condition is not None:
        parts.append("lpc_" + str(latent_per_condition))
    if means_dim is not None:
        parts.append("mdim_" + str(means_dim))
    return "_".join(parts)


def legacy_run_name(lam_MTC, sigma_noise, **_ignored):
    """The pre-`lrm_` spelling. Extra knobs are accepted and dropped, so a caller can
    hand the same arguments to either builder."""
    return "MTC_" + str(lam_MTC) + "_sigma_" + str(sigma_noise)


def run_dir_name(prefix=None, legacy=False, **knobs):
    """`5e-4_MTC_0.6_sigma_0.5_lrm_0.0005_lpc_2` - the checkpoint name without
    `_model.pt`, which is also what the logs/plots/csv directories are named.

    An empty or None `prefix` yields no prefix and, importantly, no leading underscore:
    the prefix is a free-text tag, and `""` has to mean "no tag" rather than "a tag that
    happens to be empty", or `_MTC_0.6_...` would be a third spelling for the same run.
    A trailing underscore on the prefix is tolerated, since both `5e-4` and `5e-4_` are
    natural things to write.
    """
    stem = (legacy_run_name if legacy else run_name)(**knobs)
    return prefix.rstrip("_") + "_" + stem if prefix else stem


def checkpoint_name(prefix=None, postfix="", legacy=False, **knobs):
    """`5e-4_MTC_0.6_sigma_0.5_lrm_0.0005_lpc_2_model.pt`.

    `postfix` picks a variant written alongside the main checkpoint, e.g. `_best_ood`.
    """
    return run_dir_name(prefix, legacy, **knobs) + "_model" + postfix + ".pt"


def candidate_paths(model_path, prefix=None, postfix="", **knobs):
    """Both spellings, current first - the order `resolve_checkpoint` prefers them in."""
    return [
        os.path.join(model_path, checkpoint_name(prefix, postfix, legacy=leg, **knobs))
        for leg in (False, True)
    ]


def resolve_checkpoint(model_path, prefix=None, postfix="", required=True, **knobs):
    """The checkpoint to load, accepting either spelling.

    Prefers the current spelling when both exist: that is the newer run, and silently
    loading the older one is exactly the stale-results failure this module exists to
    prevent. Returns None when nothing is found and `required` is False; otherwise raises
    with both paths tried, since "which name did it look for" is the only thing worth
    knowing at that point.
    """
    candidates = candidate_paths(model_path, prefix, postfix, **knobs)
    for path in candidates:
        if os.path.exists(path):
            return path
    if not required:
        return None
    tried = "\n  ".join(candidates)
    raise FileNotFoundError(f"No checkpoint found. Tried:\n  {tried}")


def run_settings_from_name(path):
    """`5e-3_MTC_0.3_sigma_0.1_lrm_0.0005_lpc_2_model.pt` ->
    {'lam_MTC': 0.3, 'sigma_noise': 0.1, 'lr_means': 0.0005, 'latent_per_condition': 2}.

    Read from the filename on purpose: `model_config.sigma_noise` is the *prior's* sigma
    (a tuple, e.g. `(1.0,)`), not the dequantization noise added to the data in the
    training loop - that one lives in `kwargs_loss` and never reaches the checkpoint.
    """
    name = os.path.basename(path)
    out = {}
    for key, token, cast in _FIELDS:
        if token in name:
            try:
                out[key] = cast(name.split(token)[1].split("_")[0])
            except ValueError:
                out[key] = None
    return out
