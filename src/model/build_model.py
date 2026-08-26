import schedulefree
import torch
import torch.nn as nn

from src.model.VAE.VAE_model import get_VAE
from src.model.INN.INNs_model import get_INN

from src.model.config import INNConfig, VAEConfig
from dataclasses import dataclass


def get_param_groups(model, config):
    # The prior's parameters are whatever the model says they are - one free means
    # tensor, or the per-condition factor embeddings when factorized - so this doesn't
    # have to know which parameterization is in use. Note `means_active` is a derived
    # property under factorization and never appears in named_parameters().
    prior_names = model.prior_parameter_names()
    prior_params = model.prior_parameters()

    other_params = [
        p
        for name, p in model.named_parameters()
        if name not in prior_names and name != "log_sigma"
    ]

    param_groups = [{"params": other_params, "lr": config.lr}]

    if prior_params:
        param_groups.append(
            {
                "params": prior_params,
                "lr": config.lr_means,
            }
        )

    if model.log_sigma is not None:
        param_groups.append(
            {
                "params": [model.log_sigma],
                "lr": config.lr_sigma,
            }
        )

    return param_groups


# def simple_INN_init(N_dim, N_blocks = 8, conditions = 0, N_conv_blocks=None, padding_size=1, symmetric_convolution=False, subnet_fc=None, act_func='relu', ch_hidden=None, n_hidden_layers=2, ch_hidden_conv=16, bins=10, coupling_block_type = 'GLOW', clamp=2.0, kwargs_rotations={}, permute_random=False, permute_random=True, permute_random_soft=False, householder_perms=2, use_actnorms=True, use_rotations=True, lr=1e-3, device=None):
def get_model(config, adata=None):

    if config.model_type == "inn":
        model = get_INN(config)

        param_groups = get_param_groups(model, config)

        if config.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(param_groups, lr=config.lr, weight_decay=1e-5)
        elif config.optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(param_groups, lr=config.lr, weight_decay=1e-5)
        elif config.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(param_groups, lr=config.lr, weight_decay=1e-5)
        elif config.optimizer_type == "schedulefree":
            optimizer = schedulefree.AdamWScheduleFree(
                param_groups, lr=config.lr, warmup_steps=config.warmup_steps
            )
            optimizer.train()

    elif config.model_type == "vae":
        assert adata is not None, "adata must be provided for VAE model."
        model = get_VAE(config, adata=adata)
        optimizer = None

    else:
        print(f"Unknown model_type '{config.model_type}'. Falling back to INN.")
        model = get_INN(config)

        param_groups = get_param_groups(model, config)

        if config.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(param_groups, lr=config.lr, weight_decay=1e-5)
        elif config.optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(param_groups, lr=config.lr, weight_decay=1e-5)
        elif config.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(param_groups, lr=config.lr, weight_decay=1e-5)
        elif config.optimizer_type == "schedulefree":
            optimizer = schedulefree.AdamWScheduleFree(
                param_groups, lr=config.lr, warmup_steps=config.warmup_steps
            )
            optimizer.train()

    return model, optimizer


## TODO: Remove
@dataclass
class ModelConfig(INNConfig):
    # Backward compatibility for ModelConfig, which is now an alias for INNConfig
    pass


## TODO: Remove
def remap_keys(model, state_dict):
    remapped_state = {}

    model_keys = set(model.state_dict().keys())

    for k, v in state_dict.items():

        # Old checkpoint: means -> means_active -> means_free
        # (`means_active` became a property that dispatches between the free tensor and
        # the factorized composition, so the free parameter itself is now `means_free`.)
        if k in ("means", "means_active"):
            k = "means_free"

        # Handle flow <-> model prefix mismatch
        if k.startswith("flow."):
            candidate = "model." + k[len("flow.") :]
            if candidate in model_keys:
                k = candidate

        elif k.startswith("model."):
            candidate = "flow." + k[len("model.") :]
            if candidate in model_keys:
                k = candidate

        # If no prefix but model expects model.
        elif not k.startswith("means"):
            candidate = "model." + k
            if candidate in model_keys:
                k = candidate

        remapped_state[k] = v

    # Prior-block width change. `supervise_latent_meaning` now decides whether the means
    # are restricted to a condition block at all, so a checkpoint trained with a narrower
    # block loads into a full-width model by zero-padding: the old model kept exactly
    # those zeros in its fixed `means_zero` buffer, so `means` comes out bit-for-bit the
    # same and only the tail's trainability changes.
    #
    # Narrowing is not attempted. The learned shifts are dense across the old block, so
    # truncating would silently discard fitted values - pass `means_dim` explicitly to
    # rebuild at the checkpoint's own width instead.
    model_state = model.state_dict()
    for key in [k for k in remapped_state if k == "means_free" or k.startswith("factor_emb.")]:
        old, target = remapped_state[key], model_state.get(key)
        if target is None or old.shape == target.shape:
            continue
        if old.ndim == 2 and old.shape[0] == target.shape[0] and old.shape[1] < target.shape[1]:
            pad = torch.zeros(
                old.shape[0], target.shape[1] - old.shape[1], dtype=old.dtype, device=old.device
            )
            remapped_state[key] = torch.cat([old, pad], dim=1)
            print(
                f"Zero-padded {key} from {tuple(old.shape)} to {tuple(target.shape)} "
                "(the old model pinned those dims at zero; the prior is unchanged)."
            )
        else:
            raise RuntimeError(
                f"Checkpoint's {key} is {tuple(old.shape)} but this model wants "
                f"{tuple(target.shape)}, and that is not a widening this can pad. The "
                "prior block was sized differently - rebuild with the checkpoint's own "
                "`means_dim` (and matching supervise_latent_meaning) rather than "
                "reshaping fitted means."
            )

    # Old gain checkpoint: `treatment_gain` held w directly; it is now `gain_log`, with
    # w = exp(s - mean(s)) so the geometric mean of w is pinned at 1. Rescaling w by that
    # geometric mean would change every mu, so the same factor is multiplied back into
    # the treatment embedding - `mu = a + w*b` is then bit-for-bit what it was, and only
    # the split between w and b moves.
    if "treatment_gain" in remapped_state and "gain_log" in model_keys:
        w = remapped_state.pop("treatment_gain").float()
        if (w <= 0).any():
            raise RuntimeError(
                f"Checkpoint has non-positive treatment_gain values ({w[w <= 0].tolist()}), "
                "which the log parameterization cannot represent. A negative gain means a "
                "cell type responding along the reverse of the shared treatment direction; "
                "retrain rather than converting."
            )
        log_w = torch.log(w)
        remapped_state["gain_log"] = log_w - log_w.mean()
        geo_mean = torch.exp(log_w.mean())
        emb_key = f"factor_emb.{model.control_factor}"
        if emb_key in remapped_state:
            remapped_state[emb_key] = remapped_state[emb_key] * geo_mean
            print(f"Converted treatment_gain -> gain_log (geometric mean {geo_mean:.4f} "
                  f"folded into {emb_key}; the prior is unchanged).")

    missing, unexpected = model.load_state_dict(remapped_state, strict=False)

    # The prior block is the one thing that must never be silently left at init. A
    # factorized model loading a free-means checkpoint (or the reverse) leaves the prior
    # randomly initialized while every flow weight loads fine, so training resumes on a
    # prior that encodes nothing - the exact failure this branch exists to remove. Since
    # load_state_dict runs with strict=False so genuinely fresh combos stay tolerated,
    # this has to be checked explicitly.
    prior_missing = [k for k in missing if k.startswith(("means", "factor_emb"))]
    if prior_missing and any(k.startswith("factor_emb") for k in prior_missing):
        raise RuntimeError(
            f"Checkpoint has no factorized prior ({prior_missing} missing) but the model "
            "expects one. This is a free-means checkpoint being loaded into a factorized "
            "model - they are not interchangeable. Train from scratch, or rebuild the "
            "model with factorize_means=False."
        )
    if "means_free" in prior_missing:
        raise RuntimeError(
            "Checkpoint has no free per-combo means but the model expects them - this "
            "looks like a factorized checkpoint being loaded into a free-means model. "
            "Rebuild the model with factorize_means=True."
        )

    # A gain checkpoint loaded into a plain additive model would land in `unexpected` and
    # be dropped silently, giving a *different* prior (every w reset to 1) under the same
    # filename. The other direction is fine and deliberately allowed: `treatment_gain`
    # initializes at 1, so a purely additive checkpoint loads into a gain model as exactly
    # the model it already was, which is how a gain run warm-starts from an additive one.
    carries_gain = {"treatment_gain", "gain_log"} & set(unexpected)
    if carries_gain and getattr(model, "gain_log", None) is None:
        raise RuntimeError(
            f"Checkpoint carries a learned gain ({sorted(carries_gain)}) but the model has "
            "none - its means would silently differ. Rebuild the model with "
            "treatment_gain=True."
        )

    non_prior_missing = [
        k
        for k in missing
        if not k.startswith(("means", "factor_emb", "treatment_gain", "gain_log"))
    ]

    if non_prior_missing:
        print(f"WARNING: missing keys after remapping: {non_prior_missing}")
    else:
        print(f"Loaded successfully. Missing/fresh keys: {missing}")

    if unexpected:
        print(f"WARNING: unexpected keys ignored: {unexpected}")

    return model


def init_config(config):
    ## TODO: Remove instance check
    if isinstance(config, dict):
        model_type = config["model_type"]
        config.pop("model_type", None)

        if model_type == "inn":
            config = INNConfig(**config)

        elif model_type == "vae":
            config = VAEConfig(**config)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return config
