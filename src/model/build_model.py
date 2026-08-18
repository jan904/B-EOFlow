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

    non_prior_missing = [k for k in missing if not k.startswith(("means", "factor_emb"))]

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
