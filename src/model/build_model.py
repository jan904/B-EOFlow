import schedulefree
import torch
import torch.nn as nn

from src.model.VAE.VAE_model import get_VAE
from src.model.INN.INNs_model import get_INN

from src.model.config import INNConfig, VAEConfig
from dataclasses import dataclass


def get_param_groups(model, config):
    other_params = [
        p for name, p in model.named_parameters() if name not in {"means_active", "log_sigma"}
    ]

    param_groups = [{"params": other_params, "lr": config.lr}]

    if model.means_active is not None:
        param_groups.append(
            {
                "params": [model.means_active],
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

        # Old checkpoint: means -> means_active
        if k == "means":
            k = "means_active"

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

    non_prior_missing = [k for k in missing if not k.startswith("means")]

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
