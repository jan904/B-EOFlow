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


class ModelWithMixturePrior(nn.Module):
    def __init__(
        self,
        model,
        N_dim,
        n_components=None,
        condition_type=None,
        trainable_means=False,
        trainable_sigma=False,
        init_sigma=1.0,
        means_seperation=2.0,
        latent_per_condition=None,
    ):
        super().__init__()
        self.model = model
        self.means_dim = N_dim
        self.N_dim = N_dim

        if condition_type == "mixture":
            if n_components is None:
                raise ValueError("n_components must be specified for mixture condition type.")
            if latent_per_condition is not None:
                if latent_per_condition * n_components > N_dim:
                    raise ValueError(
                        f"latent_per_condition * n_components ({latent_per_condition * n_components}) "
                        f"cannot exceed N_dim ({N_dim})."
                    )
                self.means_dim = latent_per_condition * n_components

            # learnable part: (n_components, means_dim)
            means_active = torch.zeros(n_components, self.means_dim)
            torch.nn.init.orthogonal_(means_active)
            means_active = means_active * N_dim**0.5 * means_seperation
            self.means_active = torch.nn.Parameter(means_active, requires_grad=trainable_means)

            self.log_sigma = None
            if trainable_sigma:
                log_sigma = torch.full(
                    (n_components,),
                    float(torch.log(torch.tensor(init_sigma))),
                )
                self.log_sigma = nn.Parameter(
                    log_sigma,
                    requires_grad=trainable_sigma,
                )

            # fixed zero padding: (n_components, N_dim - means_dim)
            if self.means_dim < N_dim:
                self.register_buffer(
                    "means_zero", torch.zeros(n_components, N_dim - self.means_dim)
                )
            else:
                self.means_zero = None
        else:
            self.means_active = None
            self.means_zero = None
            self.log_sigma = None

    @property
    def means(self):
        if self.means_active is None:
            return None
        if self.means_zero is None:
            return self.means_active
        return torch.cat([self.means_active, self.means_zero], dim=1)

    def forward(self, x, c=None, rev=False):
        return self.model(x, c=c, rev=rev)


# def simple_INN_init(N_dim, N_blocks = 8, conditions = 0, N_conv_blocks=None, padding_size=1, symmetric_convolution=False, subnet_fc=None, act_func='relu', ch_hidden=None, n_hidden_layers=2, ch_hidden_conv=16, bins=10, coupling_block_type = 'GLOW', clamp=2.0, kwargs_rotations={}, permute_random=False, permute_random=True, permute_random_soft=False, householder_perms=2, use_actnorms=True, use_rotations=True, lr=1e-3, device=None):
def get_model(config):

    if config.model_type == "inn":
        model = get_INN(config)
    elif config.model_type == "vae":
        model = get_VAE(config)
    else:
        print(f"Unknown model_type '{config.model_type}'. Falling back to INN.")
        model = get_INN(config)

    n_components = None
    if config.condition_type == "mixture":
        if config.n_clusters is None:
            n_components = config.condition_shapes[0]
        else:
            n_components = config.n_clusters

    model = ModelWithMixturePrior(
        model,
        N_dim=config.N_dim,
        n_components=n_components,
        condition_type=config.condition_type,
        trainable_means=config.trainable_means,
        means_seperation=config.means_seperation,
        latent_per_condition=config.latent_per_condition,
        trainable_sigma=config.trainable_sigma,
        init_sigma=config.init_sigma,
    )

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
