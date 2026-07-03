from dask import config
import torch
import torch.nn as nn

import numpy as np

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import FrEIA.modules.splines as Fms
from FrEIA.modules.coupling_layers import _BaseCouplingBlock

import schedulefree

from src.model.INNs import Orthogonal, get_subnet_fc, get_linear_INN
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    config_version: int = 1
    N_dim: int = 2000
    N_blocks: int = 12
    condition_shapes: list = None
    subnet_fc: callable = None
    act_func: str = "relu"
    ch_hidden: int = 2048
    n_hidden_layers: int = 2
    coupling_block_type: str = "GLOW"
    RQS_bins: int = 10
    clamp: float = 2.0
    pre_rotate: bool = True
    pre_normalize: bool = False
    permute_random: bool = True
    rotate_random: bool = False
    normalize: bool = True
    post_rotate: bool = True
    lr: float = 5e-4
    lr_means: float = 5e-4
    optimizer_type: str = "schedulefree"
    warmup_steps: int = 300
    condition_type: str = None
    trainable_means: bool = False
    n_clusters: int = None
    means_seperation: float = 2.0
    supervise_latent_meaning: bool = False
    ctrl_idx: int = None
    lam_supervise: float = 0.01
    latent_per_condition: int = None
    partition_divisor: int = 8

    def __post_init__(self):
        if self.condition_shapes is not None and self.condition_type is None:
            raise ValueError("condition_type must be set when condition_shapes is provided.")
        if self.condition_type is not None and self.condition_shapes is None:
            raise ValueError("condition_shapes must be set when condition_type is provided.")

        valid_condition_types = {"mixture", "normal", None}  # add yours
        if self.condition_type not in valid_condition_types:
            raise ValueError(
                f"condition_type must be one of {valid_condition_types}, got '{self.condition_type}'."
            )


def get_param_groups(model, config):
    other_params = [p for name, p in model.named_parameters() if name != "means_active"]
    param_groups = [{"params": other_params, "lr": config.lr}]

    if model.means_active is not None and model.means_active.requires_grad:
        param_groups.append({"params": [model.means_active], "lr": config.lr_means})

    return param_groups


class INNWithMixturePrior(nn.Module):
    def __init__(
        self,
        flow,
        N_dim,
        n_components=None,
        condition_type=None,
        trainable_means=False,
        means_seperation=2.0,
        latent_per_condition=None,
    ):
        super().__init__()
        self.flow = flow
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

    @property
    def means(self):
        if self.means_active is None:
            return None
        if self.means_zero is None:
            return self.means_active
        return torch.cat([self.means_active, self.means_zero], dim=1)

    def forward(self, x, c=None, rev=False):
        return self.flow(x, c=c, rev=rev)


# def simple_INN_init(N_dim, N_blocks = 8, conditions = 0, N_conv_blocks=None, padding_size=1, symmetric_convolution=False, subnet_fc=None, act_func='relu', ch_hidden=None, n_hidden_layers=2, ch_hidden_conv=16, bins=10, coupling_block_type = 'GLOW', clamp=2.0, kwargs_rotations={}, permute_random=False, permute_random=True, permute_random_soft=False, householder_perms=2, use_actnorms=True, use_rotations=True, lr=1e-3, device=None):
def get_INN(config):

    flow = Ff.SequenceINN(config.N_dim)

    subnet_fc = config.subnet_fc
    if config.subnet_fc == None:
        subnet_fc = lambda c_in, c_out: get_subnet_fc(
            c_in,
            c_out,
            act_func=config.act_func,
            ch_hidden=config.ch_hidden,
            layers=config.n_hidden_layers,
        )

    n_components = None
    if config.condition_shapes is None:
        cond = None
        cond_shape = None
    elif config.condition_type == "mixture":
        cond = None
        cond_shape = None
        if config.n_clusters is None:
            n_components = config.condition_shapes[0]
        else:
            n_components = config.n_clusters
    else:
        cond = 0
        cond_shape = config.condition_shapes

    if config.pre_normalize:
        flow.append(Fm.ActNorm)
    if config.pre_rotate:
        flow.append(Orthogonal)

    if config.coupling_block_type:
        for k in range(config.N_blocks):
            if config.coupling_block_type == "GLOW":
                flow.append(
                    Fm.GLOWCouplingBlock,
                    cond=cond,
                    cond_shape=cond_shape,
                    subnet_constructor=subnet_fc,
                    clamp=config.clamp,
                )
            elif config.coupling_block_type == "RQS":
                flow.append(
                    Fms.RationalQuadraticSpline,
                    cond=cond,
                    cond_shape=cond_shape,
                    bins=config.RQS_bins,
                    subnet_constructor=subnet_fc,
                )  # bins=3,
                # flow.append(Fm.HouseholderPerm, n_reflections=4)
            # elif coupling_block_type == 'GIN':
            #     flow.append(Fm.GINCouplingBlock, cond=cond, cond_shape=cond_shape, subnet_constructor=subnet_fc)
            # elif coupling_block_type == 'NICE':
            #     flow.append(Fm.NICECouplingBlock, cond=cond, cond_shape=cond_shape, subnet_constructor=subnet_fc)
            # elif coupling_block_type == 'None':
            #     pass
            # else:
            #     flow.append(Fm.GLOWCouplingBlock, cond=cond, cond_shape=cond_shape, subnet_constructor=subnet_fc)

            if config.permute_random:
                flow.append(Fm.PermuteRandom)
            if config.normalize:
                flow.append(Fm.ActNorm)
            if config.rotate_random and not (k == config.N_blocks - 1):
                M = torch.linalg.qr(torch.randn(config.N_dim, config.N_dim))[0]
                flow.append(Fm.FixedLinearTransform, M=M)

    if config.post_rotate:
        flow.append(Orthogonal)

    model = INNWithMixturePrior(
        flow,
        N_dim=config.N_dim,
        n_components=n_components,
        condition_type=config.condition_type,
        trainable_means=config.trainable_means,
        means_seperation=config.means_seperation,
        latent_per_condition=config.latent_per_condition,
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

    return model, optimizer  # , losses


#     optimizer_type="schedulefree",
#     warmup_steps=100,
#     n_layers=1,
# )
# flow = flow.to(device=device)
