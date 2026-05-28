from dask import config
import torch
import torch.nn as nn

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
    optimizer_type: str = "schedulefree"
    warmup_steps: int = 300
    condition_type: str = None

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


class INNWithMixturePrior(nn.Module):
    def __init__(self, flow, N_dim, n_components=None, condition_type=None, init_std=0.1):
        super().__init__()
        self.flow = flow
        if condition_type == "mixture":
            if n_components is None:
                raise ValueError("n_components must be specified for mixture condition type.")
            means = torch.zeros(n_components, N_dim)
            torch.nn.init.orthogonal_(means)
            means = means * N_dim**0.5
            self.means = torch.nn.Parameter(means, requires_grad=False)
        else:
            self.means = None

    def forward(self, x, c=None, rev=False):
        return self.flow(x, c=c, rev=rev)


# def simple_INN_init(N_dim, N_blocks = 8, conditions = 0, N_conv_blocks=None, padding_size=1, symmetric_convolution=False, subnet_fc=None, act_func='relu', ch_hidden=None, n_hidden_layers=2, ch_hidden_conv=16, bins=10, coupling_block_type = 'GLOW', clamp=2.0, kwargs_rotations={}, permute_random=False, permute_random=True, permute_random_soft=False, householder_perms=2, use_actnorms=True, use_rotations=True, lr=1e-3, device=None):
def get_INN(config):

    if config.condition_shapes is None and config.condition_type is not None:
        raise ValueError("If condition_type is specified, condition_shapes must be provided.")

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
        n_components = config.condition_shapes[0]
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
        flow, N_dim=config.N_dim, n_components=n_components, condition_type=config.condition_type
    )

    if config.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    elif config.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    elif config.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=1e-5)
    elif config.optimizer_type == "schedulefree":
        optimizer = schedulefree.AdamWScheduleFree(
            model.parameters(), lr=config.lr, warmup_steps=config.warmup_steps
        )
        optimizer.train()

    return model, optimizer  # , losses


#     optimizer_type="schedulefree",
#     warmup_steps=100,
#     n_layers=1,
# )
# flow = flow.to(device=device)
