import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import FrEIA.modules.splines as Fms
from FrEIA.modules.coupling_layers import _BaseCouplingBlock

import schedulefree

from src.model.INNs import Orthogonal, get_subnet_fc, get_linear_INN


# def simple_INN_init(N_dim, N_blocks = 8, conditions = 0, N_conv_blocks=None, padding_size=1, symmetric_convolution=False, subnet_fc=None, act_func='relu', ch_hidden=None, n_hidden_layers=2, ch_hidden_conv=16, bins=10, coupling_block_type = 'GLOW', clamp=2.0, kwargs_rotations={}, permute_random=False, permute_random=True, permute_random_soft=False, householder_perms=2, use_actnorms=True, use_rotations=True, lr=1e-3, device=None):
def get_INN(
    N_dim,
    N_blocks=8,
    condition_shapes=None,
    subnet_fc=None,
    act_func="relu",
    ch_hidden=128,
    n_hidden_layers=2,
    coupling_block_type="GLOW",
    RQS_bins=10,
    clamp=2.0,
    pre_rotate=True,
    pre_normalize=True,
    permute_random=True,
    rotate_random=True,
    normalize=True,
    post_rotate=True,
    lr=1e-3,
    optimizer_type="Adam",
    warmup_steps=0,
):

    flow = Ff.SequenceINN(N_dim)

    if subnet_fc == None:
        subnet_fc = lambda c_in, c_out: get_subnet_fc(
            c_in, c_out, act_func=act_func, ch_hidden=ch_hidden, layers=n_hidden_layers
        )

    if condition_shapes is not None:
        cond = 0
        cond_shape = condition_shapes
    else:
        cond = None
        cond_shape = None

    if pre_normalize:
        flow.append(Fm.ActNorm)
    if pre_rotate:
        flow.append(Orthogonal)

    if coupling_block_type:
        for k in range(N_blocks):
            if coupling_block_type == "GLOW":
                flow.append(
                    Fm.GLOWCouplingBlock,
                    cond=cond,
                    cond_shape=cond_shape,
                    subnet_constructor=subnet_fc,
                    clamp=clamp,
                )
            elif coupling_block_type == "RQS":
                flow.append(
                    Fms.RationalQuadraticSpline,
                    cond=cond,
                    cond_shape=cond_shape,
                    bins=RQS_bins,
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

            if permute_random:
                flow.append(Fm.PermuteRandom)
            if normalize:
                flow.append(Fm.ActNorm)
            if rotate_random and not (k == N_blocks - 1):
                M = torch.linalg.qr(torch.randn(N_dim, N_dim))[0]
                flow.append(Fm.FixedLinearTransform, M=M)

    if post_rotate:
        flow.append(Orthogonal)

    if optimizer_type == "Adam":
        optimizer_flow = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-5)
    elif optimizer_type == "AdamW":
        optimizer_flow = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-5)
    elif optimizer_type == "SGD":
        optimizer_flow = torch.optim.SGD(flow.parameters(), lr=lr, weight_decay=1e-5)
    elif optimizer_type == "schedulefree":
        optimizer_flow = schedulefree.AdamWScheduleFree(
            flow.parameters(), lr=lr, warmup_steps=warmup_steps
        )
        optimizer_flow.train()

    return flow, optimizer_flow  # , losses
