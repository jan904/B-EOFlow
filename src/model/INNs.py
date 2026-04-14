import torch
import torch.nn as nn
import torch.nn.functional as F

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import FrEIA.modules.splines as Fms
from FrEIA.modules.coupling_layers import _BaseCouplingBlock
from FrEIA.modules import InvertibleModule

from typing import Callable, Union
import geotorch
import schedulefree


class LearnableLinearTransform(InvertibleModule):
    """Fixed linear transformation for 1D input tesors. The transformation is
    :math:`y = Mx + b`. With *d* input dimensions, *M* must be an invertible *d x d* tensor,
    and *b* is an optional offset vector of length *d*."""

    def __init__(
        self, dims_in, dims_c=None, M: torch.Tensor = None, b: Union[None, torch.Tensor] = None
    ):
        """Additional args in docstring of base class FrEIA.modules.InvertibleModule.

        Args:
          M: Square, invertible matrix, with which each input is multiplied. Shape ``(d, d)``.
          b: Optional vector which is added element-wise. Shape ``(d,)``.
        """
        super().__init__(dims_in, dims_c)

        # TODO: it should be possible to give conditioning instead of M, so that the condition
        # provides M and b on each forward pass.

        if M is None:
            raise ValueError("Need to specify the M argument, the matrix to be multiplied.")

        # self.register_parameter("M", nn.Parameter(M.t()))
        # self.register_buffer("M_inv", (M.t().inverse()))

        # if b is None:
        #     self.register_buffer("b", torch.tensor(0.))
        # else:
        #     self.register_buffer("b", b.unsqueeze(0))

        # self.register_buffer("logDetM", torch.slogdet(M)[1])
        self.M = nn.Parameter(M.t())
        # self.M_inv = (self.M.inverse())
        if b is None:
            self.b = nn.Parameter(torch.zeros(1))
        else:
            self.b = nn.Parameter(b.unsqueeze(0))
        # self.logDetM = torch.slogdet(self.M)[1]

    def forward(self, x, rev=False, jac=True):
        # j = self.logDetM.expand(x[0].shape[0])
        ljd = torch.slogdet(self.M)[1].expand(x[0].shape[0])
        if not rev:
            out = x[0].mm(self.M) + self.b
            return (out,), ljd
        else:
            out = (x[0] - self.b).mm(self.M.inverse())
            return (out,), -ljd

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        return input_dims


class Orthogonal(_BaseCouplingBlock):
    def __init__(
        self,
        dims_in,
        dims_c=[],
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "ATAN",
        split_len: Union[float, int] = 0.5,
        init_identity: bool = False,
    ):

        super().__init__(dims_in, dims_c, clamp, clamp_activation, split_len=split_len)
        N_dim = dims_in[0][0]

        self.conditional = False
        self.conds = 0

        self.matrix = nn.Parameter(torch.Tensor(N_dim, N_dim))
        geotorch.orthogonal(self, "matrix")

        if init_identity:
            with torch.no_grad():
                self.matrix.copy_(torch.eye(N_dim))

    def forward(self, x_or_z, c=None, rev=False, jac=True):
        if rev:
            z = x_or_z[0]
            x = z @ self.matrix.t()
            ljd = x.new_zeros(x.shape[0])
            x_or_z = (x,)
        else:
            x = x_or_z[0]
            z = x @ self.matrix
            ljd = x.new_zeros(x.shape[0])
            x_or_z = (z,)
        return x_or_z, ljd

    def output_dims(self, input_dims):
        return input_dims


def get_subnet_fc(c_in, c_out, ch_hidden=128, act_func="relu", layers=3):
    if act_func == "relu":
        act_func = nn.ReLU()  # nn.Tanh() #nn.ReLU()
    elif act_func == "tanh":
        act_func = nn.Tanh()
    else:
        raise ValueError("Activation function not recognized.")

    subnet = nn.Sequential()
    if layers > 0:
        subnet.append(nn.Linear(c_in, ch_hidden))
        subnet.append(act_func)
        for i in range(layers - 1):
            subnet.append(nn.Linear(ch_hidden, ch_hidden))
            subnet.append(act_func)
        subnet.append(nn.Linear(ch_hidden, c_out))
    else:
        subnet.append(nn.Linear(c_in, c_out))
    subnet[-1].weight.data.zero_()
    subnet[-1].bias.data.zero_()
    return subnet


def get_linear_INN(
    N_dim,
    lr=1e-3,
    use_decomposition=False,
    init_mode="orthogonal",
    M_init=None,
    optimizer_type="SGD",
    warmup_steps=0,
    n_layers=5,
):

    ## Works well:
    # flow, optimizer_flow = get_linear_INN(
    #     N_dim,
    #     lr=1e-3,
    #     use_decomposition=False,
    #     init_mode="orthogonal",
    #     optimizer_type="schedulefree",
    #     warmup_steps=100,
    #     n_layers=1,
    # )

    # LearnableLinearTransform
    flow = Ff.SequenceINN(N_dim)

    if use_decomposition:
        flow.append(Orthogonal)
        flow.append(Fm.ActNorm)
        flow.append(Orthogonal)
    else:
        for _ in range(n_layers):
            if init_mode == "random":
                # Initialize a linear transformation with random matrix and zero bias
                M = torch.randn(N_dim, N_dim)
            if init_mode == "orthogonal":
                # Initialize a linear transformation with orthogonal matrix and zero bias
                M, _ = torch.linalg.qr(torch.randn(N_dim, N_dim))
                # Q = torch.eye(N_dim)
            elif init_mode == "unit_det":
                # Initialize a linear transformation with random matrix with unit determinant and zero bias
                A = torch.randn(N_dim, N_dim)
                det_A = torch.det(A)
                M = A / (det_A.abs() ** (1.0 / N_dim))
                if det_A < 0:
                    M[:, 0] = -M[:, 0]
            if M_init is not None:
                # scale Q by M_init (which should be diagonal)
                M = torch.tensor(M_init, dtype=torch.float32)
            # flow.append(Fm.ActNorm)
            flow.append(LearnableLinearTransform, M=M, b=torch.zeros(N_dim))

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
    else:
        raise NotImplementedError(f"Optimizer type {optimizer_type} not implemented.")

    return flow, optimizer_flow
