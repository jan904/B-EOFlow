import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import FrEIA.modules.splines as Fms
from FrEIA.modules.coupling_layers import _BaseCouplingBlock
from FrEIA.modules import InvertibleModule

from typing import Callable, Union
import geotorch
import schedulefree


def default_means_seperation(n_components):
    """Component separation for a K-component mixture with unit-variance components:

        d = 2 * sqrt(2 * ln K)

    `sqrt(2 ln K)` is the scale at which the maximum of K standard normal variables
    concentrates, so it's the distance at which a component's own samples stop being
    routinely closer to some *other* component's mean than to their own. Below it the
    components are not reliably distinguishable no matter how well the flow is trained;
    the factor 2 puts each mean that far outside the confusion radius rather than right
    on it. Growing like sqrt(ln K) means adding components costs very little extra
    spread - K=11 gives 6.2, K=198 gives 6.5, K=2000 gives 7.8.

    Note this is the *pairwise distance* between two component means, not their norm -
    `ModelWithMixturePrior` converts (orthonormal rows at norm `d/sqrt(2)` sit `d` apart).
    """
    # K=1 has nothing to separate; clamp so the rule stays well-defined (ln 1 = 0).
    return 2.0 * math.sqrt(2.0 * math.log(max(int(n_components), 2)))


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
        means_seperation=None,
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
            #
            # `means_seperation` is the pairwise distance between two component means,
            # in units of the prior's per-dimension std. None resolves to
            # `default_means_seperation(K)` = 2*sqrt(2 ln K) - see there. `orthogonal_`
            # gives unit-norm rows, and two orthogonal vectors of norm r sit r*sqrt(2)
            # apart, so the norm needed for a separation of d is d/sqrt(2).
            #
            # This used to scale by `N_dim**0.5 * means_seperation`, putting the means at
            # ~2*sqrt(N_dim) - twice the radius of a typical latent point, and sqrt(2)*
            # that apart. From there gradient descent never restructured them (the NLL
            # gradient w.r.t. mu vanishes as soon as the flow maps each combo onto its
            # assigned point), so the means stayed at their random orthogonal init: the
            # arrangement of conditions in latent space was fixed noise, and differences
            # between means - what counterfactual latent arithmetic relies on - carried
            # no information about the conditions.
            if means_seperation is None:
                means_seperation = default_means_seperation(n_components)
            self.means_seperation = float(means_seperation)  # resolved, for logging

            means_active = torch.zeros(n_components, self.means_dim)
            torch.nn.init.orthogonal_(means_active)
            means_active = means_active * (self.means_seperation / math.sqrt(2.0))
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
            self.means_seperation = None

    @property
    def means(self):
        if self.means_active is None:
            return None
        if self.means_zero is None:
            return self.means_active
        return torch.cat([self.means_active, self.means_zero], dim=1)

    def forward(self, x, c=None, rev=False):
        return self.model(x, c=c, rev=rev)


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

    if config.condition_shapes is None:
        cond = None
        cond_shape = None
    elif config.condition_type == "mixture":
        cond = None
        cond_shape = None

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
            elif config.coupling_block_type == "REALNVP":
                flow.append(
                    Fm.RNVPCouplingBlock,
                    cond=cond,
                    cond_shape=cond_shape,
                    subnet_constructor=subnet_fc,
                    clamp=config.clamp,
                )
            # elif config.coupling_block_type == 'GIN':
            #     flow.append(Fm.GINCouplingBlock, cond=cond, cond_shape=cond_shape, subnet_constructor=subnet_fc)
            # elif config.coupling_block_type == 'NICE':
            #     flow.append(Fm.NICECouplingBlock, cond=cond, cond_shape=cond_shape, subnet_constructor=subnet_fc)
            # elif config.coupling_block_type == 'None':
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

    n_components = None
    if config.condition_type == "mixture":
        if config.n_clusters is None:
            n_components = config.condition_shapes[0]
        else:
            n_components = config.n_clusters

    model = ModelWithMixturePrior(
        flow,
        N_dim=config.N_dim,
        n_components=n_components,
        condition_type=config.condition_type,
        trainable_means=config.trainable_means,
        means_seperation=config.means_seperation,
        latent_per_condition=config.latent_per_condition,
        trainable_sigma=config.trainable_sigma,
        init_sigma=config.init_sigma,
    )

    return model
