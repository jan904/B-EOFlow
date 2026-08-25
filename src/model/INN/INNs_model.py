import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import warnings

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


def build_combo_index(combo_categories, condition_categories, conditions):
    """(K, n_factors) long tensor: row k holds the per-factor level indices of combo k.

    `combo_categories` are '__'-joined labels in `conditions` order (see
    `data_utils.get_condition_vocab`, which builds them as the Cartesian product, and
    `AdataDataset`, which joins obs columns the same way), so combo "IL-2__CD8 Memory"
    with conditions ["cytokine", "cell_type"] maps to (index of IL-2, index of CD8 Memory).

    This is what lets a factorized prior compose `mu(combo) = a_cytokine + b_cell_type`
    while everything downstream keeps indexing means by a single combo id.
    """
    n_factors = len(conditions)
    level_idx = [{v: i for i, v in enumerate(condition_categories[c])} for c in conditions]

    rows = []
    for combo in combo_categories:
        parts = combo.split("__")
        if len(parts) != n_factors:
            raise ValueError(
                f"Combo label '{combo}' splits into {len(parts)} parts but {n_factors} "
                f"conditions were given ({list(conditions)}). A condition value containing "
                "'__' would do this."
            )
        try:
            rows.append([level_idx[i][p] for i, p in enumerate(parts)])
        except KeyError as e:
            raise ValueError(
                f"Combo label '{combo}' references level {e} that is not in "
                f"condition_categories - build both from the same pre-holdout adata."
            ) from e

    return torch.tensor(rows, dtype=torch.long)


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
        means_dim=None,
        combo_index=None,
        condition_shapes=None,
        control_factor=None,
        control_level=None,
        treatment_gain=False,
    ):
        """`combo_index` (with `condition_shapes`) switches the mixture prior from one
        free mean per combo to a **factorized** one:

            mu(cell_type, cytokine) = a_cell_type + b_cytokine

        Free per-combo means give every combo its own independent vector, so a combo
        held out of training keeps whatever its row was initialized to and its mean has
        to be guessed at eval time (`INN_OOD.estimate_leftout_combo_mean`) under an
        additivity assumption the training objective never enforced - measured cosine
        between a cell type's treatment shift and the average of the others' was ~0.
        Factorized, the held-out combo has no free parameter at all: both `a` and `b`
        are fitted from every *other* combo sharing that cell type / that cytokine, so
        its mean is defined by construction, and the likelihood now actively requires a
        cytokine's effect to be the same translation in every cell type - exactly the
        assumption counterfactual latent arithmetic relies on.

        `control_factor`/`control_level` pin the gauge: `a + b` is redundant (add a
        constant to every `a`, subtract it from every `b`), so the named level is fixed
        at zero. With the control condition pinned, `a_cell_type` *is* that cell type's
        control mean and `b_cytokine` *is* the latent treatment effect, which makes the
        counterfactual literally "encode control cells, add `b`, decode" - no shift
        estimation anywhere.

        `means_dim` sets the condition-carrying block directly. Free means size it as
        `latent_per_condition * n_components`, which needlessly grows with the number of
        combos; factorized only needs room for `sum(condition_shapes)` mutually
        orthogonal level vectors.

        `treatment_gain` relaxes the strict additivity to

            mu(cell_type, cytokine) = a_cell_type + w_cell_type * b_cytokine

        with one learned scalar `w` per level of the *non*-treatment factor. The
        treatment still has a single direction shared across cell types - so a held-out
        combo is still composed rather than guessed, `w` being fitted from every other
        cytokine that cell type saw and `b` from every other cell type that cytokine saw -
        but a cell type may now respond more or less strongly to it. It is a rank-1
        interaction: it rescales the shift, it cannot rotate it.

        Two things to know about the parameterization. The gauge above is what keeps `w`
        meaningful: with the control level pinned, `mu = a + w*(b - b_ctrl)` still equals
        `a_cell_type` at control whatever `w` is, so `a` stays the control mean and `w`
        reads as a response amplitude relative to the average cell type - which is why
        `control_factor`/`control_level` are required here. And `w * b` keeps one global
        scale redundancy (scale every `w` by c, every `b` by 1/c), a flat direction that
        leaves every `mu` untouched; it costs nothing during training but means the
        magnitude of a single `w` is only interpretable against the other `w`s.
        """
        super().__init__()
        self.model = model
        self.means_dim = N_dim
        self.N_dim = N_dim
        self.factorized = False

        if condition_type == "mixture":
            if n_components is None:
                raise ValueError("n_components must be specified for mixture condition type.")

            self.factorized = combo_index is not None
            if self.factorized and condition_shapes is None:
                raise ValueError("condition_shapes must be given alongside combo_index.")

            if means_dim is not None:
                self.means_dim = means_dim
            elif latent_per_condition is not None:
                self.means_dim = latent_per_condition * n_components
            elif self.factorized:
                # enough room for every level to get its own orthogonal direction, with
                # slack; `sum(condition_shapes)` is the hard floor asserted below.
                self.means_dim = 2 * int(sum(condition_shapes))

            if self.means_dim > N_dim:
                raise ValueError(f"means_dim ({self.means_dim}) cannot exceed N_dim ({N_dim}).")

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
            scale = self.means_seperation / math.sqrt(2.0)

            self.control_factor = control_factor
            self.control_level = control_level

            if self.factorized:
                if not trainable_means:
                    raise ValueError(
                        "A factorized prior has no closed-form empirical update - its factors "
                        "are only fitted by gradient descent, so trainable_means must be True."
                    )

                condition_shapes = [int(n) for n in condition_shapes]
                total_levels = int(sum(condition_shapes))
                if self.means_dim < total_levels:
                    raise ValueError(
                        f"means_dim ({self.means_dim}) must be at least sum(condition_shapes) "
                        f"({total_levels}) so every level gets its own orthogonal direction."
                    )
                if combo_index.shape != (n_components, len(condition_shapes)):
                    raise ValueError(
                        f"combo_index has shape {tuple(combo_index.shape)}, expected "
                        f"({n_components}, {len(condition_shapes)})."
                    )

                # One joint orthogonal basis sliced per factor, rather than initializing
                # each factor separately: that makes the factors' subspaces mutually
                # orthogonal, so two combos differing in exactly one factor sit
                # `scale * sqrt(2)` = means_seperation apart and the separation rule
                # carries over unchanged from the free-means case. (The gauge shift below
                # is a pure translation of one block, which leaves all pairwise distances
                # untouched.)
                basis = torch.zeros(total_levels, self.means_dim)
                torch.nn.init.orthogonal_(basis)
                basis = basis * scale

                self.factor_emb = nn.ParameterList()
                offset = 0
                for n_levels in condition_shapes:
                    self.factor_emb.append(
                        nn.Parameter(basis[offset : offset + n_levels].clone(), requires_grad=True)
                    )
                    offset += n_levels

                self.register_buffer("combo_index", combo_index.long())
                self.means_free = None
            else:
                if n_components > self.means_dim:
                    # `orthogonal_` on a tall matrix orthonormalizes columns, not rows, so
                    # K vectors in fewer than K dims cannot be mutually orthogonal and the
                    # achieved separation falls short of `means_seperation`. A factorized
                    # prior only needs room for sum(level counts) - 29 here, not 198 -
                    # which is why it does not run into this.
                    warnings.warn(
                        f"n_components ({n_components}) exceeds means_dim ({self.means_dim}): "
                        f"the means cannot be mutually orthogonal, so the initialization will "
                        f"not achieve the requested separation of {self.means_seperation:.3f}. "
                        f"Raise means_dim/latent_per_condition, or factorize the prior.",
                        stacklevel=2,
                    )
                means_free = torch.zeros(n_components, self.means_dim)
                torch.nn.init.orthogonal_(means_free)
                means_free = means_free * scale
                self.means_free = torch.nn.Parameter(means_free, requires_grad=trainable_means)
                self.factor_emb = None
                self.combo_index = None

            # per-cell-type gain on the treatment embedding (see the class docstring).
            # Stored in log space as `gain_log` and exponentiated to `w` by the
            # `treatment_gain` property; `gain_log` initializes at 0, so w = 1 and the
            # model starts as exactly the additive prior.
            self.gain_log = None
            self.gain_factor = None
            if treatment_gain:
                if not self.factorized:
                    raise ValueError(
                        "treatment_gain only applies to a factorized prior - there is no "
                        "treatment embedding to scale when every combo has a free mean."
                    )
                if len(condition_shapes) != 2:
                    raise ValueError(
                        f"treatment_gain is defined for exactly two factors (the treatment "
                        f"and the one the gain is indexed by), got {len(condition_shapes)}. "
                        "With more factors it is ambiguous which one w should depend on."
                    )
                if control_factor is None or control_level is None:
                    raise ValueError(
                        "treatment_gain needs control_factor and control_level: without the "
                        "gauge pinned, scaling one factor's embedding also rescales an "
                        "arbitrary offset, and w stops meaning 'response amplitude'. Pass "
                        "control_condition/control_label on the config."
                    )
                self.gain_factor = 1 - int(control_factor)
                self.gain_log = nn.Parameter(
                    torch.zeros(condition_shapes[self.gain_factor]), requires_grad=True
                )

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
            self.means_free = None
            self.factor_emb = None
            self.combo_index = None
            self.means_zero = None
            self.log_sigma = None
            self.means_seperation = None
            self.control_factor = None
            self.control_level = None
            self.gain_log = None
            self.gain_factor = None

    @property
    def treatment_gain(self):
        """(n_gain_levels,) per-cell-type gain `w`, or None.

        `w = exp(s - mean(s))`, which fixes the gauge `w * b` would otherwise leave free:
        scaling every w by c and every b by 1/c leaves every mu unchanged, so the raw
        parameterization let the whole vector drift (it walked from 1.0 to a median of
        1.5 within 600 epochs) and made `weight_decay` pull along a direction the
        likelihood does not see. Subtracting the mean in log space pins the geometric
        mean of w at exactly 1, so w is identifiable and a single value is readable on
        its own rather than only against the others.

        The exponential also keeps w > 0. A negative gain would mean a cell type
        responding along the *reverse* of the shared treatment direction, which is a fit
        artifact rather than something the rank-1 form is meant to express.
        """
        if self.gain_log is None:
            return None
        return torch.exp(self.gain_log - self.gain_log.mean())

    def _effective_factor(self, i):
        """Factor `i`'s level embeddings with the gauge applied.

        `mu = a + b` is redundant - shifting every `a` by a constant and every `b` by its
        negative leaves every `mu` unchanged - so the pinned level is subtracted off,
        fixing it at zero. Being a translation of one whole block, this leaves all
        pairwise distances between composed means untouched; it only chooses *which*
        decomposition of the same geometry the parameters express, so that
        `factor_emb[control_factor]` reads directly as a treatment effect relative to
        control instead of an arbitrary offset.
        """
        emb = self.factor_emb[i]
        if self.control_factor == i and self.control_level is not None:
            emb = emb - emb[self.control_level]
        return emb

    @property
    def means_active(self):
        """(K, means_dim) condition-carrying block of the prior means.

        Composed from the per-factor embeddings when factorized, otherwise the free
        per-combo parameter. Kept as a single property so every consumer - the losses,
        the `means` property, the OOD samplers - indexes means by combo id regardless of
        how they are parameterized.
        """
        if not self.factorized:
            return self.means_free

        rows = []
        for i in range(len(self.factor_emb)):
            emb = self._effective_factor(i)[self.combo_index[:, i]]
            if self.treatment_gain is not None and i == self.control_factor:
                # w indexed by the *other* factor's level, so each cell type scales the
                # one shared treatment direction by its own amount.
                emb = self.treatment_gain[self.combo_index[:, self.gain_factor]].unsqueeze(1) * emb
            rows.append(emb)
        return sum(rows)

    @property
    def means(self):
        if self.means_active is None:
            return None
        if self.means_zero is None:
            return self.means_active
        return torch.cat([self.means_active, self.means_zero], dim=1)

    def prior_parameters(self):
        """Mixture-prior parameters (factors or free means), for their own lr group."""
        if self.factorized:
            params = list(self.factor_emb.parameters())
            # gain_log, not the `treatment_gain` property: the property builds a new
            # tensor each call, so optimizing it would optimize a temporary
            if self.gain_log is not None:
                params.append(self.gain_log)
            return params
        return [] if self.means_free is None else [self.means_free]

    def prior_parameter_names(self):
        """State-dict names of `prior_parameters`, to exclude them from the base group."""
        if self.factorized:
            names = {f"factor_emb.{i}" for i in range(len(self.factor_emb))}
            if self.gain_log is not None:
                names.add("gain_log")
            return names
        return {"means_free"}

    def treatment_shift(self, level, factor=None, gain_level=None):
        """Latent translation from the pinned control level to `level`, as a full N_dim
        vector (zero-padded outside the condition block) ready to add to a latent code.

        This is what replaces `INN_OOD._average_treatment_shift` for a factorized prior.
        That function had to *estimate* the shift by averaging per-cell-type mean
        differences, which is only valid if those differences agree across cell types -
        they did not. Here the shift is a single parameter fitted jointly from every cell
        type that saw this treatment, so there is nothing to average and nothing to
        estimate, including for a combo held out of training.

        With `treatment_gain` the shift is `w_gain_level * b_level`, so it is no longer
        the same vector for every cell type and `gain_level` (the level index of the
        non-treatment factor) becomes required.
        """
        if not self.factorized:
            raise RuntimeError(
                "treatment_shift requires a factorized prior; with free per-combo means "
                "there is no single treatment vector to read off."
            )
        factor = self.control_factor if factor is None else factor
        if factor is None:
            raise ValueError("No control_factor is set - pass `factor` explicitly.")

        b = self._effective_factor(factor)[level]

        if self.treatment_gain is not None:
            if factor != self.control_factor:
                raise ValueError(
                    f"With treatment_gain, only factor {self.control_factor} (the treatment "
                    f"axis) acts by translation; moving along factor {factor} also changes "
                    "the gain, so there is no single shift vector for it."
                )
            if gain_level is None:
                raise ValueError(
                    "treatment_gain is enabled, so the shift depends on the cell type: pass "
                    f"`gain_level`, the level index into factor {self.gain_factor}."
                )
            b = self.treatment_gain[gain_level] * b

        out = torch.zeros(self.N_dim, device=b.device, dtype=b.dtype)
        out[: self.means_dim] = b
        return out

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

    combo_index = None
    factor_shapes = None
    control_factor = None
    control_level = None
    treatment_gain = bool(getattr(config, "treatment_gain", False))

    if treatment_gain and not (
        config.condition_type == "mixture" and getattr(config, "factorize_means", False)
    ):
        raise ValueError(
            "treatment_gain requires condition_type='mixture' with factorize_means=True - "
            "it scales the factorized prior's treatment embedding."
        )

    if config.condition_type == "mixture" and getattr(config, "factorize_means", False):
        if config.combo_categories is None or config.condition_categories is None:
            raise ValueError(
                "factorize_means needs combo_categories and condition_categories on the "
                "config - build both from the full pre-holdout adata via "
                "data_utils.get_condition_vocab, so every combo (including held-out ones) "
                "has a slot and every level has an index."
            )

        conditions = config.conditions or list(config.condition_categories)
        combo_index = build_combo_index(
            config.combo_categories, config.condition_categories, conditions
        )

        # Derived from condition_categories rather than taken from config.condition_shapes:
        # these must line up with combo_index exactly, and condition_shapes is computed
        # separately (from adata.obs nunique), so it can silently disagree after a holdout.
        factor_shapes = [len(config.condition_categories[c]) for c in conditions]

        if n_components != len(config.combo_categories):
            raise ValueError(
                f"n_clusters ({n_components}) must equal len(combo_categories) "
                f"({len(config.combo_categories)}) for a factorized prior."
            )

        if config.control_condition is not None:
            if config.control_condition not in conditions:
                raise ValueError(
                    f"control_condition '{config.control_condition}' not in {conditions}."
                )
            control_factor = conditions.index(config.control_condition)
            if config.control_label is not None:
                levels = list(config.condition_categories[config.control_condition])
                if config.control_label not in levels:
                    raise ValueError(
                        f"control_label '{config.control_label}' not among the levels of "
                        f"'{config.control_condition}' ({levels})."
                    )
                control_level = levels.index(config.control_label)

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
        means_dim=getattr(config, "means_dim", None),
        combo_index=combo_index,
        condition_shapes=factor_shapes,
        control_factor=control_factor,
        control_level=control_level,
        treatment_gain=treatment_gain,
    )

    return model
