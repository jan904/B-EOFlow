from dataclasses import dataclass, field


@dataclass
class BaseModelConfig:
    model_type: str = field(init=False, default="base")

    N_dim: int = 2000
    sigma_noise: float = (1.0,)
    optimizer_type: str = "schedulefree"
    warmup_steps: int = 300
    condition_shapes: list = None
    condition_type: str = None
    act_func: str = "relu"
    lr: float = 5e-4
    lr_means: float = 5e-4
    lr_sigma: float = 5e-4
    trainable_means: bool = False
    trainable_sigma: bool = False
    init_sigma: float = 1.0
    n_clusters: int = None
    means_seperation: float = 2.0
    ctrl_idx: int = None
    supervise_latent_meaning: bool = False
    lam_supervise: float = 0.01
    latent_per_condition: int = None
    partition_divisor: int = 8
    balance_classes: bool = False
    holdout_combos: list = None
    combo_categories: list = None
    condition_categories: dict = None

    def __post_init__(self):
        self._validate_condition_type()

    def _validate_condition_type(self):
        if self.condition_shapes is not None and self.condition_type is None:
            raise ValueError("condition_type must be set when condition_shapes is provided.")
        if self.condition_type is not None and self.condition_shapes is None:
            raise ValueError("condition_shapes must be set when condition_type is provided.")

        valid_condition_types = {"mixture", "normal", None}  # add yours
        if self.condition_type not in valid_condition_types:
            raise ValueError(
                f"condition_type must be one of {valid_condition_types}, got '{self.condition_type}'."
            )


@dataclass
class INNConfig(BaseModelConfig):
    model_type: str = field(init=False, default="inn")

    # Pairwise distance between two mixture-component means, in units of the prior's
    # per-dimension std. None (default) resolves at model-build time to
    # `INNs_model.default_means_seperation(n_clusters)` = 2*sqrt(2 ln K), which is the
    # point where components stop being routinely confusable; see that function.
    # Overrides BaseModelConfig's fixed 2.0 (kept there for the VAE path, whose
    # initialization still uses its own scale convention).
    means_seperation: float = None

    N_blocks: int = 12
    subnet_fc: callable = None
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

    def __post_init__(self):
        super().__post_init__()
        if self.coupling_block_type not in {"GLOW", "RQS", None}:
            raise ValueError(f"Unknown coupling_block_type '{self.coupling_block_type}'.")


@dataclass
class VAEConfig(BaseModelConfig):
    model_type: str = field(init=False, default="vae")

    condition: str = "cytokine"
    conditions: list = None
    beta: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.beta < 0:
            raise ValueError("beta must be non-negative.")
