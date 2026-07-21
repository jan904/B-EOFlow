import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence as kl
import numpy as np
from anndata import AnnData

from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.module import VAE
from scvi import REGISTRY_KEYS
from scvi.module.base import LossOutput
from scvi.data import AnnDataManager
from scvi.data.fields import (
    LayerField,
    CategoricalObsField,
    NumericalObsField,
    CategoricalJointObsField,
    NumericalJointObsField,
)

from src.model.VAE.VAE_training import MixturePriorTrainingPlan


def get_VAE(config, adata):
    adata.obs["cluster_idx"] = adata.obs[config.condition].astype("category").cat.codes

    MixturePriorSCVI.setup_anndata(adata, cluster_key="cluster_idx", layer="counts", batch_key=None)

    model = MixturePriorSCVI(
        adata,
        n_components=adata.obs["cluster_idx"].nunique(),
        n_latent=config.N_dim,
        trainable_means=config.trainable_means,
        trainable_sigma=config.trainable_sigma,
        means_seperation=config.means_seperation,
        kl_weight=config.beta,
    )

    model = SCVIMixturePriorWrapper(model)

    return model


class MixturePriorVAE(VAE):
    """scVI VAE with a Gaussian-mixture prior in latent space instead of N(0, I)."""

    def __init__(
        self,
        *args,
        n_components=None,
        trainable_means=True,
        trainable_sigma=False,
        init_sigma=1.0,
        means_seperation=2.0,
        component_key="cluster_idx",
        kl_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.component_key = component_key  # registered obs field with the cluster/condition id
        self.kl_weight = kl_weight  # weight for the KL term in the loss

        means_active = torch.zeros(n_components, self.n_latent)
        nn.init.orthogonal_(means_active)
        means_active = means_active * self.n_latent**0.5 * means_seperation
        self.means_active = nn.Parameter(means_active, requires_grad=trainable_means)

        self.log_sigma = (
            nn.Parameter(
                torch.full((n_components,), float(np.log(init_sigma))),
                requires_grad=trainable_sigma,
            )
            if trainable_sigma
            else None
        )

    def loss(self, tensors, inference_outputs, generative_outputs):
        component_idx = tensors[self.component_key].long().squeeze(-1)
        qz = inference_outputs["qz"]  # variational posterior q(z|x)

        mean_p = self.means_active[component_idx]
        sigma_p = (
            torch.exp(self.log_sigma[component_idx])
            if self.log_sigma is not None
            else torch.ones_like(mean_p[:, 0])
        )
        pz = Normal(mean_p, sigma_p.unsqueeze(-1).expand_as(mean_p))

        kl_z = kl(qz, pz).sum(dim=-1)

        # scVI's own reconstruction loss: -log p(x | z) under the decoder's likelihood
        x = tensors[REGISTRY_KEYS.X_KEY]
        reconst_loss = -generative_outputs["px"].log_prob(x).sum(-1)

        weighted_kl = self.kl_weight * kl_z
        loss = torch.mean(reconst_loss + weighted_kl)
        return LossOutput(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_z)


class MixturePriorSCVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """User-facing model class wrapping MixturePriorVAE, mirroring scvi.model.SCVI."""

    _training_plan_cls = MixturePriorTrainingPlan

    def __init__(
        self,
        adata: AnnData,
        n_components: int,
        n_latent: int = 10,
        trainable_means: bool = True,
        trainable_sigma: bool = False,
        init_sigma: float = 1.0,
        means_seperation: float = 2.0,
        **module_kwargs,
    ):
        super().__init__(adata)

        n_batch = self.summary_stats.n_batch

        self.module = MixturePriorVAE(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_latent=n_latent,
            n_components=n_components,
            trainable_means=trainable_means,
            trainable_sigma=trainable_sigma,
            init_sigma=init_sigma,
            means_seperation=means_seperation,
            component_key="cluster_idx",  # matches the registry key below
            **module_kwargs,
        )

        self._model_summary_string = (
            f"MixturePriorSCVI(n_latent={n_latent}, n_components={n_components})"
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        cluster_key: str,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())

        anndata_fields = [
            LayerField(
                REGISTRY_KEYS.X_KEY,
                layer,
                is_count_data=True,
            ),
            CategoricalObsField(
                REGISTRY_KEYS.BATCH_KEY,
                batch_key,
            ),
            CategoricalObsField(
                REGISTRY_KEYS.LABELS_KEY,
                labels_key,
            ),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY,
                size_factor_key,
                required=False,
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY,
                categorical_covariate_keys,
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY,
                continuous_covariate_keys,
            ),
            # your additional field
            CategoricalObsField(
                "cluster_idx",
                cluster_key,
            ),
        ]

        adata_manager = AnnDataManager(
            fields=anndata_fields,
            setup_method_args=setup_method_args,
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)


class SCVIMixturePriorWrapper(nn.Module):
    """Adapts a trained MixturePriorSCVI model to the ModelWithMixturePrior interface
    (`forward(x, c=None, rev=False)` + `.means`/`.means_active`/`.log_sigma`) so it can be
    dropped into `Analyzer` for analysis only, in place of an INN flow. Not trainable through
    this wrapper - use MixturePriorSCVI.train() directly for that.

    MixturePriorSCVI is trained directly on raw counts (get_VAE registers it via
    `setup_anndata(adata, cluster_key="cluster_idx", layer="counts", ...)` - see
    src/model/VAE/VAE_model.py's get_VAE), not on `adata.X`. By default
    (normalize_output=False) this wrapper matches that: `forward()` feeds/returns raw
    counts directly, with no log1p/expm1 translation, so `analyzer.adata` should also
    hold raw counts (mirroring SCGENMixturePriorWrapper's default) for a true count-space
    comparison against real data.

    Pass normalize_output=True to instead operate in log1p(normalize_total) space (the
    space `adata.X` is in when loaded with log_transform=True), for compatibility with the
    rest of the analysis pipeline as it was before this was made an option:
    - rev=False treats `x` as log1p-normalized data and inverts it to pseudo-counts via
      `expm1` before it hits the scVI encoder. Those pseudo-counts always sum to `target_sum`
      per cell, so they differ slightly from the true raw counts (variable library size) the
      model was actually trained on.
    - rev=True decodes against a fixed canonical library size (`target_sum`), since a bare
      latent `z` carries no library-size information, then reapplies `log1p` so the output
      lands back in log-normalized space comparable to a log-transformed `adata.X`.

    Other caveats vs. a real ModelWithMixturePrior, regardless of normalize_output:
    - rev=False returns the posterior mean `qz.loc`, not a reparameterized sample.
    - log_det is always zero (scVI's encoder/decoder aren't a bijective flow), same convention
      already used by the plain VAEModel drop-in.
    """

    def __init__(self, scvi_model, target_sum=1e4, default_batch_index=0, normalize_output=False):
        super().__init__()
        self.model = scvi_model
        self.module = scvi_model.module
        self.N_dim = self.module.n_latent
        self.default_batch_index = default_batch_index
        self.normalize_output = normalize_output
        self.register_buffer(
            "log_target_sum", torch.log(torch.tensor(float(target_sum))).reshape(1, 1)
        )

    @property
    def means(self):
        return self.module.means_active

    @property
    def means_active(self):
        return self.module.means_active

    @property
    def get_latent_representation(self):
        return self.model.get_latent_representation

    @property
    def log_sigma(self):
        return self.module.log_sigma

    def train(self, max_epochs, plan_kwargs=None):
        return self.model.train(max_epochs=max_epochs, plan_kwargs=plan_kwargs)

    def forward(self, x, c=None, rev=False):
        n = x.shape[0]
        batch_index = torch.full(
            (n, 1), self.default_batch_index, dtype=torch.long, device=x.device
        )

        if not rev:
            counts = torch.expm1(x) if self.normalize_output else x
            inference_out = self.module.inference(counts, batch_index)
            z = inference_out["qz"].loc
            return z, torch.zeros(n, device=x.device)
        else:
            library = self.log_target_sum.to(x.device).expand(n, 1)
            generative_out = self.module.generative(x, library, batch_index)
            px = generative_out["px"].mu
            x_hat = torch.log1p(px) if self.normalize_output else px
            return x_hat, torch.zeros(n, device=x.device)
