import numpy as np
import pandas as pd
import scgen
import scanpy as sc
import torch
import torch.nn as nn
from scgen._scgenvae import SCGENVAE
from scvi.module.base import auto_move_data
from torch.distributions import Normal

from src.utils.utils import normalize_log1p


@auto_move_data
def _inference(self, x):
    """Patched inference: adds a "qz" Distribution so scvi-tools>=1.1's
    VAEMixin.get_latent_representation (which expects MODULE_KEYS.QZ_KEY,
    "qz", rather than scgen's legacy "qz_m"/"qz_v" keys) works correctly.
    """
    qz_m, qz_v, z = self.z_encoder(x)
    return dict(z=z, qz_m=qz_m, qz_v=qz_v, qz=Normal(qz_m, qz_v.sqrt()))


SCGENVAE.inference = _inference


class SCGENMixturePriorWrapper(nn.Module):
    """Adapts a trained scgen.SCGEN model to the ModelWithMixturePrior interface
    (`forward(x, c=None, rev=False)` + `.means`/`.means_active`), mirroring
    SCVIMixturePriorWrapper (src/model/VAE/VAE_model.py), so it can be dropped into
    `Analyzer`/`generate_counterfactuals` for analysis only, in place of an INN flow
    or the scVI wrapper. Not trainable through this wrapper - use scgen.SCGEN.train()
    directly for that, then build this wrapper from the trained model.

    This model is trained directly on raw counts in `adata.X` (per scGen's official
    tutorial - setup_anndata warns if adata.X does not look like unnormalized count
    data, and setup_anndata has no `layer=` option to point it elsewhere), and
    `analyzer.adata`/`.means` are built from that same raw-count data. But like
    SCVIMixturePriorWrapper, `forward()`'s external contract is log-normalized space
    (expm1 before the encoder, library-size-normalize + log1p after the decoder) so its
    data-space output is comparable to the INN and the scVI wrapper. Note this means
    `forward()` itself expects log-normalized `x` on the rev=False path, unlike
    `analyzer.adata` which stays raw counts - the generic `generate_counterfactuals`
    (which batches directly from analyzer.adata) would need a log-normalized adata to
    feed it correctly. The actually-used scGen counterfactual path,
    generate_counterfactuals_scgen, avoids this by calling `.predict()`/
    `get_latent_representation()` directly (bypassing forward()); `.predict()` below
    applies the same library-size-normalize + log1p to its output, so callers get
    data already comparable to the "real" comparison data pulled from analyzer.adata
    (see mmd.ctrl_mmd_dists/mmd_dists, log1p_transform=True).

    This wrapper only bridges the *interface* to what Analyzer expects: scGen has no
    learned mixture-prior means parameter, so `.means` is the empirical per-condition
    mean latent representation instead, computed once at construction time.

    Caveat: SCGENVAE has no library-size term anywhere in its encoder/decoder/loss (unlike
    scVI's NB decoder, which explicitly conditions its mean on library size) - training
    directly on raw per-cell counts therefore risks conflating sequencing depth with
    condition effects. Its decoder also has no `library` input to condition on in the first
    place (unlike scVI's), so `forward(rev=True)`/`predict()` can't ask the decoder for a
    canonical-library-size reconstruction the way SCVIMixturePriorWrapper does - instead
    they rescale the decoder's raw output to sum to `target_sum` per cell after the fact
    (`normalize_log1p`), mirroring what `sc.pp.normalize_total` does to real data. Also,
    unlike scVI's NB decoder mean, SCGENVAE's decoder is a plain linear layer with no
    positivity constraint, so its raw-count-scale output can go negative; it's clamped to 0
    before that rescale to avoid NaNs.
    """

    def __init__(self, scgen_model, adata, labels_key, target_sum=1e4):
        super().__init__()
        self.model = scgen_model
        self.module = scgen_model.module
        self.N_dim = self.module.n_latent

        # order must match AdataDataset's `cats[0]` (see generate_counterfactuals),
        # i.e. the categories of adata.obs[labels_key] as pandas would sort them
        cats = pd.Categorical(adata.obs[f"{labels_key}"].astype(str))
        z = self.model.get_latent_representation(adata)
        means = np.stack([z[cats.codes == i].mean(axis=0) for i in range(len(cats.categories))])
        self.register_buffer("_means", torch.tensor(means, dtype=torch.float32))
        self.register_buffer(
            "log_target_sum", torch.log(torch.tensor(float(target_sum))).reshape(1, 1)
        )

    @property
    def means(self):
        return self._means

    @property
    def means_active(self):
        return self._means

    @property
    def log_sigma(self):
        return None

    @property
    def get_latent_representation(self):
        return self.model.get_latent_representation

    def train(self, max_epochs, **kwargs):
        return self.model.train(max_epochs=max_epochs, **kwargs)

    def predict(self, *args, **kwargs):
        """Passthrough to scgen.SCGEN.predict (official ctrl -> stim latent-arithmetic),
        used by generate_counterfactuals_scgen instead of the generic mean-shift forward().
        scgen.SCGEN.predict returns its reconstruction in raw-count scale (see class
        docstring); rescale it to library-size-normalized + log1p here so callers get
        data directly comparable to the rest of the (already log-normalized) pipeline.

        Normalizes against adata_to_predict's own real per-cell total, not the
        reconstruction's own row sum: SCGENVAE's decoder has no positivity constraint,
        so genes with a low true mean get clipped to 0 whenever reconstruction noise
        pushes them negative. That biases the clipped row sum downward (as does the
        decoder's own lack of any library-size term - see class docstring caveat), so
        normalizing each predicted cell to sum to target_sum using *that* shrunken total
        would inflate whatever survived clipping. adata_to_predict's real total, being
        actual non-negative counts, doesn't have this problem and is what the cell's
        depth "should" be regardless of how much reconstruction mass clipping wiped out.
        """
        if "adata_to_predict" not in kwargs:
            raise ValueError(
                "SCGENMixturePriorWrapper.predict requires adata_to_predict as a "
                "keyword argument, to normalize against its real per-cell library size."
            )
        adata_to_predict = kwargs["adata_to_predict"]
        source_x = adata_to_predict.X
        source_counts = source_x.toarray() if hasattr(source_x, "toarray") else source_x
        source_totals = source_counts.sum(axis=1, keepdims=True)

        pred_adata, delta = self.model.predict(*args, **kwargs)
        counts = np.clip(pred_adata.X, a_min=0, a_max=None)
        pred_adata.X = normalize_log1p(counts, totals=source_totals)
        return pred_adata, delta

    def forward(self, x, c=None, rev=False):
        n = x.shape[0]
        if not rev:
            counts = torch.expm1(x)
            z = self.module.inference(counts)["qz_m"]
            return z, torch.zeros(n, device=x.device)
        else:
            px = self.module.generative(x)["px"].clamp(min=0)
            target_sum = self.log_target_sum.exp().to(x.device)
            totals = px.sum(dim=1, keepdim=True).clamp(min=1e-8)
            x_hat = torch.log1p(px)  # torch.log1p(px / totals * target_sum)
            return x_hat, torch.zeros(n, device=x.device)
