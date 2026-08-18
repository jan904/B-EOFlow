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
    `analyzer.adata`/`.means` are built from that same raw-count data. Like
    SCVIMixturePriorWrapper, `forward()`'s external contract is log-normalized space
    (expm1 before the encoder, library-size-normalize + log1p after the decoder) so its
    data-space output is comparable to the INN and the scVI wrapper - but this path isn't
    used by the actual scGen counterfactual pipeline (see below).

    The actually-used scGen counterfactual path, generate_counterfactuals_scgen, calls
    `.predict()`/`get_latent_representation()` directly (bypassing forward()). `.predict()`
    below defaults to staying in raw-count scale (only clipping negatives to 0, no
    library-size rescale + log1p): SCGENVAE's decoder is a plain linear layer with no
    positivity constraint and no library-size term, so its raw output is only a rough,
    noisy stand-in for counts, with reconstruction noise centered near 0 for
    low-expression genes. Rescaling that to `target_sum` and log1p'ing it amplifies that
    noise floor - clipping negatives to 0 while keeping all the positive noise is a
    one-sided (rectified) bias, and blowing every cell up to `target_sum` inflates it
    further, disproportionately for genes that should be near zero. Comparing directly in
    raw-count scale (`log1p_transform=False` in mmd.ctrl_mmd_dists/mmd_dists and the
    cf_expressions plots) avoids that amplification, at the cost of no longer being on
    the same absolute scale as the log-normalized INN/scVI/Means comparisons. Pass
    `normalize_output=True` at construction to opt back into the library-size-normalize +
    log1p behavior instead (then use `log1p_transform=True` in the comparison functions
    to match).

    This wrapper only bridges the *interface* to what Analyzer expects: scGen has no
    learned mixture-prior means parameter, so `.means` is the empirical per-condition
    mean latent representation instead, computed once at construction time.

    Pass `conditions` (e.g. `["cytokine", "cell_type"]`) to index `.means` by the full
    combo instead of `labels_key` alone - the same "__"-joined combo label
    AdataDataset/get_condition_vocab use, so this can drop into the
    src.analysis.INN_OOD.estimate_leftout_combo_mean/sample_leftout_combo_normal/
    sample_leftout_combo_shift machinery built for the INN's mixture prior (nothing in
    those functions is actually INN-specific - they only touch `.means`/`.means_active`/
    `.log_sigma`/`forward()`). Pass `combo_categories` (from get_condition_vocab on the
    full, pre-holdout adata)
    alongside it so a held-out combo still reserves a slot (filled with zeros here,
    ready for estimate_leftout_combo_mean to overwrite) instead of vanishing because no
    real cells matched it in whatever `adata` this was constructed from.

    For a leave-one-combo-out setup, construct this wrapper from `train_adata` (the
    holdout-excluded set returned by split_holdout_combinations), not the full adata -
    scgen.SCGEN.setup_anndata/.train() only ever needs train_adata too, since scGen
    conditions on `batch_key`/`labels_key` as separate covariates rather than a crossed
    combo, so holding out one (condition, cell_type) pair never empties either
    covariate's own vocabulary and needs no special handling on the training side.
    Passing the full adata here instead would leak the real held-out combo's cells into
    its "estimated" mean, defeating the point of estimating it.
    """

    def __init__(
        self,
        scgen_model,
        adata,
        labels_key,
        target_sum=1e4,
        normalize_output=False,
        conditions=None,
        combo_categories=None,
    ):
        super().__init__()
        self.model = scgen_model
        self.module = scgen_model.module
        self.N_dim = self.module.n_latent
        self.normalize_output = normalize_output

        # order must match AdataDataset's `cats[0]` (see generate_counterfactuals) /
        # get_condition_vocab's combo_categories, i.e. adata.obs[conditions] joined in
        # that same column order
        conditions = list(conditions) if conditions is not None else [labels_key]
        combo_labels = adata.obs[conditions].astype(str).agg("__".join, axis=1)
        cats = pd.Categorical(combo_labels, categories=combo_categories)
        self.combo_categories = list(cats.categories)

        z = self.model.get_latent_representation(adata)
        means = np.zeros((len(cats.categories), z.shape[1]), dtype=np.float32)
        for i in range(len(cats.categories)):
            mask = cats.codes == i
            if mask.any():
                means[i] = z[mask].mean(axis=0)
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
        docstring). By default (self.normalize_output=False) this only clips negative
        values to 0 and leaves it at that raw-count scale, since rescaling to
        target_sum + log1p amplifies the decoder's reconstruction noise floor (see class
        docstring). Set normalize_output=True at construction to instead rescale to
        library-size-normalized + log1p, comparable to the log-space INN/scVI pipeline.

        When normalizing, normalizes against adata_to_predict's own real per-cell total,
        not the reconstruction's own row sum: SCGENVAE's decoder has no positivity
        constraint, so genes with a low true mean get clipped to 0 whenever
        reconstruction noise pushes them negative. That biases the clipped row sum
        downward, so normalizing each predicted cell to sum to target_sum using *that*
        shrunken total would inflate whatever survived clipping. adata_to_predict's real
        total, being actual non-negative counts, doesn't have this problem and is what
        the cell's depth "should" be regardless of how much reconstruction mass clipping
        wiped out.

        Also accepts `celltype_to_predict` (mutually exclusive with adata_to_predict, per
        the underlying scgen.SCGEN.predict) to predict every real *control* cell of that
        cell type found in the model's own registered adata - the official scGen
        leave-one-combo/cell-type-out pattern - rather than an explicit adata passed in.
        normalize_output=True still requires adata_to_predict, since there's no other
        source for each cell's real per-cell total to normalize against.
        """
        if kwargs.get("adata_to_predict") is None and kwargs.get("celltype_to_predict") is None:
            raise ValueError(
                "SCGENMixturePriorWrapper.predict requires adata_to_predict or "
                "celltype_to_predict as a keyword argument (passed straight through to "
                "scgen.SCGEN.predict)."
            )
        if self.normalize_output and kwargs.get("adata_to_predict") is None:
            raise ValueError(
                "normalize_output=True requires adata_to_predict (not celltype_to_predict) "
                "to normalize against the source cells' own real per-cell total."
            )

        pred_adata, delta = self.model.predict(*args, **kwargs)
        counts = np.clip(pred_adata.X, a_min=0, a_max=None)

        if self.normalize_output:
            adata_to_predict = kwargs["adata_to_predict"]
            source_x = adata_to_predict.X
            source_counts = source_x.toarray() if hasattr(source_x, "toarray") else source_x
            source_totals = source_counts.sum(axis=1, keepdims=True)
            pred_adata.X = normalize_log1p(counts, totals=source_totals)
        else:
            pred_adata.X = counts

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
