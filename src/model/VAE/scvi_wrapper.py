import torch
import torch.nn as nn

from scvi import REGISTRY_KEYS


class SCVIMixturePriorWrapper(nn.Module):
    """Adapts a trained MixturePriorSCVI model to the ModelWithMixturePrior interface
    (`forward(x, c=None, rev=False)` + `.means`/`.means_active`/`.log_sigma`) so it can be
    dropped into `Analyzer` for analysis only. Not trainable through this wrapper - use
    MixturePriorSCVI.train() directly for that.

    Caveats vs. a real ModelWithMixturePrior:
    - `x` passed with rev=False must be raw counts (same layer the model was trained on),
      not the log-normalized `adata.X` the rest of the analysis pipeline typically uses.
    - rev=False returns the posterior mean `qz.loc`, not a reparameterized sample.
    - rev=True decodes against a fixed canonical library size/batch (computed from the
      training adata at wrap time), since a bare latent `z` carries neither.
    - log_det is always zero (scVI's encoder/decoder aren't a bijective flow), same
      convention already used by the plain VAEModel drop-in.
    """

    def __init__(self, scvi_model, default_batch_index=0):
        super().__init__()
        self.scvi_model = scvi_model
        self.module = scvi_model.module
        self.N_dim = self.module.n_latent
        self.default_batch_index = default_batch_index

        counts = scvi_model.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        counts = torch.as_tensor(counts, dtype=torch.float32)
        log_library = torch.log(counts.sum(dim=1))
        self.register_buffer("default_log_library", log_library.mean().reshape(1, 1))

    @property
    def means(self):
        return self.module.means_active

    @property
    def means_active(self):
        return self.module.means_active

    @property
    def log_sigma(self):
        return self.module.log_sigma

    def forward(self, x, c=None, rev=False):
        n = x.shape[0]
        batch_index = torch.full(
            (n, 1), self.default_batch_index, dtype=torch.long, device=x.device
        )

        if not rev:
            if (x < 0).any():
                raise ValueError(
                    "SCVIMixturePriorWrapper expects raw counts for rev=False, got negative values."
                )
            inference_out = self.module.inference(x, batch_index)
            z = inference_out["qz"].loc
            return z, torch.zeros(n, device=x.device)
        else:
            library = self.default_log_library.to(x.device).expand(n, 1)
            generative_out = self.module.generative(x, library, batch_index)
            x_hat = generative_out["px"].mu
            return x_hat, torch.zeros(n, device=x.device)
