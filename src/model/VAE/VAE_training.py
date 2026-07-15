import torch

from scvi.train import TrainingPlan


class MixturePriorTrainingPlan(TrainingPlan):
    """TrainingPlan that optimizes the mixture-component means with their own learning rate.

    The means are kept fixed at their initial (orthogonal, well-separated) positions for the
    first `means_warmup_epochs` epochs, and only start receiving gradient updates afterwards.
    Without this, the means tend to collapse together very early in training, before the rest
    of the model has learned a latent space worth placing them in.
    """

    def __init__(
        self, *args, lr_means: float | None = None, means_warmup_epochs: int = 0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lr_means = lr_means if lr_means is not None else self.lr
        self.means_warmup_epochs = means_warmup_epochs

    def configure_optimizers(self):
        named_params = dict(self.module.named_parameters())
        means_params = [
            p for n, p in named_params.items() if n == "means_active" and p.requires_grad
        ]
        other_params = [
            p for n, p in named_params.items() if n != "means_active" and p.requires_grad
        ]

        param_groups = [{"params": other_params, "lr": self.lr}]
        if means_params:
            param_groups.append({"params": means_params, "lr": self.lr_means})

        optimizer_cls = torch.optim.AdamW if self.optimizer_name == "AdamW" else torch.optim.Adam
        optimizer = optimizer_cls(param_groups, eps=self.eps, weight_decay=self.weight_decay)
        config = {"optimizer": optimizer}
        if self.reduce_lr_on_plateau:
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )
        return config

    def on_before_optimizer_step(self, optimizer):
        # means_active stays in the optimizer's param group (with its own lr_means) the whole
        # time so Adam's moment estimates are ready to go the instant warmup ends - we just
        # zero its gradient every step during warmup so it doesn't move at all.
        if self.current_epoch < self.means_warmup_epochs:
            means_active = self.module.means_active
            if means_active.grad is not None:
                means_active.grad.zero_()
