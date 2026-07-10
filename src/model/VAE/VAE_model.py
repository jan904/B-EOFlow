import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np


class VAEModel(nn.Module):
    """Drop-in for a FrEIA flow inside INNWithMixturePrior.
    rev=False: encode x -> sample z ~ q(z|x); log_det = log|dz/deps| (reparam jacobian)
    rev=True:  decode z -> x_hat (deterministic); log_det = 0
    """

    def __init__(
        self, D_dim, N_dim, cond=0, cond_shape=None, hidden=2048, n_layers=2, act_func="relu"
    ):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "elu": nn.ELU}[act_func]

        def mlp(d_in, d_out, n_hidden_layers):
            layers, d = [], d_in
            for _ in range(n_hidden_layers):
                layers += [nn.Linear(d, hidden), act()]
                d = hidden
            layers.append(nn.Linear(d, d_out))
            return nn.Sequential(*layers)

        self.dec = mlp(N_dim + (cond_shape[cond] if cond_shape else 0), D_dim, n_layers)
        self.enc = mlp(D_dim, N_dim, n_layers)

        self.N_dim = N_dim

    def forward(self, x, c=None, rev=False):
        if c is not None and self.cond_dim > 0:
            inp = torch.cat([x, c], dim=-1)
        else:
            inp = x

        if not rev:
            z = self.enc(inp)
            return z, torch.zeros(x.shape[0], device=x.device)
        else:
            x_hat = self.dec(inp)
            return x_hat, torch.zeros(x.shape[0], device=x.device)


def get_VAE(config):
    pass
