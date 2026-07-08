"""
Encoder and dynamics models for the rollout-drift experiment.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, state_dim, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class SmallDynamics(nn.Module):
    """F2-a: 1 hidden layer, width 128."""
    def __init__(self, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, z):
        return self.net(z)


class LargeDynamics(nn.Module):
    """F2-b: 3 hidden layers, width 128, residual connections."""
    def __init__(self, latent_dim=16):
        super().__init__()
        self.l1 = nn.Linear(latent_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, latent_dim)
        self.act = nn.GELU()
        # Project input to hidden dim for residual
        self.proj = nn.Linear(latent_dim, 128)

    def forward(self, z):
        h = self.act(self.l1(z))
        h = h + self.act(self.l2(h))
        h = h + self.act(self.l3(h))
        return self.out(h)


def sigreg_loss(z):
    """
    Moment-matching SIGReg proxy:
      L = ||mu_batch||^2 + ||Sigma_batch - I||_F^2
    Flags in report: moment-matching variant used, not original characteristic-function SIGReg.
    """
    mu = z.mean(dim=0)
    z_centered = z - mu
    cov = (z_centered.T @ z_centered) / (z.shape[0] - 1)
    I = torch.eye(cov.shape[0], device=z.device)
    return (mu ** 2).sum() + ((cov - I) ** 2).sum()


def make_encoder(state_dim, latent_dim=16):
    return Encoder(state_dim, latent_dim)


def make_dynamics(latent_dim, capacity):
    if capacity == 'small':
        return SmallDynamics(latent_dim)
    elif capacity == 'large':
        return LargeDynamics(latent_dim)
    raise ValueError(capacity)
