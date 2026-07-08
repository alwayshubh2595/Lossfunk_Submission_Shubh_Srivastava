from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class StateEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class VisualEncoder(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], latent_dim: int):
        super().__init__()
        channels = obs_shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(obs))


class Dynamics(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, width: int):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, latent_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.action_dim > 0:
            z = torch.cat([z, action], dim=-1)
        return self.net(z)


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        latent_dim: int,
        visual: bool,
    ):
        super().__init__()
        if visual:
            self.encoder = VisualEncoder(obs_shape, latent_dim)
            dyn_width = 256
        else:
            self.encoder = StateEncoder(obs_shape[0], latent_dim)
            dyn_width = 128
        self.dynamics = Dynamics(latent_dim, action_dim, dyn_width)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.visual = visual

    def encode_flat(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

