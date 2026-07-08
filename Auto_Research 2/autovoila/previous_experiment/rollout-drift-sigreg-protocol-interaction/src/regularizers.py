from __future__ import annotations

import math

import torch


def moment_proxy(z: torch.Tensor) -> torch.Tensor:
    z = z.reshape(-1, z.shape[-1])
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    mean = z.mean(dim=0)
    centered = z - mean
    cov = centered.T @ centered / (z.shape[0] - 1)
    eye = torch.eye(z.shape[-1], device=z.device, dtype=z.dtype)
    return mean.square().sum() + (cov - eye).square().sum()


def sliced_epps_pulley_sigreg(
    z: torch.Tensor,
    projections: int = 256,
    gamma: float = 0.5,
    max_samples: int = 256,
    chunk: int = 32,
) -> torch.Tensor:
    """Sliced Epps-Pulley distance from projected latents to N(0, 1).

    This is a Monte Carlo implementation of the canonical SIGReg idea:
    random unit projections plus the univariate Epps-Pulley/Gaussian-kernel
    MMD statistic against a standard normal target.
    """
    z = z.reshape(-1, z.shape[-1])
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    if z.shape[0] > max_samples:
        idx = torch.randperm(z.shape[0], device=z.device)[:max_samples]
        z = z[idx]

    directions = torch.randn(z.shape[-1], projections, device=z.device, dtype=z.dtype)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    projected = z @ directions

    const_qq = 1.0 / math.sqrt(1.0 + 4.0 * gamma)
    const_pq = 1.0 / math.sqrt(1.0 + 2.0 * gamma)
    vals = []
    for start in range(0, projections, chunk):
        h = projected[:, start : start + chunk]
        diff = h[:, None, :] - h[None, :, :]
        pp = torch.exp(-gamma * diff.square()).mean(dim=(0, 1))
        pq = const_pq * torch.exp((-gamma / (1.0 + 2.0 * gamma)) * h.square()).mean(dim=0)
        vals.append(pp + const_qq - 2.0 * pq)
    return torch.cat(vals, dim=0).mean()


def regularizer_loss(
    kind: str,
    z: torch.Tensor,
    projections: int,
    gamma: float,
    max_samples: int,
) -> torch.Tensor:
    if kind == "none":
        return z.new_tensor(0.0)
    if kind == "moment":
        return moment_proxy(z)
    if kind == "sigreg":
        return sliced_epps_pulley_sigreg(
            z,
            projections=projections,
            gamma=gamma,
            max_samples=max_samples,
        )
    raise ValueError(f"unknown regularizer: {kind}")

