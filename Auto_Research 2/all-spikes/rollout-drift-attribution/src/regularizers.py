from __future__ import annotations

import math

import torch


def _moment_proxy_single(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    mean = z.mean(dim=0)
    centered = z - mean
    cov = centered.T @ centered / (z.shape[0] - 1)
    eye = torch.eye(z.shape[-1], device=z.device, dtype=z.dtype)
    return mean.square().sum() + (cov - eye).square().sum()


def _sliced_epps_pulley_single(
    z: torch.Tensor,
    projections: int,
    gamma: float,
) -> torch.Tensor:
    """Sliced Epps-Pulley distance from projected latents to N(0, 1) for a
    single (B, D) batch of samples drawn from one time point.

    This is an exact-integral Gaussian-kernel-MMD-to-N(0,1) variant of the
    Epps-Pulley statistic (LeJEPA/KerJEPA note the exact integral recovers
    kernel MMD), not the official quadrature estimator used by le-wm.
    """
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    directions = torch.randn(z.shape[-1], projections, device=z.device, dtype=z.dtype)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    h = z @ directions

    const_qq = 1.0 / math.sqrt(1.0 + 4.0 * gamma)
    const_pq = 1.0 / math.sqrt(1.0 + 2.0 * gamma)
    diff = h[:, None, :] - h[None, :, :]
    pp = torch.exp(-gamma * diff.square()).mean(dim=(0, 1))
    pq = const_pq * torch.exp((-gamma / (1.0 + 2.0 * gamma)) * h.square()).mean(dim=0)
    return (pp + const_qq - 2.0 * pq).mean()


def regularizer_loss(
    kind: str,
    z: torch.Tensor,
    projections: int,
    gamma: float,
    max_samples: int | None = None,
) -> torch.Tensor:
    """Stepwise regularizer: z is (B, T, D). The statistic is computed
    independently per time point t on the full batch B, then averaged over
    T. This equalizes exposure across protocols (same batch structure, same
    sample count, same time weighting at every t) instead of flattening
    correlated frames from different t into one pool, which would confound
    regularizer effect with protocol (see design doc v3 near-fatal issue 3).
    `max_samples` is accepted for CLI compatibility but unused: no
    subsampling is applied so R1/R2 see identical sample counts under every
    protocol.
    """
    del max_samples
    if kind == "none":
        return z.new_tensor(0.0)
    if z.dim() == 2:
        z = z.unsqueeze(1)
    bsz, t_steps, _ = z.shape
    if kind == "moment":
        return torch.stack([_moment_proxy_single(z[:, t]) for t in range(t_steps)]).mean()
    if kind == "sigreg":
        return torch.stack(
            [_sliced_epps_pulley_single(z[:, t], projections=projections, gamma=gamma) for t in range(t_steps)]
        ).mean()
    raise ValueError(f"unknown regularizer: {kind}")

