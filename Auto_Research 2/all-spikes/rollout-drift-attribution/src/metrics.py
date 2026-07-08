from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch

from .envs import SequenceDataset, observations_from_states, rollout_true_states
from .models import WorldModel


@torch.no_grad()
def evaluate_model(
    model: WorldModel,
    train: SequenceDataset,
    val: SequenceDataset,
    device: torch.device,
    eval_horizon: int,
    batch_size: int = 256,
) -> Dict[str, float]:
    model.eval()
    horizons = [h for h in [1, 5, 10, 25, 50] if h <= eval_horizon]
    metrics: Dict[str, float] = {}
    metrics.update(_latent_geometry(model, train, val, device))
    metrics.update(_rollout_errors(model, val, device, horizons, batch_size))
    metrics.update(_perturbation_amplification(model, val, device, horizons, batch_size))
    probe = _fit_linear_probe(model, train, val, device)
    metrics["linear_probe_r2"] = probe["r2"]
    metrics.update(_state_excess_amplification(model, val, device, horizons, batch_size, probe))
    metrics.update(_probe_open_loop_error(model, val, device, horizons, batch_size, probe))
    trace = metrics.get("cov_trace", 0.0)
    eff_dim = metrics.get("effective_dim", 0.0)
    rank = metrics.get("rank_1pct", 0.0)
    probe_r2 = metrics.get("linear_probe_r2", 0.0)
    metrics["collapsed"] = float(eff_dim < 2.0 or rank < 2.0 or probe_r2 < 0.5)
    if "excess_amp_25" in metrics:
        metrics["abs_log_excess_amp_25"] = float(abs(np.log(max(metrics["excess_amp_25"], 1e-12))))
    return metrics


@torch.no_grad()
def _encode_obs(model: WorldModel, obs: torch.Tensor, device: torch.device, chunk: int = 512) -> torch.Tensor:
    flat = obs.reshape(-1, *obs.shape[2:])
    outs: List[torch.Tensor] = []
    for start in range(0, flat.shape[0], chunk):
        outs.append(model.encode_flat(flat[start : start + chunk].to(device)).cpu())
    return torch.cat(outs, dim=0).reshape(*obs.shape[:2], -1)


def _latent_geometry(
    model: WorldModel,
    train: SequenceDataset,
    val: SequenceDataset,
    device: torch.device,
) -> Dict[str, float]:
    del train
    z = _encode_obs(model, val.obs[: min(128, val.obs.shape[0])], device).reshape(-1, model.latent_dim)
    z_np = z.numpy()
    cov = np.cov(z_np, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    trace = float(eigvals.sum())
    eff_dim = float((trace * trace) / (float(np.square(eigvals).sum()) + 1e-12))
    rank_1pct = float((eigvals > max(trace * 0.01, 1e-8)).sum())
    return {
        "cov_trace": trace,
        "effective_dim": eff_dim,
        "rank_1pct": rank_1pct,
    }


@torch.no_grad()
def _rollout_errors(
    model: WorldModel,
    val: SequenceDataset,
    device: torch.device,
    horizons: Iterable[int],
    batch_size: int,
) -> Dict[str, float]:
    max_h = max(horizons)
    idx, t0 = _sample_windows(val, batch_size, max_h, device)
    obs = _gather_obs(val, idx, t0, max_h, device)
    actions = _gather_actions(val, idx, t0, max_h, device)
    bsz = obs.shape[0]
    z_obs = model.encode_flat(obs.reshape(-1, *obs.shape[2:])).reshape(bsz, max_h + 1, -1)

    pred = z_obs[:, 0]
    preds = {}
    for k in range(1, max_h + 1):
        pred = model.dynamics(pred, actions[:, k - 1])
        if k in horizons:
            preds[k] = pred

    out: Dict[str, float] = {}
    for k, z_hat in preds.items():
        out[f"open_loop_mse_{k}"] = float((z_hat - z_obs[:, k]).square().mean().item())
    return out


@torch.no_grad()
def _perturbation_amplification(
    model: WorldModel,
    val: SequenceDataset,
    device: torch.device,
    horizons: Iterable[int],
    batch_size: int,
    eps: float = 1e-3,
) -> Dict[str, float]:
    max_h = max(horizons)
    idx, t0 = _sample_windows(val, batch_size, max_h, device)
    obs0 = _gather_obs(val, idx, t0, 0, device)[:, 0]
    actions = _gather_actions(val, idx, t0, max_h, device)
    z = model.encode_flat(obs0)
    direction = torch.randn_like(z)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    z_pert = z + eps * direction

    out: Dict[str, float] = {}
    for k in range(1, max_h + 1):
        z = model.dynamics(z, actions[:, k - 1])
        z_pert = model.dynamics(z_pert, actions[:, k - 1])
        if k in horizons:
            amp = (z_pert - z).norm(dim=-1) / eps
            out[f"perturb_amp_{k}"] = float(amp.mean().item())
    return out


@torch.no_grad()
def _fit_linear_probe(
    model: WorldModel,
    train: SequenceDataset,
    val: SequenceDataset,
    device: torch.device,
) -> Dict[str, np.ndarray | float]:
    n_train = min(128, train.obs.shape[0])
    n_val = min(128, val.obs.shape[0])
    z_train = _encode_obs(model, train.obs[:n_train], device).reshape(-1, model.latent_dim).numpy()
    z_val = _encode_obs(model, val.obs[:n_val], device).reshape(-1, model.latent_dim).numpy()
    y_train = train.states[:n_train].reshape(-1, train.state_dim).numpy()
    y_val = val.states[:n_val].reshape(-1, val.state_dim).numpy()

    z_mean = z_train.mean(axis=0, keepdims=True)
    z_std = z_train.std(axis=0, keepdims=True) + 1e-6
    x_train = np.concatenate([(z_train - z_mean) / z_std, np.ones((z_train.shape[0], 1))], axis=1)
    x_val = np.concatenate([(z_val - z_mean) / z_std, np.ones((z_val.shape[0], 1))], axis=1)
    ridge = 1e-3 * np.eye(x_train.shape[1], dtype=np.float64)
    weights = np.linalg.solve(x_train.T @ x_train + ridge, x_train.T @ y_train)
    pred = x_val @ weights
    ss_res = float(np.square(y_val - pred).sum())
    ss_tot = float(np.square(y_val - y_val.mean(axis=0, keepdims=True)).sum()) + 1e-12
    return {
        "weights": weights,
        "z_mean": z_mean,
        "z_std": z_std,
        "r2": 1.0 - ss_res / ss_tot,
    }


def _apply_probe(z: torch.Tensor, probe: Dict[str, np.ndarray | float]) -> np.ndarray:
    z_np = z.detach().cpu().numpy()
    z_mean = probe["z_mean"]
    z_std = probe["z_std"]
    weights = probe["weights"]
    assert isinstance(z_mean, np.ndarray)
    assert isinstance(z_std, np.ndarray)
    assert isinstance(weights, np.ndarray)
    x = np.concatenate([(z_np - z_mean) / z_std, np.ones((z_np.shape[0], 1))], axis=1)
    return x @ weights


@torch.no_grad()
def _state_excess_amplification(
    model: WorldModel,
    val: SequenceDataset,
    device: torch.device,
    horizons: Iterable[int],
    batch_size: int,
    probe: Dict[str, np.ndarray | float],
    eps: float = 1e-3,
) -> Dict[str, float]:
    max_h = max(horizons)
    idx, t0 = _sample_windows(val, batch_size, max_h, device)
    idx_np = idx.cpu().numpy()
    t0_np = t0.cpu().numpy()

    x0 = val.states[idx_np, t0_np].numpy()
    actions = _gather_actions(val, idx, t0, max_h, device)
    actions_np = actions.cpu().numpy()

    direction = np.random.normal(size=x0.shape).astype(np.float32)
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    x0_pert = x0 + eps * direction

    true_base = rollout_true_states(val.name, x0, actions_np, max_h)
    true_pert = rollout_true_states(val.name, x0_pert, actions_np, max_h)

    obs_pair = observations_from_states(val, np.stack([true_base[:, 0], true_pert[:, 0]], axis=1))
    z_base = model.encode_flat(obs_pair[:, 0].to(device))
    z_pert = model.encode_flat(obs_pair[:, 1].to(device))

    out: Dict[str, float] = {}
    for k in range(1, max_h + 1):
        z_base = model.dynamics(z_base, actions[:, k - 1])
        z_pert = model.dynamics(z_pert, actions[:, k - 1])
        if k in horizons:
            pred_base = _apply_probe(z_base, probe)
            pred_pert = _apply_probe(z_pert, probe)
            model_amp = np.linalg.norm(pred_pert - pred_base, axis=-1) / eps
            gt_amp = np.linalg.norm(true_pert[:, k] - true_base[:, k], axis=-1) / eps
            excess = model_amp / np.maximum(gt_amp, 1e-12)
            out[f"probe_state_amp_{k}"] = float(model_amp.mean())
            out[f"gt_state_amp_{k}"] = float(gt_amp.mean())
            out[f"excess_amp_{k}"] = float(model_amp.mean() / max(float(gt_amp.mean()), 1e-12))
            out[f"excess_amp_median_ratio_{k}"] = float(np.median(excess))
    return out


@torch.no_grad()
def _probe_open_loop_error(
    model: WorldModel,
    val: SequenceDataset,
    device: torch.device,
    horizons: Iterable[int],
    batch_size: int,
    probe: Dict[str, np.ndarray | float],
) -> Dict[str, float]:
    """Primary drift metric (experiment_design.md v3): probe-space open-loop
    state error along the model's own free-running rollout (no perturbation,
    just following the actual action sequence), integrated over the
    available horizons via the trapezoidal rule. This is not gameable by
    over-contraction the way a signed amplification ratio is -- a collapsed
    model that predicts a constant will have large, not small, open-loop
    state error against the true trajectory.
    """
    max_h = max(horizons)
    idx, t0 = _sample_windows(val, batch_size, max_h, device)
    idx_np = idx.cpu().numpy()
    t0_np = t0.cpu().numpy()
    obs0 = _gather_obs(val, idx, t0, 0, device)[:, 0]
    actions = _gather_actions(val, idx, t0, max_h, device)
    true_states = val.states[idx_np[:, None], t0_np[:, None] + np.arange(max_h + 1)[None, :]].numpy()

    z = model.encode_flat(obs0)
    out: Dict[str, float] = {}
    sorted_h = sorted(set(horizons) | {0})
    errors_at_h: Dict[int, float] = {0: 0.0}
    for k in range(1, max_h + 1):
        z = model.dynamics(z, actions[:, k - 1])
        if k in horizons:
            pred_state = _apply_probe(z, probe)
            err = float(np.linalg.norm(pred_state - true_states[:, k], axis=-1).mean())
            out[f"probe_open_loop_error_{k}"] = err
            errors_at_h[k] = err
    xs = [h for h in sorted_h if h == 0 or h in horizons]
    ys = [errors_at_h[h] for h in xs]
    out["integrated_probe_open_loop_error"] = _trapz(ys, xs) if len(xs) > 1 else float("nan")
    return out


def _trapz(ys: list[float], xs: list[int]) -> float:
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz to trapezoid
    return float(trapz_fn(ys, xs))


def _sample_windows(
    dataset: SequenceDataset,
    batch_size: int,
    horizon: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_t = dataset.seq_len - horizon
    if max_t < 0:
        raise ValueError(f"eval horizon {horizon} exceeds sequence length {dataset.seq_len}")
    idx = torch.randint(0, dataset.obs.shape[0], (batch_size,), device=device)
    t0 = torch.randint(0, max_t + 1, (batch_size,), device=device)
    return idx, t0


def _gather_obs(
    dataset: SequenceDataset,
    idx: torch.Tensor,
    t0: torch.Tensor,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    steps = torch.arange(horizon + 1, device=device)
    time = t0[:, None] + steps[None, :]
    return dataset.obs[idx.cpu()[:, None], time.cpu()].to(device)


def _gather_actions(
    dataset: SequenceDataset,
    idx: torch.Tensor,
    t0: torch.Tensor,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    if horizon == 0:
        return dataset.actions[:0, :0].to(device)
    steps = torch.arange(horizon, device=device)
    time = t0[:, None] + steps[None, :]
    return dataset.actions[idx.cpu()[:, None], time.cpu()].to(device)
