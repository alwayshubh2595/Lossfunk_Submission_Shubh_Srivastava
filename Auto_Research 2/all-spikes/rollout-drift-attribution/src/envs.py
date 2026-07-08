from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class SequenceDataset:
    name: str
    obs: torch.Tensor
    actions: torch.Tensor
    states: torch.Tensor
    visual: bool
    obs_mean: torch.Tensor | None = None
    obs_std: torch.Tensor | None = None

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[-1])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[-1])

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        return tuple(self.obs.shape[2:])

    @property
    def seq_len(self) -> int:
        return int(self.actions.shape[1])


def make_datasets(
    name: str,
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seq_len: int,
    image_size: int = 64,
) -> Dict[str, SequenceDataset]:
    total = n_train + n_val + n_test
    if name in {"stable-linear", "stable-linear-det"}:
        obs, actions, states = _stable_linear(total, seq_len, seed)
        visual = False
    elif name == "ou":
        obs, actions, states = _ou_process(total, seq_len, seed)
        visual = False
    elif name == "lorenz":
        obs, actions, states = _lorenz(total, seq_len, seed)
        visual = False
    elif name == "pointmass-state":
        obs, actions, states = _pointmass_state(total, seq_len, seed)
        visual = False
    elif name == "pointmass":
        obs, actions, states = _pointmass_visual(total, seq_len, seed, image_size)
        visual = True
    else:
        raise ValueError(f"unknown env: {name}")

    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, total)

    obs_mean = None
    obs_std = None
    if not visual:
        mean = obs[train_slice].reshape(-1, obs.shape[-1]).mean(axis=0)
        std = obs[train_slice].reshape(-1, obs.shape[-1]).std(axis=0)
        std = np.maximum(std, 1e-6)
        obs = (obs - mean) / std
        obs_mean = mean.astype(np.float32)
        obs_std = std.astype(np.float32)

    return {
        "train": _pack(name, obs[train_slice], actions[train_slice], states[train_slice], visual, obs_mean, obs_std),
        "val": _pack(name, obs[val_slice], actions[val_slice], states[val_slice], visual, obs_mean, obs_std),
        "test": _pack(name, obs[test_slice], actions[test_slice], states[test_slice], visual, obs_mean, obs_std),
    }


def simulate_pointmass_step(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    drift = 0.015 * np.stack(
        [np.sin(np.pi * state[..., 1]), -np.sin(np.pi * state[..., 0])],
        axis=-1,
    )
    return np.clip(state + action + drift, -1.0, 1.0)


def rollout_true_states(name: str, x0: np.ndarray, actions: np.ndarray, horizon: int) -> np.ndarray:
    states = np.zeros((x0.shape[0], horizon + 1, x0.shape[-1]), dtype=np.float32)
    states[:, 0] = x0.astype(np.float32)
    if name in {"stable-linear", "stable-linear-det"}:
        a = _stable_matrix()
        for t in range(horizon):
            states[:, t + 1] = states[:, t] @ a.T
    elif name == "ou":
        # Ground-truth sensitivity to initial condition is governed by the
        # deterministic linear map; the stochastic OU noise term is common
        # to base/perturbed branches under the standard finite-difference
        # perturbation protocol and does not affect the amplification ratio.
        a = _ou_matrix()
        for t in range(horizon):
            states[:, t + 1] = states[:, t] @ a.T
    elif name == "lorenz":
        dt = 0.01
        for t in range(horizon):
            x = states[:, t].astype(np.float64)
            k1 = _lorenz_rhs(x)
            k2 = _lorenz_rhs(x + 0.5 * dt * k1)
            k3 = _lorenz_rhs(x + 0.5 * dt * k2)
            k4 = _lorenz_rhs(x + dt * k3)
            states[:, t + 1] = (x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).astype(np.float32)
    elif name in {"pointmass", "pointmass-state"}:
        for t in range(horizon):
            states[:, t + 1] = simulate_pointmass_step(states[:, t], actions[:, t])
    else:
        raise ValueError(f"unknown env for rollout: {name}")
    return states


def observations_from_states(dataset: SequenceDataset, states: np.ndarray) -> torch.Tensor:
    if dataset.name == "pointmass":
        return torch.from_numpy(_render_pointmass(states.astype(np.float32), dataset.obs.shape[-1]))
    if dataset.obs_mean is None or dataset.obs_std is None:
        raise ValueError("state observations require stored normalization statistics")
    mean = dataset.obs_mean.numpy()
    std = dataset.obs_std.numpy()
    obs = (states.astype(np.float32) - mean) / std
    return torch.from_numpy(obs.astype(np.float32))


def _pack(
    name: str,
    obs: np.ndarray,
    actions: np.ndarray,
    states: np.ndarray,
    visual: bool,
    obs_mean: np.ndarray | None,
    obs_std: np.ndarray | None,
) -> SequenceDataset:
    return SequenceDataset(
        name=name,
        obs=torch.from_numpy(obs.astype(np.float32)),
        actions=torch.from_numpy(actions.astype(np.float32)),
        states=torch.from_numpy(states.astype(np.float32)),
        visual=visual,
        obs_mean=torch.from_numpy(obs_mean) if obs_mean is not None else None,
        obs_std=torch.from_numpy(obs_std) if obs_std is not None else None,
    )


def _stable_matrix() -> np.ndarray:
    q_rng = np.random.default_rng(12345)
    q, _ = np.linalg.qr(q_rng.normal(size=(3, 3)))
    return (q @ np.diag([0.7, 0.8, 0.9]) @ q.T).astype(np.float32)


def _ou_matrix() -> np.ndarray:
    q_rng = np.random.default_rng(54321)
    q, _ = np.linalg.qr(q_rng.normal(size=(3, 3)))
    return (q @ np.diag([0.7, 0.8, 0.9]) @ q.T).astype(np.float32)


def _ou_process(n: int, seq_len: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stationary Ornstein-Uhlenbeck-style AR(1) process: same contraction
    rates as stable-linear (eigenvalues 0.7/0.8/0.9) but with additive
    process noise calibrated so the marginal is a stationary, isotropic
    N(0, I) at every timestep -- SIGReg's favorable regime, unlike the
    deterministic stable-linear system whose marginal variance changes with
    time (Codex review, near-fatal issue 1).
    """
    rng = np.random.default_rng(seed)
    a_diag = np.array([0.7, 0.8, 0.9], dtype=np.float64)
    q_rng = np.random.default_rng(54321)
    q, _ = np.linalg.qr(q_rng.normal(size=(3, 3)))
    a = (q @ np.diag(a_diag) @ q.T).astype(np.float32)
    noise_std_rot = np.sqrt(np.maximum(1.0 - a_diag**2, 1e-8)).astype(np.float32)

    states = np.zeros((n, seq_len + 1, 3), dtype=np.float32)
    states[:, 0] = rng.normal(size=(n, 3)).astype(np.float32)  # stationary init, cov = I
    for t in range(seq_len):
        noise_rot = rng.normal(size=(n, 3)).astype(np.float32) * noise_std_rot[None, :]
        noise = noise_rot @ q.T
        states[:, t + 1] = states[:, t] @ a.T + noise

    actions = np.zeros((n, seq_len, 0), dtype=np.float32)
    return states.copy(), actions, states


def _stable_linear(n: int, seq_len: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = _stable_matrix()

    states = np.zeros((n, seq_len + 1, 3), dtype=np.float32)
    states[:, 0] = rng.normal(size=(n, 3)).astype(np.float32)
    for t in range(seq_len):
        states[:, t + 1] = states[:, t] @ a.T

    actions = np.zeros((n, seq_len, 0), dtype=np.float32)
    return states.copy(), actions, states


def _lorenz(n: int, seq_len: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.zeros((n, seq_len + 1, 3), dtype=np.float32)
    states[:, 0, 0] = rng.uniform(-12.0, 12.0, size=n)
    states[:, 0, 1] = rng.uniform(-12.0, 12.0, size=n)
    states[:, 0, 2] = rng.uniform(8.0, 32.0, size=n)

    dt = 0.01
    for t in range(seq_len):
        x = states[:, t].astype(np.float64)
        k1 = _lorenz_rhs(x)
        k2 = _lorenz_rhs(x + 0.5 * dt * k1)
        k3 = _lorenz_rhs(x + 0.5 * dt * k2)
        k4 = _lorenz_rhs(x + dt * k3)
        states[:, t + 1] = (x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).astype(np.float32)

    actions = np.zeros((n, seq_len, 0), dtype=np.float32)
    return states.copy(), actions, states


def _lorenz_rhs(x: np.ndarray) -> np.ndarray:
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    dx = sigma * (x[:, 1] - x[:, 0])
    dy = x[:, 0] * (rho - x[:, 2]) - x[:, 1]
    dz = x[:, 0] * x[:, 1] - beta * x[:, 2]
    return np.stack([dx, dy, dz], axis=-1)


def _pointmass_state(n: int, seq_len: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    states, actions = _pointmass_rollout(n, seq_len, seed)
    return states.copy(), actions, states


def _pointmass_visual(
    n: int,
    seq_len: int,
    seed: int,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    states, actions = _pointmass_rollout(n, seq_len, seed)
    obs = _render_pointmass(states, image_size)
    return obs, actions, states


def _pointmass_rollout(n: int, seq_len: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.zeros((n, seq_len + 1, 2), dtype=np.float32)
    actions = np.zeros((n, seq_len, 2), dtype=np.float32)
    states[:, 0] = rng.uniform(-0.85, 0.85, size=(n, 2)).astype(np.float32)

    prev_action = np.zeros((n, 2), dtype=np.float32)
    for t in range(seq_len):
        noise = rng.normal(scale=0.035, size=(n, 2)).astype(np.float32)
        action = np.clip(0.85 * prev_action + noise, -0.1, 0.1)
        actions[:, t] = action
        states[:, t + 1] = simulate_pointmass_step(states[:, t], action)
        prev_action = action
    return states, actions


def _render_pointmass(states: np.ndarray, image_size: int) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, image_size, dtype=np.float32)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    flat = states.reshape(-1, 2)
    dx = xx[None, :, :] - flat[:, 0, None, None]
    dy = yy[None, :, :] - flat[:, 1, None, None]
    blob = np.exp(-(dx * dx + dy * dy) / (2.0 * 0.055**2)).astype(np.float32)
    grid_pattern = 0.03 * (
        (np.abs(np.sin(4.0 * np.pi * xx))[None, :, :] > 0.96)
        | (np.abs(np.sin(4.0 * np.pi * yy))[None, :, :] > 0.96)
    ).astype(np.float32)
    frames = np.clip(blob + grid_pattern, 0.0, 1.0)
    frames = frames.reshape(states.shape[0], states.shape[1], 1, image_size, image_size)
    return frames.astype(np.float32)
