"""
Three synthetic environments for the rollout-drift causal attribution experiment.
All return trajectories as numpy arrays of shape (T, state_dim).
"""

import numpy as np
from scipy.integrate import odeint


def make_stable_linear(n_traj=50000, T=100, seed=0):
    """z_{t+1} = A z_t, eigenvalues {0.7, 0.8, 0.9}. A_k should shrink."""
    rng = np.random.RandomState(seed)
    # Construct A with eigenvalues 0.7, 0.8, 0.9 via random orthogonal basis
    Q, _ = np.linalg.qr(rng.randn(3, 3))
    A = Q @ np.diag([0.7, 0.8, 0.9]) @ Q.T
    z0 = rng.randn(n_traj, 3)
    trajs = np.zeros((n_traj, T, 3))
    trajs[:, 0, :] = z0
    for t in range(1, T):
        trajs[:, t, :] = trajs[:, t-1, :] @ A.T
    lyap = np.log(0.9)  # dominant eigenvalue log
    return trajs, A, lyap


def make_spiral_2d(n_traj=50000, T=100, seed=0):
    """2D rotation + 5% expansion per step. lambda_max ~ +0.05."""
    rng = np.random.RandomState(seed)
    theta = 2 * np.pi / 20  # ~18 degrees per step
    expansion = 1.05
    R = expansion * np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta),  np.cos(theta)]])
    z0 = rng.randn(n_traj, 2)
    trajs = np.zeros((n_traj, T, 2))
    trajs[:, 0, :] = z0
    for t in range(1, T):
        trajs[:, t, :] = trajs[:, t-1, :] @ R.T
    lyap = np.log(expansion)
    return trajs, R, lyap


def _lorenz_deriv(state, t, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def make_lorenz(n_traj=50000, T=100, dt=0.01, seed=0):
    """Lorenz system, lambda_max ~ 0.906."""
    rng = np.random.RandomState(seed)
    # Sample initial conditions near the attractor
    z0 = rng.randn(n_traj, 3) * 5 + np.array([0, 0, 25])
    t_span = np.arange(T + 1) * dt
    trajs = np.zeros((n_traj, T, 3))
    for i in range(n_traj):
        sol = odeint(_lorenz_deriv, z0[i], t_span)
        trajs[i] = sol[1:]  # drop t=0 burn-in index
    lyap = 0.906
    return trajs, None, lyap


def split_trajectories(trajs, train_frac=0.8, val_frac=0.1):
    N = len(trajs)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    return trajs[:n_train], trajs[n_train:n_train+n_val], trajs[n_train+n_val:]


ENVIRONMENTS = {
    'stable_linear': make_stable_linear,
    'spiral_2d':     make_spiral_2d,
    'lorenz':        make_lorenz,
}
