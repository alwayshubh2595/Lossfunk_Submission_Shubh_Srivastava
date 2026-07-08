"""
Three A_k measurement methods from the brief.
Design change (approved by user 2026-06-30):
  - Jacobian n_samples reduced 100→20; only computed at k≤10 (large-k products are
    numerically near-singular and covered by perturbation/traj_div methods).
  - Perturbation and traj_div unchanged at n_samples=100.
"""

import torch
import numpy as np

KS     = [1, 5, 10, 25, 50]
KS_JAC = [1, 5, 10]   # Jacobian only at short horizons


def rollout_k(dyn, z0, k):
    z = z0
    for _ in range(k):
        z = dyn(z)
    return z


# ── Method 1: Accumulated Jacobian operator norm ──────────────────────────────

def ak_jacobian(dyn, z_test, ks=KS_JAC, n_samples=20, device='cuda'):
    """
    A_k^Jac = E_{z0} [ ||prod_{i=0}^{k-1} df/dz|_{z_i}||_op ]
    n_samples=20, only at k<=10 to keep runtime tractable.
    NaN reported for k=25,50 (not computed).
    """
    dyn.eval()
    idx = torch.randperm(len(z_test))[:n_samples]
    z0_batch = z_test[idx].to(device)
    results = {k: float('nan') for k in KS}

    for k in ks:
        norms = []
        for i in range(n_samples):
            zc = z0_batch[i:i+1].detach()
            d  = zc.shape[1]
            J_prod = torch.eye(d, device=device)
            for _ in range(k):
                zc_req = zc.requires_grad_(True)
                zn = dyn(zc_req)
                # Compute Jacobian row by row
                rows = []
                for j in range(d):
                    grad = torch.autograd.grad(zn[0, j], zc_req,
                                               retain_graph=(j < d - 1))[0]
                    rows.append(grad[0])
                J = torch.stack(rows)         # (d, d)
                J_prod = J @ J_prod
                zc = zn.detach()
            sv = torch.linalg.svdvals(J_prod)
            norms.append(sv[0].item())
        results[k] = float(np.mean(norms))
    return results


# ── Method 2: Finite-difference perturbation ──────────────────────────────────

def ak_perturbation(dyn, z_test, ks=KS, eps=1e-3, n_samples=100, device='cuda'):
    dyn.eval()
    idx = torch.randperm(len(z_test))[:n_samples]
    z0_batch = z_test[idx].to(device)
    d = z0_batch.shape[1]
    results = {}
    with torch.no_grad():
        for k in ks:
            norms = []
            for i in range(n_samples):
                z0 = z0_batch[i:i+1]
                u  = torch.randn(1, d, device=device)
                u  = u / u.norm()
                zk      = rollout_k(dyn, z0, k)
                zk_pert = rollout_k(dyn, z0 + eps * u, k)
                norms.append(((zk_pert - zk).norm() / eps).item())
            results[k] = float(np.mean(norms))
    return results


# ── Method 3: Paired-trajectory divergence ────────────────────────────────────

def ak_trajectory_divergence(dyn, z_test, ks=KS, eps=1e-3, n_samples=100, device='cuda'):
    dyn.eval()
    idx = torch.randperm(len(z_test))[:n_samples]
    z0_batch = z_test[idx].to(device)
    d = z0_batch.shape[1]
    results = {}
    with torch.no_grad():
        for k in ks:
            ratios = []
            for i in range(n_samples):
                z0 = z0_batch[i:i+1]
                d1 = torch.randn(1, d, device=device) * eps
                d2 = torch.randn(1, d, device=device) * eps
                sep = (d1 - d2).norm().item()
                if sep < 1e-10:
                    continue
                zk1 = rollout_k(dyn, z0 + d1, k)
                zk2 = rollout_k(dyn, z0 + d2, k)
                ratios.append(((zk1 - zk2).norm() / sep).item())
            results[k] = float(np.mean(ratios))
    return results


# ── Ground-truth A_k ──────────────────────────────────────────────────────────

def ak_groundtruth_linear(A, ks=KS):
    results = {}
    for k in ks:
        Ak_k = np.linalg.matrix_power(A, k)
        sv = np.linalg.svd(Ak_k, compute_uv=False)
        results[k] = float(sv[0])
    return results


def ak_groundtruth_perturbation(traj_fn, ks=KS, eps=1e-3, n_samples=100, seed=42):
    rng = np.random.RandomState(seed)
    results = {}
    for k in ks:
        norms = []
        for _ in range(n_samples):
            z0 = rng.randn(1, 3) * 5 + np.array([[0, 0, 25]])
            u  = rng.randn(1, 3)
            u /= np.linalg.norm(u)
            zk      = traj_fn(z0, k)
            zk_pert = traj_fn(z0 + eps * u, k)
            norms.append(np.linalg.norm(zk_pert - zk) / eps)
        results[k] = float(np.mean(norms))
    return results


def compute_all_ak(dyn, z_test, device='cuda', ks=KS):
    return {
        'jacobian':     ak_jacobian(dyn, z_test, ks=KS_JAC, device=device),
        'perturbation': ak_perturbation(dyn, z_test, ks=ks, device=device),
        'traj_div':     ak_trajectory_divergence(dyn, z_test, ks=ks, device=device),
    }
