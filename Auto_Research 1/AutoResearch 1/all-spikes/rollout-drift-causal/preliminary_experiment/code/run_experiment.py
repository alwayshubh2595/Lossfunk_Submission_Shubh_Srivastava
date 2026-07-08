"""
Master experiment runner. Builds all 300 conditions and dispatches them.
Saves per-run JSON to results_raw/.
"""

import os, sys, json, time, itertools, traceback
import numpy as np
import torch

# Add code dir to path
sys.path.insert(0, os.path.dirname(__file__))

from environments import make_stable_linear, make_spiral_2d, make_lorenz, split_trajectories
from models import make_encoder, make_dynamics, sigreg_loss
from train import train_one
from metrics import compute_all_ak, ak_groundtruth_linear, ak_groundtruth_perturbation, KS

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(BASE, 'results_raw')
os.makedirs(RAW_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

SEEDS = list(range(5))  # reduced from 10 per user approval 2026-06-30
ENVS  = ['stable_linear', 'spiral_2d', 'lorenz']
ENV_FNS = {
    'stable_linear': make_stable_linear,
    'spiral_2d':     make_spiral_2d,
    'lorenz':        make_lorenz,
}

# ── Pre-generate all trajectories once ───────────────────────────────────────

print("Generating trajectories...")
env_data = {}
env_gt   = {}  # ground-truth transition matrix (linear) or None
env_lyap = {}

for env_name in ENVS:
    trajs, gt, lyap = ENV_FNS[env_name](n_traj=50000, T=100)
    tr, va, te = split_trajectories(trajs, 0.8, 0.1)
    env_data[env_name] = (tr, va, te)
    env_gt[env_name]   = gt
    env_lyap[env_name] = lyap
    print(f"  {env_name}: state_dim={trajs.shape[2]}, lyap={lyap:.3f}")


def run_key(env, reg, cap, proto, seed, latent_dim=16, oracle=False, side_sweep=False):
    tag = f"{'oracle' if oracle else 'main'}__{env}__reg{int(reg)}__cap{cap}__proto{proto}__dz{latent_dim}__seed{seed}"
    return tag


def run_condition(env_name, use_sigreg, capacity, protocol, seed,
                  latent_dim=16, oracle=False):
    tag = run_key(env_name, use_sigreg, capacity, protocol, seed, latent_dim, oracle)
    out_path = os.path.join(RAW_DIR, tag + '.json')

    if os.path.exists(out_path):
        print(f"  SKIP (exists): {tag}")
        return json.load(open(out_path))

    tr, va, te = env_data[env_name]
    state_dim = tr.shape[2]

    config = {
        'env_name':    env_name,
        'latent_dim':  state_dim if oracle else latent_dim,
        'capacity':    capacity,
        'use_sigreg':  use_sigreg,
        'protocol':    protocol,
        'seed':        seed,
        'oracle':      oracle,
        'n_steps':     20000,
        'batch_size':  8192,
        'lr':          3e-4,
        'wd':          1e-4,
        'lam_reg':     1.0,
    }

    print(f"  RUN: {tag}")
    t0 = time.time()

    result, enc, dyn, encode_fn = train_one(config, tr, va, DEVICE, out_path)

    # Compute A_k metrics on test latents
    te_t = torch.tensor(te[:, 0, :], dtype=torch.float32, device=DEVICE)
    if enc is not None:
        enc.eval()
        with torch.no_grad():
            z_test = enc(te_t)
    else:
        z_test = te_t

    ak_metrics = compute_all_ak(dyn, z_test, device=DEVICE)

    # Ground-truth A_k for normalization
    gt_ak = None
    if env_gt[env_name] is not None:
        gt_ak = ak_groundtruth_linear(env_gt[env_name])
    elif env_name == 'lorenz':
        from scipy.integrate import odeint
        def lorenz_fn(z0_np, k, dt=0.01):
            from environments import _lorenz_deriv
            t_span = np.arange(k + 1) * dt
            out = []
            for z in z0_np:
                sol = odeint(_lorenz_deriv, z, t_span)
                out.append(sol[-1])
            return np.array(out)
        gt_ak = ak_groundtruth_perturbation(lorenz_fn)

    result['ak_metrics'] = ak_metrics
    result['gt_ak'] = gt_ak
    result['total_time_s'] = time.time() - t0

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"    done in {result['total_time_s']:.1f}s | val_mse={result['final_val_mse']:.4f}")
    return result


# ── Main factorial: 8 conditions × 3 envs × 10 seeds = 240 runs ──────────────

print("\n=== MAIN FACTORIAL ===")
n_done = 0
for env, reg, cap, proto, seed in itertools.product(
        ENVS, [False, True], ['small', 'large'], ['1step', '5step'], SEEDS):
    try:
        run_condition(env, reg, cap, proto, seed)
        n_done += 1
    except Exception as e:
        print(f"  ERROR in {env} reg={reg} cap={cap} proto={proto} seed={seed}: {e}")
        traceback.print_exc()

print(f"\nMain factorial done: {n_done} runs")

# ── Oracle baseline: 1 condition × 3 envs × 10 seeds = 30 runs ───────────────

print("\n=== ORACLE BASELINE ===")
for env, seed in itertools.product(ENVS, SEEDS):
    try:
        run_condition(env, False, 'large', '5step', seed, oracle=True)
    except Exception as e:
        print(f"  ERROR oracle {env} seed={seed}: {e}")
        traceback.print_exc()

# ── Latent-width side sweep: 3 widths × 10 seeds on spiral_2d ────────────────

print("\n=== LATENT WIDTH SIDE SWEEP ===")
for dz, seed in itertools.product([2, 8, 32], SEEDS):
    tag = run_key('spiral_2d', True, 'small', '1step', seed, dz)
    out_path = os.path.join(RAW_DIR, tag + '.json')
    if os.path.exists(out_path):
        print(f"  SKIP (exists): {tag}")
        continue
    try:
        run_condition('spiral_2d', True, 'small', '1step', seed, latent_dim=dz)
    except Exception as e:
        print(f"  ERROR side_sweep dz={dz} seed={seed}: {e}")
        traceback.print_exc()

print("\n=== ALL RUNS COMPLETE ===")
