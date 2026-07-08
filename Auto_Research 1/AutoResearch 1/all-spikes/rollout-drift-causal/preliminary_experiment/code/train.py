"""
Training loop for one condition (one seed, one config).
"""

import torch
import torch.optim as optim
import numpy as np
import json, os, time
from models import make_encoder, make_dynamics, sigreg_loss


def train_one(config, train_trajs, val_trajs, device, out_path):
    """
    config keys:
      env_name, latent_dim, capacity (small|large), use_sigreg, protocol (1step|5step),
      seed, n_steps=20000, batch_size=8192, lr=3e-4, wd=1e-4, lam_reg=1.0, oracle=False
    """
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    state_dim = train_trajs.shape[2]
    latent_dim = config.get('latent_dim', 16)

    # Build tensors
    # train_trajs: (N, T, D)
    train_t = torch.tensor(train_trajs, dtype=torch.float32, device=device)
    val_t   = torch.tensor(val_trajs,   dtype=torch.float32, device=device)

    if config.get('oracle', False):
        # Oracle: dynamics trained directly on ground-truth states
        enc = None
        dyn = make_dynamics(state_dim, config['capacity']).to(device)
        def encode(x): return x
    else:
        enc = make_encoder(state_dim, latent_dim).to(device)
        dyn = make_dynamics(latent_dim, config['capacity']).to(device)
        def encode(x): return enc(x)

    params = list(dyn.parameters())
    if enc is not None:
        params += list(enc.parameters())
    opt = optim.AdamW(params, lr=config.get('lr', 3e-4), weight_decay=config.get('wd', 1e-4))

    N, T, D = train_t.shape
    protocol = config.get('protocol', '1step')
    use_sigreg = config.get('use_sigreg', False)
    lam_reg = config.get('lam_reg', 1.0)
    n_steps = config.get('n_steps', 20000)
    bs = config.get('batch_size', 8192)

    K = 5 if protocol == '5step' else 1

    train_losses, val_losses = [], []
    t0 = time.time()

    for step in range(n_steps):
        # Sample random (trajectory, timestep) pairs
        idx_n = torch.randint(0, N, (bs,), device=device)
        idx_t = torch.randint(0, T - K, (bs,), device=device)

        loss = torch.tensor(0.0, device=device)

        if protocol == '1step':
            x_t  = train_t[idx_n, idx_t]
            x_t1 = train_t[idx_n, idx_t + 1]
            z_t  = encode(x_t)
            z_t1_pred = dyn(z_t)
            z_t1_tgt  = encode(x_t1).detach()
            loss = loss + ((z_t1_pred - z_t1_tgt) ** 2).mean()
            if use_sigreg and enc is not None:
                loss = loss + lam_reg * sigreg_loss(z_t)

        else:  # 5step BPTT
            x_t = train_t[idx_n, idx_t]
            z = encode(x_t)
            for k in range(1, K + 1):
                x_tk = train_t[idx_n, idx_t + k]
                z_pred = dyn(z)
                z_tgt  = encode(x_tk).detach()
                loss = loss + ((z_pred - z_tgt) ** 2).mean()
                z = z_pred
            if use_sigreg and enc is not None:
                # Regularise the initial latent
                z0 = encode(train_t[idx_n, idx_t])
                loss = loss + lam_reg * sigreg_loss(z0)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 2000 == 0 or step == n_steps - 1:
            train_losses.append(loss.item())

    # Validation MSE (1-step)
    with torch.no_grad():
        Nv, Tv, Dv = val_t.shape
        all_mse = []
        chunk = 512
        for start in range(0, Nv, chunk):
            vt = val_t[start:start+chunk]
            x_v  = vt[:, :-1].reshape(-1, Dv)
            x_v1 = vt[:, 1:].reshape(-1, Dv)
            zv  = encode(x_v)
            zvp = dyn(zv)
            zvt = encode(x_v1)
            all_mse.append(((zvp - zvt) ** 2).mean().item())
        val_mse = float(np.mean(all_mse))

    # Marginal KL from N(0,I) — compute on a batch of train latents
    kl_val = None
    if enc is not None:
        with torch.no_grad():
            sample = train_t[:2000, 0, :]
            z_sample = encode(sample)
            mu = z_sample.mean(0)
            var = z_sample.var(0)
            # KL(q || N(0,I)) = 0.5 * sum(var + mu^2 - 1 - log(var))
            kl_val = 0.5 * (var + mu**2 - 1 - torch.log(var + 1e-8)).sum().item()

    # Effective latent dim via PCA
    eff_dim = None
    if enc is not None:
        with torch.no_grad():
            sample = train_t[:2000, 0, :]
            z_sample = encode(sample).cpu().numpy()
            cov = np.cov(z_sample.T)
            evals = np.linalg.eigvalsh(cov)
            evals = evals[evals > 0]
            total = evals.sum()
            cumsum = np.cumsum(sorted(evals, reverse=True))
            eff_dim = int(np.searchsorted(cumsum, 0.95 * total)) + 1

    result = {
        'config': config,
        'final_train_loss': train_losses[-1],
        'final_val_mse': val_mse,
        'marginal_kl': kl_val,
        'eff_latent_dim': eff_dim,
        'wall_time_s': time.time() - t0,
    }

    # Save checkpoint
    ckpt = {'dyn': dyn.state_dict()}
    if enc is not None:
        ckpt['enc'] = enc.state_dict()
    torch.save(ckpt, out_path.replace('.json', '.pt'))

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result, enc, dyn, encode
