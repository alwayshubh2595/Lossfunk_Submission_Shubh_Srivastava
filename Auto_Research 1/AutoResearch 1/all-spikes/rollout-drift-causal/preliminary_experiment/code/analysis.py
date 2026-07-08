"""
Post-run analysis: load all per-run JSONs, build tables, run ANOVA, generate plots.
Outputs go to tables/ and figures/.
"""

import os, sys, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

BASE    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(BASE, 'results_raw')
FIG_DIR = os.path.join(BASE, 'figures')
TAB_DIR = os.path.join(BASE, 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

ENVS  = ['stable_linear', 'spiral_2d', 'lorenz']
KS    = [1, 5, 10, 25, 50]
METHODS = ['jacobian', 'perturbation', 'traj_div']


# ── Load all results ──────────────────────────────────────────────────────────

def load_all():
    rows = []
    for path in glob.glob(os.path.join(RAW_DIR, '*.json')):
        try:
            d = json.load(open(path))
        except:
            continue
        cfg = d.get('config', {})
        row = {
            'env':        cfg.get('env_name', ''),
            'use_sigreg': cfg.get('use_sigreg', False),
            'capacity':   cfg.get('capacity', ''),
            'protocol':   cfg.get('protocol', ''),
            'seed':       cfg.get('seed', -1),
            'latent_dim': cfg.get('latent_dim', 16),
            'oracle':     cfg.get('oracle', False),
            'val_mse':    d.get('final_val_mse', np.nan),
            'train_loss': d.get('final_train_loss', np.nan),
            'marginal_kl': d.get('marginal_kl', np.nan),
            'eff_latent_dim': d.get('eff_latent_dim', np.nan),
        }
        ak = d.get('ak_metrics', {})
        gt_ak = d.get('gt_ak', {})
        for method in METHODS:
            for k in KS:
                # JSON keys are strings; try both str and int
                v = ak.get(method, {}).get(str(k), ak.get(method, {}).get(k, np.nan))
                if v is None:
                    v = np.nan
                row[f'ak_{method}_k{k}'] = v
                gt_v = gt_ak.get(str(k), gt_ak.get(k, None)) if gt_ak else None
                if gt_v and gt_v > 0 and not np.isnan(v):
                    row[f'ak_{method}_k{k}_norm'] = v / gt_v
                else:
                    row[f'ak_{method}_k{k}_norm'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def condition_label(row):
    reg = 'SIGReg' if row['use_sigreg'] else 'NoReg'
    cap = 'Large' if row['capacity'] == 'large' else 'Small'
    proto = '5step' if row['protocol'] == '5step' else '1step'
    return f"{reg}_{cap}_{proto}"


# ── Table 1: Master results ───────────────────────────────────────────────────

def make_master_table(df):
    main = df[~df['oracle'] & (df['latent_dim'] == 16)].copy()
    main['condition'] = main.apply(condition_label, axis=1)

    rows = []
    for env in ENVS:
        for cond in sorted(main['condition'].unique()):
            sub = main[(main['env'] == env) & (main['condition'] == cond)]
            if len(sub) == 0:
                continue
            row = {'environment': env, 'condition': cond, 'n_seeds': len(sub)}
            for method in METHODS:
                for k in KS:
                    col = f'ak_{method}_k{k}'
                    vals = sub[col].dropna()
                    if len(vals) > 0:
                        m, s = vals.mean(), vals.std()
                        ci = 1.96 * s / np.sqrt(len(vals))
                        row[f'{method}_k{k}_mean'] = round(m, 3)
                        row[f'{method}_k{k}_ci95'] = round(ci, 3)
            rows.append(row)

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB_DIR, 'tab01_master_results.csv'), index=False)
    t.to_markdown(os.path.join(TAB_DIR, 'tab01_master_results.md'), index=False)
    print("tab01 done")
    return t


# ── Table 2: Normalized A_k ───────────────────────────────────────────────────

def make_normalized_table(df):
    main = df[~df['oracle'] & (df['latent_dim'] == 16)].copy()
    main['condition'] = main.apply(condition_label, axis=1)

    rows = []
    for env in ENVS:
        for cond in sorted(main['condition'].unique()):
            sub = main[(main['env'] == env) & (main['condition'] == cond)]
            if len(sub) == 0:
                continue
            row = {'environment': env, 'condition': cond}
            for method in METHODS:
                for k in KS:
                    col = f'ak_{method}_k{k}_norm'
                    vals = sub[col].dropna()
                    if len(vals) > 0:
                        row[f'{method}_k{k}_norm_mean'] = round(vals.mean(), 3)
                        row[f'{method}_k{k}_norm_ci95'] = round(1.96 * vals.std() / np.sqrt(len(vals)), 3)
            rows.append(row)

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB_DIR, 'tab02_normalized_ak.csv'), index=False)
    t.to_markdown(os.path.join(TAB_DIR, 'tab02_normalized_ak.md'), index=False)
    print("tab02 done")
    return t


# ── Table 3: Variance decomposition (ANOVA) ───────────────────────────────────

def make_variance_table(df):
    from itertools import combinations
    main = df[~df['oracle'] & (df['latent_dim'] == 16)].copy()
    main['F1'] = main['use_sigreg'].astype(int)
    main['F2'] = (main['capacity'] == 'large').astype(int)
    main['F3'] = (main['protocol'] == '5step').astype(int)

    rows = []
    for env in ENVS:
        sub = main[main['env'] == env].copy()
        y_col = 'ak_perturbation_k50'
        sub = sub.dropna(subset=[y_col])
        if len(sub) < 8:
            print(f"  WARNING: not enough data for ANOVA on {env}")
            continue
        sub['log_y'] = np.log(sub[y_col].clip(1e-8))

        # Simple variance decomposition via SS
        grand_mean = sub['log_y'].mean()
        SS_total = ((sub['log_y'] - grand_mean) ** 2).sum()

        effects = {}
        for factor in ['F1', 'F2', 'F3']:
            group_means = sub.groupby(factor)['log_y'].mean()
            group_counts = sub.groupby(factor)['log_y'].count()
            SS = sum(group_counts[g] * (group_means[g] - grand_mean) ** 2 for g in group_means.index)
            effects[factor] = SS

        for f1, f2 in combinations(['F1', 'F2', 'F3'], 2):
            group_means = sub.groupby([f1, f2])['log_y'].mean()
            group_counts = sub.groupby([f1, f2])['log_y'].count()
            SS_int = sum(group_counts[g] * (group_means[g] - grand_mean) ** 2 for g in group_means.index)
            SS_int -= effects[f1] + effects[f2]
            effects[f'{f1}x{f2}'] = max(SS_int, 0)

        SS_explained = sum(effects.values())
        SS_resid = max(SS_total - SS_explained, 0)

        row = {'environment': env, 'SS_total': round(SS_total, 4)}
        for k, v in effects.items():
            eta2 = v / SS_total if SS_total > 0 else 0
            row[f'SS_{k}'] = round(v, 4)
            row[f'eta2_{k}'] = round(eta2, 4)
        row['SS_residual'] = round(SS_resid, 4)
        row['eta2_residual'] = round(SS_resid / SS_total, 4) if SS_total > 0 else 0
        rows.append(row)

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB_DIR, 'tab03_variance_decomp.csv'), index=False)
    t.to_markdown(os.path.join(TAB_DIR, 'tab03_variance_decomp.md'), index=False)
    print("tab03 done")
    return t


# ── Table 4: Diagnostics ──────────────────────────────────────────────────────

def make_diagnostics_table(df):
    main = df[~df['oracle'] & (df['latent_dim'] == 16)].copy()
    main['condition'] = main.apply(condition_label, axis=1)

    rows = []
    for env in ENVS:
        for cond in sorted(main['condition'].unique()):
            sub = main[(main['env'] == env) & (main['condition'] == cond)]
            if len(sub) == 0:
                continue
            row = {
                'environment': env,
                'condition': cond,
                'val_mse_mean': round(sub['val_mse'].mean(), 5),
                'val_mse_std':  round(sub['val_mse'].std(), 5),
                'train_loss_mean': round(sub['train_loss'].mean(), 5),
                'marginal_kl_mean': round(sub['marginal_kl'].mean(), 3),
                'eff_latent_dim_mean': round(sub['eff_latent_dim'].mean(), 1),
            }
            rows.append(row)

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB_DIR, 'tab04_diagnostics.csv'), index=False)
    t.to_markdown(os.path.join(TAB_DIR, 'tab04_diagnostics.md'), index=False)
    print("tab04 done")
    return t


# ── Figure 1: A_k(k) curves ───────────────────────────────────────────────────

def plot_ak_curves(df):
    main = df[~df['oracle'] & (df['latent_dim'] == 16)].copy()
    main['condition'] = main.apply(condition_label, axis=1)
    conditions = sorted(main['condition'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, env in zip(axes, ENVS):
        for cond, color in zip(conditions, colors):
            sub = main[(main['env'] == env) & (main['condition'] == cond)]
            means, cis = [], []
            for k in KS:
                vals = sub[f'ak_perturbation_k{k}'].dropna()
                if len(vals) == 0:
                    means.append(np.nan); cis.append(np.nan)
                    continue
                means.append(vals.mean())
                cis.append(1.96 * vals.std() / np.sqrt(len(vals)))
            means, cis = np.array(means), np.array(cis)
            means_pos = np.where(np.isfinite(means) & (means > 0), means, np.nan)
            ax.plot(KS, means_pos, label=cond, color=color, marker='o')
            lower = np.where(np.isfinite(means_pos), np.maximum(means_pos - cis, 1e-8), np.nan)
            upper = np.where(np.isfinite(means_pos), means_pos + cis, np.nan)
            ax.fill_between(KS, lower, upper, alpha=0.15, color=color)
        ax.set_yscale('log')
        ax.set_xlabel('k (rollout steps)')
        ax.set_ylabel('A_k (perturbation method)')
        ax.set_title(env.replace('_', ' ').title())
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig01_ak_curves.png'), dpi=150)
    plt.close(fig)
    print("fig01 done")


# ── Figure 2: Variance decomposition ─────────────────────────────────────────

def plot_variance_decomp(vt):
    if vt is None or len(vt) == 0:
        print("fig02 skipped (no variance data)")
        return
    factors = ['F1', 'F2', 'F3', 'F1xF2', 'F1xF3', 'F2xF3', 'residual']
    labels  = ['F1 (SIGReg)', 'F2 (capacity)', 'F3 (protocol)',
               'F1×F2', 'F1×F3', 'F2×F3', 'Residual']
    fig, axes = plt.subplots(1, len(vt), figsize=(5 * len(vt), 4))
    if len(vt) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, vt.iterrows()):
        vals = [row.get(f'eta2_{f}', 0) for f in factors]
        bars = ax.bar(labels, vals, color=plt.cm.Set2(np.linspace(0, 1, len(labels))))
        ax.set_ylim(0, 1)
        ax.set_ylabel('η² (fraction of variance)')
        ax.set_title(row['environment'].replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=30)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig02_variance_decomp.png'), dpi=150)
    plt.close(fig)
    print("fig02 done")


# ── Figure 3: Sanity check (stable-linear) ────────────────────────────────────

def plot_sanity_check(df):
    sub = df[(df['env'] == 'stable_linear') & ~df['oracle'] & (df['latent_dim'] == 16)].copy()
    sub['condition'] = sub.apply(condition_label, axis=1)
    conditions = sorted(sub['condition'].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    for cond, color in zip(conditions, colors):
        csub = sub[sub['condition'] == cond]
        means = [csub[f'ak_perturbation_k{k}'].mean() for k in KS]
        ax.plot(KS, means, label=cond, color=color, marker='o')
    ax.axhline(1.0, color='black', linestyle='--', label='A_k=1 (sanity threshold)')
    ax.set_xlabel('k')
    ax.set_ylabel('A_k (perturbation method)')
    ax.set_title('Sanity Check: Stable-Linear Environment\n(all A_k should be < 1)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig03_sanity_check.png'), dpi=150)
    plt.close(fig)
    print("fig03 done")


# ── Figure 4: Latent-width side sweep ─────────────────────────────────────────

def plot_latent_sweep(df):
    sub = df[(df['env'] == 'spiral_2d') & df['use_sigreg'] &
             (df['capacity'] == 'small') & (df['protocol'] == '1step') &
             ~df['oracle']].copy()
    dz_vals = sorted(sub['latent_dim'].unique())
    fig, ax = plt.subplots(figsize=(6, 4))
    for dz in dz_vals:
        vals = sub[sub['latent_dim'] == dz]['ak_perturbation_k50'].dropna()
        if len(vals) == 0:
            continue
        m, ci = vals.mean(), 1.96 * vals.std() / np.sqrt(len(vals))
        ax.bar(str(dz), m, yerr=ci, capsize=5, label=f'd_z={dz}')
    ax.set_xlabel('Latent dimension d_z')
    ax.set_ylabel('A_50 (perturbation, mean ± 95% CI)')
    ax.set_title('Latent Width Side Sweep (2D Spiral, SIGReg+Small+1step)')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig04_latent_sweep.png'), dpi=150)
    plt.close(fig)
    print("fig04 done")


# ── Figure 5: Method agreement ────────────────────────────────────────────────

def plot_method_agreement(df):
    main = df[~df['oracle'] & (df['latent_dim'] == 16)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, env in zip(axes, ENVS):
        sub = main[main['env'] == env]
        jac  = sub['ak_jacobian_k50'].dropna()
        pert = sub['ak_perturbation_k50'].dropna()
        traj = sub['ak_traj_div_k50'].dropna()
        common = sub.dropna(subset=['ak_jacobian_k50', 'ak_perturbation_k50', 'ak_traj_div_k50'])
        ax.scatter(common['ak_jacobian_k50'], common['ak_perturbation_k50'],
                   alpha=0.5, s=10, label='Jac vs Pert')
        ax.scatter(common['ak_perturbation_k50'], common['ak_traj_div_k50'],
                   alpha=0.5, s=10, label='Pert vs Traj', marker='x')
        lim = max(common[['ak_jacobian_k50','ak_perturbation_k50','ak_traj_div_k50']].max().max(), 1)
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.4)
        ax.set_xlabel('A_50 method 1/2')
        ax.set_ylabel('A_50 method 2/3')
        ax.set_title(f'Method Agreement: {env}')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig05_method_agreement.png'), dpi=150)
    plt.close(fig)
    print("fig05 done")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading results...")
    df = load_all()
    print(f"  Loaded {len(df)} runs")

    print("\nBuilding tables...")
    mt = make_master_table(df)
    nt = make_normalized_table(df)
    vt = make_variance_table(df)
    dt = make_diagnostics_table(df)

    print("\nGenerating figures...")
    plot_ak_curves(df)
    plot_variance_decomp(vt)
    plot_sanity_check(df)
    plot_latent_sweep(df)
    plot_method_agreement(df)

    # Sanity check: flag if any stable-linear A_k50 > 1
    sl = df[(df['env'] == 'stable_linear') & ~df['oracle']]
    violations = sl[sl['ak_perturbation_k50'] > 1.0]
    if len(violations) > 0:
        print(f"\n⚠️  SANITY CHECK FAILURE: {len(violations)} stable-linear runs have A_k50 > 1")
        print(violations[['condition' if 'condition' in violations else 'capacity', 'seed', 'ak_perturbation_k50']])
    else:
        print("\n✓ Sanity check passed: all stable-linear A_k50 ≤ 1")

    print("\nAnalysis complete.")
