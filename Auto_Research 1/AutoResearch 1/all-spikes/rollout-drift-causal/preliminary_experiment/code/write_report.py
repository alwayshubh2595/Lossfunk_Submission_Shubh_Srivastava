"""
Generates REPORT.md from computed tables and figures.
Run after analysis.py has completed.
"""
import os, json, glob
import pandas as pd

BASE    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TAB_DIR = os.path.join(BASE, 'tables')
FIG_DIR = os.path.join(BASE, 'figures')
RAW_DIR = os.path.join(BASE, 'results_raw')
OUT     = os.path.join(BASE, 'REPORT.md')

def read_md(path):
    if os.path.exists(path):
        return open(path).read()
    return f"*(table not found: {os.path.basename(path)})*"

def count_runs():
    return len(glob.glob(os.path.join(RAW_DIR, '*.json')))

def sanity_warning(tab01_path):
    """Check if any stable-linear A_k50 > 1 and return warning string if so."""
    if not os.path.exists(tab01_path):
        return ""
    df = pd.read_csv(tab01_path)
    sl = df[df['environment'] == 'stable_linear']
    col = 'perturbation_k50_mean'
    if col not in sl.columns:
        return ""
    bad = sl[sl[col] > 1.0]
    if len(bad) == 0:
        return ""
    lines = ["**WARNING: SANITY CHECK FAILURE**\n"]
    lines.append(f"The following {len(bad)} condition(s) on the stable-linear environment have A_k50 > 1, ")
    lines.append("which should not occur if the implementation is correct. This is a critical finding.\n\n")
    for _, row in bad.iterrows():
        lines.append(f"- Condition: {row.get('condition','?')} | A_50 = {row[col]:.3f}\n")
    lines.append("\nSee Section 5 for discussion.\n\n")
    return "".join(lines)

tab01 = os.path.join(TAB_DIR, 'tab01_master_results.md')
tab02 = os.path.join(TAB_DIR, 'tab02_normalized_ak.md')
tab03 = os.path.join(TAB_DIR, 'tab03_variance_decomp.md')
tab04 = os.path.join(TAB_DIR, 'tab04_diagnostics.md')

report = f"""# Preliminary Experiment: Causal Attribution of Rollout Drift

## 1. Question

This experiment tests whether rollout amplification (`A_k`) in latent world models is primarily caused by the marginal encoder regularizer (SIGReg), by the dynamics model capacity, or by the training-time rollout protocol. A 2×2×2 factorial design (F1: SIGReg on/off; F2: dynamics capacity small/large; F3: 1-step vs. 5-step BPTT) was run across three synthetic environments (stable-linear, 2D spiral, Lorenz) with 10 seeds each, plus an oracle baseline and a latent-width side sweep, totaling 300 runs. The central question: of the total amplification `A_k` observed, what fraction of variance is attributable to each factor? See `brief.md` for full design detail.

## 2. Design Summary

**Factorial:** 2 (SIGReg) × 2 (dynamics capacity) × 2 (training protocol) = 8 conditions.
**Environments:** stable-linear (sanity check, A_k should shrink), 2D spiral (replicates prior toy result), Lorenz (chaotic stress test).
**Metric:** A_k at k ∈ {{1,5,10,25,50}} via three methods: Jacobian operator norm, finite-difference perturbation (ε=1e-3), paired-trajectory divergence.
**Seeds:** 10 per condition. **Total runs:** 300 ({count_runs()} completed).
**Note on SIGReg implementation:** moment-matching proxy used (`L = ||μ||² + ||Σ-I||²_F`), not the original characteristic-function SIGReg from LeJEPA. See Section 5.
See `brief.md` for full detail.

## 3. Results

### 3.1 Master Results Table

Rows: condition × environment. Columns: A_k mean ± 95% CI across 10 seeds, for each measurement method and rollout length.

{read_md(tab01)}

### 3.2 Variance Decomposition

Fraction of variance in `log(A_50^pert)` explained by each factor and interaction, per environment (η²).

{read_md(tab03)}

![Variance decomposition](figures/fig02_variance_decomp.png)

### 3.3 Sanity Checks

{sanity_warning(os.path.join(TAB_DIR, 'tab01_master_results.csv'))}All conditions on the stable-linear environment should produce `A_k < 1` at all k, since eigenvalues are {{0.7, 0.8, 0.9}}.

![Sanity check: stable-linear A_k curves](figures/fig03_sanity_check.png)

### 3.4 Method Agreement

Do the three A_k measurement methods agree? Comparison of Jacobian, perturbation, and trajectory-divergence methods at k=50, per environment.

![Method agreement scatter](figures/fig05_method_agreement.png)

### 3.5 Latent Width Sweep

Effect of latent dimension `d_z ∈ {{2, 8, 32}}` on A_50, for the SIGReg + small-dynamics + 1-step condition on the 2D spiral.

![Latent width sweep](figures/fig04_latent_sweep.png)

### 3.6 A_k Curves

`A_k(k)` curves per condition, log-scale y-axis, shaded 95% CI.

![A_k curves](figures/fig01_ak_curves.png)

## 4. Diagnostics

Final train loss, validation MSE, marginal KL from N(0,I), and effective latent dimension (95% PCA variance) per condition.

{read_md(tab04)}

## 5. Caveats and Confounds

The following limitations apply to all results (from brief.md §11):

1. **Synthetic-only environments.** Results may not transfer to JEPA-family models trained on images.
2. **No image encoder.** SIGReg's interaction with image encoder pathologies (collapse-to-prior) is not tested.
3. **No action conditioning.** Autonomous dynamics may behave differently from action-conditioned dynamics.
4. **Limited latent dimensions.** Behavior at d_z = 256 or 1024 (realistic for vision models) is not directly tested.
5. **Fixed-compute training.** Findings about F1 may depend on whether SIGReg converges slower than no-reg.
6. **Moment-matching SIGReg.** The proxy `L = ||μ||² + ||Σ-I||²_F` was used instead of the original characteristic-function implementation. The two may behave differently.
7. **A_k measures local sensitivity, not basin-of-attraction structure.** Two models with identical A_k could have very different global trajectory behavior.
8. **Encoder warping changes geometry.** Normalized A_k partially addresses this, but ground-truth Lyapunov exponent of the state space ≠ Lyapunov exponent of the learned latent space.

## 6. Raw Numbers Appendix

All per-run JSONs are in `results_raw/`. Each file contains the full config, all A_k values for all three methods and all five k values, diagnostics, and wall time.

Normalized A_k table (A_k^model / A_k^ground_truth):

{read_md(tab02)}

Per-seed data: see `results_raw/*.json`.
"""

with open(OUT, 'w') as f:
    f.write(report)

print(f"REPORT.md written to {OUT}")
