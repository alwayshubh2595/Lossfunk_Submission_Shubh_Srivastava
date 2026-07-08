# Experiment Design

## Working Title

Does Canonical SIGReg Change the Causal Attribution of Rollout Drift?

## Research Question

In latent world models trained for imagination/planning, is rollout drift primarily explained by marginal regularizer choice, training protocol, or their interaction once SIGReg is implemented in its canonical sliced Epps-Pulley form rather than as a moment-matching proxy?

## Core Claim Under Test

The previous study found that training protocol dominated drift, but that result is not fully interpretable because its SIGReg condition was a covariance/moment proxy. This spike tests whether the conclusion survives the strongest correction: canonical SIGReg.

The experiment is designed so any of these outcomes can win:

- Canonical SIGReg materially reduces drift under 1-step teacher forcing, overturning the previous conclusion.
- Multi-step closed-loop training remains the dominant factor even under canonical SIGReg.
- Canonical SIGReg and multi-step training interact: either helps only in the presence of the other, or one worsens drift under the other.
- Drift is mostly explained by encoder geometry/collapse rather than either headline factor.
- The answer is environment-dependent.

## Single Selected Design

Run a pre-registered main 3 x 2 x 4 factorial plus two small controls:

```text
Regularizer R:
  R0: none
  R1: moment proxy from prior paper
  R2: canonical sliced Epps-Pulley SIGReg

Main training protocol P:
  P0: 1-step teacher forcing
  P1: H-step closed-loop BPTT

Environment E:
  E0: stable-linear state environment
  E1: Lorenz/chaotic state environment
  E2: state-observed action-conditioned point-mass environment
  E3: visual action-conditioned point-mass environment
```

Seeds:

```text
E0/E1/E2 state environments: 5 seeds per condition
E3 visual-action environment: 3 seeds per condition
```

Total runs:

```text
main state: 3 envs x 3 regs x 2 protocols x 5 seeds = 90 runs
main visual: 1 env x 3 regs x 2 protocols x 3 seeds = 18 runs
main total = 108 training runs
```

Control runs:

```text
P0.5 H-step teacher forcing on {E0, E3} x {R1, R2}
E0: 2 regs x 5 seeds = 10 runs
E3: 2 regs x 3 seeds = 6 runs
control total = 16 runs
```

The total planned sweep is therefore 124 runs before mandatory `M=1024` sensitivity cells.

## Why This Is The Right Next Design

The prior result's most serious weakness is not sample size or lack of another synthetic environment. It is that the central regularizer was not the regularizer in the relevant papers. A follow-up that adds more environments but still uses the proxy would not answer the main objection.

The chosen design fixes that by keeping the old moment proxy as a bridge condition while adding canonical SIGReg as a separate condition. The moment-proxy condition answers whether the new implementation reproduces the old paper. The canonical condition answers whether the old conclusion survives the correction.

The design keeps protocol changes architecturally controlled: both training protocols use the same encoder and one-step transition function. This isolates protocol more cleanly than adding Fast-LeWorldModel/prefix prediction to the main factorial, because prefix prediction changes both the training loss and the inference interface/model class.

## Regularizers

### R0: None

No marginal anti-collapse regularizer:

```text
L = L_pred
```

This condition may collapse, especially in the pixel/action environment. Collapse is not treated as low drift. Collapse diagnostics are primary safety checks.

### R1: Moment Proxy

This reproduces the previous paper's proxy:

```text
L_reg = ||mu_batch||_2^2 + ||Sigma_batch - I||_F^2
L = L_pred + lambda L_reg
```

This is not canonical SIGReg. It is included only to bridge the previous result.

### R2: Canonical Sliced Epps-Pulley SIGReg

This is the main corrected regularizer. Given latents `Z in R^{n x d}` and `M` random unit vectors `u_m`, project:

```text
h_m = Z u_m,   h_m in R^n
```

For each projection, estimate the Epps-Pulley/Gaussian-kernel MMD distance to `N(0,1)`:

```text
EP_gamma(h) =
  mean_ij exp(-gamma (h_i - h_j)^2)
  + 1 / sqrt(1 + 4 gamma)
  - 2 mean_i [ exp(-gamma h_i^2 / (1 + 2 gamma)) / sqrt(1 + 2 gamma) ]
```

Then:

```text
SIGReg(Z) = (1 / M) sum_m EP_gamma(Z u_m)
L = L_pred + lambda SIGReg(Z)
```

Default implementation:

```text
M = 256 random projections per batch
gamma = 0.5 unless pilot diagnostics show numerical saturation
```

Lambda protocol:

```text
R1 and R2 get separately tuned lambda values per (regularizer, environment), not per protocol.
Grid: {0.01, 0.03, 0.1, 0.3, 1.0}
Tuning seed: seed 0 only.
Tuning criterion: validation one-step prediction MSE subject to non-collapse diagnostics.
Forbidden criterion: perturbation amplification or any rollout drift metric.
Selected values are frozen across P0, P0.5, and P1 for that (R, E).
```

This prevents lambda tuning from manufacturing an apparent regularizer x protocol interaction.

Important caveat: `M=256` is a Monte Carlo approximation to the sliced Epps-Pulley objective. It is not a moment proxy. A sensitivity check will rerun the highest-leverage canonical SIGReg cells with `M=1024`, matching the LeWorldModel scale more closely.

## Training Protocols

### P0: 1-Step Teacher Forcing

For a sequence `(o_t, a_t, o_{t+1})`:

```text
z_t = enc(o_t)
z_{t+1} = enc(o_{t+1})
z_hat_{t+1} = f(z_t, a_t)
L_pred = ||z_hat_{t+1} - z_{t+1}||_2^2
```

Gradients follow the same convention across all conditions. No stop-gradient, EMA target network, reconstruction decoder, or auxiliary reward loss is introduced.

### P1: H-Step Closed-Loop BPTT

For a sequence horizon `H=8`:

```text
z_t = enc(o_t)
z_hat_{t+1} = f(z_t, a_t)
z_hat_{t+2} = f(z_hat_{t+1}, a_{t+1})
...
z_hat_{t+H} = f(z_hat_{t+H-1}, a_{t+H-1})

L_pred = (1 / H) sum_{k=1}^H w_k ||z_hat_{t+k} - enc(o_{t+k})||_2^2
```

Weights:

```text
w_k = 1 by default
```

This directly tests whether supervising the model on its self-induced latent states reduces drift.

Regularizer placement:

```text
R1/R2 are applied only to encoded latents enc(o_t), enc(o_{t+1}), ..., enc(o_{t+H}).
They are not applied to rolled-out predictions z_hat_{t+k} in the main experiment.
```

This keeps the regularizer a marginal encoder regularizer. Directly regularizing rollout predictions would be a different intervention that could mechanically attack drift and inflate the apparent R x P interaction.

### P0.5: H-Step Teacher Forcing Control

This control is not in the full main factorial. It is run on selected cells to separate dense multi-horizon supervision from closed-loop exposure.

For horizon `H=8`:

```text
z_{t+k-1} = enc(o_{t+k-1})
z_hat_{t+k} = f(z_{t+k-1}, a_{t+k-1})
L_pred = (1 / H) sum_{k=1}^H ||z_hat_{t+k} - enc(o_{t+k})||_2^2
```

If P0.5 behaves like P0, the gain from P1 is likely from exposure to self-generated states. If P0.5 behaves like P1, the gain is more likely dense multi-horizon supervision rather than closed-loop exposure.

## Environments

### E0: Stable-Linear State

Purpose: sanity check.

```text
x_{t+1} = A x_t
eigenvalues(A) = {0.7, 0.8, 0.9}
observation = x_t
action = none or zero-vector
```

Any faithful latent dynamics should contract perturbations over long horizons. If a model produces `A_50 >> 1`, that is evidence of learned latent geometry instability rather than true system instability.

### E1: Lorenz/Chaotic State

Purpose: nonlinear/chaotic stress test.

```text
sigma = 10
rho = 28
beta = 8/3
dt = 0.01
observation = x_t
action = none or zero-vector
```

This checks whether regularizer effects become stronger under chaotic dynamics.

### E2: State-Observed Action-Conditioned Point Mass

Purpose: isolate action conditioning without a pixel encoder.

State:

```text
x_t in [-1, 1]^2
a_t in [-0.1, 0.1]^2
x_{t+1} = clip(x_t + a_t + small_drift(x_t), [-1, 1])
observation = x_t
```

This environment shares the same transition dynamics as E3. Comparing E2 and E3 isolates the effect of replacing vector observations with a learned pixel encoder.

### E3: Visual Action-Conditioned Point Mass

Purpose: add pixel encoder and action conditioning without exceeding budget.

State:

```text
x_t in [-1, 1]^2
a_t in [-0.1, 0.1]^2
x_{t+1} = clip(x_t + a_t + small_drift(x_t), [-1, 1])
```

Observation:

```text
64 x 64 grayscale image with a rendered Gaussian blob at x_t
optional static distractor dot or faint grid
```

Offline data:

```text
random smooth action trajectories
train/val/test split by trajectory
```

The environment is intentionally simple. The point is not benchmark performance; it is to test whether the state-observed point-mass conclusion survives when an encoder must infer state from pixels.

## Model

State environments:

```text
encoder: 2-layer MLP, width 128, GELU, output dim d=16
dynamics: 2-layer MLP over [z, a], width 128, output dim d=16
```

Visual environment:

```text
encoder: small CNN, 4 conv blocks, output dim d=32
dynamics: 2-layer MLP over [z, a], width 256, output dim d=32
```

No decoder is used. No reward/value model is used.

## Metrics

Primary:

```text
excess perturbation amplification log(A_model,H / A_gt,H) at H in {10, 25, 50}
probe-space open-loop state error growth slope
```

`A_gt,H` is the finite-difference amplification of the ground-truth environment under the same initial state, perturbation scale, and action sequence. This is mandatory for Lorenz, where faithful dynamics should amplify perturbations. Raw latent amplification is still logged as a diagnostic, but it is not the primary cross-environment drift metric.

Secondary:

```text
raw latent perturbation amplification
Jacobian spectral norm product for H in {1, 5, 10}
trajectory-divergence agreement
linear probe R^2 from z_t to true state x_t
effective latent dimension from covariance spectrum
collapse score: rank/effective-dimension threshold plus probe quality
visual planning proxy: CEM terminal state error on point-mass goals
```

Collapse rule:

```text
If effective dimension < 2, rank_1pct < 2, or linear_probe_r2 < 0.5 on state-recoverable environments,
drift metrics for that run are marked invalid/collapsed rather than counted as successful stability.
```

## Analysis

Fit a factorial model separately per environment and jointly:

```text
y = log excess_A_H
y ~ R + P + R:P
```

For the joint model:

```text
y ~ E + R + P + E:R + E:P + R:P + E:R:P
```

Report:

```text
omega^2 where possible, otherwise eta^2 with bootstrap CIs for R, P, and R:P
bootstrap confidence intervals over seeds
condition-wise means with 95% CI
collapsed-run counts
Type-III sums of squares when collapsed-run exclusion unbalances the design
```

## Pre-Registered Decision Criteria

The previous conclusion survives if:

```text
effect_size(P) >= 2 * effect_size(R)
and multi-step BPTT reduces excess A_H in at least 3 of 4 environments
and canonical SIGReg does not remove the 1-step drift failure on its own.
```

The conclusion is overturned if:

```text
canonical SIGReg under 1-step teacher forcing reduces A_H to the same range as multi-step BPTT
in at least 3 of 4 environments, without collapse.
Same range means within a factor of 1.25 on excess amplification or overlapping 95% bootstrap CIs.
```

The interaction claim is supported if:

```text
R:P interaction effect size >= 0.15 in at least 2 environments with bootstrap CI excluding 0
or the canonical SIGReg effect changes sign between 1-step and BPTT with bootstrap support.
```

The encoder-geometry mechanism is supported if:

```text
high drift correlates with effective dimension, Jacobian norms, or state-probe distortion
after controlling for validation one-step MSE.
```

This is mediation evidence, not decisive causality. A small frozen-encoder swap is added as a mechanism probe:

```text
On E0 with R2, freeze encoders from P0 and P1 runs.
Retrain only dynamics under the opposite protocol for seed 0.
If drift follows the frozen encoder rather than the retrained dynamics protocol, encoder geometry is upgraded from correlational to interventional evidence.
```

Middle-region verdict:

```text
If canonical SIGReg improves 1-step drift but does not reach the equivalence margin with BPTT,
the conclusion is "mixed": regularizer matters, but training protocol remains necessary.
```

## Minimum Bar For A Workshop-Quality Paper

The paper should be treated as viable only if it has:

- A reproduced bridge condition showing whether the prior moment-proxy result still appears.
- A canonical SIGReg condition with explicit formula and sensitivity check.
- A clear collapse analysis, so "stable because collapsed" is not mistaken for a good world model.
- At least one action-conditioned pixel environment.
- A result that distinguishes at least two plausible explanations, not just a single averaged performance table.
- Ground-truth-normalized drift on Lorenz.
- The P0.5 control or an explicit statement downgrading "closed-loop exposure" to a coarse protocol claim.

## Known Risks

- No-reg may collapse in the visual environment. This is expected and must be analyzed explicitly.
- The visual point-mass task is not a standard benchmark. It improves over state-only experiments but does not prove transfer to rich robotics/video settings.
- `M=256` projections is a budgeted approximation to canonical SIGReg. The `M=1024` sensitivity check is mandatory for the main cells.
- Multi-step BPTT may reduce drift by changing optimization difficulty, not only by correcting deployment mismatch. Training loss and validation one-step loss must be reported to separate these.
- The experiment will not by itself test Fast-LeWorldModel-style prefix prediction, although that is an obvious next protocol if BPTT is the dominant factor.
