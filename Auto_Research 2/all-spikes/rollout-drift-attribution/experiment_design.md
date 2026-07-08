# Experiment Design (v3 — after Codex xhigh review)

## Working Title

Does Canonical SIGReg Change the Causal Attribution of Rollout Drift?

## Provenance

v1: inherited from interrupted prior session (already survived one review round).
v2: this session's budget-driven strengthening (5 visual seeds, encoder swap, rollout-reg ablation).
v3: this version, fixing five near-fatal issues from Codex xhigh review
(`codex_feedback_v2.md`). Changes marked [v3].

## Research Question

In latent world models trained for imagination/planning, is rollout drift primarily
explained by marginal regularizer choice, training protocol, their interaction, or
encoder geometry — once SIGReg is implemented faithfully to its Epps-Pulley
formulation rather than as a moment-matching proxy?

## Outcomes the design can produce (all acceptable)

Regularizer effect / protocol effect / interaction / encoder-geometry mechanism
(interventional) / rollout-distribution mechanism / environment- or
dimension-dependence / pre-registered null (equivalence, not absence-of-significance).

## Factors

```text
Regularizer R:
  R0: none
  R1: moment proxy (bridge to prior paper): ||mu||^2 + ||Sigma - I||_F^2
  R2: SIGReg, exact-integral Epps-Pulley variant [v3: naming corrected, see below]

Protocol P:
  P0: 1-step teacher forcing
  P1: H-step closed-loop BPTT (H=8)
  P0.5: H-step teacher forcing (control, subset)

Environment E [v3: restructured]:
  E0-det: deterministic stable-linear (eigs 0.7/0.8/0.9) — BRIDGE CELLS ONLY
  E0:     stationary OU / stable-linear + process noise (stationary Gaussian
          marginal — SIGReg's favorable regime) — primary sanity environment
  E1:     Lorenz chaotic (sigma=10, rho=28, beta=8/3, dt=0.01)
  E2:     state point mass, action-conditioned
  E3:     visual 64x64 point mass, action-conditioned (CNN)

Latent dimension D [v3: new explicit factor on a subset]:
  primary factorial runs at d=16 (state) / d=32 (visual) — the overcomplete
  regime where the prior result lives;
  matched-dimension sensitivity (d = intrinsic dim: 3 for E0/E1, 2 for E2)
  on E0 and E2 with R0/R2 x P0/P1, so a negative SIGReg result cannot be
  dismissed as "Gaussian target infeasible at rank-deficient d=16".
  Per-environment marginal Gaussianity and stationarity are measured and reported.
```

## R2: SIGReg — exact-integral Epps-Pulley variant [v3]

Per time point t, latents Z_t in R^{B x d}, M random unit vectors, h_m = Z_t u_m:

```text
EP_gamma(h) = mean_ij exp(-gamma (h_i - h_j)^2) + 1/sqrt(1+4 gamma)
            - 2 mean_i exp(-gamma h_i^2/(1+2 gamma)) / sqrt(1+2 gamma)
SIGReg(Z_t) = (1/M) sum_m EP_gamma(Z_t u_m)
L_reg = mean over time points t of SIGReg(Z_t)        [v3: stepwise, averaged
                                                       over time — matches official
                                                       le-wm; never a flattened
                                                       mixture of correlated frames]
```

[v3] Naming: this closed-form Gaussian-kernel-MMD-to-N(0,1) expression is an
exact-integral EP variant (LeJEPA notes the exact integral recovers kernel MMD),
not the official quadrature estimator. The paper will say exactly this. Before
the main sweep, a loss-and-gradient parity test against the official le-wm
implementation (github.com/lucas-maes/le-wm) runs on Gaussian, collapsed,
heavy-tailed, and anisotropic inputs; results logged.

[v3] Equalized exposure: under ALL protocols the regularizer sees the same
per-time-point batch structure, the same sample count, and the same time
weighting (average over the time points available to that protocol's loss,
computed stepwise). M=256 everywhere including lambda pilots. M=1024 sensitivity
reduced to 2 cells (literature reports projection-count insensitivity).

## Run manifest

```text
Bridge (E0-det):            R0/R1 x P0/P1 x 5 seeds                  = 20
Main state (E0,E1,E2):      3 envs x 3 regs x 2 protocols x 5 seeds  = 90
Visual (E3):                3 regs x 2 protocols x 5 seeds           = 30
P0.5 control:               {E0,E3} x {R1,R2} x 5 seeds              = 20
Encoder swap [v3: full 2x2]:
  {E0,E2} x (enc-source P0/P1 x dyn-retrain P0/P1) x 3 seeds         = 24
  (ALL four cells retrain dynamics on a frozen encoder, including the
   diagonal, so the intervention is compared against retrained controls)
Rollout-reg ablation [v3: with dose control, contingent]:
  {E0,E2} x {enc(lambda)+rollout(lambda), enc-only(2*lambda)} x 3 sd = 12
  (run only if main factorial shows P1 or R:P matters; success requires
   improved probe-space state error, not lower amplification alone)
Matched-dim sensitivity [v3]: {E0,E2} x R0/R2 x P0/P1 x 3 seeds      = 24
Lambda sensitivity [v3]: R2 x {low, selected, high} lambda x P0/P1
  on {E0,E2}, 3 seeds (selected-lambda cells reused from main)       = 24 extra
M=1024 sensitivity [v3: reduced]:                                    = 2
Lambda tuning short runs: 2 regs x 5 envs x 5 lambdas, seed 0        = 50 short

Full-length runs: 246 (only 40 are visual/CNN; the rest are tiny MLPs)
```

## Lambda protocol (pre-registered)

Per (R, E), grid {0.01, 0.03, 0.1, 0.3, 1.0}, seed 0, selected on VALIDATION
one-step MSE subject to non-collapse [v3: code currently uses final training
loss — must be fixed before execution]; never on drift; frozen across protocols.

## Metrics [v3: primary outcome changed]

Primary (single pre-registered outcome):
```text
probe-space open-loop state error, integrated over horizon 1..50 (area under
the error curve, log scale), per run. Growth slope reported alongside.
```

Secondary:
```text
amplification fidelity |log(A_model,H / A_gt,H)| at the single primary horizon
H=25 [v3: absolute value so over-contraction cannot masquerade as success;
overshoot and undershoot also reported separately], plus H=10/50 as descriptive.
raw latent amplification, Jacobian norm products, trajectory divergence,
linear-probe R^2, effective dimension, collapse score, CEM planning error (E3),
per-env marginal Gaussianity + stationarity diagnostics [v3].
```

Evaluation uses a fixed, saved evaluation suite per (environment, seed) —
identical windows, perturbation directions, and action sequences across all
treatments [v3]. Final reporting on the test split, not validation [v3].

## Collapse handling [v3: outcome, not exclusion]

Collapse (eff dim < 2, rank_1pct < 2, or probe R^2 < 0.5 on state-recoverable
envs) is a treatment outcome. Report: (1) collapse probability by condition,
(2) drift conditional on viability, (3) bounded sensitivity analysis for the
selection effect. No discretionary reruns of collapsed cells; any added budget
buys complete matched seed blocks across all conditions.

## Analysis [v3: contrast-based, seed-blocked]

Pre-registered same-scale contrasts on the primary outcome:

```text
C_R = (R2 - R0) under P0          — can the regularizer alone fix drift?
C_P = (P1 - P0) averaged over R0, R2   — protocol effect
C_I = (R2 - R0)_P1 - (R2 - R0)_P0      — interaction
```

Seeds are paired blocks (same seed = same data + init across conditions);
inference via cluster bootstrap over seed blocks / mixed models. Factorial
omega^2 tables are reported as descriptive supplements only. E2 and E3 share
dynamics and are counted as one dynamics family in any "k of n environments"
statement [v3].

## Pre-registered decision criteria [v3: rewritten]

Let m = log(1.25) be the practical-equivalence margin on the primary outcome.

- Regularizer fixes drift: C_R CI excludes 0 in the improving direction on
  >= 2 of 3 dynamics families, and P0+R2 vs P1 contrast CI lies entirely
  inside [-m, m] (TOST-style equivalence) without collapse.
- Protocol dominates (prior conclusion survives): C_P CI excludes 0 in >= 2
  of 3 families AND C_R equivalence-to-zero holds (CI inside [-m, m]).
- Interaction: C_I CI excludes 0 in >= 2 families, or CI-supported sign flip
  of C_R between protocols.
- Encoder geometry: in the 2x2 swap, primary outcome follows encoder source
  rather than dynamics-retraining protocol (interaction contrast CI).
- Null/equivalence verdict: C_R, C_P, C_I CIs all inside [-m, m].
- Mixed verdict: anything else, reported as such with all contrast CIs.
```

## Execution stop conditions [v3]

Before any paid sweep:
1. Scaffold brought to parity with this document (5-seed E3, OU env, stepwise
   equalized regularizer, validation-based lambda selection, fixed eval suites,
   checkpoint saving for swaps, contrast-based analysis, test-split reporting).
2. SIGReg parity test vs official le-wm passes.
3. Benchmark of representative R0/R2 x P0/P1 state + visual runs at intended
   concurrency on the actual GPU; extrapolate with 25-30% reserve; projected
   core completion must be under ~50 A30-hours or cells are dropped by the
   pre-registered priority list (drop order: lambda-sensitivity extremes,
   matched-dim seeds 3->2, M=1024, rollout ablation, P0.5 E0 cells).
```
