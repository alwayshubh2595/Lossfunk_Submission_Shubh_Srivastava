# Causal Attribution of Rollout Drift in Latent World Models
## A Preliminary Experimental Brief for Autoresearch Execution

---

## 0. How to Read This Document

This is a brief for an autoresearch system (Claude Code, Autovoila, or similar) to design, implement, run, and report a controlled experiment. It is **not** a prompt to interpret loosely. The factorial design, the metrics, the predictions, and the decision criteria are specified upfront and should not be modified by the executing system without explicit human approval.

If you (the AI) find yourself wanting to "improve" the design — expand the factor set, change the environments, swap the metric — **stop and flag it instead of acting**. The design choices encode judgment calls that are the human researcher's contribution. Silently changing them defeats the purpose of the exercise.

This is a **scoping experiment** with a budget of ~$5–10. It exists to answer one question:

> *Is the rollout-drift problem we observe in latent world models primarily caused by the marginal regularizer (SIGReg), or by something else?*

The answer determines whether the larger research project on regularizer design is worth pursuing.

---

## 1. Background and Motivation

### 1.1 The broader research context

Latent world models (DreamerV3, JEPA-family models like LeWM/LeJEPA, DINO-WM) represent observations in a compressed latent space `z` and learn a dynamics model `f` such that `z_{t+1} ≈ f(z_t, a_t)`. A central failure mode is **representational collapse**: the encoder maps distinct observations to the same point, trivially satisfying any temporal-prediction objective.

To prevent collapse, several **marginal regularizers** are used. They constrain the distribution of `z_t` aggregated across many trajectories. Examples:

- **SIGReg** (LeJEPA): pushes the empirical marginal `q(z)` toward `N(0, I)`.
- **VICReg**: enforces variance, invariance, and covariance constraints on `z`.
- **Barlow Twins**: decorrelates feature dimensions across two views.

All three operate on the **marginal** `q(z)` — the distribution of latents at any single timestep, aggregated. None of them places explicit constraints on the **conditional** `p(z_{t+1} | z_t, a_t)`, which governs trajectory-level behavior.

### 1.2 The hypothesis under test

The original hypothesis (from the proposal): *marginal regularizers like SIGReg do not bound rollout drift, because they constrain marginals but not conditionals; therefore, latent rollouts in JEPA-family world models can drift catastrophically even when the regularizer is "succeeding" by its own metric.*

### 1.3 What the existing preliminary toy experiment showed

A small synthetic experiment was already run on a 2D mildly-expanding spiral dynamical system. Two world models were trained, identical except for SIGReg:

| k  | KL (SIGReg) | A_k (SIGReg) | KL (no reg) | A_k (no reg) |
|----|-------------|--------------|-------------|--------------|
| 1  | 1.48        | 1.13         | 0.43        | 1.02         |
| 50 | 17.68       | 18.0         | 87.05       | 2.78         |

Where `A_k` measures how a small perturbation `ε` to `z_0` is amplified after `k` rollout steps.

**Two observations from this preliminary data:**

1. **SIGReg keeps the marginal Gaussian** (KL stays ~5× lower than no-reg).
2. **The SIGReg model amplifies perturbations *more*, not less, than the unregularized model** (`A_50 = 18.0` vs `2.78`).

This already partially undercuts the original framing. The unregularized model also amplifies (`A_50 = 2.78`), so the regularizer is not the *sole* cause. And the regularized model amplifies more, suggesting the regularizer may even be *harming* trajectory stability while improving marginal cleanliness.

### 1.4 Why a scoping experiment is needed before the main project

The toy result is suggestive but not causally interpretable, for at least four reasons:

- **One environment (2D spiral)**, which is rigged: any positive Lyapunov exponent in the underlying dynamics guarantees A_k growth.
- **One seed** (per condition).
- **Dynamics model capacity and training protocol were not varied**, so we cannot tell whether the A_k difference is encoder-attributable, dynamics-attributable, or protocol-attributable.
- **No positive control** (e.g., a stable linear system where A_k should *shrink*) — so we cannot rule out measurement artifacts.

The scoping experiment fixes all four issues.

---

## 2. Research Question (Precise)

> **Of the total rollout amplification `A_k` observed in latent world models, what fraction is attributable to (a) the encoder regularizer, (b) the dynamics model's capacity/specification, and (c) the training-time rollout protocol?**

Subsidiary questions, answered by the same experiment:

- Does the effect persist on a chaotic system (Lorenz) and reverse on a stable-linear system, as it should if `A_k` is a meaningful measure?
- Do the three different `A_k` measurement methods (Jacobian, perturbation, trajectory divergence) agree?
- Does normalized `A_k` (`A_k_model / A_k_ground_truth`) behave differently than raw `A_k`?

---

## 3. Hypothesis Space

All of the following are **live hypotheses** at the start of the experiment. The result must distinguish among them.

- **H1 (original):** Encoder regularization (SIGReg) is the primary driver of `A_k` growth.
- **H2:** Dynamics-model under-specification (low capacity) is the primary driver, and the regularizer effect is incidental.
- **H3:** Training-protocol exposure bias (1-step teacher-forced training) is the primary driver; multi-step BPTT during training would substantially reduce `A_k`.
- **H4:** Interaction effects dominate (e.g., SIGReg only hurts when dynamics are under-capacity; with adequate dynamics, SIGReg is neutral or helpful).
- **H5 (null):** None of the three factors explains a meaningful fraction of variance in `A_k`; the drift is intrinsic to the environment's Lyapunov exponent and the regularizer is a red herring.

---

## 4. Experimental Design

### 4.1 Factorial structure (2 × 2 × 2)

| Factor | Level (a) | Level (b) |
|--------|-----------|-----------|
| **F1: Encoder regularization** | SIGReg | None |
| **F2: Dynamics MLP depth** | 1 hidden layer (under-capacity) | 3 hidden layers (well-specified) |
| **F3: Training rollout protocol** | 1-step teacher-forced | 5-step BPTT rollout loss |

→ **8 conditions** per environment.

### 4.2 Environments (3)

| Environment | Dimension | Type | Ground-truth `λ_max` | Role |
|-------------|-----------|------|----------------------|------|
| **Stable linear** | 3D | `z_{t+1} = A z_t`, eigenvalues `|λ| ∈ {0.7, 0.8, 0.9}` | < 0 | Sanity check: `A_k` should *shrink* |
| **2D spiral** | 2D | Rotation + 5% expansion per step | ≈ +0.05 | Replicate prior toy result |
| **Lorenz** | 3D | `σ=10, ρ=28, β=8/3`, `dt=0.01` | ≈ +0.906 | Chaotic stress test |

For each environment, generate 50,000 trajectories of length 100 from random initial conditions in the relevant basin. Train/val/test split 80/10/10.

**Note on the stable-linear environment:** if any condition reports `A_k > 1` at `k=50` for the stable-linear system, this indicates either (a) a bug in the implementation, or (b) a pathological effect of the training procedure that should be flagged as a finding in itself.

### 4.3 Positive control: oracle dynamics

Add one extra condition per environment: **oracle baseline** — train a dynamics model directly on the ground-truth state (not on learned latents), with the same MLP architecture and protocol as the "large dynamics + 5-step BPTT" condition. Measure `A_k`. This is the floor of dynamics-model-attributable error. Anything worse than this is encoder-attributable.

### 4.4 Side sweep: latent dimension

For the **2D spiral environment only**, additionally sweep latent dimension `d_z ∈ {2, 8, 32}` with the SIGReg + small-dynamics + 1-step condition. This tests whether the observed effect depends on the latent being over- or under-dimensioned relative to the true state.

**Rationale:** SIGReg's effective pressure scales with how much room the latent has. The original toy result may be entirely an artifact of a particular `d_z` choice.

### 4.5 Total run count

- Main factorial: 8 conditions × 3 environments × 10 seeds = **240 runs**
- Oracle baseline: 1 condition × 3 environments × 10 seeds = **30 runs**
- Latent-width side sweep: 3 widths × 1 environment × 10 seeds = **30 runs**

**Total: ~300 runs.**

---

## 5. Metrics

### 5.1 Primary metric: A_k via three methods

For each trained model, compute `A_k` at `k ∈ {1, 5, 10, 25, 50}` using **all three** methods below. Report all three.

**Method 1: Accumulated Jacobian operator norm**

```
A_k^Jac = E_{z_0} [ ||∏_{i=0}^{k-1} ∂f/∂z |_{z_i}||_op ]
```

Where `z_i` is the deterministic rollout from `z_0`. Use the largest singular value of the product matrix. Average over `N = 100` initial conditions `z_0` sampled from the test-set encoder output distribution.

**Method 2: Finite-difference perturbation**

```
A_k^pert(ε) = E_{z_0, u} [ ||f^k(z_0 + ε·u) - f^k(z_0)|| / ε ]
```

Where `u` is a unit vector sampled uniformly on the sphere. Compute for `ε ∈ {1e-4, 1e-3, 1e-2, 1e-1}`. Report the full `A_k(ε)` curve, not just one value.

**Method 3: Paired-trajectory divergence**

```
A_k^traj = E_{z_0} [ ||f^k(z_0 + δ_1) - f^k(z_0 + δ_2)|| / ||δ_1 - δ_2|| ]
```

Where `δ_1, δ_2 ~ N(0, ε² I)` with `ε = 1e-3`. This measures how the *spread* of nearby initial conditions evolves, which is closer to what happens during planning/imagination.

**If the three methods disagree substantially, report this as a finding.** Do not pick one and discard the others. Disagreement is information about the geometry of the latent space.

### 5.2 Normalized A_k

For environments where ground-truth dynamics are known (all three in this experiment), compute the same `A_k` metrics on the *ground-truth* dynamics and report:

```
A_k^norm = A_k^model / A_k^ground_truth
```

Interpretation:
- `A_k^norm ≈ 1`: model correctly captures sensitivity of true dynamics
- `A_k^norm >> 1`: spurious amplification (latent space is artificially unstable)
- `A_k^norm << 1`: artificial smoothing (model is over-stable; may indicate collapse)

**Raw `A_k` alone is nearly meaningless without normalization.** A model with `A_k = 18` is not necessarily bad if the ground-truth has `A_k = 20`.

### 5.3 Secondary diagnostics

Report for each condition:

- **1-step prediction MSE** on held-out test trajectories (to detect under-training)
- **Marginal KL** from `N(0, I)` (to verify SIGReg is doing its job)
- **Effective latent dimension** via PCA of test-set latents (to detect collapse)
- **Final training loss** (to verify convergence)

### 5.4 Statistical analysis

- For each cell (condition × environment), report **mean, std, and 95% CI** across 10 seeds.
- Run **two-way ANOVA** on `log(A_50^pert)` with factors F1, F2, F3 within each environment.
  - Use `log(A_k)` because amplification is multiplicative.
  - Report **effect sizes (η² and partial η²)** for main effects and all two-way interactions, *not just p-values*.
- Report a **variance decomposition**: what fraction of variance in `log(A_50)` is explained by each main effect, each interaction, and residual.

---

## 6. Implementation Specifications

### 6.1 Encoder architecture

- Input: raw environment state (2D, 3D, or 3D depending on env)
- MLP: 2 hidden layers, width 256, GELU activation, LayerNorm before final projection
- Output: latent `z ∈ R^{d_z}`
- Default `d_z = 16` for all factorial conditions; varied only in the side sweep

**No image encoder.** Since the environments are low-dimensional state, the encoder is an MLP applied to state coordinates. This is *intentional* — adding an image encoder would conflate encoder-quality issues with regularization issues and is out of scope for the scoping experiment.

### 6.2 Dynamics model

- Input: `z_t` (no action conditioning, since environments are autonomous)
- Small (F2-a): 1 hidden layer, width 128, GELU
- Large (F2-b): 3 hidden layers, width 128 each, GELU, with residual connections
- Output: `z_{t+1}` (deterministic prediction)

### 6.3 SIGReg implementation

Implement SIGReg from the LeJEPA paper (arXiv:2603.19312 or the referenced LeWM paper). The loss term is a closed-form characteristic-function distance between the empirical batch latent distribution and `N(0, I)`. If unsure of exact implementation, use a moment-matching variant:

```
L_SIGReg = ||μ_batch||² + ||Σ_batch - I||_F²
```

This is not the original SIGReg but is a reasonable proxy if the original implementation is unavailable. **Flag in the report which version was used.**

### 6.4 Training loss

For 1-step teacher-forced (F3-a):
```
L = MSE(f(enc(x_t)), enc(x_{t+1})) + λ_reg · L_SIGReg
```

For 5-step BPTT (F3-b):
```
L = Σ_{k=1}^{5} MSE(f^k(enc(x_t)), enc(x_{t+k})) + λ_reg · L_SIGReg
```

`λ_reg = 1.0` when SIGReg is on, `0` when off.

### 6.5 Convergence criterion

- **Fixed compute**: 50,000 gradient steps for all conditions, batch size 256, AdamW with `lr=3e-4, wd=1e-4`.
- **Report**: final train and val 1-step prediction MSE per condition. If any condition has val MSE substantially above the median, flag as potentially under-converged.
- Do **not** use early stopping. We want differences in `A_k` at matched compute, not at matched validation loss.

### 6.6 Seeds

10 seeds per condition. Use seeds `[0, 1, ..., 9]`. Same seeds across conditions for paired analysis.

### 6.7 Reproducibility

- Fix all random seeds (`torch.manual_seed`, `numpy.random.seed`, `random.seed`, `torch.use_deterministic_algorithms(True)` where possible).
- Save model checkpoints, all measured metrics, and configuration files for every run.
- Save a `config.json` per run with all hyperparameters and a git commit hash.

---

## 7. Compute Budget

- Per run: ~5 minutes on a single GPU (synthetic low-D environments train fast).
- 300 runs × 5 min ≈ 25 GPU-hours.
- Budget cap: **$10**. Stop if cost projection exceeds this.

If runtime is significantly higher than estimated, reduce seed count from 10 → 5 *before* dropping any factor. Do not silently drop conditions.

---

## 8. Pre-Registered Predictions and Decision Criteria

These predictions are recorded **before** the experiment runs and will be compared against actual results in the critique. Filling these out is the user's responsibility — examples below indicate the format.

### 8.1 Predicted ranking of factor effects on `log(A_50)`

> **(User: fill in your prior ranking here. Example:)**
> 1. F2 (dynamics capacity) — predicted largest effect: η² ≈ 0.4
> 2. F3 (training protocol) — predicted η² ≈ 0.25
> 3. F1 (regularizer) — predicted η² ≈ 0.10
> 4. F1×F2 interaction — predicted η² ≈ 0.05

### 8.2 Specific point predictions

> **(User: fill in. Example:)**
> - For SIGReg + large dynamics + 5-step BPTT on Lorenz at `k=50`: predicted `A_k^pert(ε=1e-3) ≈ 8 ± 3`
> - For oracle baseline on Lorenz at `k=50`: predicted `A_k^pert ≈ 6 ± 2`
> - For unregularized + large dynamics + 5-step BPTT on stable linear at `k=50`: predicted `A_k ≈ 0.5 ± 0.2`

### 8.3 Predicted method agreement

> **(User: fill in. Example:)**
> - Jacobian and perturbation methods will agree within 20% for small `ε`.
> - Trajectory-divergence method will diverge from the other two at large `k` due to nonlinearity.

### 8.4 Decision criteria

These are committed in advance to prevent post-hoc rationalization:

- **"F1 (regularizer) is the primary cause"** ⇔ F1 main effect explains ≥ 50% of `log(A_50)` variance averaged across environments.
- **"The original framing should be redirected toward dynamics models"** ⇔ F2 main effect explains ≥ 2× the variance of F1.
- **"The original framing should be redirected toward training protocols"** ⇔ F3 main effect explains ≥ 2× the variance of F1.
- **"The regularizer effect is conditional on dynamics capacity"** ⇔ F1×F2 interaction explains ≥ F1 main effect.
- **"The null hypothesis holds"** ⇔ No main effect or interaction explains ≥ 20% of variance; environment Lyapunov exponent dominates.

### 8.5 What would falsify the broader research direction

> If F1 explains < 15% of `log(A_50)` variance and F2 or F3 explains > 40%, the broader project should be redirected. The "regularizers can't bound rollout drift" framing is technically correct but practically irrelevant: other factors matter more.

---

## 9. Deliverables (What the AI Should Produce)

### 9.1 Required tables

1. **Master results table:** rows = (condition × environment), columns = (`A_1, A_5, A_10, A_25, A_50` for each of the 3 measurement methods), with mean ± 95% CI across 10 seeds.
2. **Normalized A_k table:** same structure but reporting `A_k^norm`.
3. **ANOVA variance decomposition table:** per environment, fraction of variance in `log(A_50)` attributable to each main effect and interaction.
4. **Diagnostic table:** per condition, final train MSE, final val MSE, marginal KL from `N(0,I)`, effective latent dim.
5. **Sanity-check table:** stable-linear results showing `A_k` < 1 across all conditions (if not, this is a finding).

### 9.2 Required plots

1. `A_k(k)` curves per condition, one panel per environment, log-scale y-axis, with shaded 95% CI.
2. `A_k(ε)` curves at fixed `k=50`, per condition, log-log axes.
3. Factor effect plots: main effect of F1, F2, F3 on `log(A_50)`, with error bars.
4. Interaction plots: F1×F2, F1×F3, F2×F3 on `log(A_50)`.
5. Latent-width side sweep: `A_50` as a function of `d_z` for the 2D spiral.

### 9.3 Required written sections

The artifact (paper draft or report) should contain:

1. **Methods** — exact implementation, all hyperparameters, exact metric definitions.
2. **Results** — tables and plots above, with no interpretation.
3. **Statistical analysis** — ANOVA results, effect sizes, CI overlap.
4. **Discussion of method agreement** — do the three `A_k` methods give the same answer? Where do they disagree?
5. **Honest caveats section** — list every confound the experiment doesn't control for. See section 11.

### 9.4 What the AI should NOT write

- Do **not** write an "Interpretation" or "Conclusion" section that says which hypothesis is supported. That is the human researcher's job to write after reading the results.
- Do **not** write speculation about what a fix would look like. That is the next experiment, not this one.
- Do **not** soften negative results. If F1 explains 5% of variance, report 5%, not "F1 plays an important role."

---

## 10. What the AI Should NOT Do (Constraint List)

### 10.1 Do not expand the experimental scope

- Do **not** add additional regularizers (VICReg, Barlow Twins). The scoping experiment is binary on F1 for a reason.
- Do **not** add real environments (DMC, Atari, robotics). Scope is intentionally synthetic.
- Do **not** add longer horizons beyond `k=50`. The curve shape is established.
- Do **not** add additional latent widths beyond the specified side sweep.

If the AI thinks an expansion is critical, **flag it in the report and ask for human approval** instead of running it silently.

### 10.2 Do not change measurement definitions

- The three `A_k` methods are defined exactly above. Do not introduce new variants.
- If a definition is ambiguous, ask for clarification rather than picking a default.

### 10.3 Do not skip the sanity checks

- The stable-linear environment is non-negotiable. If `A_k > 1` for any condition on stable-linear at `k=50`, this is a critical finding that must be reported at the top of the results.
- The oracle baseline is non-negotiable. Without it, encoder-vs-dynamics attribution is impossible.

### 10.4 Do not conflate prediction error with latent geometry

- 1-step prediction MSE and `A_k` are different quantities measuring different things. A model can have low MSE and high `A_k` (the dynamics are smooth at the data points but the latent geometry is unstable elsewhere).
- Report both. Do not use one to justify claims about the other.

### 10.5 Do not "fix" the prior result

- The toy result from the proposal showed SIGReg `A_50 = 18`, no-reg `A_50 = 2.78`. The scoping experiment should *test* whether this reverses or persists, not assume it.
- If the result fully reverses (SIGReg now reduces `A_k`), do not adjust the experimental setup to "match the prior." Report the reversal.

### 10.6 Do not interpret null results as support

- If no factor explains a meaningful fraction of variance, the conclusion is "the null hypothesis holds." It is not "all factors matter equally."

---

## 11. Honest Caveats and Failure Modes the Final Report Must Flag

The report must explicitly acknowledge the following limitations. The AI should add to this list any additional confounds it observes.

1. **Synthetic-only environments.** Results may not transfer to JEPA-family models trained on images. The scoping experiment is specifically about isolating mechanism, not about realism.
2. **No image encoder.** SIGReg's interaction with image encoder pathologies (e.g., collapse-to-prior issues) is not tested here.
3. **No action conditioning.** Autonomous dynamics may behave differently from action-conditioned dynamics.
4. **Limited latent dimensions.** Even with the side sweep, behavior at `d_z = 256` or `d_z = 1024` (more realistic for vision models) is not directly tested.
5. **Fixed-compute training.** Findings about F1 may depend on whether SIGReg converges slower than no-reg. Reporting both train and val MSE is the partial mitigation.
6. **Moment-matching SIGReg may behave differently from the original.** If the original characteristic-function implementation is unavailable, the proxy is not a perfect substitute.
7. **`A_k` measures local sensitivity, not basin-of-attraction structure.** Two models with identical `A_k` could have very different global trajectory behavior.
8. **One-step Lyapunov exponent of the *ground truth* may not equal the Lyapunov exponent of the *latent-space* dynamics.** Encoder warping changes geometry. Normalized `A_k` partially addresses this.

---

## 12. Workflow Summary for the Autoresearch System

1. **Read this brief in full.** If anything is ambiguous, stop and ask before proceeding.
2. **Verify environment.** Confirm GPU access, dependencies (PyTorch, NumPy, SciPy for ANOVA, matplotlib).
3. **Implement environments** (stable linear, 2D spiral, Lorenz). Verify ground-truth Lyapunov exponents match expected values.
4. **Implement encoder, dynamics model, SIGReg loss.** Verify SIGReg drives marginal KL to small values on a quick test run.
5. **Implement A_k measurement (all three methods).** Verify on ground-truth dynamics of the stable-linear system: A_k should be < 1.
6. **Run main factorial: 8 conditions × 3 environments × 10 seeds = 240 runs.**
7. **Run oracle baseline: 3 environments × 10 seeds = 30 runs.**
8. **Run latent-width side sweep: 3 widths × 10 seeds = 30 runs (on 2D spiral).**
9. **Compute all metrics**, save all per-run data as JSON or Parquet.
10. **Generate all required tables and plots.**
11. **Run ANOVA on `log(A_50^pert)` per environment.** Compute effect sizes.
12. **Write the report** following section 9. Do not write interpretation or conclusions beyond what the data directly states.
13. **Flag in the report:** any conditions that failed to converge, any sanity-check failures, any AI-flagged design concerns, any deviations from this brief.

---

## 13. Definition of "Success" for the Scoping Experiment

The scoping experiment succeeds — independent of which hypothesis is supported — if it produces:

- A clean variance decomposition for `log(A_50)` per environment.
- Reliable, reproducible numbers with reported uncertainty.
- A clear answer to: "Is F1 the dominant factor, or not?"
- Honest documentation of every confound and limitation.

The experiment **fails** if:

- The result is ambiguous due to insufficient seeds or non-converged training (report which and re-run if budget allows).
- The sanity check (stable-linear) is violated and the violation is not explained.
- The AI silently changed the design.

---

## 14. Final Note to the Autoresearch System

Speed is not the goal. Correctness is. A clean, narrow, well-controlled answer to the question "is F1 the primary cause" is worth far more than a sprawling exploration of related questions.

If during execution you discover a design flaw in this brief, **flag it before continuing**. The human researcher would rather pause and revise than receive a contaminated result.

The deliverable is data + tables + plots + honest documentation. The interpretation, the conclusion, the next-step decisions — those are human work.

---
*End of brief.*
