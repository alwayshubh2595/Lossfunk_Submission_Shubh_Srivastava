# Reviewer Gate Before Experiments

## Verdict

Proceed to implementation and remote experiment setup only after the user provides GPU access. The design is worth running because it attacks the main validity threat in the prior paper: moment-proxy SIGReg.

## What Would Make This Paper Interesting

The interesting result is not "BPTT is better than teacher forcing." That is too predictable. The paper becomes useful if it can show one of the following with clean attribution:

- Canonical SIGReg changes the prior causal attribution relative to the moment proxy.
- Canonical SIGReg and closed-loop training interact in a way that neither factor explains alone.
- Closed-loop training remains dominant even when the regularizer is corrected, which would make the prior conclusion more credible.
- Apparent low drift can be separated from representation collapse.
- Encoder geometry metrics explain why stable state dynamics become unstable in latent rollouts.

## Main Confounds And Required Controls

### Collapse

No-reg JEPA-style models may collapse, especially in the visual condition. The analysis must not count collapse as successful low drift. This is why effective dimension, covariance trace, and linear probe quality are mandatory.

### Canonical SIGReg Approximation

The experiment uses the canonical sliced Epps-Pulley objective, but with `M=256` projections by default. This is computationally budgeted. To avoid a weak paper, the highest-leverage canonical cells must be rerun with `M=1024`.

### Protocol Versus Optimization Time

H-step BPTT uses more loss terms per sequence than 1-step teacher forcing. The paper must report matched optimizer steps, wall-clock, prediction loss, and if possible matched-gradient-update controls. If BPTT wins only because it gets denser supervision, that is still a training-protocol result, but it should be stated honestly.

The H-step teacher-forced control is required on selected cells before making a mechanistic claim about closed-loop exposure. Without it, the conclusion must be phrased as a coarse training-protocol effect.

### Visual Environment Simplicity

The visual point-mass task is not enough to claim real-world robotics generalization. It is enough to test whether the prior state-only conclusion survives the introduction of a pixel encoder and actions. The paper must phrase this scope carefully.

Add the state-observed point-mass environment with the same action-conditioned dynamics. This separates action conditioning from pixel encoder geometry at very low cost.

### Fast-LeWorldModel Omission

Fast-LeWorldModel is highly relevant but not part of the main factorial because it changes the architecture and inference interface. If the results point strongly toward protocol/interface as the driver, prefix prediction should be framed as the obvious follow-up rather than silently ignored.

## Minimum Acceptance Bar

The experiments pass the minimum workshop bar only if:

- The code reproduces the previous moment-proxy pattern on at least the stable-linear sanity check.
- Canonical SIGReg is clearly implemented and documented mathematically.
- The main tables separate collapsed and non-collapsed runs.
- The result includes interaction terms, not only condition means.
- Lorenz drift is reported as excess amplification over ground-truth amplification, not raw amplification.
- Lambda selection is pre-registered per `(regularizer, environment)` and not tuned on drift.
- BPTT regularizer placement is fixed to encoded latents only in the main experiment.
- At least one plot shows drift versus encoder geometry diagnostics.
- Compute budget and deviations from the plan are logged.

## Stop Conditions

Stop or redesign before spending the full budget if:

- The canonical SIGReg implementation is numerically unstable and cannot pass a synthetic Gaussian sanity test.
- More than half of all visual runs collapse across all regularizers and protocols.
- The runtime smoke test implies the full sweep would exceed 55 A30 GPU-hours.
- The bridge reproduction contradicts the prior result so strongly that the implementation is likely wrong.
- Ground-truth-normalized drift metrics cannot be implemented for Lorenz before the main sweep.
