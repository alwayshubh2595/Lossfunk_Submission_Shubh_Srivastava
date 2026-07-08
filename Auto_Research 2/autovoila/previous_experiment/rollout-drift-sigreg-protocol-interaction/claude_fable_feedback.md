# Claude Fable Feedback

Claude Fable was invoked for design feedback after the experiment design, compute budget, and reviewer gate were drafted.

## Verbatim Review

I've read all five documents. Here is my review as an AI/ML reviewer, before GPU execution.

## Verdict

The design is directionally sound -- the bridge condition, collapse handling, and pre-registration instinct are all good -- but as written it **cannot cleanly deliver the five-way attribution it promises** (regularizer vs protocol vs interaction vs encoder geometry vs environment). Four issues are near-fatal for interpretation and all are fixable cheaply before spending GPU money. Compute budget is plausible, if anything over-provisioned.

## Near-fatal issues

**1. The drift metric is invalid on the Lorenz environment as specified.** The Lorenz system has a positive Lyapunov exponent, so a *faithful* latent model must show perturbation amplification -- high `A_H` on E1 is correct behavior, not drift. Unless `log A_H` is normalized against the ground-truth system's amplification over the same horizon (i.e., measure excess amplification), E1's primary metric conflates fidelity with drift, and E1 is one of the two environments most decision criteria vote on. E0 has the mirror problem: it's a sanity check where every non-broken model contracts, so it contributes near-degenerate votes to "2 of 3 environments." In practice your environmental vote may reduce to E2 alone.

**2. lambda tuning is a confound that can manufacture or hide the interaction effect.** The grid {0.03, 0.1, 0.3} "on one seed" doesn't specify: per regularizer? per protocol? per environment? selected on what criterion? R1 (Frobenius-norm penalty) and R2 (MMD in [0, ~1]) have completely different loss scales, so a shared grid is meaningless -- and if lambda is tuned under P0 only, an apparent R:P interaction may just be lambda mis-calibration under P1. You must pre-register: lambda tuned per (R, E) cell minimum, on a criterion that is *not* the outcome metric `A_H` (tuning on the outcome makes the comparison circular), with the selected values logged.

**3. Where the regularizer is applied under P1 is unspecified, and it is itself an R x P design choice.** Under H-step BPTT, is SIGReg applied only to encoded latents `enc(o_t)`, or also to rolled-out predictions `z_hat_{t+k}`? These are different experiments: regularizing the rollout distribution directly attacks drift and would inflate the interaction term. Whichever you choose changes the headline interaction result, so it must be pre-specified and justified (or included as a cheap ablation on a few cells).

**4. "Training protocol" is a bundle, and the design can't unbundle it.** P0 to P1 simultaneously changes (a) supervision horizon/density (8 loss terms vs 1), (b) exposure to self-generated states, and (c) gradient flow through the rollout. The risks section acknowledges this but provides no control. Add **P0.5: H-step teacher forcing** (multi-step targets, but each step conditioned on `enc(o_{t+k-1})`, no closed loop) on a subset of cells. Without it, "protocol dominates" is an honest but coarse claim; with it, you can say whether closed-loop exposure specifically -- the mechanistically interesting part -- is what matters. This is a small marginal cost given how tiny the models are.

## Serious but non-fatal

**Encoder geometry is not distinguishable by this design as written.** The criterion is purely correlational across trained runs, and geometry metrics are downstream of both R and P -- so geometry could be a symptom of whatever helps, not a mechanism. To make "the fix lies in encoder geometry" an outcome the design *could* actually produce, add an interventional swap: freeze the encoder from one condition, retrain only the dynamics under another (a 2 x 2 on one environment is a few GPU-minutes). Otherwise, downgrade that outcome to "mediation evidence" explicitly.

**Decision criteria are not exhaustive and have soft edges.** The survive/overturn conditions can both fail simultaneously (e.g., canonical SIGReg halves drift but doesn't match BPTT) -- there is no pre-registered verdict for the middle region, which is the most likely region. "Reduces A_H to the same range" needs a quantitative equivalence margin (e.g., overlapping bootstrap CIs or within a stated factor). "Sign of the effect changes" with n=3-5 seeds can be noise -- require the flip to be significant. Also: partial eta^2 of a factor (P) vs a *contrast* (R2 vs R1/R0) is not an apples-to-apples comparison; and eta^2 with 5 seeds/cell is upward-biased -- report bias-corrected omega^2 or at least CIs on the effect sizes. Specify Type-III sums of squares now, because collapsed-run exclusion will unbalance the design (if R0 collapses in most visual runs, the visual factorial silently loses a level).

**Environment dependence is confounded by construction.** E2 differs from E0/E1 on four axes at once: observation modality, action-conditioning, encoder class, and latent dim. Any E2-specific finding can't be attributed to "pixels" vs "actions" vs "CNN." Cheap partial fix: add a state-based action-conditioned variant of the point mass (same dynamics, vector observation) -- that isolates the pixel/encoder axis and costs almost nothing.

**Drift metrics should be measured in probe/state space, not raw latent space, for cross-condition comparison.** SIGReg forces near-unit marginals; R0 latents have arbitrary scale and anisotropy. `log A_H` is invariant to global rescaling but not to the geometry differences the regularizers themselves induce. Reporting drift via the linear probe into true state (which you already compute) makes conditions commensurable.

**Underspecified essentials that will bite mid-run:** dataset sizes/trajectory counts for E0/E1, optimizer/LR/batch size, the perturbation protocol for `A_H` (magnitude, direction sampling, number of probes), and the fact that P1 trains at H=8 but is evaluated at H=50 (fine, but state it -- extrapolation beyond training horizon is part of the claim). The collapse threshold `trace(cov(z)) < 1e-3` is scale-dependent and meaningless for R0, which has no scale constraint; use a relative or rank-based criterion.

## Compute budget

Plausible and honestly reserved. The 19-36h estimate against a 55h envelope on an A30 at $0.41/hr is realistic; the models are tiny and the O(n^2 M) Epps-Pulley pairwise term (~batch^2 x 256 exp ops/step) is trivial on GPU. Two notes:

- The state-space MLP runs (2-layer, width 128, d=16) barely need a GPU at all. Running them locally on CPU, or vectorizing many runs in parallel on one GPU, would free hours for the additions above (P0.5 control, encoder swap, state-based action-conditioned env, more visual seeds -- visual has only 3 seeds/cell and is your most decision-relevant environment).
- The lambda grid pre-runs and the bridge-failure contingency don't appear as line items in the stage budgets. Add them; the reviewer gate's stop condition treats a failed bridge as "implementation wrong," but it could equally mean the prior result is fragile -- pre-register how you'd tell those apart (e.g., match the prior codebase's hyperparameters exactly for the bridge cells).

## Strongest changes before GPU execution, ranked

1. Normalize `A_H` by ground-truth amplification (fixes E1; harmless elsewhere).
2. Pre-register the lambda protocol: per-(R,E) tuning on a non-outcome criterion.
3. Pre-specify where SIGReg applies under P1 (encoded vs rolled-out latents).
4. Add the H-step teacher-forcing control (P0.5) on a subset of cells.
5. Add the frozen-encoder swap so "encoder geometry" is an outcome the design can actually produce, and close the gap in the decision criteria (define the middle-region verdict and an equivalence margin).
6. Move state-space runs to CPU/parallel and spend the freed GPU-hours on visual seeds, since E2 carries most of the inferential weight.

None of these change the question or the honest-adjudication framing the prompt asked for -- they're what makes the five advertised outcomes genuinely reachable rather than nominally listed.

