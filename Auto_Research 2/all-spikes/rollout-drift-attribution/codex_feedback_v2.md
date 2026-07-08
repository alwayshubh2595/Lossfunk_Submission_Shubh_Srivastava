
Furthermore, none of E0–E3 is a stationary Gaussian world:

- E0 is deterministic contraction, so its marginal variance changes with time.
- Lorenz and clipped point-mass trajectories are strongly non-Gaussian.
- The point-mass policy induces a policy-dependent, bounded visitation distribution.

A negative SIGReg result could therefore be dismissed as testing SIGReg only outside its favorable regime.

Required fix:

- Replace or augment E0 with a stationary Gaussian OU system.
- Use matched latent dimensions—3, 3, 2, 2—as the primary configuration.
- Treat d=16/32 as an overcomplete-latent sensitivity, not the only setting.
- Explicitly measure marginal Gaussianity and stationarity per environment.

This also turns “environment dependence” into a mechanistic hypothesis instead of four loosely related environments.

### Near-fatal 2: The primary metric can label a bad model as improved

The primary outcome is signed `log(A_model/A_gt)` ([design](/home/shubh/Documents/Lossfunk/third-trial/all-spikes/rollout-drift-attribution/experiment_design.md:108)). Lower is treated as better. But a model that collapses sensitivity toward zero receives a large negative value and appears better than a faithful model at zero.

This is especially dangerous for E0, where ground-truth amplification at H=50 is already very small. It also allows rollout SIGReg to “succeed” by making dynamics overly contractive.

Required fix:

- Make probe-space rollout state error, integrated over horizons or its growth slope, the primary outcome.
- For amplification fidelity use `abs(log(A_model/A_gt))`, or separately report overshoot and undershoot.
- Preselect one primary horizon or one trajectory-level summary; H={10,25,50} plus a slope currently leaves outcome multiplicity unresolved.
- Evaluate absolute trajectory bias as well as local perturbation sensitivity. Amplification alone is not rollout drift.

The implementation currently calculates a ratio of mean amplification and does not implement the documented probe-space error-growth slope ([metrics.py](/home/shubh/Documents/Lossfunk/third-trial/all-spikes/rollout-drift-attribution/src/metrics.py:168)).

### Near-fatal 3: Regularizer exposure is confounded with protocol

The design says regularization applies to `enc(o_t...o_t+H)`, but the code regularizes:

- two time points under P0;
- nine under P0.5/P1.

See [train.py](/home/shubh/Documents/Lossfunk/third-trial/all-spikes/rollout-drift-attribution/src/train.py:192). The moment proxy uses all those samples, while SIGReg flattens and subsamples 256. Thus R × P can be generated merely by changing which marginal distribution and how many frames the regularizer sees.

The official LeWorldModel implementation applies SIGReg separately across the independent batch at each time point, then averages across time—not to a flattened mixture of correlated frames ([official implementation](https://github.com/lucas-maes/le-wm/blob/main/module.py)).

The closed-form Gaussian-MMD formula is mathematically a legitimate exact-integral Epps–Pulley variant; [LeJEPA](https://arxiv.org/html/2511.08544) notes that the exact integral recovers kernel MMD. But it is not identical to the official quadrature estimator, sample scaling, or temporal aggregation. Calling it “canonical” without a parity test invites criticism.

Required fix:

- Apply the same regularizer batch, time weighting, and sample count under every protocol.
- Prefer stepwise `(T,B,D)` evaluation with averaging across time.
- Run loss-and-gradient parity tests against the official implementation on Gaussian, collapsed, heavy-tailed, and anisotropic inputs.
- Describe the closed-form implementation as an exact-integral EP variant if retaining it.

### Near-fatal 4: The decision rules do not honestly permit equivalence or null outcomes

“Within factor 1.25 or overlapping 95% CIs” is not an equivalence test. Wide intervals overlap precisely when evidence is weak. Equivalence requires the entire contrast CI to lie inside ±log(1.25), or a preregistered TOST.

There is also no null criterion despite null being advertised. Failure to reject with five seeds is not evidence that effects are negligible.

Other problems:

- `effect(P) >= 2 × effect(R)` compares a one-degree protocol factor to a three-level, two-degree regularizer factor whose variance depends on including the bridge condition.
- The interaction threshold has limited power. With 30 runs per environment, a conventional fixed-effect calculation gives only about 48% power for partial η²=0.15 before multiplicity; requiring confirmation in two environments reduces it further.
- E2 and E3 share dynamics, so “3 of 4 environments” is not four independent replications.
- Five seeds should be treated as paired seed/data blocks, not independent cell observations.

Required fix:

Pre-register same-scale contrasts:

- \(C_R = R2-R0\) under P0.
- \(C_P = P1-P0\), averaged over specified R levels.
- \(C_I = (R2-R0)_{P1}-(R2-R0)_{P0}\).

Use seed-blocked or mixed models and cluster-bootstrap entire seed blocks. Define null as all relevant CIs lying inside practical-equivalence margins.

Collapse must be a treatment outcome, not merely excluded. Conditioning on non-collapse is post-treatment selection; Type-III sums of squares do not repair that. Report:

1. collapse probability by condition;
2. drift conditional on viability;
3. a composite or bounded sensitivity analysis.

Do not use “rerun collapsed cells” as a discretionary budget sink. Add complete matched seed blocks across all conditions instead.

### Near-fatal 5: The execution scaffold does not match v2

The current sweep still launches three visual seeds and only 124 main/control runs ([run_sweep.py](/home/shubh/Documents/Lossfunk/third-trial/all-spikes/rollout-drift-attribution/src/run_sweep.py:34)). It has no rollout-regularization, encoder-swap, or M=1024 stages and does not save the checkpoints needed for swaps.

Additional mismatches:

- Lambda selection uses final training-batch loss, not validation one-step MSE ([select_lambdas.py](/home/shubh/Documents/Lossfunk/third-trial/all-spikes/rollout-drift-attribution/src/select_lambdas.py:31)).
- Lambda pilots use M=128 rather than the main M=256.
- Analysis includes collapsed runs and implements neither omega², Type-III SS, blocked bootstrap, joint modeling, nor the preregistered contrasts ([analyze.py](/home/shubh/Documents/Lossfunk/third-trial/all-spikes/rollout-drift-attribution/src/analyze.py:92)).
- Evaluation uses one random window/direction batch whose RNG state can differ by treatment. Save a fixed evaluation suite per environment/seed.
- Validation is used for final reporting while the generated test split is unused.

This is an execution stop condition.

## 3. Is the extra $5 well spent?

Partly.

- Five visual seeds: yes. This is the best v2 addition, though it remains weak for formal equivalence.
- Strengthened encoder swap: right motivation, incomplete design. Two off-diagonal swaps do not form a clean intervention because they are compared against jointly trained, non-retrained diagonal models. Retrain dynamics for the complete `encoder-source protocol × dynamics-training protocol` 2×2. Also specify R, lambda, initialization, and the contrast.
- Rollout-regularization ablation: not in its present form. “Encoder + rollout SIGReg” increases total regularization as well as changing placement. Add an encoder-only `2λ` control or hold total regularizer weight fixed. Success must require improved state error/planning, not merely lower amplification. Three seeds are descriptive, not CI-grade.

I would retain the visual seeds, complete the swap, and make the rollout ablation contingent. Spend remaining capacity on:

1. matched-dimension/stationary-OU controls;
2. low/selected/high lambda sensitivity for R2 under both P0 and P1;
3. full seed blocks rather than selective reruns.

Six expensive M=1024 reruns are lower priority. LeJEPA recommends 1024, but reports competitive results with 512, LeWorldModel reports projection-count insensitivity, and the recent identifiability experiments use 256. One or two parity/sensitivity cells should be sufficient unless M changes conclusions.

The v2 change arithmetic is also wrong: raising E3 from three to five seeds adds 12 main runs plus four P0.5 runs—16, not 14.

## 4. Compute-budget realism

The hourly prices are correct: Jarvis currently lists A30 at $0.41/hour and L4 at $0.44/hour with per-minute billing ([official pricing](https://jarvislabs.ai/pricing)).

The runtime estimate is not yet reliable:

- The listed high bounds sum to 66 hours, not 60, when contingency and analysis are included.
- The previous A4000 calibration did not include this quadratic pairwise SIGReg. At n=M=256, the implementation evaluates roughly 16.8 million pairwise kernel values per training step before autograd. The official characteristic-function implementation is linear in batch size.
- Each visual step currently encodes 256×9=2,304 images at once. Eight concurrent jobs on a 24GB GPU require an empirical memory/throughput test.
- The launcher stops starting jobs at the deadline but allows all active jobs to finish, so it is not a hard dollar cutoff ([parallel_runner.py](/home/shubh/Documents/Lossfunk/third-trial/all-spikes/rollout-drift-attribution/src/parallel_runner.py:60)).
- Dropping visual training from 15k to 10k steps after seeing timing can create unequal convergence. Calibrate before preregistration, or drop secondary cells rather than changing the primary treatment dose.

A realistic approval gate is:

1. Implement the exact manifest and analysis.
2. Benchmark representative R0/R2 × P0/P1 state and visual runs at intended concurrency.
3. Extrapolate from measured step times with a 25–30% reserve.
4. Require projected core completion under about 50 A30 hours; leave the remainder for failed jobs and predetermined full seed blocks.

## 5. Likely workshop rejection reasons

A reviewer could reasonably reject for:

1. SIGReg being evaluated only where its Gaussian target is incompatible with latent dimension and data distribution.
2. The signed amplification metric rewarding over-contraction.
3. Calling the implementation canonical without official loss/gradient parity.
4. Invalid equivalence-by-overlapping-CIs and unsupported null claims.
5. Differential collapse being excluded post-treatment.
6. Underpowered interaction/mechanism conclusions from three to five seeds.
7. Overclaiming general world-model conclusions from one Gaussian-blob visual task.
8. Missing recent literature on Gaussian identifiability, low-dimensional bias, Sub-JEPA, and Fast-LeWM.
9. Code, run manifest, and analysis not matching the preregistration.

The design is close in spirit, but not in identification. Fixing the first four items before execution matters more than adding further runs.
