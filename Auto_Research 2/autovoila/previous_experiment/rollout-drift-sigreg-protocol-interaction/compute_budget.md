# Compute Budget

## Hard Constraint

Maximum spend: 25 USD on Jarvis Labs.

## Current Jarvis Labs Prices Observed

Observed on 2026-07-03:

```text
A30 24GB:       0.41 USD/hr
L4 24GB:        0.44 USD/hr
A100 40GB:     0.89 USD/hr
A100 80GB:     1.49 USD/hr
H100 80GB:     2.69 USD/hr
```

Selected primary GPU:

```text
1 x NVIDIA A30 24GB
```

Fallback if A30 is unavailable:

```text
1 x NVIDIA L4 24GB
```

An A100 is not justified for this design. The experiment uses small CNN/MLP models, not large transformers. The bottleneck is many small runs and careful logging, not single-run memory.

## Budget Arithmetic

At A30 pricing:

```text
25 / 0.41 = 60.98 GPU-hours maximum
```

Reserve for storage and mistakes:

```text
usable cap = 55 GPU-hours
estimated cost = 55 * 0.41 = 22.55 USD
remaining margin = 2.45 USD
```

At L4 fallback pricing:

```text
25 / 0.44 = 56.82 GPU-hours maximum
usable cap = 50 GPU-hours
estimated cost = 50 * 0.44 = 22.00 USD
remaining margin = 3.00 USD
```

## Runtime Estimate

The prior PDF reports:

```text
150 state-space runs
RTX A4000 16GB
about 6 hours wall-clock
20k training steps/run
```

This follow-up:

```text
90 main state-space MLP runs
18 main visual CNN runs
16 H-step teacher-forcing control runs
6-8 frozen-encoder / M=1024 sensitivity runs
canonical SIGReg adds projection/MMD cost
```

Conservative estimate:

```text
state-space MLP runs:
  90 runs * 20k steps = 1.8M optimizer steps
  expected wall-clock on A30: 6-12 hours

visual CNN runs:
  18 runs * 15k steps = 270k optimizer steps
  expected wall-clock on A30: 8-14 hours

H-step teacher-forcing controls:
  16 mixed state/visual runs
  expected wall-clock on A30: 3-6 hours

lambda pre-runs:
  one seed per (regularizer, environment) for R1/R2
  expected wall-clock on A30: 2-4 hours

analysis, plots, reruns:
  4-8 hours

M=1024 canonical SIGReg sensitivity:
  6 high-leverage reruns * 15k steps = 90k optimizer steps
  expected wall-clock: 3-6 hours
```

Expected total:

```text
26-50 GPU-hours
```

Safety envelope:

```text
55 GPU-hours requested on A30
```

This is intentionally below the absolute 60.98h maximum to avoid crossing the 25 USD ceiling under per-minute billing, setup time, failed runs, or small storage costs.

## Planned Execution Stages

### Stage 0: Remote Setup And Smoke Test

Budget:

```text
1-2 GPU-hours
```

Actions:

```text
verify CUDA
install dependencies
run one tiny training job for each environment
verify logging, metrics, and checkpoint paths
```

Stop condition:

```text
If canonical SIGReg is slower than estimated by more than 2x, reduce visual training steps before the full sweep and log the deviation.
```

### Stage 1: Bridge Reproduction

Budget:

```text
8-12 GPU-hours
```

Run the state-space environments with R0/R1 and P0/P1. This verifies whether the implementation can reproduce the previous moment-proxy conclusion before spending budget on the corrected canonical condition.

If the bridge reproduction contradicts the prior result, do not automatically assume implementation failure. First compare hyperparameters, horizon, batch size, regularizer scale, and metric normalization against the prior artifact. If those match, log the contradiction as a result and continue only with a reduced canonical sweep.

### Stage 1.5: Lambda Pre-Runs

Budget:

```text
2-4 GPU-hours
```

Tune lambda per `(regularizer, environment)` for R1/R2 using validation one-step prediction MSE subject to non-collapse. Do not tune on rollout amplification.

### Stage 2: Canonical SIGReg Main Sweep

Budget:

```text
8-16 GPU-hours
```

Run R2 canonical SIGReg across state-space conditions and compare against the bridge cells.

### Stage 3: Visual Action-Conditioned Sweep

Budget:

```text
8-14 GPU-hours
```

Run the 18 visual point-mass conditions.

### Stage 3.5: Controls

Budget:

```text
3-6 GPU-hours
```

Run the selected H-step teacher-forcing controls and the small frozen-encoder swap.

### Stage 4: Sensitivity And Reruns

Budget:

```text
4-8 GPU-hours
```

Mandatory:

```text
M=1024 canonical SIGReg sensitivity on selected high-leverage cells
```

Possible if budget remains:

```text
increase seeds for cells with wide confidence intervals
rerun failed/collapsed cells to distinguish real collapse from optimizer accident
```

### Stage 5: Analysis And Paper Artifacts

Budget:

```text
2-4 GPU-hours if done remotely
0 GPU-hours if analysis can be copied back locally
```

## Requested Access

Ask user for:

```text
Jarvis Labs SSH access to one A30 24GB instance for up to 55 GPU-hours.
Fallback: one L4 24GB instance for up to 50 GPU-hours.
```

Do not start with A100 unless A30/L4 are unavailable or the smoke test shows the visual CNN jobs are unexpectedly slow by more than 3x.

## Spend Guardrails

- Keep a local `budget_meter.md` with start/stop timestamps and estimated cost.
- Stop the instance when not actively running experiments.
- Treat 22.50 USD as the operational stop point, leaving 2.50 USD for storage/per-minute rounding.
- Do not launch multi-GPU instances.
- Do not run extra benchmarks outside the pre-registered cells unless a critical result is uninterpretable.
