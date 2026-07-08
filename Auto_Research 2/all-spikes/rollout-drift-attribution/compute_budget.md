# Compute Budget (v10 FINAL, $30 ceiling, single 7x L4 instance)

## v10: actual provisioned hardware is a 7-GPU L4 instance

The user confirmed the instance actually being provisioned is a single L4
24GB instance with **7 GPUs**. This supersedes v9's RTX A5000 plan (that
GPU is not what's being provisioned) — v9 kept for audit trail below.

Using the previously-established real L4 rate (Rs 41.31/hr per GPU, ~95.5
INR/USD):

```text
Total GPU-hours affordable at L4 rate: 2865 / 41.31 = 69.4 h (any instance shape)
7 GPUs -> wall-clock for the full $30 budget = 69.4 / 7 = 9.91 h
Manifest (66 GPU-hours, ~$28.55) -> wall-clock = 66 / 7 = 9.43 h
```

9.43h leaves ~2.6h margin under the 12h requirement for the Stage 0
benchmark, the lambda-tuning sync point, and monitoring/restart overhead.
Even spending the entire $30 ceiling stays at 9.91h — still comfortably
inside 12h. Single instance, no cross-machine sync needed.

## OPERATIVE PLAN: single instance, 7x L4 24GB (final)

Reasoning carries over unchanged: the manifest (~246 mostly-tiny MLP jobs +
40 small-CNN visual jobs) is bottlenecked by per-step/launch overhead and job
count, not per-GPU FLOPs. L4 remains the only tier that affords enough total
GPU-hours for the design (69.4h vs 16.0h/11.2h/7.6h for RTX PRO 6000/H100/H200
at the same $30). 24GB VRAM is ample headroom for these models.

## Execution plan (single 7-GPU instance)

```text
Stage 0: smoke tests + mandatory on-GPU benchmark across all 7 GPUs (~0.5-1h).
         Benchmark gates everything after it — see approval gate below.
Stage 1.5: lambda pre-runs (40 short runs) sharded across the 7 GPUs (~0.5-1h).
           HARD SYNC POINT before the main sweep starts.
Stage 2-3.7: main state sweep, visual sweep, P0.5 controls, encoder swap
             (full 2x2), rollout-reg ablation — pre-sharded across the 7
             GPUs by run-manifest chunks assigned before launch so no GPU
             idles waiting on another (~6-7h).
Stage 4: M=1024 sensitivity (6 reruns) + analysis, run on 1-2 GPUs while
         the rest spin down (~0.5-1h).
```

## Approval gate

Before the paid main sweep: benchmark representative R0/R2 x P0/P1 state and
visual runs at intended per-GPU concurrency on real L4 hardware; extrapolate
with 25-30% reserve. Projected core completion must fit under **9-9.5 hours
wall-clock on 7 GPUs (~66 GPU-hours)**, else drop cells by the pre-registered
priority list from `experiment_design.md`: lambda-sensitivity extremes ->
matched-dim seeds 3->2 -> M=1024 (back to 2) -> rollout ablation -> P0.5 E0
cells. Never change training steps of primary cells after seeing timing —
drop whole cells instead.

## Request to user [v10 — FINAL]

```text
SSH access to ONE instance with 7x L4 24GB GPUs.
Operational budget cap: ~66 GPU-hours (~Rs 2726, ~$28.55 of $30).
Expected wall-clock: ~9-9.5 hours, comfortably under the 12h requirement.
Worst case using the entire budget: ~9.9h wall-clock.
```

## Guardrails

- `budget_meter.md` with start/stop timestamps for the single instance and
  running cumulative cost in USD/INR.
- Stop the instance as soon as all assigned work finishes; do not leave GPUs
  idle mid-session.
- Operational stop point: ~$28.55 spent (~$1.45 margin).
- No runs outside pre-registered cells except the explicit contingency uses
  (extra seeds for wide-CI cells, matched seed blocks for collapsed cells).
- All 7 GPUs' work is pre-sharded before launch; the only hard sync point is
  lambda-tuning -> main sweep.

---

## SUPERSEDED: v9 (single 10x RTX A5000 instance) — not what's being provisioned

## v9: A5000 plan reinstated as final (v7/v8 L4 topology voided)

The L4-based plans (v7: 2x4, v8: 4x2) were built on a Jarvis-style INR price
list that kept shrinking in usable instance size (4 GPUs -> 2 GPUs -> 1 GPU
per instance), each revision degrading wall-clock. The user then returned to
the original larger USD price/spec table and asked to choose a final GPU
from it, excluding L4, B300, H100, and H200 specifically (all now off the
table). Among the remaining options, **RTX A5000 (24GB, $0.27/hr, up to 10
GPUs in one instance)** is confirmed as the final choice — see comparison
below. This is the same selection as the earlier (previously voided) v6;
v6 is now un-voided and renumbered v9 as the operative plan.

## Why A5000 over the other remaining options

```text
RTX A5000     $0.27/hr, max10 GPUs -> 111.1 total GPU-hrs -> 11.11h worst-case wall-clock @ 10 GPUs
RTX A4500     $0.25/hr, max 1 GPU  -> 120.0 total GPU-hrs -> 120.00h worst-case (no parallelism)
RTX A4000     $0.25/hr, max 1 GPU  -> 120.0 total GPU-hrs -> 120.00h worst-case (no parallelism)
RTX 2000 Ada  $0.24/hr, max 2 GPUs ->  125.0 total GPU-hrs -> 62.50h worst-case
RTX 4000 Ada  $0.26/hr, max 2 GPUs ->  115.4 total GPU-hrs -> 57.69h worst-case
A40           $0.44/hr, max 7 GPUs ->   68.2 total GPU-hrs ->  9.74h worst-case
RTX A6000     $0.49/hr, max 5 GPUs ->   61.2 total GPU-hrs -> 12.24h worst-case
```

A4500/A4000/2000 Ada/4000 Ada are marginally cheaper per GPU-hour but cap at
1-2 GPUs per instance — reaching 8-10x parallelism would require many
separate concurrent instances, adding real coordination overhead (this
session already hit shrinking per-instance GPU caps twice; more instances is
the wrong lever now that a single 10-GPU option exists). A5000 dominates:
cheapest per-hour option with a real multi-GPU single instance, 24GB is
ample for these small MLP/CNN models, and it removes cross-instance
sharding/sync complexity entirely.

## OPERATIVE PLAN: single instance, 10x RTX A5000 24GB (final)

Reasoning (unchanged from the original v6 analysis):

- The manifest (~246 mostly-tiny 2-layer-MLP jobs + 40 small-CNN visual jobs)
  is bottlenecked by per-step/launch overhead and job count, not per-GPU
  FLOPs, so a single fast/expensive GPU would not meaningfully speed up the
  bulk of runs — confirmed earlier: H100-class pricing buys far fewer total
  GPU-hours for no offsetting per-run speedup.
- 10 GPUs in **one instance** means one filesystem, one job queue, one
  `parallel_runner.py` process managing all 10 physical GPUs — no
  cross-machine sharding/sync needed.
- Spending the entire $30 ceiling caps at 111.1 GPU-hours -> **11.1h
  wall-clock on 10 GPUs**, inside the 12h requirement even in the worst case.
  The actual manifest (~66 GPU-hours expected) finishes in **~6.6h for
  ~$17.82**, leaving real slack.

```text
budget / price = 30 / 0.27  = 111.1 GPU-hours absolute max (11.1h wall-clock @ 10 GPUs)
operational cap = 100 GPU-hours (~$27, 10.0h wall-clock @ 10 GPUs) — 3 USD margin
current manifest estimate (66 GPU-hours) = 6.6h wall-clock, ~$17.82
```

## Using the extra headroom

The 100-GPU-hour operational cap is well above the ~66-GPU-hour manifest.
Rather than leave it idle:

- **M=1024 sensitivity: 6 reruns** (not trimmed to 2).
- **Rollout-regularization ablation (12 runs, dose-controlled): unconditional**,
  not gated on an intermediate result.

Revised manifest: ~256 full runs, still comfortably under the 100 GPU-hour
cap at expected ~40-66 GPU-hours; remaining slack (34-60 GPU-hours) is
contingency for collapsed-cell reruns and wide-CI extra seeds, per the
existing pre-registered rule (never used to change primary-cell training
steps).

## Execution plan (single 10-GPU instance)

```text
Stage 0: smoke tests + mandatory on-GPU benchmark across all 10 GPUs (~0.5-1h).
         Benchmark gates everything after it — see approval gate below.
Stage 1.5: lambda pre-runs (40 short runs) sharded across the 10 GPUs (~0.5h).
           HARD SYNC POINT before the main sweep starts.
Stage 2-3.7: main state sweep, visual sweep, P0.5 controls, encoder swap
             (full 2x2), rollout-reg ablation — pre-sharded across the 10
             GPUs by run-manifest chunks assigned before launch so no GPU
             idles waiting on another (~4-6h).
Stage 4: M=1024 sensitivity (6 reruns) + analysis, run on a few GPUs while
         the rest spin down (~0.5-1h).
```

## Approval gate

Before the paid main sweep: benchmark representative R0/R2 x P0/P1 state and
visual runs at intended per-GPU concurrency on real A5000 hardware;
extrapolate with 25-30% reserve. Projected core completion must fit under
**10 hours wall-clock on 10 GPUs (~100 GPU-hours)**, else drop cells by the
pre-registered priority list from `experiment_design.md`: lambda-sensitivity
extremes -> matched-dim seeds 3->2 -> M=1024 (back down to 2) -> rollout
ablation -> P0.5 E0 cells. Never change training steps of primary cells after
seeing timing — drop whole cells instead.

## Request to user [v9 — FINAL]

```text
SSH access to ONE instance with 10x RTX A5000 24GB GPUs.
Operational budget cap: 100 GPU-hours (~$27 of $30).
Expected actual usage: ~40-66 GPU-hours (~$11-18), 4-6.6h wall-clock.
Worst case if the full cap is used: 10h wall-clock, well inside the 12h requirement.
```

## Guardrails

- `budget_meter.md` with start/stop timestamps for the single instance and
  running cumulative cost in USD.
- Stop the instance as soon as all assigned work finishes; do not leave GPUs
  idle mid-session.
- Operational stop point: $27 spent (~$3 margin).
- No runs outside pre-registered cells except the explicit contingency uses
  (extra seeds for wide-CI cells, matched seed blocks for collapsed cells).
- All 10 GPUs' work is pre-sharded before launch; the only hard sync point is
  lambda-tuning -> main sweep.

---

## Provenance

- v1-v3: original design-phase estimates against a stale/assumed price list
  (A30 $0.41/hr), before real pricing was available.
- v4: corrected for a 12h wall-clock requirement by proposing 6x parallel
  instances (still on the stale price list).
- v5/v7: corrected to a Jarvis-style INR price list (no A30 tier; L4 only),
  2 instances x 4 GPUs = 8 GPUs total.
- v8: L4 instance size corrected down to 2 GPUs max; 4 instances x 2 GPUs
  = 8 GPUs total, same cost/wall-clock target as v7.
- v6/v9: user supplied a different, fuller provider price/spec table in USD
  with per-instance GPU-count limits up to 10, and after L4 access
  constraints kept shrinking (4 -> 2 -> 1 GPU per instance), asked for a
  final choice excluding L4/B300/H100/H200. RTX A5000 selected: cheapest
  remaining per-GPU-hour option with real multi-GPU (10) single-instance
  support.
- v10 (FINAL): user confirmed the actual provisioned hardware is a single
  L4 24GB instance with 7 GPUs, superseding v9's A5000 plan. Same L4 rate
  established in v5/v7/v8 (Rs 41.31/hr per GPU) applies: ~66 GPU-hour
  manifest -> ~9.43h wall-clock, ~$28.55 of $30, well inside the 12h
  requirement. The 246-run manifest and per-stage design logic from v3/v4
  carry over unchanged, including the two items restored to full scope
  (M=1024 at 6 reruns, unconditional rollout-reg ablation).
