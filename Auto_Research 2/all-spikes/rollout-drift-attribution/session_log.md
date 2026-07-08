# Session Log (third-trial)

## 2026-07-04 — Session start (Claude Fable, planning phase)

- Read `voila.md`, `research-philosophy.md`, prior paper (`second-trial/autovoila/main.pdf`),
  and all artifacts of the interrupted prior session under
  `autovoila/previous_experiment/rollout-drift-sigreg-protocol-interaction/`.
- User's starting prompt (verbatim in voila.md): rollout drift in latent world
  models may be caused by regularizer choice AND training protocol, interacting;
  design must be able to honestly produce any outcome (regularizer / protocol /
  interaction / mechanism / environment-dependence / negative result).
- Budget this session: $30 Jarvis Labs hard ceiling (prior session assumed $25).

## Decision: adopt inherited design with four changes (v2)

The inherited design already fixed the major issues from one review round
(GT-normalized excess amplification, per-(R,E) lambda protocol, regularizer
placement pre-specified, P0.5 dense-TF control, collapse rules, middle-region
verdict). Rather than restart, I adopted its core. Changes, with reasons:

1. Visual env seeds 3 -> 5 (E3 carries the most external validity, was most
   under-seeded).
2. Frozen-encoder swap 1 seed x 1 env -> 3 seeds x 2 envs x 2 directions
   (mechanism claim needs interventional evidence beyond seed 0).
3. New 6-run ablation: SIGReg applied to rolled-out latents under BPTT, so
   "regularize the rollout distribution" is a reachable outcome.
4. Budget recomputed for $30: request one A30 24GB, ~66 GPU-hour cap
   (~$27), expected usage 31-60 h. Fallback L4 24GB.

Copied prior `src/` scaffold (already smoke-tested in prior session) into this
spike; it will need extension for: 5-seed E3, both-direction encoder swaps,
rollout-reg ablation flag.

## External feedback

- Codex review #1 (xhigh reasoning, web search allowed) on v2 design completed
  2026-07-04; saved verbatim in `codex_feedback_v2.md` (note: first Codex
  attempt failed on git-repo trust check; rerun with --skip-git-repo-check.
  Output captured via `tail -150`, so the review's opening philosophy section
  was truncated; all ranked findings were captured).

## Codex review #1 findings and v3 responses (all adopted)

1. SIGReg tested outside its favorable regime (rank-3 data in d=16, no
   stationary Gaussian env) -> added OU environment as primary sanity env
   (E0-det kept for bridge cells only) + matched-dim sensitivity on E0/E2 +
   Gaussianity/stationarity diagnostics.
2. Signed log(A_model/A_gt) rewards over-contraction -> primary outcome is now
   integrated probe-space open-loop state error; amplification fidelity uses
   |log ratio| at single primary horizon H=25.
3. Regularizer exposure confounded with protocol (2 vs 9 time points; flattened
   correlated frames) -> stepwise per-time-point regularization averaged over
   time, equal sample counts under all protocols; parity test vs official
   le-wm required; renamed "exact-integral EP variant" not "canonical".
4. Decision rules didn't permit honest equivalence/null -> pre-registered
   same-scale contrasts C_R/C_P/C_I, seed-blocked cluster bootstrap, TOST-style
   equivalence margin log(1.25), collapse treated as outcome not exclusion,
   E2/E3 counted as one dynamics family.
5. Scaffold/preregistration mismatches (3 visual seeds, training-loss lambda
   selection, M=128 pilots, no checkpoints/swap/ablation code, no hard dollar
   cutoff, val used as test) -> logged as execution stop conditions; sweep may
   not start until scaffold matches design and an on-GPU benchmark projects
   core completion under ~50 A30-hours.
6. v2 arithmetic error (14 vs 16 runs) acknowledged; superseded by v3 manifest.
7. Encoder swap upgraded to full 2x2 with retrained diagonals; rollout-reg
   ablation given a 2*lambda dose control and made contingent; M=1024 cut to 2.

## Planning phase closed

Design frozen as experiment_design.md v3. Compute request: one Jarvis Labs
A30 24GB, ~66 GPU-hour cap (~$27 of $30), expected 36-58 h. Next: user
confirms GPU + model switch to Sonnet for execution phase.

## 2026-07-04 — v4: wall-clock constraint, switch to Sonnet

- User: 66 hours wall-clock is too long to wait. Clarified via AskUserQuestion
  that under ~12h is required.
- Since cost = GPU-hours x $/hr regardless of split, and the manifest is
  mostly independent tiny MLP jobs, fix is parallel instances not a stronger
  single GPU (state runs are launch-overhead bound, not FLOP bound; only the
  40 visual/CNN runs would benefit from more compute per-run).
- Revised compute_budget.md to v4: 6x A30 24GB in parallel, same ~66 GPU-hour
  / ~$27 cap, target wall-clock ~10-12h. Execution order: smoke+benchmark on
  all 6 -> lambda tuning sharded across 6 (hard sync point) -> main sweep
  sharded across 6 -> M=1024 + analysis on 1 instance while others spin down.
- User switched model to Sonnet 5 for the execution phase per voila.md.

## 2026-07-04 — v5: real Jarvis pricing, confirmed 2-instance topology

- User supplied actual Jarvis pricing table: no A30 tier; RTX PRO 6000
  Rs179.01/hr, H100 Rs255.15/hr, H200 Rs378.27/hr, L4 Rs41.31/hr; instances
  available in 1/2/4 GPU sizes.
- Recomputed budget in INR using ~95.5 INR/USD (July 2026 spot, x-rates.com /
  bookmyforex.com). $30 ceiling = ~Rs 2865.
- Confirmed stronger GPUs still wrong choice: at H100 pricing $30 buys only
  ~11 GPU-hours total vs the ~66 GPU-hours the manifest needs; workload is
  launch-overhead-bound not FLOP-bound.
- Asked user whether 2 concurrent 4-GPU instances are possible; initial answer
  was "only 1 at a time," then user confirmed "let's take two instances."
  Committed to: 2 instances x 4x L4 each = 8 GPUs total, concurrent.
- v5 compute_budget.md: 66 GPU-hour cap (~Rs2726/~$28.55), wall-clock target
  ~8-8.5h (well under the 12h requirement). Approval gate before paid sweep:
  Stage 0 benchmark must project core completion under 8h/64 GPU-hours on the
  8-GPU topology, else drop cells via the pre-registered priority list.

## 2026-07-04 — v6: final GPU choice from full provider price table

- User supplied a full, different provider price/spec table (USD, per-GPU
  hourly, max-GPUs-per-instance column), superseding v5's Jarvis-INR numbers.
- Selected RTX A5000 24GB at $0.27/hr, up to 10 GPUs in a single instance:
  cheapest per-GPU-hour by a wide margin, 24GB is ample for these small
  models, and a single 10-GPU instance avoids the 2-instance cross-machine
  sync complexity from v5 entirely.
- Arithmetic: spending the full $30 ceiling caps at 111.1 GPU-hours = 11.1h
  wall-clock on 10 GPUs (inside the 12h requirement even in the worst case).
  Current ~66 GPU-hour manifest actually finishes in ~6.6h for ~$17.82.
- Used the resulting slack to restore two previously-trimmed items to full
  scope: M=1024 sensitivity reruns (2 -> 6) and made the rollout-regularization
  ablation (12 runs) unconditional rather than contingent.
- Operational cap set at 100 GPU-hours (~$27, 10h worst-case wall-clock),
  $3 margin. compute_budget.md rewritten as v6 with full provenance of v1-v6
  numbers kept for audit trail.
- Chose this GPU/topology directly per user's "choose from this" — no further
  clarifying question needed, this was a compute-selection judgment call
  within already-agreed constraints ($30 ceiling, <12h wall-clock).

## 2026-07-04 — v7: reverted to real Jarvis Labs pricing (v6 voided)

- User re-shared the original Jarvis Labs table (RTX PRO 6000/H100/H200/L4 in
  INR) and said "sorry these" — the RTX A5000/B200/etc. USD table used for v6
  was not actually the real available option set on this platform.
- v6 marked void in compute_budget.md (kept for audit trail only).
- Reverted to v5 as the operative plan, now labeled v7: two concurrent
  instances, 4x L4 24GB each (8 GPUs total), ~66 GPU-hour cap (~Rs2726/~$28.55
  of $30), target wall-clock ~8-8.5h, well under the 12h requirement.
- This was already the confirmed topology from the earlier "let's take two
  instances and do" exchange, so no new question asked — just restored it as
  canonical and voided the intervening A5000 detour.

## 2026-07-04 — v9 FINAL: RTX A5000 reinstated after L4 access kept shrinking

- L4 instance access degraded twice: first "only 1 at a time" -> "2 instances"
  (v7, 2x4=8 GPUs), then corrected to "2 GPUs per instance max" (v8, 4x2=8
  GPUs), then "only 1 GPU per instance" (would need 8 separate 1-GPU
  instances). Before resolving that, user returned to the earlier, fuller
  USD price/spec table and asked to choose a final GPU excluding L4, B300,
  H100, H200.
- Compared remaining options by total-affordable-GPU-hours vs max-GPUs-per-
  instance: RTX A5000 ($0.27/hr, max 10 GPUs/instance) dominates — cheaper
  alternatives (A4500/A4000 $0.25, 2000 Ada $0.24, 4000 Ada $0.26) cap at 1-2
  GPUs/instance, implying 57-120h worst-case wall-clock unless split across
  many concurrent tiny instances, which repeatedly proved unreliable this
  session.
- Reinstated the RTX A5000 single-10-GPU-instance plan (previously drafted as
  v6, then voided when the user said the Jarvis INR table was the real one).
  Un-voided it and renumbered v9 as FINAL. Numbers unchanged from v6: 100
  GPU-hour operational cap (~$27 of $30), expected ~40-66 GPU-hours / 4-6.6h
  wall-clock, worst-case 10h if the full cap is used — comfortably under the
  12h requirement.
- compute_budget.md cleaned up: removed the duplicate old v6 body text,
  kept a single canonical v9 write-up plus a provenance log of v1-v9.
- This is presented as the final GPU/topology decision; no further GPU-choice
  questions pending. Next step is SSH access to the 10x A5000 instance.

## 2026-07-04 — v10 FINAL: actual provisioned hardware is 7x L4

- User confirmed the instance actually being provisioned is a single L4
  24GB instance with 7 GPUs, superseding the v9 A5000 plan.
- Using the established L4 rate (Rs41.31/hr/GPU, ~95.5 INR/USD): the ~66
  GPU-hour manifest costs ~$28.55 and finishes in ~9.43h wall-clock on 7
  GPUs; full-budget worst case is ~9.91h. Both comfortably inside the 12h
  requirement.
- compute_budget.md rewritten as v10 (final), with v9 kept below as
  superseded/audit-trail. This is believed to be the final GPU/topology
  decision. Next step: SSH access to the 7x L4 instance, then begin
  bringing the code scaffold up to the v3 design's parity requirements.

## 2026-07-04 — Compute instance live: RunPod, 7x L4 confirmed

- SSH access initially given as a Jarvis-style address (root@66.92.198.226)
  which rejected our key; turned out the actual provisioned instance is on
  RunPod (`qgcqpj2crtkw6h-6441171b@ssh.runpod.io`), which worked immediately.
  Note for future reference: RunPod's SSH gateway ignores trailing remote
  commands in non-interactive `ssh host "cmd"` form — it always spawns an
  interactive shell; commands must be piped via stdin (heredoc) instead.
- Confirmed via nvidia-smi: 7x NVIDIA L4 24GB (driver 570.195.03), 128 vCPUs,
  755GB RAM, CUDA 12.8, Python 3.12.3, PyTorch 2.8.0+cu128 already installed.
  Root disk (/) only 30GB, but /workspace is a 196TB network volume with
  108TB free — all code/data/checkpoints will live under /workspace, not /.
  git, tmux, rsync all present.
- This matches the v10 compute plan exactly (7x L4). Ready to start bringing
  the code scaffold up to design parity and run the Stage 0 benchmark.

## 2026-07-04 — Code scaffold brought to v3/v9 design parity (partial)

Given time constraints, prioritized the fixes that block the Stage 0
approval-gate benchmark (correctness-critical, per Codex review):

Done and smoke-tested locally (CPU, all env/reg/protocol combos pass):
- regularizers.py: rewritten to stepwise application (per-timestep stat
  averaged over T) instead of flattening correlated frames across time;
  removed inconsistent subsampling; docstring clarifies "exact-integral
  Epps-Pulley variant" naming (not "canonical"/official quadrature).
- train.py: compute_prediction_loss now returns (B,T,D) encoded latents
  (not flattened) plus optional (B,H,D) rollout predictions; added
  --reg-target {encoded, encoded-and-rollout} for the rollout-reg ablation;
  added --load-encoder-from to freeze an encoder from a checkpoint and train
  only dynamics (encoder-swap mechanism probe).
- envs.py: added "ou" (stationary AR(1)/OU process, same 0.7/0.8/0.9
  contraction as stable-linear but calibrated additive noise so the marginal
  is stationary isotropic N(0,I) at every t -- SIGReg's favorable regime)
  and "stable-linear-det" (deterministic, for bridge cells only).
- select_lambdas.py: selection criterion changed from final training-batch
  loss to validation one-step MSE (open_loop_mse_1 on the val split),
  subject to non-collapse -- matches the pre-registered lambda protocol.
- parallel_runner.py: added --hard-kill-after-hours, a real dollar cutoff
  that SIGTERMs then SIGKILLs any still-running job past the deadline via a
  watchdog thread (previously only stopped launching new jobs).
- run_sweep.py: rebuilt manifest to bridge (stable-linear-det, R0/R1, P0/P1,
  5 seeds) + main factorial (ou/lorenz/pointmass-state/pointmass x
  none/moment/sigreg x 1step/bptt, 5 seeds each including visual, up from 3)
  + P0.5 control (ou/pointmass x moment/sigreg x hstep-tf, 5 seeds).
  Dry-run --quick produces 160 commands as expected (20+120+20).

Smoke-tested locally on CPU: all 4 state envs x 3 regs x 2 protocols, visual
pointmass, hstep-tf, the encoded-and-rollout ablation flag, and the
load-encoder-from swap flag against a real checkpoint. All pass.

Deferred (not yet implemented, tracked for before the main paid sweep):
matched-latent-dimension sensitivity arm, lambda-sensitivity sweep script,
full 2x2 encoder-swap orchestration script (train.py supports it; the
sweep/orchestration script that runs both directions x envs x seeds is not
yet written), M=1024 sensitivity script, analyze.py rewrite (integrated
probe-space primary metric, C_R/C_P/C_I contrasts, seed-blocked bootstrap,
Gaussianity/stationarity diagnostics). These are needed before the full
paid sweep but not before the Stage 0 smoke test + benchmark, which only
needs train.py/regularizers.py/envs.py to be correct.

Remote instance confirmed live (7x L4, RunPod, PyTorch 2.8.0+cu128
pre-installed). Next: sync code to /workspace, run Stage 0 smoke test +
on-GPU benchmark there to validate the approval gate before proceeding to
write the remaining deferred pieces.

## 2026-07-04 — Stage 0 benchmark complete: APPROVAL GATE PASSED

Code synced to /workspace/rollout-drift-attribution on the 7x L4 instance via
base64-over-SSH-stdin (rsync/scp unavailable: no rsync locally, RunPod's SSH
gateway doesn't expose the SFTP subsystem or accept non-interactive remote
commands -- only interactive shell with piped stdin works). md5sum-verified
transfer integrity.

Benchmarked representative cells on real GPU hardware (single L4, 2000
train steps for state envs, 1000 for visual, extrapolated to full step
counts):

```
state none/1step:   0.00459 s/step
state moment/bptt:  0.02047 s/step
state sigreg/bptt:  0.07086 s/step  (confirms Codex's quadratic-SIGReg concern)
visual none/1step:  0.07517 s/step
visual sigreg/bptt: 0.14065 s/step
```

Extrapolated core manifest (160 runs: 20 bridge + 120 main factorial + 20
P0.5 controls, using an additive cost model fit to the 5 measured points):

- State total: ~18.72 GPU-hours (1-GPU-equivalent sum)
- Visual total: ~17.99 GPU-hours
- **Core manifest: ~36.7 GPU-hours -> ~5.24h wall-clock on 7 GPUs, ~$15.88**

Against the ~66 GPU-hour / ~9.4h approval-gate cap from compute_budget.md
v10, this leaves **~29.3 GPU-hours (~$12.67) of headroom** for the stages
not yet built: lambda-tuning pre-runs (cheap, 40 short runs), matched-dim
sensitivity, encoder-swap orchestration, rollout-reg-ablation orchestration,
M=1024 sensitivity. No cells need to be dropped from the pre-registered
priority list -- the design fits.

**APPROVAL GATE: PASSED.** No training-step reductions or cell drops needed.
Reported benchmark results to user; next steps are (1) finish the deferred
orchestration scripts (matched-dim, swap, ablation, M1024) and the
analyze.py rewrite (integrated probe-space primary metric, C_R/C_P/C_I
contrasts, seed-blocked bootstrap), all well within remaining budget/time,
then (2) launch the lambda-tuning stage (hard sync point) followed by the
full sweep across all 7 GPUs.

## 2026-07-04 — All deferred scripts finished and verified on remote

Completed the remaining pieces from the last checkpoint:

- **run_sweep.py**: added --stage {matched-dim, lambda-sensitivity, m1024, all}.
  matched-dim (24 runs: {ou,pointmass-state} x {none,sigreg} x {1step,bptt}
  x 3 seeds at intrinsic dim 3/2). lambda-sensitivity (24 runs: R2 low/high
  0.3x/3x multiples on {ou,pointmass-state} x {1step,bptt} x 3 seeds;
  selected-lambda point reused from main sweep, not rerun). m1024 (2 reruns
  of the highest-leverage R2/bptt cells at 1024 projections).
- **run_swap.py** (new): full 2x2 encoder-swap orchestration -- trains 12
  source models (2 envs x 2 protocols x 3 seeds, checkpointed), then all 4
  (encoder-source-protocol x dynamics-retrain-protocol) combinations
  including the two retrained diagonals, per Codex's requirement that the
  intervention be compared against retrained controls, not the original
  jointly-trained models.
- **run_ablation.py** (new): rollout-regularization ablation with a
  dose-matched control (encoder+rollout at lambda vs encoder-only at 2x
  lambda), per Codex's note that a naive comparison would confound placement
  with total regularization mass.
- **metrics.py**: added `_probe_open_loop_error`, the pre-registered primary
  metric -- integrated (trapezoidal) probe-space state error along the
  model's own free-running rollout, not gameable by over-contraction the way
  the signed amplification ratio was. Also added `abs_log_excess_amp_25` as
  the non-gameable amplification-fidelity secondary metric.
- **analyze.py**: rewritten with PRIMARY_METRIC = integrated_probe_open_loop_error,
  pre-registered C_R/C_P/C_I contrasts with seed-blocked bootstrap CIs
  (C_I is a same-length-matched proxy given time constraints -- true
  interaction contrast needs seed-paired 2x2 differencing, noted as a known
  simplification), a collapse report treating collapse as a treatment
  outcome (probability by condition, drift conditional on viability) rather
  than silent exclusion, and eta^2 factorial table kept as descriptive-only.

Fixed one bug caught during local smoke-testing: `np.trapz` was removed in
numpy 2.x (renamed `np.trapezoid`); added a compatibility shim.

All new stages smoke-tested locally end-to-end (matched-dim, ablation arms,
swap command generation, and analyze.py against real (tiny) run outputs --
contrasts.json and collapse_report.csv both produce sensible numbers).

Synced updated code to /workspace/rollout-drift-attribution on the remote
7x L4 instance (same base64-over-SSH-stdin method as before, atomic
swap via a `_new`/`_old` rename so the running instance was never left in a
half-updated state). md5sum-verified three key files match exactly between
local and remote. Confirmed on remote: py_compile passes, dry-run stage
counts match local (core=160, matched-dim=24).

**All deferred scripts are now complete and verified.** Next: launch the
lambda-tuning pre-run stage (hard sync point per compute_budget.md v10),
then the full sweep across all 7 GPUs.

## 2026-07-04 — Lambda-pilot stage launched (with a caught GPU-assignment bug)

Added `--stage lambda-pilot` to run_sweep.py (45 commands: 9 (env,reg)
combos x 5-point lambda grid, seed 0, reduced step count 8000/6000
state/visual since pilots only need relative ranking by validation MSE, not
full convergence -- estimated ~3.85 GPU-hours / ~33min wall-clock on 7 GPUs).

**Bug caught before it wasted GPU time**: parallel_runner.py had no
per-job GPU assignment logic -- my first launch attempt tried to set
CUDA_VISIBLE_DEVICES via a shell for-loop before starting the runner, which
is wrong (it would've just left every job on whatever GPU the loop last set,
i.e. all 45 jobs contending for GPU 6 while 0-5 sat idle). Caught via
nvidia-smi showing 0 utilization on 6 of 7 GPUs. Fixed properly: added a
thread-safe GPU pool (queue.Queue) to parallel_runner.py -- each job
acquires a free GPU id, sets CUDA_VISIBLE_DEVICES for just that subprocess,
and releases the id on completion. Verified locally with a 4-job/2-GPU test
(correct alternation, no collisions) before resyncing to remote (md5
verified: 7827e82918c1e811773f6a9125ee110b).

Second false start: after fixing the runner, I pre-stripped the `python3`
interpreter prefix from the generated commands file (thinking parallel_runner
needed bare `-m src.train ...`), which broke every job immediately
(FileNotFoundError: '-m', since Popen has no shell to resolve that as an
executable). Caught within ~20 seconds via the runner log. Fixed by not
stripping anything -- the commands file already contains the full
interpreter path parallel_runner needs.

Confirmed working: relaunched in a detached tmux session (`lambda_pilot`),
nvidia-smi shows all 7 GPUs active (282 MiB, 0-2% util each, consistent with
7 concurrent tiny-model training jobs having just started). Running with
--hard-kill-after-hours 1.5 as a safety margin (expected ~0.55h).

Per user instruction, proceeding through the remaining stages (select
lambdas, main sweep, matched-dim, lambda-sensitivity, swap, ablation,
m1024, analysis) autonomously without further check-ins, checking back only
if something requires a judgment call outside pre-registered scope.

## 2026-07-04 — v4 REPLAN: 8-hour hard wall-clock deadline (user instruction)

User set a hard requirement mid-run: everything must finish within 8 hours
wall-clock on the 7 GPUs. Empirical concurrent-load rates (state moment
0.059-0.070 s/step, sigreg 0.047 s/step) were ~10x slower than the isolated
single-GPU benchmark, making the v3 manifest (~90-100 GPU-hours realistic)
infeasible. Root cause identified: CPU thread thrashing -- the isolated
benchmark showed real 9s / user 5m17s, i.e. each PyTorch process spawns ~35
threads, so 7 concurrent jobs oversubscribe the 128 vCPUs ~2x. 

v4 replan decisions (all logged before seeing any outcome data -- only
timing data informed these):
1. OMP_NUM_THREADS=12 / MKL_NUM_THREADS=12 for every job (the thread fix).
2. Training steps reduced UNIFORMLY: state 20k->10k, visual 15k->8k. Uniform
   dose change across all conditions cannot bias between conditions, unlike
   selective cell drops. This supersedes the earlier pre-registered
   "never change steps" rule, which assumed a budget that no longer exists;
   the alternative (dropping half the factorial) would cost more inference.
3. Visual seeds 5->3 (reverted to v1 scope; visual is the costliest tier).
4. Dropped per pre-registered priority list: lambda-sensitivity (24 runs),
   matched-dim (24), M=1024 (2), rollout-reg ablation (12).
5. Encoder swap kept but restructured: the core sweep's ou/sigreg cells
   (seeds 0-2, both protocols) now save checkpoints and double as phase-A
   swap sources, eliminating 12 dedicated source runs; only the 12 phase-B
   dynamics-retrains on ou remain.
6. Visual pilot cells reduced 6000->4000 steps (same relative-ranking logic).

Worst-case arithmetic with NO thread-fix credit: bridge 3.5 + main-state
18.75 + controls 2.8 + visual 10.7 + swap 3.3 = ~39 GPU-hours = ~5.6h
wall-clock on 7 GPUs, plus ~0.5-1h pilot remainder = ~6.5h, inside the 8h
deadline with margin. With the thread fix, likely 3-4h.

Execution: single chained driver.sh in tmux (pilot remainder -> select_lambdas
-> core sweep 144 runs -> swap phase B 12 runs -> analysis), stage-level
hard-kill guards (1.0h pilot / 5.5h core / 1.0h swap). Killed the old
un-thread-capped pilot (31/45 cells already complete, kept), requeued the
remaining 14 via remote_inventory.py. Confirmed all 7 GPUs active after
launch; GPUs 4-6 show 2.6GB allocations = visual cells now measurable.

Also logged: core manifest is now 144 runs (was 160): 20 bridge + 108 main
(90 state + 18 visual) + 16 P0.5 controls (10 ou + 6 visual).

## 2026-07-04 10:36 UTC — Core sweep launched; thread fix confirmed (~3x)

Stages 1-2 complete. Thread-fixed concurrent rates (vs pre-fix):
- state sigreg 1step: 0.0162 s/step (was 0.047 -- 2.9x faster)
- visual moment 1step: 0.064 s/step; visual sigreg 1step: 0.079 s/step
- GPU utilization 96-98% across all 7 (was 0-16% pre-fix).

Selected lambdas (validation one-step MSE, non-collapse constraint):
  stable-linear-det:moment 0.01; ou:moment 0.03; ou:sigreg 1.0;
  lorenz:moment 0.1; lorenz:sigreg 1.0; pointmass-state:moment 0.03;
  pointmass-state:sigreg 0.03; pointmass:moment 0.1; pointmass:sigreg 0.01.

NOTABLE: pointmass-state:sigreg and pointmass:sigreg were ALL-COLLAPSED --
every lambda in the grid collapsed under sigreg/1-step on the point-mass
envs at pilot step counts (val MSE ~1e-8/1e-9, consistent with trivial
predictions). select_lambdas fell back to best-val-MSE as pre-registered.
This is itself a result signal (collapse is a treatment outcome in the
analysis); the sweep runs those cells regardless and the collapse report
will quantify it. To watch during analysis: whether sigreg cells on
point-mass collapse at full 10k/8k steps too.

Core sweep (144 runs) started ~10:21 UTC; 42/144 complete at 10:36 (~15
min). Early completions are cheap state cells; visual/bptt cells are
slower. Rough projection: core done ~11:30-12:00 UTC, swap +30 min,
analysis minutes. Full pipeline ETA ~12:30 UTC vs 18:01 deadline -- large
margin.

## 2026-07-04 12:40 UTC — PIPELINE COMPLETE. Results pulled and interpreted.

All stages done: 45 lambda pilots, 144 core runs, 12 swap runs, 0 failures.
Wall-clock ~2.5h of the 8h deadline. Results in results/ locally.

### Headline empirical pattern (from collapse_report + condition_summary)

1. NO-REGULARIZER (R0) COLLAPSED 100% -- every cell, every env, both
   protocols. Anti-collapse regularization is necessary in this regime.
2. SIGREG (exact-integral EP variant) collapsed 100% on ou, pointmass-state,
   pointmass (all protocols; hstep-tf on ou 80%) at its val-MSE-selected
   lambda. It survived ONLY on Lorenz (0% collapse). Note: on OU swap runs
   probe R2 was 0.67-0.79 (state still decodable) -- the flag fires on
   effective-dimension/rank, i.e. dimensional collapse, not full
   informational collapse. Wording in the paper must reflect this.
3. MOMENT PROXY was the most collapse-robust (0% most cells; 60% ou/bptt,
   80% stable-linear-det/bptt).
4. LORENZ (only env where R1 and R2 both fully viable): protocol dominates
   -- moment: 9249 -> 123 (1step -> bptt); sigreg: 573 -> 134. C_P over R2:
   -1.20 log units, bootstrap CI [-1.75, -0.65], excludes 0, far beyond the
   log(1.25) margin. AND a strong descriptive RxP interaction: sigreg cuts
   1-step drift 16x vs moment (9249 -> 573) while under BPTT the regularizer
   is irrelevant (123 vs 134). The regularizer matters only under the bad
   protocol -- directly the "neither can be studied in isolation" pattern
   the user hypothesized.
5. POINT-MASS state+visual, conditional on viability (moment only): BPTT
   reduces drift (4.61 -> 2.70 state; 37.5 -> 8.8 visual).
6. SWAP 2x2 (ou, sigreg): primary metric flat ~80+-5 across all four
   encoder-source x dynamics-protocol cells; all cells rank-collapsed;
   OU sits near its noise floor (~76 even for best cells) -- OU turned out
   uninformative for drift attribution (though informative for collapse).
7. BRIDGE caveat: stable-linear-det moment/bptt collapsed 4/5 seeds, so the
   prior paper's cleanest environment did not reproduce cleanly at 10k
   steps with the newly selected lambda.

### Formal pre-registered verdicts

- C_R (regularizer fixes drift): UNEVALUABLE -- R0 collapsed everywhere, so
  no non-collapsed R2-R0 seed pairs exist in any env.
- C_P (protocol): SUPPORTED where evaluable (Lorenz, CI excludes 0 beyond
  margin; point-mass descriptively consistent under moment).
- C_I: formally unevaluable; strong descriptive interaction on Lorenz.
- Encoder-geometry via swap: null/unevaluable (all swap cells collapsed;
  flat primary metric).
- OVERALL: MIXED verdict with an honest headline: in this regime the
  regularizer's first-order causal role is anti-collapse, not drift
  control; differential collapse largely pre-empts the drift attribution
  question; conditional on survival, protocol dominates drift; and on the
  one environment where both regularizers survive, the regularizer matters
  only under 1-step training (interaction).

### Limitations to carry into the paper

- lambda-sensitivity arm was dropped in the 8h replan: sigreg's collapse
  rates could be lambda artifacts (ou:sigreg lambda=1.0 was selected on a
  suspiciously tiny 4e-7 val MSE). Must be a stated limitation.
- Steps reduced uniformly 20k->10k / 15k->8k (logged, uniform, unbiased
  between conditions, but absolute collapse rates may be step-dependent).
- Collapse flag is rank/eff-dim based on OU (probe R2 remained > 0.5).
- The prior-paper bridge did not reproduce cleanly (moment/bptt collapse).

Next: reviewer-gate self-review, Codex feedback on findings, then paper.

## 2026-07-04 — Paper complete: compiled and verified

Post-results phase completed in sequence:
1. Codex results-stage review (xhigh) returned 5 ranked objections + analyses;
   saved verbatim in codex_results_review.md. All findings adopted:
   - Title/framing changed to censoring-by-collapse ("When Collapse Censors
     Drift"); "protocol dominates" and "SIGReg fails" explicitly disclaimed.
   - 16x corrected to geometric 6.2x; C_P renamed C_{P|R2}/C_{P|R1}; C_R/C_P/
     C_I all reported as formally censored; interaction labeled exploratory.
   - Added exact paired sign-flip tests (p=0.0625 both protocol effects, all
     5 seeds consistent), leave-one-seed-out ranges, horizon-resolved
     tradeoff (BPTT worse at H=1, far better at H=50).
   - Collapse-mode decomposition (variance/dimensional/informational:
     R2 = 19/5/9 of 33) + eff-dim-vs-probe-R2 figure.
   - EP stationary-point theory VERIFIED: value 1+1/sqrt(3)-sqrt(2) = 0.1631
     at gamma=1/2; median final reg loss of the 33 collapsed R2 runs =
     0.1631 exactly. Bounded-vs-unbounded penalty contrast with moment
     (=d=16 at collapse) presented as candidate mechanism + Appendix D
     derivation. This became claim C3.
   - Lambda-frontier audit figure (fig7): R2 survivors are boundary-only
     (OU/Lorenz) or empty (point-mass); collapsed pilots show spuriously
     tiny val MSE -- documented why validation error alone cannot select
     weights.
   - Three prereg violations (val-as-test, per-run eval windows, shared RNG)
     disclosed in a full deviation ledger (Appendix B).
2. Figures: 7 publication figures generated from logged data using the
   validated dataviz palette (fixed identity encoding, CVD-safe, markers +
   hatching so identity is never color-alone); visually verified.
3. Paper: main.tex (7 content pages, under the 8-page CAISC limit),
   appendix.tex (prereg criteria, deviation ledger, lambda audit,
   stationary-point derivation, extra results, compute account, human-AI
   session flow incl. verbatim human brief and error log), checklists.tex
   (both mandatory CAISC checklists completed -- AI Involvement: B/D/D/D
   with iteration ratings; Reproducibility: all justified).
4. Compiled with latexmk: 16 pages total, zero undefined references/
   citations, bibliography renders, figures embed correctly (visually
   checked pages 4 and 10). Final artifact:
   all-spikes/rollout-drift-attribution/paper/main.pdf
