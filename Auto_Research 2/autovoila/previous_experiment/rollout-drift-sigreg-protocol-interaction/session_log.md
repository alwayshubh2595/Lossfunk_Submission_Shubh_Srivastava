# Session Log

## Setup

- Started from `voila.md` on 2026-07-03.
- Research philosophy read from `research-philosophy.md`.
- Prior session artifact read from `main.pdf`.
- Project directory: `all-spikes/rollout-drift-sigreg-protocol-interaction`.

## Human Starting Prompt

The prompt embedded in `voila.md` states that rollout drift in latent world models may be caused by both marginal regularizer choice and training protocol, and that the two may interact. The requested research stance is not to design experiments that confirm a preferred answer, but to distinguish real possibilities: regularizer effect, training-protocol effect, interaction, mechanism, environment dependency, or a negative result.

The same prompt sets a hard compute ceiling of 25 USD on Jarvis Labs, asks for a rigorous compute calculation before requesting access, and asks that the design explicitly state which regularizers are used and whether canonical formulations or approximations are used.

## Previous Paper Summary

`main.pdf` reported a 2 x 2 x 2 factorial study over synthetic environments. Its headline result was that training protocol dominated rollout drift in two of three environments, with 5-step BPTT correcting drift that appeared under 1-step teacher forcing. The most important caveat for the next study is that the "SIGReg" condition was a moment-matching proxy:

```text
L_reg = ||mu_batch||^2 + ||Sigma_batch - I||_F^2
```

The prior artifact also had no pixel encoder, no action-conditioned task, and no canonical characteristic-function/Epps-Pulley SIGReg implementation.

## Local Resource Snapshot

- CPU: Intel Core Ultra 5 125H, 18 logical CPUs.
- RAM: 15 GiB total, about 6.6 GiB available at inspection.
- Local GPU: Intel Arc integrated graphics; no NVIDIA CUDA GPU detected locally.
- Python: 3.13.5.
- PyTorch: 2.12.1+cu130 installed locally, but `torch.cuda.is_available()` is false.
- PDF tooling: `pdftotext`, `pdfinfo`, `pdflatex`, and `latexmk` are available.
- Claude Code and Codex CLI are installed locally.

## External Review Attempts

- Claude Code review attempt started with a high-effort prompt asking it to read `research-philosophy.md` and `voila.md` and critique the follow-up design. It did not return output within about two minutes, so it was interrupted to avoid blocking progress.
- A second shorter Claude Code review attempt was run with a 75 second timeout after `experiment_design.md` and `compute_budget.md` were written. It also returned no output before timeout. For this planning step, Claude feedback is logged as unavailable rather than silently omitted.
- User asked why Claude could not be accessed. Diagnostics showed the `claude` CLI was installed, but sandboxed model calls timed out. An escalated minimal call worked after raising the per-call budget cap from 0.05 USD to 0.25 USD.
- User requested Claude Fable specifically for feedback. Claude Fable was invoked with `--model fable`, `--effort medium`, and a 2.00 USD cap. The review returned successfully and was saved in `claude_fable_feedback.md`.

## Literature/Context Notes

- LeWorldModel uses a next-embedding prediction objective plus SIGReg over random projections of latent embeddings.
- Its SIGReg regularizer is based on testing one-dimensional random projections for Gaussianity using an Epps-Pulley/characteristic-function statistic, not merely matching first and second moments.
- Fast LeWorldModel is directly relevant because it argues that autoregressive one-step LeWM rollout accumulates error and replaces it with dense action-prefix prediction. This supports treating the training/inference interface as a serious alternative explanation, but adding that architecture to the main factorial would confound training protocol with model class. It is therefore not in the main factorial for this spike.
- Jarvis Labs pricing observed on 2026-07-03: A30 is 0.41 USD/hr, L4 is 0.44 USD/hr, A100 40GB is 0.89 USD/hr, A100 80GB is 1.49 USD/hr. The selected budget request uses one A30 if available and one L4 as fallback.

## Current Decision

Commit to a single follow-up design:

```text
regularizer in {none, moment proxy, canonical sliced Epps-Pulley SIGReg}
main training protocol in {1-step teacher forcing, H-step closed-loop BPTT}
control protocol on selected cells: H-step teacher forcing
environment in {stable-linear state, Lorenz/chaotic state, state point mass, visual action-conditioned point mass}
```

The design deliberately excludes Fast-LeWorldModel/prefix prediction from the main factorial because it changes the architecture/interface as well as the training protocol. It will be mentioned as related work and a possible second-stage experiment only if the main factorial points there.

## Changes After Claude Fable Review

Implemented the following before GPU execution:

- Added ground-truth-normalized excess amplification as the primary drift metric.
- Added `pointmass-state` to separate action conditioning from pixel/CNN encoding.
- Added `hstep-tf` as the P0.5 dense teacher-forced control.
- Fixed regularizer placement under BPTT: R1/R2 apply only to encoded latents in the main experiment, not rolled-out predictions.
- Added a pre-registered lambda protocol per `(regularizer, environment)`, selected on validation one-step prediction MSE subject to non-collapse, never on drift.
- Added a middle-region verdict for cases where canonical SIGReg helps but does not match BPTT.

## Code/Smoke Verification

Created an executable PyTorch scaffold under `src/`:

- `envs.py`: stable-linear, Lorenz, pointmass-state, and visual pointmass datasets.
- `regularizers.py`: moment proxy and canonical sliced Epps-Pulley SIGReg.
- `train.py`: single-run trainer for `1step`, `hstep-tf`, and `bptt`.
- `metrics.py`: latent amplification, probe-space excess amplification, open-loop error, collapse/probe diagnostics.
- `run_sweep.py`: dry-run/full sweep command generator.
- `analyze.py`: CSV summaries and simple factorial effect-size table.

Verification completed locally:

```text
python3 -m py_compile src/*.py
stable-linear SIGReg/BPTT smoke run passed
visual pointmass moment/hstep-tf smoke run passed
run_sweep --dry-run --quick passed
analyze.py loaded smoke metrics and wrote CSV summaries
```
