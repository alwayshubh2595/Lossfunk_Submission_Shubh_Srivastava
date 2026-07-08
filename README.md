# AUTORESEARCH_SUBMISSION

Two end-to-end autonomous research runs on the causes of **rollout drift in latent world models**, produced with **Autovoila** — Lossfunk's autonomous AI/ML research agent prompt for Claude Code ([github.com/paraschopra/autovoila](https://github.com/paraschopra/autovoila)). Each run went from a research prompt through experiment design, implementation, execution, and paper writing, without a human writing code or prose directly.

## Contents

- `Paper 1.pdf` — output of `Auto_Research 1`
- `Paper 2.pdf` — output of `Auto_Research 2`
- `LOSSFUNK.pdf` — presentation deck for this submission
- `Auto_Research 1/AutoResearch 1/` — first run: the Autovoila prompt files + generated spike (`all-spikes/rollout-drift-causal/`)
- `Auto_Research 2/autovoila/` + `Auto_Research 2/all-spikes/` — second run: the modified Autovoila prompt + generated spike (`all-spikes/rollout-drift-attribution/`)

**Presentation:** [Google Slides](https://docs.google.com/presentation/d/14EAu93B6xOYcvSHlQ3nRFcwFitSCbiRUY2xR4BCFWgM/edit?usp=sharing)

## Paper 1 — *Training Protocol Dominates Rollout Drift in Latent World Models: A Causal Attribution Study*

Produced by `Auto_Research 1/AutoResearch 1`. Starting from a blank Autovoila session (topic left open per the standard `voila.md` phases: exploration → question sharpening → locked experiment brief → execution → paper), the agent converged on a controlled 2×2×2 factorial study across three synthetic environments (stable-linear, 2D spiral, Lorenz), attributing latent-rollout amplification (A_k) to encoder regularization, dynamics model capacity, and training protocol (1-step teacher forcing vs. 5-step BPTT).

**Finding:** training protocol (not the marginal regularizer) is the dominant factor in most environments — but a sanity check revealed all 1-step teacher-forcing models produced geometrically unstable latents even on a trivially stable linear system, meaning the regularizer used ("SIGReg") was implemented as a moment-matching proxy rather than its canonical form.

Deliverables: `all-spikes/rollout-drift-causal/preliminary_experiment/` (code, results, REPORT.md) and `all-spikes/rollout-drift-causal/paper/` (`main.tex` / `main.pdf`).

## Paper 2 — *When Collapse Censors Drift: Differential Encoder Collapse Pre-empts Regularizer–Protocol Attribution in Latent World Models*

Produced by `Auto_Research 2`, seeded explicitly with Paper 1 as prior work. This run pre-registered a follow-up study using the **canonical sliced Epps–Pulley statistic** (rather than the moment-matching proxy from Paper 1) in a 3-regularizer × 2-protocol × 5-environment factorial (201 runs), with ground-truth-normalized metrics, collapse-as-outcome tracking, and an interventional encoder-swap probe.

**Finding:** the planned regularizer/protocol attribution was largely confounded by *differential encoder collapse* — unregularized runs collapsed entirely, and the faithful Epps–Pulley regularizer collapsed on 3 of 4 non-bridge environments at standard-validation-selected weights, while the crude moment proxy from Paper 1 survived almost everywhere (masking rather than solving the instability). The paper identifies a candidate mechanism (collapse as a stationary point of the Epps–Pulley loss matching observed collapsed-run loss to four digits) and argues collapse must be reported as a primary outcome, analogous to truncation-by-death, whenever marginal regularizers are compared.

Deliverables: `all-spikes/rollout-drift-attribution/` (`src/`, `results/`, `compute_budget.md`, `experiment_design.md`, `session_log.md`, `literature_notes.md`, Codex review notes) and `all-spikes/rollout-drift-attribution/paper/` (`main.tex`, `appendix.tex`, `checklists.tex`, `main.pdf`).

## Input files & differences between the two runs

**Auto_Research 1** used the *stock* Autovoila prompt (`voila.md` follows Autovoila's default five-phase process: Setup → Exploration → Question Sharpening → locked Experiment Brief → Execution → Paper Writing, entirely driven by the agent surfacing and scoring candidate claims itself). No prior artifact was supplied — the agent started from scratch and chose its own research question.

**Auto_Research 2** used a **customized `voila.md`** that replaced the generic phase-by-phase discovery process with a fixed, pre-specified session:
- The research question was given directly by the user up front (rollout drift caused by regularizer × training protocol interaction), instead of being discovered by the agent in Phase 1–2.
- The agent was pointed at **Paper 1's output** (`previous_experiment/`, including its `README_EXPERIMENTS.md`) as reference material to adopt, modify, or reject — rather than starting cold.
- A **hard compute budget** ($30 on Jarvis Labs) was specified up front, requiring the agent to size GPU/hour usage rigorously *before* design was finalized, rather than estimating cost after the fact.
- It removed the rigid Phase 0–5 structure/templates (claim-scoring rubric, sharpened-question template, fixed experiment-brief/report skeleton) in favor of a shorter, looser instruction set that leans on continuous Codex feedback and user judgment calls instead of fixed checklists.
- It added an explicit model-switch protocol: plan on Claude Fable, then prompt the user to switch to Claude Sonnet once execution (code/training/analysis) begins, to conserve credits.
- It explicitly asked the agent to keep clarifying with the user until fully confident, but avoid pinging on minor implementation choices — logging those decisions instead.

**Code differences (`rollout-drift-causal` vs. `rollout-drift-attribution`):**
- Auto_Research 1's spike keeps all code inside `preliminary_experiment/code/` (11 files: environments, models, training, metrics, experiment runner, analysis), following the standard Autovoila deliverable layout (`REPORT.md`, `figures/`, `tables/`, `results_raw/`).
- Auto_Research 2's spike uses a flatter, more elaborate layout at the project root: `src/` (models, envs, regularizers, training, metrics, lambda-selection/grid search, ablation/sweep/swap runners, a parallel runner), plus top-level `driver.sh`, `remote_inventory.py`, `gen_swap_b.py`, `compute_budget.md`, `experiment_design.md`, `literature_notes.md`, `session_log.md`, and Codex review logs (`codex_feedback_v2.md`, `codex_results_review.md`). This reflects the larger factorial design (201 runs across 3 regularizers × 2 protocols × 5 environments, plus a lambda-selection stage and an encoder-swap intervention) and the requirement to run real remote GPU jobs within a fixed cost ceiling, which the first run did not need.

## System

Both runs use **Autovoila** ([github.com/paraschopra/autovoila](https://github.com/paraschopra/autovoila)) — Lossfunk's Claude Code prompt for autonomous ML research: an agent that reads a research philosophy, designs and locks an experiment brief, executes it, gets Codex feedback at checkpoints, and writes up a paper in the provided LaTeX format.
