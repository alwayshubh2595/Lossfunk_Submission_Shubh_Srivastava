#!/bin/bash
# v4 replan driver: finishes the whole experiment pipeline within the 8h deadline.
# Runs inside tmux on the RunPod instance. All stages chained; hard-kill guards throughout.
set -x
cd /workspace/rollout-drift-attribution

# CPU-thread fix: 7 concurrent jobs on 128 vCPUs were thrashing (each torch
# process spawned ~35 threads). Cap every child at 12 threads.
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

DEADLINE_HOURS_PILOT=1.0
DEADLINE_HOURS_CORE=5.5
DEADLINE_HOURS_SWAP=1.0

# ---- Stage 1: finish remaining lambda pilots ----
python3 remote_inventory.py
python3 -m src.parallel_runner --commands-file lambda_pilot_remaining.txt \
  --jobs 7 --num-gpus 7 --log-dir lambda_pilot_logs2 \
  --status-file lambda_pilot_status2.jsonl \
  --hard-kill-after-hours $DEADLINE_HOURS_PILOT > lambda_pilot_runner2.log 2>&1
echo STAGE1_PILOT_DONE

# ---- Stage 2: select lambdas (hard sync point) ----
python3 -m src.select_lambdas --runs-dir lambda_runs --out lambda_map.json \
  --report lambda_selection_report.json > select_lambdas.log 2>&1
echo STAGE2_SELECT_DONE

# ---- Stage 3: core sweep (v4: 10k/8k steps, 3 visual seeds, checkpoints on ou sigreg) ----
python3 -m src.run_sweep --dry-run --stage core --lambda-json lambda_map.json \
  --output-dir runs > core_commands.txt 2>&1
wc -l core_commands.txt
python3 -m src.parallel_runner --commands-file core_commands.txt \
  --jobs 7 --num-gpus 7 --log-dir core_logs --status-file core_status.jsonl \
  --hard-kill-after-hours $DEADLINE_HOURS_CORE > core_runner.log 2>&1
echo STAGE3_CORE_DONE

# ---- Stage 4: encoder-swap phase B only (sources = ou sigreg checkpoints from core) ----
python3 gen_swap_b.py > swap_commands.txt 2>&1
wc -l swap_commands.txt
python3 -m src.parallel_runner --commands-file swap_commands.txt \
  --jobs 7 --num-gpus 7 --log-dir swap_logs --status-file swap_status.jsonl \
  --hard-kill-after-hours $DEADLINE_HOURS_SWAP > swap_runner.log 2>&1
echo STAGE4_SWAP_DONE

# ---- Stage 5: analysis (CPU, cheap) ----
python3 -m src.analyze --runs-dir runs --out-dir analysis > analyze.log 2>&1
echo STAGE5_ANALYSIS_DONE
echo ALL_STAGES_DONE
