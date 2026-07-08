# Experiment Runner

This folder contains the first executable scaffold for the rollout-drift follow-up.

## One-Run Smoke Test

The directory names contain hyphens, so run commands from this spike directory:

```bash
cd all-spikes/rollout-drift-sigreg-protocol-interaction
python3 -m src.train --env stable-linear --regularizer sigreg --protocol bptt --steps 5 --batch-size 8 --n-train 32 --n-val 8 --n-test 8 --seq-len 12 --horizon 3 --eval-horizon 5 --sigreg-projections 8 --sigreg-max-samples 16 --output-dir smoke_runs
```

## Dry-Run Full Sweep Commands

```bash
cd all-spikes/rollout-drift-sigreg-protocol-interaction
python3 -m src.run_sweep --dry-run
```

## Analyze Runs

```bash
cd all-spikes/rollout-drift-sigreg-protocol-interaction
python3 -m src.analyze --runs-dir runs --out-dir analysis
```

## Notes

- `src/regularizers.py` contains the canonical sliced Epps-Pulley SIGReg approximation used in the design.
- `src/train.py` logs every run to `metrics.json` and `history.jsonl`.
- The current scaffold uses no pandas dependency.
