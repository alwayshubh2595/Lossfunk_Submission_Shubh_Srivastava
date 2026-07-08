#!/usr/bin/env python3
"""Inventory completed lambda-pilot cells and emit the remaining commands."""
import json
from pathlib import Path

GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
COMBOS = [
    ("stable-linear-det", "moment"),
    ("ou", "moment"), ("ou", "sigreg"),
    ("lorenz", "moment"), ("lorenz", "sigreg"),
    ("pointmass-state", "moment"), ("pointmass-state", "sigreg"),
    ("pointmass", "moment"), ("pointmass", "sigreg"),
]

done = set()
for p in Path("lambda_runs").glob("*/metrics.json"):
    m = json.loads(p.read_text())
    done.add((m["env"], m["regularizer"], round(float(m["lambda_reg"]), 4)))

remaining = []
for env, reg in COMBOS:
    visual = env == "pointmass"
    steps = 4000 if visual else 8000  # visual pilot reduced 6000->4000 (v4 replan)
    n_train, n_val, n_test = (800, 128, 128) if visual else (3000, 512, 512)
    for lam in GRID:
        if (env, reg, round(lam, 4)) in done:
            continue
        remaining.append(
            f"/usr/bin/python3 -m src.train --env {env} --regularizer {reg} "
            f"--protocol 1step --seed 0 --lambda-reg {lam} --steps {steps} "
            f"--n-train {n_train} --n-val {n_val} --n-test {n_test} "
            f"--output-dir lambda_runs"
        )

Path("lambda_pilot_remaining.txt").write_text("\n".join(remaining) + "\n")
print(f"done cells: {len(done)}, remaining: {len(remaining)}")
