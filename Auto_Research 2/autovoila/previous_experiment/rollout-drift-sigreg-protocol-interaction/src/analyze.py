from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_metrics(Path(args.runs_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "metrics_flat.csv")
    write_group_summary(rows, out_dir / "condition_summary.csv")
    write_anova(rows, out_dir / "anova_eta2.csv")
    print(f"loaded {len(rows)} runs")


def load_metrics(runs_dir: Path) -> List[Dict[str, object]]:
    rows = []
    for path in sorted(runs_dir.glob("*/metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        row["run_dir"] = str(path.parent)
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_group_summary(rows: List[Dict[str, object]], path: Path) -> None:
    groups: Dict[tuple, list] = defaultdict(list)
    for row in rows:
        key = (row["env"], row["regularizer"], row["protocol"])
        groups[key].append(row)

    fields = [
        "env",
        "regularizer",
        "protocol",
        "n",
        "collapsed_rate",
        "excess_amp_50_mean",
        "excess_amp_50_ci95",
        "perturb_amp_50_mean",
        "open_loop_mse_50_mean",
        "effective_dim_mean",
        "linear_probe_r2_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (env, reg, protocol), group in sorted(groups.items()):
            writer.writerow(
                {
                    "env": env,
                    "regularizer": reg,
                    "protocol": protocol,
                    "n": len(group),
                    "collapsed_rate": mean(group, "collapsed"),
                    "excess_amp_50_mean": mean(group, "excess_amp_50"),
                    "excess_amp_50_ci95": ci95(group, "excess_amp_50"),
                    "perturb_amp_50_mean": mean(group, "perturb_amp_50"),
                    "open_loop_mse_50_mean": mean(group, "open_loop_mse_50"),
                    "effective_dim_mean": mean(group, "effective_dim"),
                    "linear_probe_r2_mean": mean(group, "linear_probe_r2"),
                }
            )


def write_anova(rows: List[Dict[str, object]], path: Path) -> None:
    fields = ["env", "metric", "eta2_regularizer", "eta2_protocol", "eta2_interaction", "n"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for env in sorted({str(row["env"]) for row in rows}):
            subset = [row for row in rows if row["env"] == env and "excess_amp_50" in row]
            if len(subset) < 6:
                continue
            eta = eta2_factorial(subset, "excess_amp_50")
            writer.writerow(
                {
                    "env": env,
                    "metric": "log_excess_amp_50",
                    "eta2_regularizer": eta["regularizer"],
                    "eta2_protocol": eta["protocol"],
                    "eta2_interaction": eta["interaction"],
                    "n": len(subset),
                }
            )


def eta2_factorial(rows: List[Dict[str, object]], metric: str) -> Dict[str, float]:
    y = np.log(np.array([float(row[metric]) for row in rows], dtype=np.float64) + 1e-12)
    regs = sorted({str(row["regularizer"]) for row in rows})
    protocols = sorted({str(row["protocol"]) for row in rows})

    x_full = design_matrix(rows, regs, protocols, include_reg=True, include_protocol=True, include_interaction=True)
    sse_full = sse_ols(x_full, y)
    sst = float(np.square(y - y.mean()).sum()) + 1e-12

    x_no_reg = design_matrix(rows, regs, protocols, include_reg=False, include_protocol=True, include_interaction=False)
    x_no_protocol = design_matrix(rows, regs, protocols, include_reg=True, include_protocol=False, include_interaction=False)
    x_no_inter = design_matrix(rows, regs, protocols, include_reg=True, include_protocol=True, include_interaction=False)

    return {
        "regularizer": max(0.0, (sse_ols(x_no_reg, y) - sse_full) / sst),
        "protocol": max(0.0, (sse_ols(x_no_protocol, y) - sse_full) / sst),
        "interaction": max(0.0, (sse_ols(x_no_inter, y) - sse_full) / sst),
    }


def design_matrix(
    rows: List[Dict[str, object]],
    regs: List[str],
    protocols: List[str],
    include_reg: bool,
    include_protocol: bool,
    include_interaction: bool,
) -> np.ndarray:
    cols = [np.ones(len(rows), dtype=np.float64)]
    if include_reg:
        for reg in regs[1:]:
            cols.append(np.array([str(row["regularizer"]) == reg for row in rows], dtype=np.float64))
    if include_protocol:
        for protocol in protocols[1:]:
            cols.append(np.array([str(row["protocol"]) == protocol for row in rows], dtype=np.float64))
    if include_interaction:
        for reg in regs[1:]:
            for protocol in protocols[1:]:
                cols.append(
                    np.array(
                        [
                            str(row["regularizer"]) == reg and str(row["protocol"]) == protocol
                            for row in rows
                        ],
                        dtype=np.float64,
                    )
                )
    return np.stack(cols, axis=1)


def sse_ols(x: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return float(np.square(resid).sum())


def values(rows: Iterable[Dict[str, object]], key: str) -> np.ndarray:
    vals = [float(row[key]) for row in rows if key in row and row[key] is not None]
    return np.array(vals, dtype=np.float64)


def mean(rows: Iterable[Dict[str, object]], key: str) -> float:
    vals = values(rows, key)
    return float(vals.mean()) if vals.size else float("nan")


def ci95(rows: Iterable[Dict[str, object]], key: str) -> float:
    vals = values(rows, key)
    if vals.size <= 1:
        return float("nan")
    return float(1.96 * vals.std(ddof=1) / np.sqrt(vals.size))


if __name__ == "__main__":
    main()
