from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="lambda_runs")
    parser.add_argument("--out", default="lambda_map.json")
    parser.add_argument("--report", default="lambda_selection_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(Path(args.runs_dir).glob("*/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["regularizer"] == "none":
            continue
        key = f"{row['env']}:{row['regularizer']}"
        row["_path"] = str(path)
        groups[key].append(row)

    selected: dict[str, float] = {}
    report: dict[str, Any] = {}
    for key, rows in sorted(groups.items()):
        noncollapsed = [row for row in rows if float(row.get("collapsed", 1.0)) == 0.0]
        candidates = noncollapsed if noncollapsed else rows
        best = min(candidates, key=lambda row: float(row.get("final_pred_loss", float("inf"))))
        selected[key] = float(best["lambda_reg"])
        report[key] = {
            "selected_lambda": selected[key],
            "all_collapsed": not bool(noncollapsed),
            "selected_final_pred_loss": best.get("final_pred_loss"),
            "selected_linear_probe_r2": best.get("linear_probe_r2"),
            "selected_path": best.get("_path"),
            "candidates": [
                {
                    "lambda": row.get("lambda_reg"),
                    "collapsed": row.get("collapsed"),
                    "final_pred_loss": row.get("final_pred_loss"),
                    "linear_probe_r2": row.get("linear_probe_r2"),
                    "path": row.get("_path"),
                }
                for row in sorted(rows, key=lambda r: float(r["lambda_reg"]))
            ],
        }

    Path(args.out).write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(selected, sort_keys=True))


if __name__ == "__main__":
    main()

