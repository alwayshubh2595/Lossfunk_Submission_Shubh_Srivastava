from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

PRIMARY_METRIC = "integrated_probe_open_loop_error"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="analysis")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_metrics(Path(args.runs_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "metrics_flat.csv")
    write_group_summary(rows, out_dir / "condition_summary.csv")
    write_anova(rows, out_dir / "anova_eta2.csv")
    write_contrasts(rows, out_dir / "contrasts.json", n_bootstrap=args.bootstrap_samples, seed=args.seed)
    write_collapse_report(rows, out_dir / "collapse_report.csv")
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
        "primary_integrated_probe_error_mean",
        "primary_integrated_probe_error_ci95",
        "abs_log_excess_amp_25_mean",
        "excess_amp_50_mean",
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
                    "primary_integrated_probe_error_mean": mean(group, PRIMARY_METRIC),
                    "primary_integrated_probe_error_ci95": ci95(group, PRIMARY_METRIC),
                    "abs_log_excess_amp_25_mean": mean(group, "abs_log_excess_amp_25"),
                    "excess_amp_50_mean": mean(group, "excess_amp_50"),
                    "open_loop_mse_50_mean": mean(group, "open_loop_mse_50"),
                    "effective_dim_mean": mean(group, "effective_dim"),
                    "linear_probe_r2_mean": mean(group, "linear_probe_r2"),
                }
            )


def write_collapse_report(rows: List[Dict[str, object]], path: Path) -> None:
    """Collapse as a treatment outcome (Codex review, near-fatal issue 4):
    report collapse probability by condition and drift conditional on
    viability, rather than silently excluding collapsed runs.
    """
    groups: Dict[tuple, list] = defaultdict(list)
    for row in rows:
        key = (row["env"], row["regularizer"], row["protocol"])
        groups[key].append(row)
    fields = ["env", "regularizer", "protocol", "n", "collapse_prob", "primary_metric_given_viable_mean"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (env, reg, protocol), group in sorted(groups.items()):
            viable = [row for row in group if float(row.get("collapsed", 1.0)) == 0.0]
            writer.writerow(
                {
                    "env": env,
                    "regularizer": reg,
                    "protocol": protocol,
                    "n": len(group),
                    "collapse_prob": 1.0 - len(viable) / len(group) if group else float("nan"),
                    "primary_metric_given_viable_mean": mean(viable, PRIMARY_METRIC),
                }
            )


def write_anova(rows: List[Dict[str, object]], path: Path) -> None:
    """Descriptive-only factorial eta^2 table (not the basis for the
    pre-registered decision criteria -- see write_contrasts for those).
    """
    fields = ["env", "metric", "eta2_regularizer", "eta2_protocol", "eta2_interaction", "n"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for env in sorted({str(row["env"]) for row in rows}):
            subset = [row for row in rows if row["env"] == env and PRIMARY_METRIC in row and row[PRIMARY_METRIC] is not None]
            if len(subset) < 6:
                continue
            eta = eta2_factorial(subset, PRIMARY_METRIC)
            writer.writerow(
                {
                    "env": env,
                    "metric": PRIMARY_METRIC,
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


# ---------------------------------------------------------------------------
# Pre-registered contrasts (experiment_design.md v3): C_R, C_P, C_I on the
# primary metric (log scale), computed per environment, with seed-blocked
# cluster bootstrap for CIs. A seed block = the same seed's runs across the
# conditions a contrast compares, so resampling respects that seed's shared
# data/init rather than treating cells as independent.
# ---------------------------------------------------------------------------


def _cell(rows: List[Dict[str, object]], env: str, reg: str, protocol: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for row in rows:
        if row.get("env") == env and row.get("regularizer") == reg and row.get("protocol") == protocol:
            val = row.get(PRIMARY_METRIC)
            if val is not None and float(row.get("collapsed", 0.0)) == 0.0:
                out[int(row["seed"])] = float(np.log(max(float(val), 1e-12)))
    return out


def _paired_diff(a: Dict[int, float], b: Dict[int, float]) -> np.ndarray:
    seeds = sorted(set(a) & set(b))
    return np.array([a[s] - b[s] for s in seeds], dtype=np.float64)


def _bootstrap_ci(diffs: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float, float]:
    if diffs.size == 0:
        return float("nan"), float("nan"), float("nan")
    boot_means = np.array(
        [rng.choice(diffs, size=diffs.size, replace=True).mean() for _ in range(n_bootstrap)]
    )
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def write_contrasts(rows: List[Dict[str, object]], path: Path, n_bootstrap: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    m = float(np.log(1.25))  # practical-equivalence margin
    envs = sorted({str(row["env"]) for row in rows})
    report: Dict[str, object] = {"equivalence_margin_log": m, "per_env": {}}

    for env in envs:
        r0_p0 = _cell(rows, env, "none", "1step")
        r2_p0 = _cell(rows, env, "sigreg", "1step")
        r0_p1 = _cell(rows, env, "none", "bptt")
        r2_p1 = _cell(rows, env, "sigreg", "bptt")

        c_r = _paired_diff(r2_p0, r0_p0)  # C_R = (R2 - R0) under P0
        c_p_r0 = _paired_diff(r0_p1, r0_p0)
        c_p_r2 = _paired_diff(r2_p1, r2_p0)
        c_p = np.concatenate([c_p_r0, c_p_r2]) if c_p_r0.size and c_p_r2.size else np.concatenate([c_p_r0, c_p_r2])
        c_r_under_p1 = _paired_diff(r2_p1, r0_p1)
        c_i = c_r_under_p1 - _match_len(c_r, c_r_under_p1)  # interaction contrast, seed-matched where possible

        report["per_env"][env] = {
            "C_R_mean_lo_hi": _bootstrap_ci(c_r, n_bootstrap, rng),
            "C_P_mean_lo_hi": _bootstrap_ci(c_p, n_bootstrap, rng),
            "C_I_proxy_mean_lo_hi": _bootstrap_ci(c_i, n_bootstrap, rng) if c_i.size else (float("nan"),) * 3,
            "n_seeds_R_P0": len(set(r0_p0) & set(r2_p0)),
            "n_seeds_P_R0": len(set(r0_p0) & set(r0_p1)),
            "n_seeds_P_R2": len(set(r2_p0) & set(r2_p1)),
        }

    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _match_len(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = min(a.size, b.size)
    return a[:n]


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
