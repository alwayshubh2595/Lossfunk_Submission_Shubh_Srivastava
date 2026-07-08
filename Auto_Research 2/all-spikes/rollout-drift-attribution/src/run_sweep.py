from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--lambda-json", default=None)
    parser.add_argument(
        "--stage",
        choices=["core", "matched-dim", "lambda-sensitivity", "m1024", "lambda-pilot", "all"],
        default="core",
        help="core = bridge+main+P0.5 (160 runs); matched-dim/lambda-sensitivity/m1024 are the "
        "additional sensitivity stages from experiment_design.md v3; all = every flat stage "
        "concatenated (encoder-swap and rollout-ablation are separate multi-phase scripts: "
        "run_swap.py, run_ablation.py)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "core":
        commands = build_commands(args)
    elif args.stage == "matched-dim":
        commands = build_matched_dim_commands(args)
    elif args.stage == "lambda-sensitivity":
        commands = build_lambda_sensitivity_commands(args)
    elif args.stage == "m1024":
        commands = build_m1024_commands(args)
    elif args.stage == "lambda-pilot":
        commands = build_lambda_pilot_commands(args)
    elif args.stage == "all":
        commands = (
            build_commands(args)
            + build_matched_dim_commands(args)
            + build_lambda_sensitivity_commands(args)
            + build_m1024_commands(args)
        )
    else:
        raise ValueError(f"unknown stage: {args.stage}")
    if args.limit is not None:
        commands = commands[: args.limit]
    for cmd in commands:
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


def _base_steps_args(args: argparse.Namespace, env: str) -> list[str]:
    # v4 replan (8h hard wall-clock deadline on 7 GPUs): training steps reduced
    # UNIFORMLY across every condition (state 20k->10k, visual 15k->8k). A
    # global dose change applied identically to all cells cannot bias between
    # conditions, unlike selective cell drops; logged in session_log.md.
    if args.quick:
        return quick_args()
    if env == "pointmass":
        return ["--steps", "8000", "--n-train", "800", "--n-val", "128", "--n-test", "128"]
    return ["--steps", "10000", "--n-train", "3000", "--n-val", "512", "--n-test", "512"]


def _make_cmd(
    args: argparse.Namespace,
    lambda_map: dict[str, float],
    env: str,
    reg: str,
    protocol: str,
    seed: int,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "src.train",
        "--env",
        env,
        "--regularizer",
        reg,
        "--protocol",
        protocol,
        "--seed",
        str(seed),
        "--output-dir",
        args.output_dir,
    ]
    cmd += _base_steps_args(args, env)
    cmd += lambda_args(lambda_map, env, reg)
    if extra:
        cmd += extra
    return cmd


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    """Builds the v9 run manifest (experiment_design.md v3 + compute_budget.md
    v9/v10): bridge cells on the deterministic stable-linear system, the main
    factorial on {ou, lorenz, pointmass-state, pointmass} x {none, moment,
    sigreg} x {1step, bptt}, and the P0.5 dense-teacher-forcing control.
    Matched-dim, lambda-sensitivity, encoder-swap, rollout-reg-ablation, and
    M=1024 sensitivity stages are separate scripts (see experiment_design.md)
    and are not generated here.
    """
    lambda_map = load_lambda_map(args.lambda_json)
    commands: list[list[str]] = []

    # Bridge (E0-det): R0/R1 x P0/P1 x 5 seeds, deterministic stable-linear.
    for reg, protocol in product(["none", "moment"], ["1step", "bptt"]):
        for seed in range(5):
            commands.append(_make_cmd(args, lambda_map, "stable-linear-det", reg, protocol, seed))

    # Main factorial: state envs at 5 seeds; pointmass (visual) at 3 seeds
    # (v4 replan: visual reverted 5->3 under the 8h deadline -- visual is the
    # most expensive tier per run).
    main_envs = ["ou", "lorenz", "pointmass-state", "pointmass"]
    for env, reg, protocol in product(main_envs, ["none", "moment", "sigreg"], ["1step", "bptt"]):
        seeds = 3 if env == "pointmass" else 5
        for seed in range(seeds):
            extra = None
            if env == "ou" and reg == "sigreg" and seed < 3:
                # Checkpoints double as encoder-swap phase-A sources (v4:
                # reusing them eliminates the 12 dedicated source runs).
                extra = ["--save-model"]
            commands.append(_make_cmd(args, lambda_map, env, reg, protocol, seed, extra=extra))

    # P0.5 control: dense H-step teacher forcing (no closed loop), on {ou,
    # pointmass} x {moment, sigreg}; visual at 3 seeds (v4).
    if not args.skip_controls:
        for env, reg in product(["ou", "pointmass"], ["moment", "sigreg"]):
            seeds = 3 if env == "pointmass" else 5
            for seed in range(seeds):
                commands.append(_make_cmd(args, lambda_map, env, reg, "hstep-tf", seed))

    return commands


def build_matched_dim_commands(args: argparse.Namespace) -> list[list[str]]:
    """Matched-latent-dimension sensitivity: {ou, pointmass-state} x {none,
    sigreg} x {1step, bptt} x 3 seeds, at the environment's intrinsic
    dimension (3 for ou, 2 for pointmass-state) instead of the overcomplete
    d=16 used in the main factorial. Guards against a negative SIGReg result
    being dismissed as an artifact of rank-deficient overcomplete latents.
    """
    lambda_map = load_lambda_map(args.lambda_json)
    intrinsic_dim = {"ou": 3, "pointmass-state": 2}
    commands: list[list[str]] = []
    for env, reg, protocol in product(["ou", "pointmass-state"], ["none", "sigreg"], ["1step", "bptt"]):
        for seed in range(3):
            cmd = _make_cmd(
                args,
                lambda_map,
                env,
                reg,
                protocol,
                seed,
                extra=["--latent-dim", str(intrinsic_dim[env]), "--output-dir", f"{args.output_dir}_matched_dim"],
            )
            commands.append(cmd)
    return commands


def build_lambda_sensitivity_commands(args: argparse.Namespace) -> list[list[str]]:
    """Lambda sensitivity for R2 (sigreg) on {ou, pointmass-state} x {1step,
    bptt} x 3 seeds at low (0.3x) and high (3x) multiples of the selected
    lambda from the main lambda-tuning stage. The selected-lambda point
    itself is reused from the main sweep's results, not rerun here.
    """
    lambda_map = load_lambda_map(args.lambda_json)
    commands: list[list[str]] = []
    for env in ["ou", "pointmass-state"]:
        base_lambda = lambda_map.get(f"{env}:sigreg")
        if base_lambda is None:
            continue
        for protocol in ["1step", "bptt"]:
            for mult, tag in [(0.3, "low"), (3.0, "high")]:
                for seed in range(3):
                    cmd = _make_cmd(
                        args,
                        lambda_map,
                        env,
                        "sigreg",
                        protocol,
                        seed,
                        extra=[
                            "--lambda-reg",
                            str(base_lambda * mult),
                            "--output-dir",
                            f"{args.output_dir}_lambda_sensitivity_{tag}",
                        ],
                    )
                    commands.append(cmd)
    return commands


def build_m1024_commands(args: argparse.Namespace) -> list[list[str]]:
    """M=1024 sensitivity: rerun the two highest-leverage R2/bptt cells
    (one state, one action-conditioned state env) at seed 0 with 4x the
    default projection count, per LeJEPA's guidance and to check the M=256
    Monte Carlo approximation isn't materially different from a denser one.
    """
    lambda_map = load_lambda_map(args.lambda_json)
    commands: list[list[str]] = []
    for env in ["ou", "pointmass-state"]:
        cmd = _make_cmd(
            args,
            lambda_map,
            env,
            "sigreg",
            "bptt",
            0,
            extra=["--sigreg-projections", "1024", "--output-dir", f"{args.output_dir}_m1024"],
        )
        commands.append(cmd)
    return commands


LAMBDA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
PILOT_STATE_STEPS = 8000  # reduced from full 20000: pilots only need relative
PILOT_VISUAL_STEPS = 6000  # ranking of lambda values by validation MSE, not full convergence


def build_lambda_pilot_commands(args: argparse.Namespace) -> list[list[str]]:
    """Lambda tuning pre-runs (experiment_design.md v3): grid search per
    (regularizer, environment), seed 0 only, selection criterion is
    validation one-step MSE (open_loop_mse_1) subject to non-collapse --
    never any drift/rollout metric (this is enforced downstream in
    select_lambdas.py, not here). Step count is reduced relative to the main
    sweep (8000/6000 vs 20000/15000): ranking candidate lambdas by relative
    validation MSE does not require full convergence, and using the same
    reduced budget for every candidate in a cell keeps the comparison fair.
    Protocol is fixed to 1step for all pilots (matches the val-MSE
    criterion's name and keeps the pilot stage protocol-independent, since
    the selected lambda is frozen across P0/P0.5/P1 for that (R, E) cell).
    """
    combos = [
        ("stable-linear-det", "moment"),
        ("ou", "moment"),
        ("ou", "sigreg"),
        ("lorenz", "moment"),
        ("lorenz", "sigreg"),
        ("pointmass-state", "moment"),
        ("pointmass-state", "sigreg"),
        ("pointmass", "moment"),
        ("pointmass", "sigreg"),
    ]
    commands: list[list[str]] = []
    for env, reg in combos:
        visual = env == "pointmass"
        steps = PILOT_VISUAL_STEPS if visual else PILOT_STATE_STEPS
        n_train = 800 if visual else 3000
        n_val = 128 if visual else 512
        n_test = 128 if visual else 512
        for lam in LAMBDA_GRID:
            cmd = [
                args.python,
                "-m",
                "src.train",
                "--env",
                env,
                "--regularizer",
                reg,
                "--protocol",
                "1step",
                "--seed",
                "0",
                "--lambda-reg",
                str(lam),
                "--steps",
                str(steps),
                "--n-train",
                str(n_train),
                "--n-val",
                str(n_val),
                "--n-test",
                str(n_test),
                "--output-dir",
                args.output_dir,
            ]
            if args.quick:
                cmd = cmd[:-2] + quick_args() + ["--output-dir", args.output_dir]
            commands.append(cmd)
    return commands


def load_lambda_map(path: str | None) -> dict[str, float]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): float(v) for k, v in data.items()}


def lambda_args(lambda_map: dict[str, float], env: str, reg: str) -> list[str]:
    value = lambda_map.get(f"{env}:{reg}")
    if value is None:
        return []
    return ["--lambda-reg", str(value)]


def quick_args() -> list[str]:
    return [
        "--steps",
        "20",
        "--batch-size",
        "16",
        "--n-train",
        "64",
        "--n-val",
        "16",
        "--n-test",
        "16",
        "--seq-len",
        "16",
        "--horizon",
        "4",
        "--eval-horizon",
        "8",
        "--sigreg-projections",
        "16",
        "--sigreg-max-samples",
        "32",
    ]


if __name__ == "__main__":
    main()
