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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    if args.limit is not None:
        commands = commands[: args.limit]
    for cmd in commands:
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    lambda_map = load_lambda_map(args.lambda_json)
    envs = ["stable-linear", "lorenz", "pointmass-state", "pointmass"]
    regs = ["none", "moment", "sigreg"]
    protocols = ["1step", "bptt"]
    commands: list[list[str]] = []
    for env, reg, protocol in product(envs, regs, protocols):
        seeds = range(3) if env == "pointmass" else range(5)
        for seed in seeds:
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
            if args.quick:
                cmd += quick_args()
            elif env == "pointmass":
                cmd += ["--steps", "15000", "--n-train", "800", "--n-val", "128", "--n-test", "128"]
            else:
                cmd += ["--steps", "20000", "--n-train", "3000", "--n-val", "512", "--n-test", "512"]
            cmd += lambda_args(lambda_map, env, reg)
            commands.append(cmd)
    if not args.skip_controls:
        for env, reg in product(["stable-linear", "pointmass"], ["moment", "sigreg"]):
            seeds = range(3) if env == "pointmass" else range(5)
            for seed in seeds:
                cmd = [
                    args.python,
                    "-m",
                    "src.train",
                    "--env",
                    env,
                    "--regularizer",
                    reg,
                    "--protocol",
                    "hstep-tf",
                    "--seed",
                    str(seed),
                    "--output-dir",
                    args.output_dir,
                ]
                if args.quick:
                    cmd += quick_args()
                elif env == "pointmass":
                    cmd += ["--steps", "15000", "--n-train", "800", "--n-val", "128", "--n-test", "128"]
                else:
                    cmd += ["--steps", "20000", "--n-train", "3000", "--n-val", "512", "--n-test", "512"]
                cmd += lambda_args(lambda_map, env, reg)
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
