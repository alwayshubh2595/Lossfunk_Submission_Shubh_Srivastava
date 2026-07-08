from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="lambda_runs")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for cmd in build_commands(args):
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    envs = ["stable-linear", "lorenz", "pointmass-state", "pointmass"]
    regs = ["moment", "sigreg"]
    lambdas = ["0.01", "0.03", "0.1", "0.3", "1.0"]
    commands: list[list[str]] = []
    for env, reg, lam in product(envs, regs, lambdas):
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
            lam,
            "--output-dir",
            args.output_dir,
            "--sigreg-projections",
            "128",
        ]
        if env == "pointmass":
            cmd += ["--steps", "2000", "--n-train", "800", "--n-val", "128", "--n-test", "128"]
        else:
            cmd += ["--steps", "3000", "--n-train", "3000", "--n-val", "512", "--n-test", "512"]
        commands.append(cmd)
    return commands


if __name__ == "__main__":
    main()

