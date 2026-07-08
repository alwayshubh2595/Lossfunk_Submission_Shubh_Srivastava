from __future__ import annotations

import argparse
import subprocess
import sys

from .run_sweep import load_lambda_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", default="ablation_runs")
    parser.add_argument("--lambda-json", default=None)
    parser.add_argument("--envs", nargs="+", default=["ou", "pointmass-state"])
    parser.add_argument("--seeds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Rollout-regularization ablation (experiment_design.md v3): under bptt,
    compare (a) R2 applied to encoded latents AND rolled-out predictions at
    the standard lambda, against (b) R2 applied to encoded latents only at
    2x lambda -- a dose-matched control so a positive result reflects where
    the regularizer is applied, not merely that more total regularization
    was used. Success requires improved probe-space state error, not just
    lower amplification (Codex review note on this ablation).
    """
    args = parse_args()
    lambda_map = load_lambda_map(args.lambda_json)
    commands: list[list[str]] = []
    for env in args.envs:
        base_lambda = lambda_map.get(f"{env}:sigreg")
        if base_lambda is None:
            continue
        for seed in range(args.seeds):
            # Arm A: encoder + rollout regularization at standard lambda.
            cmd_a = [
                args.python,
                "-m",
                "src.train",
                "--env",
                env,
                "--regularizer",
                "sigreg",
                "--protocol",
                "bptt",
                "--seed",
                str(seed),
                "--output-dir",
                f"{args.output_dir}_enc_and_rollout",
                "--reg-target",
                "encoded-and-rollout",
                "--lambda-reg",
                str(base_lambda),
                "--steps",
                "20000",
                "--n-train",
                "3000",
                "--n-val",
                "512",
                "--n-test",
                "512",
            ]
            # Arm B (dose-matched control): encoder-only at 2x lambda.
            cmd_b = [
                args.python,
                "-m",
                "src.train",
                "--env",
                env,
                "--regularizer",
                "sigreg",
                "--protocol",
                "bptt",
                "--seed",
                str(seed),
                "--output-dir",
                f"{args.output_dir}_enc_only_double",
                "--reg-target",
                "encoded",
                "--lambda-reg",
                str(base_lambda * 2.0),
                "--steps",
                "20000",
                "--n-train",
                "3000",
                "--n-val",
                "512",
                "--n-test",
                "512",
            ]
            commands.append(cmd_a)
            commands.append(cmd_b)

    for cmd in commands:
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
