from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .run_sweep import load_lambda_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", default="swap_runs")
    parser.add_argument("--source-dir", default="swap_sources")
    parser.add_argument("--lambda-json", default=None)
    parser.add_argument("--envs", nargs="+", default=["ou", "pointmass-state"])
    parser.add_argument("--seeds", type=int, default=3)
    return parser.parse_args()


def _base_args(env: str) -> list[str]:
    if env == "pointmass-state":
        return ["--steps", "20000", "--n-train", "3000", "--n-val", "512", "--n-test", "512"]
    return ["--steps", "20000", "--n-train", "3000", "--n-val", "512", "--n-test", "512"]


def main() -> None:
    """Full 2x2 encoder-swap mechanism probe (experiment_design.md v3):
    for each env and seed, train source models under P0 and P1 (R2/sigreg,
    saving checkpoints), then retrain dynamics under all four
    (encoder-source-protocol, dynamics-retrain-protocol) combinations,
    including the two "diagonal" combinations (retrained, not reused from
    the source run) so the intervention is compared against retrained
    controls rather than the originally jointly-trained models.
    """
    args = parse_args()
    lambda_map = load_lambda_map(args.lambda_json)
    source_dir = Path(args.source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    commands: list[tuple[str, list[str]]] = []

    # Phase A: source models, one per (env, protocol, seed), with checkpoint saved.
    source_paths: dict[tuple[str, str, int], str] = {}
    for env in args.envs:
        for protocol in ["1step", "bptt"]:
            for seed in range(args.seeds):
                run_name = f"{env}__sigreg__{protocol}__seed{seed}"
                lam = lambda_map.get(f"{env}:sigreg")
                cmd = [
                    args.python,
                    "-m",
                    "src.train",
                    "--env",
                    env,
                    "--regularizer",
                    "sigreg",
                    "--protocol",
                    protocol,
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(source_dir),
                    "--save-model",
                ]
                cmd += _base_args(env)
                if lam is not None:
                    cmd += ["--lambda-reg", str(lam)]
                commands.append(("phase_a", cmd))
                source_paths[(env, protocol, seed)] = str(source_dir / run_name / "model.pt")

    # Phase B: full 2x2 swap. For every (encoder_protocol, dynamics_protocol)
    # pair -- including the two diagonals, which are retrained from a frozen
    # encoder just like the off-diagonals, not reused from phase A directly.
    for env in args.envs:
        for seed in range(args.seeds):
            for encoder_protocol in ["1step", "bptt"]:
                for dynamics_protocol in ["1step", "bptt"]:
                    encoder_ckpt = source_paths[(env, encoder_protocol, seed)]
                    lam = lambda_map.get(f"{env}:sigreg")
                    cmd = [
                        args.python,
                        "-m",
                        "src.train",
                        "--env",
                        env,
                        "--regularizer",
                        "sigreg",
                        "--protocol",
                        dynamics_protocol,
                        "--seed",
                        str(seed),
                        "--output-dir",
                        args.output_dir,
                        "--load-encoder-from",
                        encoder_ckpt,
                    ]
                    cmd += _base_args(env)
                    if lam is not None:
                        cmd += ["--lambda-reg", str(lam)]
                    commands.append(("phase_b", cmd))

    for phase, cmd in commands:
        print(f"[{phase}] " + " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    manifest_path = Path(args.output_dir).parent / "swap_manifest.json"
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "swap_source_paths.json").write_text(
        json.dumps({f"{k[0]}:{k[1]}:{k[2]}": v for k, v in source_paths.items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
