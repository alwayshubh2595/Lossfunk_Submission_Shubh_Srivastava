from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands-file", required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--log-dir", default="launcher_logs")
    parser.add_argument("--status-file", default="launcher_status.jsonl")
    parser.add_argument("--stop-after-hours", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = [
        line.strip()
        for line in Path(args.commands_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status_file)
    start = time.time()
    deadline = None if args.stop_after_hours is None else start + args.stop_after_hours * 3600.0

    def run_one(index: int, command: str) -> dict[str, object]:
        log_path = log_dir / f"job_{index:04d}.log"
        t0 = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"$ {command}\n")
            log.flush()
            proc = subprocess.run(
                shlex.split(command),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        return {
            "index": index,
            "returncode": proc.returncode,
            "seconds": round(time.time() - t0, 3),
            "log": str(log_path),
            "command": command,
        }

    next_index = 0
    running = {}
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        while next_index < len(commands) or running:
            while next_index < len(commands) and len(running) < args.jobs:
                if deadline is not None and time.time() >= deadline:
                    break
                future = pool.submit(run_one, next_index, commands[next_index])
                running[future] = next_index
                next_index += 1
            if not running:
                break
            done, _ = wait(running.keys(), timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                running.pop(future)
                result = future.result()
                completed += 1
                failed += int(result["returncode"] != 0)
                result["completed"] = completed
                result["failed"] = failed
                result["elapsed_hours"] = round((time.time() - start) / 3600.0, 4)
                with status_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, sort_keys=True) + "\n")
                print(json.dumps(result, sort_keys=True), flush=True)
            if deadline is not None and time.time() >= deadline and next_index < len(commands):
                print(f"deadline reached; not starting remaining {len(commands) - next_index} commands", flush=True)
                break

    summary = {
        "total_commands": len(commands),
        "started": next_index,
        "completed": completed,
        "failed": failed,
        "elapsed_hours": round((time.time() - start) / 3600.0, 4),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
