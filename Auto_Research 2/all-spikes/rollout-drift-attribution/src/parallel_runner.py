from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands-file", required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="physical GPU count to round-robin CUDA_VISIBLE_DEVICES across (default: same as --jobs). "
        "Each launched job blocks on a thread-safe GPU pool, so this is safe even if --jobs > --num-gpus "
        "(multiple jobs will then share a GPU, waiting their turn).",
    )
    parser.add_argument("--log-dir", default="launcher_logs")
    parser.add_argument("--status-file", default="launcher_status.jsonl")
    parser.add_argument("--stop-after-hours", type=float, default=None)
    parser.add_argument(
        "--hard-kill-after-hours",
        type=float,
        default=None,
        help="hard dollar cutoff: SIGTERM (then SIGKILL) any still-running job past this wall-clock, "
        "not just stop launching new ones",
    )
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
    hard_deadline = None if args.hard_kill_after_hours is None else start + args.hard_kill_after_hours * 3600.0

    active_procs: dict[int, subprocess.Popen] = {}
    active_lock = threading.Lock()
    stop_watchdog = threading.Event()

    num_gpus = args.num_gpus if args.num_gpus is not None else args.jobs
    gpu_pool: "queue.Queue[int]" = queue.Queue()
    for gpu_id in range(num_gpus):
        gpu_pool.put(gpu_id)

    def watchdog() -> None:
        if hard_deadline is None:
            return
        while not stop_watchdog.is_set():
            if time.time() >= hard_deadline:
                with active_lock:
                    procs = list(active_procs.values())
                for proc in procs:
                    if proc.poll() is None:
                        proc.terminate()
                time.sleep(5)
                with active_lock:
                    procs = list(active_procs.values())
                for proc in procs:
                    if proc.poll() is None:
                        proc.kill()
                print(
                    json.dumps({"event": "hard_kill_after_hours_reached", "killed": len(procs)}),
                    flush=True,
                )
                return
            time.sleep(5)

    def run_one(index: int, command: str) -> dict[str, object]:
        log_path = log_dir / f"job_{index:04d}.log"
        t0 = time.time()
        gpu_id = gpu_pool.get()
        try:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"$ CUDA_VISIBLE_DEVICES={gpu_id} {command}\n")
                log.flush()
                proc = subprocess.Popen(
                    shlex.split(command),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
                with active_lock:
                    active_procs[index] = proc
                returncode = proc.wait()
                with active_lock:
                    active_procs.pop(index, None)
        finally:
            gpu_pool.put(gpu_id)
        return {
            "index": index,
            "returncode": returncode,
            "gpu": gpu_id,
            "seconds": round(time.time() - t0, 3),
            "log": str(log_path),
            "command": command,
        }

    watchdog_thread = threading.Thread(target=watchdog, daemon=True)
    watchdog_thread.start()

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
            if hard_deadline is not None and time.time() >= hard_deadline:
                print("hard kill deadline reached; stopping loop", flush=True)
                break

    stop_watchdog.set()
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
