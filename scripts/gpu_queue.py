"""Run `vifeedback` commands one at a time, each only when the machine is free.

The owner's rule for the shared laptop: one heavy job at a time, and never on top of another
process's GPU work. Before every step this waits until there is no other GPU compute process, the
GPU is below 70 C and the CPU below 50%, on two checks one minute apart. It never touches another
process. It stops at the first failing step.

    python scripts/gpu_queue.py "study h12p-control --seed 1337" "study h12p-run --seed 1337"

A step that starts with `!` runs without waiting (cheap steps such as `!study h12p-status`).
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOISE = ("it/s]", "s/it]", "UNEXPECTED", "MISSING", "Notes:", "can be ignored", "newly initialized")


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def machine_state() -> tuple[bool, str]:
    """(free, description): no other GPU compute process, GPU < 70 C, CPU < 50%."""
    q = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    temp, util = (int(x) for x in q.split(",")[:2])
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    cpu = 0.0
    if sys.platform == "win32":
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        try:
            cpu = float(out)
        except ValueError:
            cpu = 0.0
    free = not apps and temp < 70 and cpu < 50
    return free, f"GPU {temp} C {util}% other GPU processes {apps or '-'} CPU {cpu:.0f}%"


def wait_until_free() -> None:
    last = ""
    while True:
        free, state = machine_state()
        if free:
            time.sleep(60)
            free, state = machine_state()
            if free:
                log(f"free: {state}")
                return
        if state != last:
            log(f"waiting: {state}")
            last = state
        time.sleep(60)


def main(steps: list[str]) -> int:
    for raw in steps:
        cheap = raw.startswith("!")
        step = raw.lstrip("!")
        if not cheap:
            wait_until_free()
        log(f"start: {step}")
        t0 = time.time()
        proc = subprocess.Popen(
            [sys.executable, "-m", "vifeedback.cli", *shlex.split(step)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if not any(n in line for n in NOISE):
                print(line, end="", flush=True)
        rc = proc.wait()
        log(f"end: {step} (exit {rc}, {time.time() - t0:.0f} s)")
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
