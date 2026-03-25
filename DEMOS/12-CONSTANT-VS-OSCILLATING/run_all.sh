#!/usr/bin/env python3
"""Run DEMO 12: Constant vs Oscillating load simulations in parallel."""

import sys
import time
import subprocess
import shutil
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

DEMO_DIR = Path(__file__).resolve().parent
SRC_RUN = DEMO_DIR.parents[1] / "SRC" / "run.py"
RUNS_DIR = DEMO_DIR / "runs"
LOG_DIR = DEMO_DIR / "log"

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"

def _up(n):
    return f"\033[{n}A"

def _clr():
    return "\033[2K\r"

def _fmt(s):
    m, s2 = divmod(int(s), 60)
    return f"{m}m {s2:02d}s" if m else f"{s2}s"

@dataclass
class Job:
    label: str
    log: Path
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    log_fh: Optional[object] = field(default=None, repr=False)
    start: Optional[float] = None
    end: Optional[float] = None
    retcode: Optional[int] = None

    @property
    def status(self):
        if self.proc is None:
            return "pending"
        if self.retcode is None:
            return "running"
        return "done" if self.retcode == 0 else "failed"

    @property
    def elapsed(self):
        if self.start is None:
            return "—"
        t = (self.end or time.time()) - self.start
        m, s = divmod(int(t), 60)
        return f"{m}m {s:02d}s" if m else f"{s}s"

    def status_cell(self):
        return {
            "pending": f"{YELLOW}  waiting{RESET}",
            "running": f"{YELLOW}⟳ running{RESET}",
            "done": f"{GREEN}✓ done   {RESET}",
            "failed": f"{RED}✗ FAILED {RESET}",
        }[self.status]

    def last_log_line(self):
        try:
            with open(self.log, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return ""
                buf, pos = b"", size - 1
                while pos >= 0:
                    f.seek(pos)
                    ch = f.read(1)
                    if ch == b"\n" and buf.strip():
                        break
                    buf = ch + buf
                    pos -= 1
            return buf.decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

_TABLE_ROWS = 0

def _draw_table(jobs, t_elapsed, first):
    global _TABLE_ROWS
    col = max(len(j.label) for j in jobs)
    sep = f"  {'─' * (col + 2)}┼{'─' * 11}┼{'─' * 9}┼{'─' * 50}"
    hdr = f"  {BOLD}{'Job':<{col}}  {'Status':9}  {'Elapsed':7}  {'Last log line':<50}{RESET}"
    wall = f"  {CYAN}Wall-clock elapsed: {_fmt(t_elapsed)}{RESET}"

    lines = [wall, hdr, sep]
    for j in jobs:
        hint = (
            j.last_log_line()
            if j.status == "running"
            else f"completed in {j.elapsed}"
            if j.status == "done"
            else f"FAILED after {j.elapsed}  (rc={j.retcode})"
            if j.status == "failed"
            else ""
        )
        hint = (hint[:48] + "…") if len(hint) > 49 else hint
        lines.append(
            f"  {BOLD}{j.label:<{col}}{RESET}  "
            f"{j.status_cell()}  "
            f"{j.elapsed:>7}  "
            f"{RESET}{hint:<50}"
        )
    lines.append(sep)

    _TABLE_ROWS = len(lines)
    if not first:
        sys.stdout.write(_up(_TABLE_ROWS))
    for line in lines:
        sys.stdout.write(_clr() + line + "\n")
    sys.stdout.flush()

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        ("constant_5mpa",      "config_constant.yaml",    "Constant 5 MPa"),
        ("oscillating_0_10mpa","config_oscillating.yaml", "Oscillating 0-10 MPa (10 cycles)"),
    ]

    print(f"\n{BOLD}DEMO 12: Constant vs Oscillating Load — launching {len(cases)} parallel runs{RESET}")
    print(f"  solver : {SRC_RUN}")
    print(f"  logs   : {LOG_DIR.relative_to(DEMO_DIR)}/")
    for label, cfg_file, desc in cases:
        print(f"  • {label}: {desc}")
    print()

    jobs = []
    for label, cfg_file, _ in cases:
        run_dir = RUNS_DIR / label
        log_path = LOG_DIR / f"{label}.log"
        (run_dir / "outputs").mkdir(parents=True, exist_ok=True)

        # Copy the template config into the run directory
        src_cfg = DEMO_DIR / cfg_file
        dst_cfg = run_dir / "config.yaml"
        shutil.copy2(src_cfg, dst_cfg)

        job = Job(label=label, log=log_path)
        job.log_fh = open(log_path, "w")
        job.proc = subprocess.Popen(
            [sys.executable, "-u", str(SRC_RUN)],
            cwd=str(run_dir),
            stdout=job.log_fh,
            stderr=subprocess.STDOUT,
        )
        job.start = time.time()
        jobs.append(job)
        print(f"  {CYAN}▶{RESET} {label:<25}  → PID {job.proc.pid}  log: {log_path.relative_to(DEMO_DIR)}")

    print()
    t_start = time.time()
    first = True

    try:
        while True:
            for j in jobs:
                if j.retcode is None and j.proc is not None:
                    rc = j.proc.poll()
                    if rc is not None:
                        j.retcode = rc
                        j.end = time.time()
            _draw_table(jobs, time.time() - t_start, first)
            first = False
            if all(j.retcode is not None for j in jobs):
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted — terminating child processes…{RESET}")
        for j in jobs:
            j.proc.terminate()
            if j.end is None:
                j.end = time.time()

    for j in jobs:
        j.log_fh.close()

    print()
    n_ok = sum(1 for j in jobs if j.retcode == 0)
    n_fail = len(jobs) - n_ok
    t_wall = max(j.end or 0 for j in jobs) - min(j.start for j in jobs)

    if n_fail == 0:
        print(f"{GREEN}{BOLD}✓ All {len(jobs)} runs completed successfully ({_fmt(t_wall)} wall-clock){RESET}")
        print("  Now run: python plot_comparison.py")
    else:
        print(f"{RED}{BOLD}✗ {n_fail} run(s) failed — check logs below.{RESET}")
        for j in jobs:
            if j.retcode != 0:
                print(f"  {RED}{j.label}: rc={j.retcode}  →  {j.log}{RESET}")
    print()

if __name__ == "__main__":
    main()
