#!/usr/bin/env python3
"""Generic parallel sweep runner.

Discovers all runs/*/config.yaml under the current demo directory and runs
the FEM solver in each case in parallel, displaying a live progress table.

USAGE (from a demo directory):
    ./run_sweep.py                  # run all cases found in runs/
    ./run_sweep.py --dry-run        # preview without running
    ./run_sweep.py --n-jobs 4       # limit parallel workers (default: all at once)
    ./run_sweep.py --rerun-failed   # re-run only failed/missing cases

The solver executed per case is <repo-root>/SRC/run.py.  Each run is launched
with its run directory as cwd so that config.yaml is picked up automatically.
"""

import argparse
import sys
import time
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Resolve paths ──────────────────────────────────────────────────────────────
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _TOOLS_DIR.parent
SRC_RUN     = _REPO_ROOT / "SRC" / "run.py"

# ── ANSI helpers ───────────────────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD  = "\033[1m";  DIM  = "\033[2m"
GREEN  = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"

def _up(n):  return f"\033[{n}A"
def _clr():  return "\033[2K\r"
def _fmt_t(s):
    m, s2 = divmod(int(s), 60)
    return f"{m}m {s2:02d}s" if m else f"{s2}s"


# ── Job dataclass ──────────────────────────────────────────────────────────────
@dataclass
class Job:
    label:   str
    run_dir: Path
    log:     Path
    proc:    Optional[subprocess.Popen] = field(default=None, repr=False)
    log_fh:  Optional[object]           = field(default=None, repr=False)
    start:   Optional[float]            = None
    end:     Optional[float]            = None
    retcode: Optional[int]              = None

    @property
    def status(self):
        if self.proc is None:    return "pending"
        if self.retcode is None: return "running"
        return "done" if self.retcode == 0 else "failed"

    @property
    def elapsed(self):
        if self.start is None: return "—"
        t = (self.end or time.time()) - self.start
        m, s = divmod(int(t), 60)
        return f"{m}m {s:02d}s" if m else f"{s}s"

    def status_cell(self):
        return {"pending": f"{DIM}  waiting {RESET}",
                "running": f"{YELLOW}⟳ running{RESET}",
                "done":    f"{GREEN}✓ done   {RESET}",
                "failed":  f"{RED}✗ FAILED {RESET}"}[self.status]

    def last_log_line(self):
        try:
            with open(self.log, "rb") as f:
                f.seek(0, 2); size = f.tell()
                if size == 0: return ""
                buf, pos = b"", size - 1
                while pos >= 0:
                    f.seek(pos); ch = f.read(1)
                    if ch == b"\n" and buf.strip(): break
                    buf = ch + buf; pos -= 1
            return buf.decode("utf-8", errors="replace").strip()
        except OSError:
            return ""


# ── Progress table ─────────────────────────────────────────────────────────────
_TABLE_ROWS = 0

def _draw_table(jobs, t_elapsed, first):
    global _TABLE_ROWS
    col = max(len(j.label) for j in jobs)
    sep = f"  {'─'*(col+2)}┼{'─'*11}┼{'─'*9}┼{'─'*50}"
    hdr = f"  {BOLD}{'Job':<{col}}  {'Status':9}  {'Elapsed':7}  {'Last log line':<50}{RESET}"
    wall = f"  {DIM}Wall-clock elapsed: {_fmt_t(t_elapsed)}{RESET}"

    lines = [wall, hdr, sep]
    for j in jobs:
        hint = (j.last_log_line() if j.status == "running"
                else f"completed in {j.elapsed}" if j.status == "done"
                else f"FAILED after {j.elapsed}  (rc={j.retcode})" if j.status == "failed"
                else "")
        hint = (hint[:48] + "…") if len(hint) > 49 else hint
        lines.append(f"  {BOLD}{j.label:<{col}}{RESET}  "
                     f"{j.status_cell()}  "
                     f"{j.elapsed:>7}  "
                     f"{DIM}{hint:<50}{RESET}")
    lines.append(sep)

    _TABLE_ROWS = len(lines)
    if not first:
        sys.stdout.write(_up(_TABLE_ROWS))
    for line in lines:
        sys.stdout.write(_clr() + line + "\n")
    sys.stdout.flush()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-jobs", type=int, default=0,
                        help="Max parallel jobs (0 = unlimited, default)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print cases that would run, then exit")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="Only re-run cases whose outputs/fem_timeseries.nc is missing")
    parser.add_argument("--runs-dir", default="runs",
                        help="Directory containing per-case subdirectories (default: runs/)")
    args = parser.parse_args()

    demo_dir = Path.cwd()
    runs_dir = demo_dir / args.runs_dir
    log_dir  = demo_dir / "log"

    if not runs_dir.exists():
        print(f"{RED}Error:{RESET} runs directory not found: {runs_dir}")
        print("  Run py/gen_configs.py first.")
        sys.exit(1)

    # Discover cases
    cases = sorted(p.parent for p in runs_dir.glob("*/config.yaml"))
    if not cases:
        print(f"{RED}Error:{RESET} no config.yaml files found under {runs_dir}")
        sys.exit(1)

    if args.rerun_failed:
        cases = [c for c in cases
                 if not (c / "outputs" / "fem_timeseries.nc").exists()]
        if not cases:
            print(f"{GREEN}All cases already have outputs — nothing to rerun.{RESET}")
            return

    print(f"\n{BOLD}Sweep runner — {len(cases)} case(s)  |  solver: {SRC_RUN}{RESET}")
    for c in cases:
        print(f"  {c.relative_to(demo_dir)}")
    print()

    if args.dry_run:
        print(f"{DIM}--dry-run: exiting without running.{RESET}")
        return

    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    pending = list(cases)
    n_slots = args.n_jobs if args.n_jobs > 0 else len(cases)

    # Launch initial batch
    while pending and len(jobs) < n_slots:
        run_dir = pending.pop(0)
        label   = run_dir.name
        log_path = log_dir / f"{label}.log"
        (run_dir / "outputs").mkdir(parents=True, exist_ok=True)

        job = Job(label=label, run_dir=run_dir, log=log_path)
        job.log_fh = open(log_path, "w")
        job.proc   = subprocess.Popen(
            [sys.executable, "-u", str(SRC_RUN)],
            cwd=str(run_dir),
            stdout=job.log_fh,
            stderr=subprocess.STDOUT,
        )
        job.start = time.time()
        jobs.append(job)
        print(f"  {GREEN}▶{RESET} {label:<30}  PID {job.proc.pid}")

    print()
    t_start = time.time()
    first   = True

    try:
        while True:
            for j in jobs:
                if j.retcode is None:
                    rc = j.proc.poll()
                    if rc is not None:
                        j.retcode = rc
                        j.end     = time.time()
                        # Launch next pending job if slot freed
                        if pending:
                            run_dir  = pending.pop(0)
                            label    = run_dir.name
                            log_path = log_dir / f"{label}.log"
                            (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
                            nj = Job(label=label, run_dir=run_dir, log=log_path)
                            nj.log_fh = open(log_path, "w")
                            nj.proc   = subprocess.Popen(
                                [sys.executable, "-u", str(SRC_RUN)],
                                cwd=str(run_dir),
                                stdout=nj.log_fh,
                                stderr=subprocess.STDOUT,
                            )
                            nj.start = time.time()
                            jobs.append(nj)

            _draw_table(jobs, time.time() - t_start, first)
            first = False
            if all(j.retcode is not None for j in jobs) and not pending:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted — terminating child processes…{RESET}")
        for j in jobs:
            if j.retcode is None:
                j.proc.terminate()
                j.end = time.time()

    for j in jobs:
        j.log_fh.close()

    print()
    n_ok   = sum(1 for j in jobs if j.retcode == 0)
    n_fail = len(jobs) - n_ok
    t_wall = max(j.end or 0 for j in jobs) - min(j.start for j in jobs if j.start)

    if n_fail == 0:
        print(f"{GREEN}{BOLD}✓ All {len(jobs)} runs completed  ({_fmt_t(t_wall)} wall-clock){RESET}")
    else:
        print(f"{RED}{BOLD}✗ {n_fail} run(s) failed — check logs in log/{RESET}")
        for j in jobs:
            if j.retcode != 0:
                print(f"  {RED}{j.label}: rc={j.retcode}  →  {j.log}{RESET}")
    print()


if __name__ == "__main__":
    main()
