#!/usr/bin/env python3
"""
Launch simulation cases in parallel and display a live status table.

Usage:
    python run_parallel.py [--main PATH] [--pattern PATTERN] [--jobs N]

Each job writes stdout+stderr to its own file inside log/ in the current directory.
The status table refreshes every second until all jobs finish.
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# ── Two-phase job definitions ─────────────────────────────────────────────────

JOBS_FVM = [
    {"name": "FVM",    "config": "config_twophase_fvm.yaml",    "log": "fvm.log"},
]
JOBS_CG_FVM = [
    {"name": "CG_FVM", "config": "config_twophase_cg_fvm.yaml", "log": "cg_fvm.log"},
]
JOBS_EG = [
    {"name": "EG",     "config": "config_twophase_eg.yaml",     "log": "eg.log"},
]
JOBS_FVM_4W = [
    {"name": "FVM_4w",    "config": "config_twophase_fvm_4well.yaml",    "log": "fvm_4well.log"},
]
JOBS_CG_FVM_4W = [
    {"name": "CG_FVM_4w", "config": "config_twophase_cg_fvm_4well.yaml", "log": "cg_fvm_4well.log"},
]
JOBS_EG_4W = [
    {"name": "EG_4w",     "config": "config_twophase_eg_4well.yaml",     "log": "eg_4well.log"},
]

JOBS_2WELL = JOBS_FVM + JOBS_CG_FVM + JOBS_EG
JOBS_4WELL = JOBS_FVM_4W + JOBS_CG_FVM_4W + JOBS_EG_4W
JOBS_ALL   = JOBS_2WELL + JOBS_4WELL

JOBS_FVM_ALL    = JOBS_FVM    + JOBS_FVM_4W
JOBS_CG_FVM_ALL = JOBS_CG_FVM + JOBS_CG_FVM_4W
JOBS_EG_ALL     = JOBS_EG     + JOBS_EG_4W

# ── Miscible job definitions (auto-generated from directory tree) ──────────────

_MISC_SCENARIOS = [
    "2well-homogeneous", "2well-nugget", "2well-squarehole",
    "4well-homogeneous", "4well-nugget", "4well-squarehole",
]
_MISC_ENGINES = ["EG", "EG_SUPG", "EG_UPWIND", "EG_UPWIND_CWD", "DG1", "CG", "FVM", "WeakEG", "CG_EG", "CG_EG_UPWIND_CWD"]

# Each job runs in its scenario subdirectory so that output_dir resolves correctly.
def _misc_jobs(scenarios, engines=_MISC_ENGINES):
    return [
        {
            "name":   f"{s[:7]}-{e}",
            "config": f"config_{e}.yaml",
            "log":    f"{s}-{e}.log",
            "cwd":    s,
        }
        for s in scenarios for e in engines
    ]

JOBS_MISCIBLE_ALL       = _misc_jobs(_MISC_SCENARIOS)
JOBS_MISCIBLE_2WELL     = _misc_jobs([s for s in _MISC_SCENARIOS if s.startswith("2well")])
JOBS_MISCIBLE_4WELL     = _misc_jobs([s for s in _MISC_SCENARIOS if s.startswith("4well")])
JOBS_MISCIBLE_HOMO      = _misc_jobs([s for s in _MISC_SCENARIOS if "homogeneous" in s])
JOBS_MISCIBLE_NUGGET    = _misc_jobs([s for s in _MISC_SCENARIOS if "nugget"      in s])
JOBS_MISCIBLE_SQUAREHOLE= _misc_jobs([s for s in _MISC_SCENARIOS if "squarehole"  in s])

# Per-engine patterns across all scenarios
_MISC_ENGINE_PATTERNS = {
    f"miscible_{e.lower()}": _misc_jobs(_MISC_SCENARIOS, [e]) for e in _MISC_ENGINES
}

JOBS_MISCIBLE_EG_FAMILY = _misc_jobs(_MISC_SCENARIOS, ["EG", "EG_SUPG", "EG_UPWIND", "EG_UPWIND_CWD"])
JOBS_MISCIBLE_NEW       = _misc_jobs(_MISC_SCENARIOS, ["EG_UPWIND_CWD", "DG1", "CG_EG_UPWIND_CWD"])

PATTERN_MAP = {
    # ── two-phase ─────────────────────────────────────────────────────────────
    "2well":              JOBS_2WELL,
    "4well":              JOBS_4WELL,
    "all":                JOBS_ALL,
    "fvm":                JOBS_FVM,
    "cg_fvm":             JOBS_CG_FVM,
    "eg":                 JOBS_EG,
    "fvm_4well":          JOBS_FVM_4W,
    "cg_fvm_4well":       JOBS_CG_FVM_4W,
    "eg_4well":           JOBS_EG_4W,
    "fvm_all":            JOBS_FVM_ALL,
    "cg_fvm_all":         JOBS_CG_FVM_ALL,
    "eg_all":             JOBS_EG_ALL,
    # ── miscible ──────────────────────────────────────────────────────────────
    "miscible_all":         JOBS_MISCIBLE_ALL,
    "miscible_2well":       JOBS_MISCIBLE_2WELL,
    "miscible_4well":       JOBS_MISCIBLE_4WELL,
    "miscible_homogeneous": JOBS_MISCIBLE_HOMO,
    "miscible_nugget":      JOBS_MISCIBLE_NUGGET,
    "miscible_squarehole":  JOBS_MISCIBLE_SQUAREHOLE,
    "miscible_eg_family":   JOBS_MISCIBLE_EG_FAMILY,
    "miscible_new":         JOBS_MISCIBLE_NEW,
    **_MISC_ENGINE_PATTERNS,
}

DEFAULT_MAIN = "../../bin/main.py"

# ── ANSI helpers ─────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
DIM    = "\033[2m"

def _up(n):   return f"\033[{n}A"
def _clr():   return "\033[2K\r"


# ── Status model ─────────────────────────────────────────────────────────────

@dataclass
class Job:
    name:    str
    config:  str
    log:     str
    cwd:     Optional[str]              = None   # subdirectory to run in (None = caller's cwd)
    proc:    Optional[subprocess.Popen] = None
    start:   Optional[float]            = None
    end:     Optional[float]            = None
    retcode: Optional[int]              = None

    @property
    def status(self) -> str:
        if self.proc is None:         return "pending"
        if self.retcode is None:      return "running"
        return "done" if self.retcode == 0 else "failed"

    @property
    def elapsed(self) -> str:
        if self.start is None:
            return "—"
        t = (self.end or time.time()) - self.start
        m, s = divmod(int(t), 60)
        return f"{m}m {s:02d}s" if m else f"{s}s"

    def status_icon(self) -> str:
        return {
            "pending": f"{DIM}  waiting{RESET}",
            "running": f"{YELLOW}⟳ running{RESET}",
            "done":    f"{GREEN}✓ done   {RESET}",
            "failed":  f"{RED}✗ FAILED {RESET}",
        }[self.status]


# ── Drawing ───────────────────────────────────────────────────────────────────

TABLE_ROWS = None   # set once drawn


def _draw_table(jobs: list[Job], first: bool):
    global TABLE_ROWS
    col_name = max(len(j.name) for j in jobs)
    col_log  = max(len(j.log)  for j in jobs)

    lines = []
    sep   = f"  {'─'*(col_name+2)}┼{'─'*11}┼{'─'*9}┼{'─'*(col_log+2)}"
    hdr   = (f"  {BOLD}{'Job':<{col_name}}  {'Status':9}  {'Elapsed':7}  "
             f"{'Log':<{col_log}}{RESET}")

    lines.append(hdr)
    lines.append(sep)
    for j in jobs:
        lines.append(
            f"  {BOLD}{j.name:<{col_name}}{RESET}  "
            f"{j.status_icon()}  "
            f"{j.elapsed:>7}  "
            f"{DIM}{j.log:<{col_log}}{RESET}"
        )
    lines.append(sep)

    TABLE_ROWS = len(lines)

    if not first:
        sys.stdout.write(_up(TABLE_ROWS))

    for line in lines:
        sys.stdout.write(_clr() + line + "\n")
    sys.stdout.flush()


# ── Runner ────────────────────────────────────────────────────────────────────

def run(main_script: str, job_defs: list = JOBS_2WELL, max_parallel: int = 6):
    # Resolve main_path to absolute so it works regardless of subprocess cwd.
    main_path = Path(main_script).resolve()
    if not main_path.exists():
        print(f"{RED}Error: main script not found: {main_path}{RESET}")
        sys.exit(1)

    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)

    jobs  = [Job(**{**j, "log": str(log_dir / j["log"])}) for j in job_defs]
    queue = list(jobs)  # jobs not yet launched

    print(f"\n{BOLD}Parallel launcher{RESET}")
    print(f"  main        : {main_path}")
    print(f"  jobs        : {len(jobs)}")
    print(f"  max parallel: {max_parallel}")
    print()

    def _launch(j: Job):
        log_fh  = open(j.log, "w")
        run_cwd = Path(j.cwd) if j.cwd else None
        j.proc  = subprocess.Popen(
            ["python", str(main_path), j.config],
            stdout=log_fh, stderr=log_fh,
            cwd=run_cwd,
        )
        j.start = time.time()
        cwd_str = f"  [{j.cwd}]" if j.cwd else ""
        print(f"  {CYAN}▶{RESET} {j.name:20s} → PID {j.proc.pid}{cwd_str}  log: {j.log}")

    # ── launch first batch ───────────────────────────────────────────────────
    for j in queue[:max_parallel]:
        _launch(j)
    queue = queue[max_parallel:]
    print()

    # ── poll until all done ──────────────────────────────────────────────────
    first = True
    while True:
        for j in jobs:
            if j.proc is not None and j.retcode is None:
                rc = j.proc.poll()
                if rc is not None:
                    j.retcode = rc
                    j.end     = time.time()
                    # slot freed — launch next queued job if any
                    if queue:
                        _launch(queue.pop(0))

        _draw_table(jobs, first)
        first = False

        if all(j.retcode is not None for j in jobs if j.proc is not None) and not queue:
            break

        time.sleep(1)

    # ── summary ──────────────────────────────────────────────────────────────
    print()
    n_ok   = sum(1 for j in jobs if j.retcode == 0)
    n_fail = len(jobs) - n_ok
    total  = max((j.end or 0) for j in jobs) - min(j.start for j in jobs)
    m, s   = divmod(int(total), 60)

    if n_fail == 0:
        print(f"{GREEN}{BOLD}✓ All {len(jobs)} jobs completed successfully "
              f"({m}m {s:02d}s wall-clock){RESET}")
    else:
        print(f"{RED}{BOLD}✗ {n_fail} job(s) failed — check the log files above.{RESET}")

    for j in jobs:
        if j.retcode != 0:
            print(f"  {RED}{j.name}: exit code {j.retcode}  →  tail {j.log}{RESET}")

    print()
    return n_fail


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", default=DEFAULT_MAIN,
                        help=f"Path to main.py (default: {DEFAULT_MAIN})")
    parser.add_argument("--pattern", choices=list(PATTERN_MAP), default="2well",
                        help="Job group to run (default: 2well)")
    parser.add_argument("--jobs", "-j", type=int, default=6,
                        help="Maximum simultaneous jobs (default: 6)")
    args = parser.parse_args()

    sys.exit(run(args.main, PATTERN_MAP[args.pattern], max_parallel=args.jobs))
