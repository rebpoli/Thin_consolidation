#!/usr/bin/env python
"""Permeability sweep: 5 parallel consolidation runs varying permeability in log-scale.

perm:   logspace from 1e-20 to 1e-16 (5 steps)
alpha:  0.75 (fixed)
Loads:  L0=0 MPa (relaxed, t<50s)  L1=1 MPa  L2=5 MPa  period=100s  5 cycles
Logs:   log/perm_<value>.log
Output: runs/perm_<value>/outputs/
"""
import sys
import time
import subprocess
import yaml
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
DEMO_DIR = Path(__file__).resolve().parent
SRC_RUN  = DEMO_DIR.parents[1] / "SRC" / "run.py"
RUNS_DIR = DEMO_DIR / "runs"
LOG_DIR  = DEMO_DIR / "log"

# ── Case generation ───────────────────────────────────────────────────────────
# Shared material params (used to compute c_v → end_time_tv per case)
_E, _nu, _alpha = 1.44e10, 0.2, 0.75
_M, _visc, _H   = 1.35e10, 1.0e-3, 0.02
_mu = _E / (2 * (1 + _nu))
_T_TARGET = 600.0   # desired physical end time [s]

def _make_cases():
    perm_values = np.logspace(-20, -16, 5)
    cases = []
    for perm in perm_values:
        S          = 1/_M + _alpha**2 * (1 - 2*_nu) / (2 * _mu * (1 - _nu))
        c_v        = perm / (_visc * S)
        end_time_tv = _T_TARGET * c_v / (4 * _H**2)
        cases.append({"perm": float(perm), "end_time_tv": float(end_time_tv)})
    return cases

CASES = _make_cases()

def _fmt_perm(perm: float) -> str:
    """Format perm as a compact string suitable for directory names, e.g. '1e-20'."""
    s = f"{perm:.2e}"
    mantissa, exp = s.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"

# ── ANSI ─────────────────────────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD   = "\033[1m";  DIM    = "\033[2m"
GREEN  = "\033[32m"; YELLOW = "\033[33m"; RED    = "\033[31m"; CYAN   = "\033[36m"

def _up(n):  return f"\033[{n}A"
def _clr():  return "\033[2K\r"

# ── Job dataclass ─────────────────────────────────────────────────────────────

@dataclass
class Job:
    perm:    float
    log:     Path
    proc:    Optional[subprocess.Popen] = field(default=None, repr=False)
    log_fh:  Optional[object]           = field(default=None, repr=False)
    start:   Optional[float]            = None
    end:     Optional[float]            = None
    retcode: Optional[int]              = None

    @property
    def name(self) -> str:
        return f"perm={_fmt_perm(self.perm)}"

    @property
    def status(self) -> str:
        if self.proc is None:    return "pending"
        if self.retcode is None: return "running"
        return "done" if self.retcode == 0 else "failed"

    @property
    def elapsed(self) -> str:
        if self.start is None:
            return "—"
        t = (self.end or time.time()) - self.start
        m, s = divmod(int(t), 60)
        return f"{m}m {s:02d}s" if m else f"{s}s"

    def status_cell(self) -> str:
        return {
            "pending": f"{DIM}  waiting {RESET}",
            "running": f"{YELLOW}⟳ running{RESET}",
            "done":    f"{GREEN}✓ done   {RESET}",
            "failed":  f"{RED}✗ FAILED {RESET}",
        }[self.status]

    def last_log_line(self) -> str:
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

# ── Table drawing ─────────────────────────────────────────────────────────────

_TABLE_ROWS = 0

def _draw_table(jobs: list, t_elapsed: float, first: bool):
    global _TABLE_ROWS
    col = max(len(j.name) for j in jobs)
    sep  = f"  {'─'*(col+2)}┼{'─'*11}┼{'─'*9}┼{'─'*50}"
    hdr  = (f"  {BOLD}{'Job':<{col}}  {'Status':9}  {'Elapsed':7}  "
            f"{'Last log line':<50}{RESET}")
    wall = f"  {DIM}Wall-clock elapsed: {_fmt(t_elapsed)}{RESET}"

    lines = [wall, hdr, sep]
    for j in jobs:
        if j.status == "running":
            hint = j.last_log_line()
            hint = (hint[:48] + "…") if len(hint) > 49 else hint
        elif j.status == "done":
            hint = f"completed in {j.elapsed}"
        elif j.status == "failed":
            hint = f"FAILED after {j.elapsed}  (rc={j.retcode})"
        else:
            hint = ""
        lines.append(
            f"  {BOLD}{j.name:<{col}}{RESET}  "
            f"{j.status_cell()}  "
            f"{j.elapsed:>7}  "
            f"{DIM}{hint:<50}{RESET}"
        )
    lines.append(sep)

    _TABLE_ROWS = len(lines)
    if not first:
        sys.stdout.write(_up(_TABLE_ROWS))
    for line in lines:
        sys.stdout.write(_clr() + line + "\n")
    sys.stdout.flush()

def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"

_TV_COMMENT = "  # T_v = c_v·t/(4H²)  — dimensionless consolidation time; t_end[s] = 4H²/c_v · T_v"

def _write_config(path, cfg):
    import io, re
    buf = io.StringIO()
    yaml.dump(cfg, buf, default_flow_style=False, sort_keys=False)
    text = re.sub(r"(  end_time_tv:.*)", r"\1" + _TV_COMMENT, buf.getvalue())
    with open(path, "w") as f:
        f.write(text)

# ── Config generation ─────────────────────────────────────────────────────────

def make_config(perm: float, end_time_tv: float) -> dict:
    return {
        "general": {
            "description": f"Perm sweep perm={_fmt_perm(perm)} alpha={_alpha}",
            "tags": "perm_sweep periodic",
            "run_dir": "",
            "run_id":  f"perm_{_fmt_perm(perm)}",
        },
        "mesh": {"Re": 0.1, "H": _H, "N": 100},
        "materials": {
            "E":     _E,
            "nu":    _nu,
            "alpha": _alpha,
            "perm":  perm,
            "visc":  _visc,
            "M":     _M,
        },
        "boundary_conditions": {
            "bottom": {
                "Pressure": 0.0,
                "U_r": 0.0,
                "periodic_load": {
                    "L0":         0.0,
                    "L1":        -5.0e6,
                    "t_start":   50.0,
                    "period":   100.0,
                    "duty_cycle": 0.5,
                    "n_periods": -1,
                },
            },
            "top":   {"U_z": 0.0},
            "right": {"U_r": 0.0},
            "left":  {"U_r": 0.0},
        },
        "numerical": {
            "theta_cn":   0.75,
            "end_time_tv": end_time_tv,
            "dt_min_s":   0.01,
            "dt_max_s":   1.0,
            "dt_factor":  1.5,
        },
        "output": {
            "results":    "outputs/results.bp",
            "timeseries": "outputs/fem_timeseries.nc",
        },
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}Perm sweep — launching {len(CASES)} parallel runs{RESET}")
    print(f"  main  : {SRC_RUN}")
    print(f"  logs  : {LOG_DIR.relative_to(DEMO_DIR)}/")
    print(f"  alpha : {_alpha}  (fixed)")
    print(f"  perm  : logspace {_fmt_perm(CASES[0]['perm'])} → {_fmt_perm(CASES[-1]['perm'])}")
    print()

    jobs = []
    for case in CASES:
        perm     = case["perm"]
        run_dir  = RUNS_DIR / f"perm_{_fmt_perm(perm)}"
        log_path = LOG_DIR  / f"perm_{_fmt_perm(perm)}.log"

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "outputs").mkdir(exist_ok=True)
        _write_config(run_dir / "config.yaml", make_config(perm, case["end_time_tv"]))

        job = Job(perm=perm, log=log_path)
        job.log_fh = open(log_path, "w")
        job.proc   = subprocess.Popen(
            [sys.executable, "-u", str(SRC_RUN)],
            cwd=str(run_dir),
            stdout=job.log_fh,
            stderr=subprocess.STDOUT,
        )
        job.start = time.time()
        jobs.append(job)
        print(f"  {CYAN}▶{RESET} {job.name:<16}  → PID {job.proc.pid}"
              f"  log: {log_path.relative_to(DEMO_DIR)}")

    print()
    t_start = time.time()
    first   = True

    try:
        while True:
            for j in jobs:
                if j.retcode is None and j.proc is not None:
                    rc = j.proc.poll()
                    if rc is not None:
                        j.retcode = rc
                        j.end     = time.time()

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
    n_ok   = sum(1 for j in jobs if j.retcode == 0)
    n_fail = len(jobs) - n_ok
    t_wall = max(j.end or 0 for j in jobs) - min(j.start for j in jobs)

    if n_fail == 0:
        print(f"{GREEN}{BOLD}✓ All {len(jobs)} runs completed successfully "
              f"({_fmt(t_wall)} wall-clock){RESET}")
        print("  Run ./plot_uz_comparison.py to visualise.")
    else:
        print(f"{RED}{BOLD}✗ {n_fail} run(s) failed — check logs below.{RESET}")
        for j in jobs:
            if j.retcode != 0:
                print(f"  {RED}{j.name}: rc={j.retcode}  →  {j.log}{RESET}")
    print()


if __name__ == "__main__":
    main()
