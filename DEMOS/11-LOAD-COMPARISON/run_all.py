#!/usr/bin/env python
"""Load-strategy comparison: 10 parallel runs (5 perms × 2 setups).

Setup A — stress-then-relax : t_start=50s, 500s at L=-5 MPa, then 1000s at L=-1 MPa
Setup B — constant baseline  : t_start=50s, 1500s at L=-1 MPa, then return to 0

Geometry  : Re=0.1 m, H=0.02 m
Materials : E=1.44e10, nu=0.2, alpha=0.75, M=1.35e10, visc=1e-3
Perm sweep: logspace(1e-20, 1e-16, 5)

Output: runs/<setup>_perm_<value>/outputs/
Logs  : log/<setup>_perm_<value>.log
"""
import sys
import time
import subprocess
import yaml
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEMO_DIR = Path(__file__).resolve().parent
SRC_RUN  = DEMO_DIR.parents[1] / "SRC" / "run.py"
RUNS_DIR = DEMO_DIR / "runs"
LOG_DIR  = DEMO_DIR / "log"

# ── Shared constants ──────────────────────────────────────────────────────────
_E, _nu   = 1.44e10, 0.2
_alpha    = 0.75
_M        = 1.35e10
_visc     = 1.0e-3
_H        = 0.02
_mu       = _E / (2 * (1 + _nu))

# Protocol timing
_T_START      = 50.0
_T_ON         = 500.0    # duration at L1 (setup A)
_T_CONST      = 1500.0   # duration at L1 (setup B)
_T_CYCLE_END  = _T_START + _T_CONST          # 1550s — end of active protocol
_T_MIN        = 2.0 * _T_CYCLE_END           # 3100s — ensure observation window
_T_PHYS       = _T_MIN                        # effective simulation end time [s]

# Load levels
_L_BASELINE   = -1.0e6   # -1 MPa  (setup A off-phase and post-cycle; setup B level)
_L_OVERSTRESS = -5.0e6   # -5 MPa  (setup A on-phase)

PERM_VALUES = np.array([1e-22, 1e-21, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16])

def _cv(perm):
    S = 1/_M + _alpha**2 * (1 - 2*_nu) / (2 * _mu * (1 - _nu))
    return perm / (_visc * S)

def _end_time_tv(c_v):
    return float(_T_PHYS * c_v / (4 * _H**2))

def _fmt_perm(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

# ── Case definitions ──────────────────────────────────────────────────────────

def _all_cases():
    cases = []
    for perm in PERM_VALUES:
        cv  = _cv(float(perm))
        etv = _end_time_tv(cv)
        tag = _fmt_perm(float(perm))

        # Setup A: relaxed until t_start, then 500s at -5 MPa,
        # then stepped transition to -1 MPa (5 steps × 100 s), held forever
        cases.append({
            "setup":      "A",
            "label":      f"A_perm_{tag}",
            "perm":       float(perm),
            "end_time_tv": etv,
            "periodic_load": {
                "L0":                  0.0,
                "L1":                  _L_OVERSTRESS,
                "L_after":             _L_BASELINE,
                "t_start":             _T_START,
                "period":              _T_ON,    # 500s — single on-phase, then L_after
                "duty_cycle":          1.0,
                "n_periods":           1,
                "transition_steps":    5,
                "transition_step_dur": 100.0,
            },
        })

        # Setup B: relaxed until t_start, then constant -1 MPa forever
        cases.append({
            "setup":      "B",
            "label":      f"B_perm_{tag}",
            "perm":       float(perm),
            "end_time_tv": etv,
            "periodic_load": {
                "L0":         0.0,
                "L1":         _L_BASELINE,
                "L_after":    _L_BASELINE,
                "t_start":    _T_START,
                "period":     _T_CONST,   # 1500s
                "duty_cycle": 1.0,
                "n_periods":  1,
            },
        })
    return cases

ALL_CASES = _all_cases()

# ── ANSI ──────────────────────────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD   = "\033[1m";  DIM    = "\033[2m"
GREEN  = "\033[32m"; YELLOW = "\033[33m"; RED    = "\033[31m"
CYAN   = "\033[36m"; BLUE   = "\033[34m"

def _up(n):  return f"\033[{n}A"
def _clr():  return "\033[2K\r"
def _fmt(s): m, s2 = divmod(int(s), 60); return f"{m}m {s2:02d}s" if m else f"{s2}s"

_TV_COMMENT = "  # T_v = c_v·t/(4H²)  — dimensionless consolidation time; t_end[s] = 4H²/c_v · T_v"

def _write_config(path, cfg):
    import io, re
    buf = io.StringIO()
    yaml.dump(cfg, buf, default_flow_style=False, sort_keys=False)
    text = re.sub(r"(  end_time_tv:.*)", r"\1" + _TV_COMMENT, buf.getvalue())
    with open(path, "w") as f:
        f.write(text)

SETUP_COLOR = {"A": CYAN, "B": BLUE}

# ── Job dataclass ─────────────────────────────────────────────────────────────

@dataclass
class Job:
    label:   str
    setup:   str
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

# ── Table ─────────────────────────────────────────────────────────────────────

_TABLE_ROWS = 0

def _draw_table(jobs, t_elapsed, first):
    global _TABLE_ROWS
    col  = max(len(j.label) for j in jobs)
    sep  = f"  {'─'*(col+2)}┼{'─'*11}┼{'─'*9}┼{'─'*50}"
    hdr  = f"  {BOLD}{'Job':<{col}}  {'Status':9}  {'Elapsed':7}  {'Last log line':<50}{RESET}"
    wall = f"  {DIM}Wall-clock elapsed: {_fmt(t_elapsed)}{RESET}"

    lines = [wall, hdr, sep]
    prev_setup = None
    for j in jobs:
        if j.setup != prev_setup:
            lines.append(f"  {SETUP_COLOR[j.setup]}{DIM}── Setup {j.setup} {'─'*(col+50)}{RESET}")
            prev_setup = j.setup
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

# ── Config builder ────────────────────────────────────────────────────────────

def _make_config(case):
    return {
        "general": {
            "description": f"Load comparison {case['label']}",
            "tags":        f"load_comparison perm_sweep setup_{case['setup']}",
            "run_dir":     "",
            "run_id":      case["label"],
        },
        "mesh": {"Re": 0.1, "H": _H, "N": 100},
        "materials": {
            "E":     _E,
            "nu":    _nu,
            "alpha": _alpha,
            "perm":  case["perm"],
            "visc":  _visc,
            "M":     _M,
        },
        "boundary_conditions": {
            "bottom": {
                "Pressure": 0.0,
                "U_r":      0.0,
                "periodic_load": case["periodic_load"],
            },
            "top":   {"U_z": 0.0},
            "right": {"U_r": 0.0},
            "left":  {"U_r": 0.0},
        },
        "numerical": {
            "theta_cn":    0.75,
            "end_time_tv": case["end_time_tv"],
            "dt_min_s":    0.01,
            "dt_max_s":    1.0,
            "dt_factor":   1.5,
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

    n_A = sum(1 for c in ALL_CASES if c["setup"] == "A")
    n_B = sum(1 for c in ALL_CASES if c["setup"] == "B")
    print(f"\n{BOLD}Load-strategy comparison — launching {len(ALL_CASES)} parallel runs{RESET}")
    print(f"  solver : {SRC_RUN}")
    print(f"  logs   : {LOG_DIR.relative_to(DEMO_DIR)}/")
    print(f"  Setup A (stress-then-relax): {n_A} cases")
    print(f"  Setup B (constant baseline): {n_B} cases")
    print(f"  t_phys : {_T_PHYS:.0f} s  (≥ 2× protocol end = {_T_CYCLE_END:.0f} s)")
    print()

    jobs = []
    for case in ALL_CASES:
        run_dir  = RUNS_DIR / case["label"]
        log_path = LOG_DIR  / f"{case['label']}.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "outputs").mkdir(exist_ok=True)
        _write_config(run_dir / "config.yaml", _make_config(case))

        job = Job(label=case["label"], setup=case["setup"], log=log_path)
        job.log_fh = open(log_path, "w")
        job.proc   = subprocess.Popen(
            [sys.executable, "-u", str(SRC_RUN)],
            cwd=str(run_dir),
            stdout=job.log_fh,
            stderr=subprocess.STDOUT,
        )
        job.start = time.time()
        jobs.append(job)
        c = SETUP_COLOR[case["setup"]]
        print(f"  {c}▶{RESET} {case['label']:<25}  → PID {job.proc.pid}"
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
            if j.end is None: j.end = time.time()

    for j in jobs:
        j.log_fh.close()

    print()
    n_ok   = sum(1 for j in jobs if j.retcode == 0)
    n_fail = len(jobs) - n_ok
    t_wall = max(j.end or 0 for j in jobs) - min(j.start for j in jobs)

    if n_fail == 0:
        print(f"{GREEN}{BOLD}✓ All {len(jobs)} runs completed successfully "
              f"({_fmt(t_wall)} wall-clock){RESET}")
        print("  Run ./plot_comparison.py to visualise.")
    else:
        print(f"{RED}{BOLD}✗ {n_fail} run(s) failed — check logs below.{RESET}")
        for j in jobs:
            if j.retcode != 0:
                print(f"  {RED}{j.label}: rc={j.retcode}  →  {j.log}{RESET}")
    print()


if __name__ == "__main__":
    main()
