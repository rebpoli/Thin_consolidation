#!/usr/bin/env python
"""Full parameter sweep: 15 parallel runs across three sub-batches.

  Sub-batch 1 — Biot sweep  : α ∈ {0.2, 0.4, 0.6, 0.8, 1.0}   (M=1.35e10, perm=1e-20)
  Sub-batch 2 — M sweep     : M ∈ logspace(1e8, 1.5e10, 5)      (α=0.75,    perm=1e-20)
  Sub-batch 3 — Perm sweep  : k ∈ logspace(1e-20, 1e-16, 5)     (α=0.75,    M=1.35e10)

Geometry  : Re=0.1 m, H=0.02 m
Load      : L0=-1 MPa (baseline), L1=-5 MPa (over-stress), duty_cycle=90%,
             t_start=50 s, period=100 s, n_periods=5, L_after=L0
Timestepping: dt_min=0.01 s, dt_max=1 s, factor=1.5

Output: runs/<group>_<value>/outputs/
Logs  : log/<group>_<value>.log
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
_H, _visc = 0.02, 1.0e-3
_mu       = _E / (2 * (1 + _nu))
_T_TARGET = 600.0        # desired physical end time [s] (may be raised by load constraint)

_M_REF    = 1.35e10   # used when M is fixed
_PERM_REF = 1.0e-20   # used when perm is fixed
_ALPHA_REF= 0.75      # used when alpha is fixed

# ── Load parameters (single source of truth) ──────────────────────────────────
_LOAD_L0         = -1.0e6   # baseline / off-phase level [Pa]
_LOAD_L1         = -5.0e6   # over-stress / on-phase level [Pa]
_LOAD_L_AFTER    = _LOAD_L0 # return to baseline after all cycles
_LOAD_T_START    = 50.0
_LOAD_PERIOD     = 100.0
_LOAD_DUTY_CYCLE = 0.9      # fraction of period at over-stress L1
_LOAD_N_PERIODS  = 5

# End time must be at least 2× the total cycle duration
_T_CYCLE_END = _LOAD_T_START + _LOAD_N_PERIODS * _LOAD_PERIOD   # 550 s
_T_MIN       = 2.0 * _T_CYCLE_END                                # 1100 s
_T_PHYS      = max(_T_TARGET, _T_MIN)                            # effective target [s]

def _cv(perm, M, alpha):
    S = 1/M + alpha**2 * (1 - 2*_nu) / (2 * _mu * (1 - _nu))
    return perm / (_visc * S)

def _end_time_tv(c_v):
    return _T_PHYS * c_v / (4 * _H**2)

def _fmt_alpha(v): return f"{v}"
def _fmt_M(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"
def _fmt_perm(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

# ── Case definitions ──────────────────────────────────────────────────────────

def _biot_cases():
    cases = []
    for alpha in [0.2, 0.4, 0.6, 0.8, 1.0]:
        cv = _cv(_PERM_REF, _M_REF, alpha)
        cases.append({"group": "biot", "key": alpha, "label": f"biot_{_fmt_alpha(alpha)}",
                      "end_time_tv": float(_end_time_tv(cv)),
                      "alpha": alpha, "M": _M_REF, "perm": _PERM_REF})
    return cases

def _M_cases():
    cases = []
    for M in np.logspace(np.log10(1e8), np.log10(1.5e10), 5):
        cv = _cv(_PERM_REF, float(M), _ALPHA_REF)
        cases.append({"group": "M", "key": float(M), "label": f"M_{_fmt_M(M)}",
                      "end_time_tv": _end_time_tv(cv),
                      "alpha": _ALPHA_REF, "M": float(M), "perm": _PERM_REF})
    return cases

def _perm_cases():
    cases = []
    for perm in np.logspace(-20, -16, 5):
        cv = _cv(float(perm), _M_REF, _ALPHA_REF)
        cases.append({"group": "perm", "key": float(perm), "label": f"perm_{_fmt_perm(perm)}",
                      "end_time_tv": _end_time_tv(cv),
                      "alpha": _ALPHA_REF, "M": _M_REF, "perm": float(perm)})
    return cases

ALL_CASES = _biot_cases() + _M_cases() + _perm_cases()

# ── ANSI ──────────────────────────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD   = "\033[1m";  DIM    = "\033[2m"
GREEN  = "\033[32m"; YELLOW = "\033[33m"; RED    = "\033[31m"; CYAN   = "\033[36m"
BLUE   = "\033[34m"; MAGENTA= "\033[35m"

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

GROUP_COLOR = {"biot": CYAN, "M": MAGENTA, "perm": BLUE}

# ── Job dataclass ─────────────────────────────────────────────────────────────

@dataclass
class Job:
    label:   str
    group:   str
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
    prev_group = None
    for j in jobs:
        if j.group != prev_group:
            lines.append(f"  {GROUP_COLOR[j.group]}{DIM}── {j.group} sweep {'─'*(col+50)}{RESET}")
            prev_group = j.group
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
            "description": f"Full sweep {case['label']}",
            "tags":        f"{case['group']}_sweep periodic",
            "run_dir":     "",
            "run_id":      case["label"],
        },
        "mesh": {"Re": 0.1, "H": _H, "N": 100},
        "materials": {
            "E":     _E,
            "nu":    _nu,
            "alpha": case["alpha"],
            "perm":  case["perm"],
            "visc":  _visc,
            "M":     case["M"],
        },
        "boundary_conditions": {
            "bottom": {
                "Pressure": 0.0,
                "U_r":      0.0,
                "periodic_load": {
                    "L0":         _LOAD_L0,
                    "L1":         _LOAD_L1,
                    "L_after":    _LOAD_L_AFTER,
                    "t_start":    _LOAD_T_START,
                    "period":     _LOAD_PERIOD,
                    "duty_cycle": _LOAD_DUTY_CYCLE,
                    "n_periods":  _LOAD_N_PERIODS,
                },
            },
            "top":   {"U_z": 0.0},
            "right": {"U_r": 0.0},
            "left":  {"U_r": 0.0},
        },
        "numerical": {
            "theta_cn":   0.75,
            "end_time_tv": case["end_time_tv"],
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

    print(f"\n{BOLD}Full parameter sweep — launching {len(ALL_CASES)} parallel runs{RESET}")
    print(f"  solver : {SRC_RUN}")
    print(f"  logs   : {LOG_DIR.relative_to(DEMO_DIR)}/")
    print(f"  groups : biot ({len(_biot_cases())})  |  M ({len(_M_cases())})  |  perm ({len(_perm_cases())})")
    print()

    jobs = []
    for case in ALL_CASES:
        run_dir  = RUNS_DIR / case["label"]
        log_path = LOG_DIR  / f"{case['label']}.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "outputs").mkdir(exist_ok=True)
        _write_config(run_dir / "config.yaml", _make_config(case))

        job = Job(label=case["label"], group=case["group"], log=log_path)
        job.log_fh = open(log_path, "w")
        job.proc   = subprocess.Popen(
            [sys.executable, "-u", str(SRC_RUN)],
            cwd=str(run_dir),
            stdout=job.log_fh,
            stderr=subprocess.STDOUT,
        )
        job.start = time.time()
        jobs.append(job)
        c = GROUP_COLOR[case["group"]]
        print(f"  {c}▶{RESET} {case['label']:<20}  → PID {job.proc.pid}"
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
