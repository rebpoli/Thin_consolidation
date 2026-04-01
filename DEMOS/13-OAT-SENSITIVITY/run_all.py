#!/usr/bin/env python3
"""Demo 13 — One-at-a-time (OAT) sensitivity study.

Base case: phi=0.10, alpha=0.50, perm=1e-18 m²
  M is derived from porosity: M = Kf/phi  (Kf=2.2e9 Pa, Ks>>Kf)

Three independent sweeps:
  phi   : {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}   fixed alpha=0.50, perm=1e-18
  alpha : {0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0} fixed phi=0.10, perm=1e-18
  perm  : logspace(1e-21, 1e-16, 6)               fixed phi=0.10, alpha=0.50

Geometry  : Re=0.025 m, H=0.005 m
Materials : E=3e9, nu=0.2, visc=1e-3 (water)
Load      : single step to −10 MPa at t_start=50 s, held to t_end=3600 s
"""

import sys, time, subprocess, yaml, io, re
import numpy as np


# ── Compact YAML float formatter ──────────────────────────────────────────────
def _yaml_float(v):
    """Format float: scientific notation for |v| >= 1e4 or |v| < 1e-3.

    PyYAML's implicit float resolver requires a decimal point in the mantissa
    (pattern: [-+]?[0-9]+.[0-9]*[eE][-+][0-9]+).  Numbers without a decimal
    (e.g. '3e+9') are treated as strings and get an ugly !!float tag.
    We therefore ensure a '.0' is present when the mantissa is an integer.
    """
    if v == 0.0:
        return "0.0"
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        s = f"{v:.6g}"                                    # e.g. "4.4e+10", "1e-18"
        s = re.sub(r"^([-+]?\d+)(e)", r"\1.0\2", s)     # 3e+9 → 3.0e+9  (add decimal)
        s = re.sub(r"e\+0*(\d+)", r"e+\1", s)            # e+010 → e+10
        s = re.sub(r"e-0*(\d+)", r"e-\1", s)             # e-03  → e-3
        return s
    return str(v)

class _CompactDumper(yaml.Dumper):
    pass

_CompactDumper.add_representer(
    float,
    lambda d, v: d.represent_scalar("tag:yaml.org,2002:float", _yaml_float(v)),
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEMO_DIR = Path(__file__).resolve().parent
SRC_RUN  = DEMO_DIR.parents[1] / "SRC" / "run.py"
RUNS_DIR = DEMO_DIR / "runs"
LOG_DIR  = DEMO_DIR / "log"

# ── Shared constants ──────────────────────────────────────────────────────────
_E     = 3.0e9
_nu    = 0.2
_visc  = 1.0e-3          # water viscosity
_Kf    = 2.2e9           # water bulk modulus
_H     = 0.005           # half-height [m]
_Re    = 0.025           # radius [m]
_N     = 80              # mesh elements
_mu    = _E / (2 * (1 + _nu))

# Base case parameters
_PHI_BASE   = 0.10
_ALPHA_BASE = 0.50
_PERM_BASE  = 1.0e-20

# Load
_L0      = 0.0
_L1      = -10.0e6       # −10 MPa step load
_T_START = 50.0          # ramp-up time [s]
_T_END   = 3600.0        # physical end time [s] = 1 hour

# Sweep values
PHI_VALUES   = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
ALPHA_VALUES = np.array([0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
PERM_VALUES  = np.array([1e-21, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16])


def _M(phi):
    """Biot modulus from porosity: M = Kf/phi  (assumes Ks >> Kf)."""
    return _Kf / phi


def _S(alpha, phi):
    M = _M(phi)
    return 1.0 / M + alpha**2 * (1 - 2 * _nu) / (2 * _mu * (1 - _nu))


def _cv(perm, alpha, phi):
    return perm / (_visc * _S(alpha, phi))


def _end_time_tv(perm, alpha, phi):
    cv = _cv(perm, alpha, phi)
    return float(cv * _T_END / (4 * _H**2))


def _fmt_perm(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"


def _load_block():
    """Periodic-load config for a single held step (n_periods=-1, duty_cycle=1)."""
    return {
        "L0":          _L0,
        "L1":          _L1,
        "t_start":     _T_START,
        "period":      7200.0,   # > t_end − t_start; never completes within simulation
        "duty_cycle":  1.0,
        "n_periods":   -1,       # infinite → stays at L1 forever after t_start
    }


_TV_COMMENT = ("  # T_v = c_v·t/(4H²) — dimensionless consolidation time; "
               "t_end[s] = 4H²/c_v · T_v")


def _write_config(path, cfg):
    buf = io.StringIO()
    yaml.dump(cfg, buf, Dumper=_CompactDumper, default_flow_style=False, sort_keys=False)
    text = re.sub(r"(  end_time_tv:.*)", r"\1" + _TV_COMMENT, buf.getvalue())
    with open(path, "w") as f:
        f.write(text)


def _make_config(label, perm, alpha, phi):
    M = _M(phi)
    return {
        "general": {
            "description": f"OAT sensitivity {label}",
            "tags":        "oat_sensitivity demo13",
            "run_dir":     "",
            "run_id":      label,
        },
        "mesh": {"Re": float(_Re), "H": float(_H), "N": _N},
        "materials": {
            "E":     _E,
            "nu":    _nu,
            "alpha": float(alpha),
            "perm":  float(perm),
            "visc":  _visc,
            "M":     float(M),
        },
        "boundary_conditions": {
            "bottom": {
                "Pressure":      0.0,
                "U_r":           0.0,
                "periodic_load": _load_block(),
            },
            "top":   {"U_z": 0.0},
            "right": {"U_r": 0.0},
            "left":  {"U_r": 0.0},
        },
        "numerical": {
            "theta_cn":    0.75,
            "end_time_tv": _end_time_tv(perm, alpha, phi),
            "dt_min_s":    0.01,
            "dt_max_s":    20.0,
            "dt_factor":   1.5,
        },
        "output": {
            "results":    "outputs/results.bp",
            "timeseries": "outputs/fem_timeseries.nc",
        },
    }


# ── Case list ─────────────────────────────────────────────────────────────────

def _all_cases():
    cases = []

    # φ sweep — alpha and perm fixed at base
    for phi in PHI_VALUES:
        lbl = f"phi_{phi:.2f}"
        cases.append(dict(sweep="phi", label=lbl,
                          perm=_PERM_BASE, alpha=_ALPHA_BASE, phi=phi))

    # α sweep — phi and perm fixed at base
    for alpha in ALPHA_VALUES:
        lbl = f"alpha_{alpha:.2f}"
        cases.append(dict(sweep="alpha", label=lbl,
                          perm=_PERM_BASE, alpha=alpha, phi=_PHI_BASE))

    # k sweep — phi and alpha fixed at base
    for perm in PERM_VALUES:
        lbl = f"perm_{_fmt_perm(perm)}"
        cases.append(dict(sweep="perm", label=lbl,
                          perm=perm, alpha=_ALPHA_BASE, phi=_PHI_BASE))

    return cases


ALL_CASES = _all_cases()

# ── ANSI ──────────────────────────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD   = "\033[1m";  DIM    = "\033[2m"
GREEN  = "\033[32m"; YELLOW = "\033[33m"; RED    = "\033[31m"
CYAN   = "\033[36m"; BLUE   = "\033[34m"; MAGENTA = "\033[35m"

SWEEP_COLOR = {"phi": CYAN, "alpha": BLUE, "perm": MAGENTA}

def _up(n):  return f"\033[{n}A"
def _clr():  return "\033[2K\r"
def _fmt(s): m, s2 = divmod(int(s), 60); return f"{m}m {s2:02d}s" if m else f"{s2}s"


@dataclass
class Job:
    label:   str
    sweep:   str
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


_TABLE_ROWS = 0

def _draw_table(jobs, t_elapsed, first):
    global _TABLE_ROWS
    col  = max(len(j.label) for j in jobs)
    sep  = f"  {'─'*(col+2)}┼{'─'*11}┼{'─'*9}┼{'─'*50}"
    hdr  = f"  {BOLD}{'Job':<{col}}  {'Status':9}  {'Elapsed':7}  {'Last log line':<50}{RESET}"
    wall = f"  {DIM}Wall-clock elapsed: {_fmt(t_elapsed)}{RESET}"

    lines = [wall, hdr, sep]
    prev_sweep = None
    for j in jobs:
        if j.sweep != prev_sweep:
            lines.append(f"  {SWEEP_COLOR[j.sweep]}{DIM}── {j.sweep} sweep {'─'*(col+50)}{RESET}")
            prev_sweep = j.sweep
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


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    n_by_sweep = {s: sum(1 for c in ALL_CASES if c["sweep"] == s)
                  for s in ("phi", "alpha", "perm")}
    print(f"\n{BOLD}Demo 13 — OAT Sensitivity — {len(ALL_CASES)} parallel runs{RESET}")
    print(f"  solver  : {SRC_RUN}")
    print(f"  logs    : {LOG_DIR.relative_to(DEMO_DIR)}/")
    print(f"  base    : phi={_PHI_BASE}, alpha={_ALPHA_BASE}, perm={_fmt_perm(_PERM_BASE)}")
    print(f"  E={_E:.1e} Pa   nu={_nu}   Kf={_Kf:.1e} Pa   M=Kf/phi")
    print(f"  H={_H*100:.1f} cm   Re={_Re*100:.1f} cm   t_end={_T_END:.0f} s (1 h)")
    for s, n in n_by_sweep.items():
        print(f"  {s} sweep: {n} cases")
    print()

    jobs = []
    for case in ALL_CASES:
        run_dir  = RUNS_DIR / case["label"]
        log_path = LOG_DIR  / f"{case['label']}.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "outputs").mkdir(exist_ok=True)
        cfg = _make_config(case["label"], case["perm"], case["alpha"], case["phi"])
        _write_config(run_dir / "config.yaml", cfg)

        job = Job(label=case["label"], sweep=case["sweep"], log=log_path)
        job.log_fh = open(log_path, "w")
        job.proc   = subprocess.Popen(
            [sys.executable, "-u", str(SRC_RUN)],
            cwd=str(run_dir),
            stdout=job.log_fh,
            stderr=subprocess.STDOUT,
        )
        job.start = time.time()
        jobs.append(job)
        c = SWEEP_COLOR[case["sweep"]]
        print(f"  {c}▶{RESET} {case['label']:<25}  PID {job.proc.pid}"
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
