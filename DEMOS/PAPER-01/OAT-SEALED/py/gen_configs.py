#!/usr/bin/env python3
"""Generate per-case config.yaml files for the OAT-SEALED sweep.

OAT (one-at-a-time) sensitivity study — sealed lateral boundary (U_r=0 on right).
Base case: phi=0.10, alpha=0.50, perm=1e-20 m²

Three independent sweeps:
  phi   : {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}        fixed alpha=0.50, perm=1e-20
  alpha : {0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00}  fixed phi=0.10,   perm=1e-20
  perm  : logspace(1e-21, 1e-16, 6)                    fixed phi=0.10,   alpha=0.50

Writes one runs/<label>/config.yaml per case.
Run TOOLS/run_sweep.py afterwards to execute all cases.

USAGE:
    ./py/gen_configs.py
    ./py/gen_configs.py --dry-run
"""

import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
DEMO_DIR = Path(__file__).resolve().parents[1]   # OAT-SEALED/
RUNS_DIR = DEMO_DIR / "runs"

# ── Compact YAML float formatter ───────────────────────────────────────────────
def _yaml_float(v):
    """Ensure PyYAML writes floats with a decimal point in scientific notation."""
    if v == 0.0:
        return "0.0"
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        s = f"{v:.6g}"
        s = re.sub(r"^([-+]?\d+)(e)", r"\1.0\2", s)
        s = re.sub(r"e\+0*(\d+)", r"e+\1", s)
        s = re.sub(r"e-0*(\d+)", r"e-\1", s)
        return s
    return str(v)

class _CompactDumper(yaml.Dumper):
    pass

_CompactDumper.add_representer(
    float,
    lambda d, v: d.represent_scalar("tag:yaml.org,2002:float", _yaml_float(v)),
)

# ── Shared material / geometry constants ───────────────────────────────────────
_E     = 5.0e9
_nu    = 0.40
_visc  = 1.0e-3          # water viscosity [Pa·s]
_Kf    = 2.2e9           # water bulk modulus [Pa]
_H     = 0.010           # full specimen height [m] (mesh spans 0 to H/2)
_Re    = 0.025           # radius [m]
_N     = 30              # cells per direction
_mu    = _E / (2 * (1 + _nu))

# Base case
_PHI_BASE   = 0.10
_ALPHA_BASE = 0.75
_PERM_BASE  = 1.0e-20

# Load
_L0      = 0.0
_L1      = -10.0e6       # −10 MPa step
_T_START = 50.0          # ramp-up time [s]
_T_END   = 36000.0       # end time [s] = 10 hours

# Sweep values
PHI_VALUES   = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
ALPHA_VALUES = np.array([0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90])
PERM_VALUES  = np.array([1e-18, 1e-19, 1e-20, 1e-21, 1e-22])

# ── Derived quantities ─────────────────────────────────────────────────────────
def _M(phi):
    return _Kf / phi

def _storativity(alpha, phi):
    M = _M(phi)
    return 1.0 / M + alpha**2 * (1 - 2 * _nu) / (2 * _mu * (1 - _nu))

def _cv(perm, alpha, phi):
    return perm / (_visc * _storativity(alpha, phi))

def _end_time_tv(perm, alpha, phi):
    H_dr = _H / 2
    return float(_cv(perm, alpha, phi) * _T_END / H_dr**2)

def _fmt_perm(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

# ── Config builder ─────────────────────────────────────────────────────────────
_TV_COMMENT = ("  # T_v = c_v·t/(4H²) — dimensionless consolidation time; "
               "t_end[s] = 4H²/c_v · T_v")

def _write_config(path, cfg):
    buf = io.StringIO()
    yaml.dump(cfg, buf, Dumper=_CompactDumper, default_flow_style=False, sort_keys=False)
    text = re.sub(r"(  end_time_tv:.*)", r"\1" + _TV_COMMENT, buf.getvalue())
    path.write_text(text)

def _make_config(label, perm, alpha, phi):
    return {
        "general": {
            "description": f"OAT-SEALED {label}",
            "tags":        "oat_sensitivity sealed",
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
            "M":     float(_M(phi)),
        },
        "boundary_conditions": {
            "bottom": {"U_z": 0.0},
            "top": {
                "Pressure":      0.0,
                "U_z_rigid":     1,
                "periodic_load": {
                    "L0":         _L0,
                    "L1":         _L1,
                    "t_start":    _T_START,
                    "period":     72000.0,
                    "duty_cycle": 1.0,
                    "n_periods":  -1,
                },
            },
            "right": {"U_r": 0.0},   # sealed: no lateral displacement or drainage
            "left":  {"U_r": 0.0},   # axis of symmetry
        },
        "numerical": {
            "theta_cn":    0.75,
            "end_time_tv": _end_time_tv(perm, alpha, phi),
            "dt_min_s":    0.001,
            "dt_max_s":    600.0,
            "dt_factor":   1.5,
        },
        "output": {
            "results":    "outputs/results.bp",
            "timeseries": "outputs/fem_timeseries.nc",
        },
    }

# ── Case list ──────────────────────────────────────────────────────────────────
def _all_cases():
    cases = []
    for phi in PHI_VALUES:
        cases.append(dict(sweep="phi",   label=f"phi_{phi:.2f}",
                          perm=_PERM_BASE, alpha=_ALPHA_BASE, phi=phi))
    for alpha in ALPHA_VALUES:
        cases.append(dict(sweep="alpha", label=f"alpha_{alpha:.2f}",
                          perm=_PERM_BASE, alpha=alpha, phi=_PHI_BASE))
    for perm in PERM_VALUES:
        cases.append(dict(sweep="perm",  label=f"perm_{_fmt_perm(perm)}",
                          perm=perm, alpha=_ALPHA_BASE, phi=_PHI_BASE))
    return cases

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print cases without writing files")
    args = parser.parse_args()

    cases = _all_cases()
    print(f"OAT-SEALED — {len(cases)} cases → {RUNS_DIR.relative_to(DEMO_DIR)}/")
    print(f"  phi   sweep: {len(PHI_VALUES)} cases")
    print(f"  alpha sweep: {len(ALPHA_VALUES)} cases")
    print(f"  perm  sweep: {len(PERM_VALUES)} cases")
    print()

    for case in cases:
        run_dir = RUNS_DIR / case["label"]
        cfg_path = run_dir / "config.yaml"
        print(f"  {'(dry) ' if args.dry_run else ''}writing {cfg_path.relative_to(DEMO_DIR)}")
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_config(cfg_path, _make_config(
                case["label"], case["perm"], case["alpha"], case["phi"]))

    if not args.dry_run:
        print(f"\n{len(cases)} config files written. Run TOOLS/run_sweep.py to execute.")

if __name__ == "__main__":
    main()
