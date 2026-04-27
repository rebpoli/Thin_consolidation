#!/usr/bin/env python3
"""Generate per-case config.yaml files for the THIN-DISC-VS-TRADITIONAL comparison.

Two test protocols, same material, with permeability and alpha sweeps:

  THIN-DISC   — OAT-DRAINED protocol
                H = 1.0 cm, Re = 2.5 cm
                Right boundary: drained (P=0, mechanically free)

  TRADITIONAL — OAT-SEALED protocol
                H = 2" = 50.8 mm, Re = 0.5" = 12.7 mm
                Right boundary: sealed (U_r=0, no drainage)

Reference material (shared):
  E=5 GPa, nu=0.40, alpha=0.75, phi=0.10, Kf=2.2 GPa

Permeabilities: 1e-20, 1e-21, 1e-22 m²
Alpha values:   0.70, 0.80, 0.90 (at fixed k = 1e-20 m²)

End time: 30 days

Writes one runs/<label>/config.yaml per case.
"""

import argparse
import io
import re
from pathlib import Path

import numpy as np
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
DEMO_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = DEMO_DIR / "runs"

# ── YAML helpers ───────────────────────────────────────────────────────────────
def _yaml_float(v):
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

# ── Material constants (shared) ────────────────────────────────────────────────
_E     = 5.0e9
_nu    = 0.40
_visc  = 1.0e-3
_Kf    = 2.2e9
_phi   = 0.10
_alpha_base = 0.75
_mu    = _E / (2 * (1 + _nu))
_M     = _Kf / _phi           # Biot modulus

# ── Geometry ───────────────────────────────────────────────────────────────────
_THIN = dict(H=0.010,           Re=0.025,          N=30)   # 1 cm × 2.5 cm
_TRAD = dict(H=2 * 0.0254,      Re=0.5 * 0.0254,   N=30)   # 2" × 0.5"  (H=50.8mm, Re=12.7mm)

# ── Load ───────────────────────────────────────────────────────────────────────
_L0       = 0.0
_L1       = -10.0e6        # −10 MPa step
_T_START  = 0.0            # load applied at t=0
_T_END    = 30 * 86400.0   # 30 days [s]

# ── Permeabilities ─────────────────────────────────────────────────────────────
PERM_VALUES = [1e-20, 1e-21, 1e-22]
ALPHA_VALUES = [0.70, 0.80, 0.90]

# ── Derived helpers ────────────────────────────────────────────────────────────
def _storativity(alpha):
    return 1.0 / _M + alpha**2 * (1 - 2*_nu) / (2 * _mu * (1 - _nu))

def _cv(perm, alpha):
    return perm / (_visc * _storativity(alpha))

def _end_time_tv(perm, alpha, H):
    H_dr = H / 2
    return float(_cv(perm, alpha) * _T_END / H_dr**2)

def _fmt_perm(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

_TV_COMMENT = ("  # T_v = c_v·t/(4H²) — dimensionless consolidation time; "
               "t_end[s] = 4H²/c_v · T_v")

def _write_config(path, cfg):
    buf = io.StringIO()
    yaml.dump(cfg, buf, Dumper=_CompactDumper, default_flow_style=False, sort_keys=False)
    text = re.sub(r"(  end_time_tv:.*)", r"\1" + _TV_COMMENT, buf.getvalue())
    path.write_text(text)

# ── Config builders ────────────────────────────────────────────────────────────
def _make_thin_disc(label, perm, alpha):
    geo = _THIN
    return {
        "general": {
            "description": f"THIN-DISC {label}",
            "tags":        "thin_disc_vs_traditional thin_disc",
            "run_dir":     "",
            "run_id":      label,
        },
        "mesh": {"Re": float(geo["Re"]), "H": float(geo["H"]), "N": geo["N"]},
        "materials": {
            "E":     _E,
            "nu":    _nu,
            "alpha": float(alpha),
            "perm":  float(perm),
            "visc":  _visc,
            "M":     _M,
        },
        "boundary_conditions": {
            "bottom": {"U_z": 0.0},
            "top": {
                "Pressure":      0.0,
                "U_r":           0.0,
                "U_z_rigid":     1,
                "periodic_load": {
                    "L0": _L0, "L1": _L1,
                    "t_start": _T_START, "period": _T_END * 2,
                    "duty_cycle": 1.0, "n_periods": -1,
                },
            },
            "right": {"Pressure": 0.0},   # drained, mechanically free
            "left":  {"U_r": 0.0},
        },
        "numerical": {
            "theta_cn":    0.75,
            "end_time_tv": _end_time_tv(perm, alpha, geo["H"]),
            "dt_min_s":    0.1,
            "dt_max_s":    3600.0,
            "dt_factor":   1.5,
        },
        "output": {
            "results":    "outputs/results.bp",
            "timeseries": "outputs/fem_timeseries.nc",
        },
    }

def _make_traditional(label, perm, alpha):
    geo = _TRAD
    return {
        "general": {
            "description": f"TRADITIONAL {label}",
            "tags":        "thin_disc_vs_traditional traditional",
            "run_dir":     "",
            "run_id":      label,
        },
        "mesh": {"Re": float(geo["Re"]), "H": float(geo["H"]), "N": geo["N"]},
        "materials": {
            "E":     _E,
            "nu":    _nu,
            "alpha": float(alpha),
            "perm":  float(perm),
            "visc":  _visc,
            "M":     _M,
        },
        "boundary_conditions": {
            "bottom": {"U_z": 0.0},
            "top": {
                "Pressure":      0.0,
                "U_z_rigid":     1,
                "periodic_load": {
                    "L0": _L0, "L1": _L1,
                    "t_start": _T_START, "period": _T_END * 2,
                    "duty_cycle": 1.0, "n_periods": -1,
                },
            },
            "right": {"U_r": 0.0},   # sealed — traditional oedometer
            "left":  {"U_r": 0.0},
        },
        "numerical": {
            "theta_cn":    0.75,
            "end_time_tv": _end_time_tv(perm, alpha, geo["H"]),
            "dt_min_s":    0.1,
            "dt_max_s":    3600.0,
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
    for perm in PERM_VALUES:
        p = _fmt_perm(perm)
        cases.append(dict(label=f"thin_disc_{p}", builder=_make_thin_disc,
                          perm=perm, alpha=_alpha_base))
        cases.append(dict(label=f"traditional_{p}", builder=_make_traditional,
                          perm=perm, alpha=_alpha_base))

    for alpha in ALPHA_VALUES:
        a = f"{alpha:.2f}"
        cases.append(dict(label=f"thin_disc_alpha_{a}", builder=_make_thin_disc,
                          perm=1e-20, alpha=alpha))
        cases.append(dict(label=f"traditional_alpha_{a}", builder=_make_traditional,
                          perm=1e-20, alpha=alpha))
    return cases

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = _all_cases()
    print(f"THIN-DISC-VS-TRADITIONAL — {len(cases)} cases → {RUNS_DIR.relative_to(DEMO_DIR)}/")
    for case in cases:
        run_dir  = RUNS_DIR / case["label"]
        cfg_path = run_dir / "config.yaml"
        print(f"  {'(dry) ' if args.dry_run else ''}writing {cfg_path.relative_to(DEMO_DIR)}")
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_config(cfg_path, case["builder"](case["label"], case["perm"], case["alpha"]))

    if not args.dry_run:
        print(f"\n{len(cases)} config files written. Run TOOLS/run_sweep.py to execute.")

if __name__ == "__main__":
    main()
