#!/usr/bin/env python3
"""Aspect-ratio sweep — Traditional Consolidation (laterally drained).

Generates one config.yaml per H/Re value.  Load and drainage are at the
bottom face; the lateral boundary (r=Re) is also drained (Pressure=0).
This is the "unjacketed" configuration where fluid can escape both axially
and radially.

Each run is sized so the simulation covers T_v = 3 consolidation time
factors regardless of specimen geometry.  dt_max is set per case so the
adaptive stepper takes at least ~100 steps.

Sweep: 10 logarithmically spaced H/Re values from 0.01 to 10.
Labels:  ar_0.01, ar_0.02, …, ar_10.00   (ar = H/Re)

USAGE:
    python py/gen_configs.py
    python py/gen_configs.py --dry-run
"""
import argparse
import numpy as np
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = DEMO_DIR / "runs"

# ── Sweep definition ───────────────────────────────────────────────────────────
H_OVER_RE_VALUES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00, 10.00]

Re = 0.025   # fixed radius [m]

# ── Fixed material properties ──────────────────────────────────────────────────
E     = 1.44e10   # Pa
nu    = 0.2
alpha = 0.78
perm  = 1.0e-20   # m²
visc  = 1.0e-3    # Pa·s
M     = 1.35e10   # Pa

# ── Derived consolidation coefficient ─────────────────────────────────────────
mu = E / (2 * (1 + nu))
S  = 1.0 / M + alpha**2 * (1 - 2*nu) / (2 * mu * (1 - nu))
cv = perm / (visc * S)

# ── Timestepping target ────────────────────────────────────────────────────────
T_V_TARGET = 3.0   # run until Tv=3 (>95% consolidation)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dry-run", action="store_true",
                    help="Print cases without writing files")
args = parser.parse_args()


def _make_config(label, H_over_Re):
    H    = Re * H_over_Re
    H_dr = H / 2.0

    t_end       = T_V_TARGET * 4.0 * H_dr**2 / cv
    end_time_tv = T_V_TARGET
    dt_max_s    = t_end / 100.0
    dt_min_s    = min(dt_max_s / 1000.0, 0.001)

    return f"""\
general:
  description: "AR-sweep-drained {label}  (H/Re={H_over_Re:.2f})"
  tags: "aspect_ratio_sweep drained"
  run_dir: "{RUNS_DIR / label}"
  run_id: "{label}"

mesh:
  Re: {Re}
  H: {H:.6g}
  N: 6             # fallback only
  Nr: 3            # quasi-1D: minimal r-elements
  Nz: 50           # vertical resolution
  grade_r: false
  grade_z_bottom: false   # drainage at z=0 (bottom)

materials:
  E: {E:.4g}
  nu: {nu}
  alpha: {alpha}
  perm: {perm:.4g}
  visc: {visc:.4g}
  M: {M:.4g}

# DRAINED LATERAL — fluid escapes at bottom face and at r=Re
boundary_conditions:
  top:  
    sig_zz: -1.0e5
    Pressure: 0.0
    U_r: 0.0

  bottom:
    U_z: 0.0

  right:  
    U_r: 0.0

  left:
    U_r: 0.0

numerical:
  theta_cn: 0.75
  end_time_tv: {end_time_tv:.4f}   # T_v = c_v·t/(4·H_dr²); t_end = {t_end:.3g} s
  dt_min_s: {dt_min_s:.4g}
  dt_max_s: {dt_max_s:.4g}
  dt_factor: 1.5

output:
  results: "outputs/results.bp"
  timeseries: "outputs/fem_timeseries.nc"
"""


# ── Generate ───────────────────────────────────────────────────────────────────
print(f"\nAspect-ratio sweep — Traditional Consolidation (laterally drained)")
print(f"  Re = {Re} m  |  cv = {cv:.3e} m²/s  |  T_v target = {T_V_TARGET}")
print(f"  {len(H_OVER_RE_VALUES)} cases\n")

for H_over_Re in H_OVER_RE_VALUES:
    H    = Re * H_over_Re
    H_dr = H / 2.0
    t_end = T_V_TARGET * 4.0 * H_dr**2 / cv

    label = f"ar_{H_over_Re:.2f}"
    print(f"  {label:<12}  H={H:.4g} m  H_dr={H_dr:.4g} m  t_end={t_end:.3g} s")

    if args.dry_run:
        continue

    run_dir = RUNS_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(_make_config(label, H_over_Re))

if args.dry_run:
    print(f"\n--dry-run: no files written.")
else:
    print(f"\nGenerated {len(H_OVER_RE_VALUES)} cases in {RUNS_DIR}/")
