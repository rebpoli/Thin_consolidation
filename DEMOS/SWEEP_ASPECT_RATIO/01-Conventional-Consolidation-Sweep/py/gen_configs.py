#!/usr/bin/env python3
"""Aspect-ratio sweep — Thin Disc Consolidation (sealed lateral boundary).

Generates one config.yaml per H/Re value.  Load and drainage are at the
bottom face; the lateral boundary (r=Re) is sealed (U_r=0).  This is the
"jacketed" configuration where fluid can only escape axially — the 1D
Terzaghi problem exactly.  Errors should remain low across all aspect ratios.

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

# ── Fixed material properties (match PAPER-01 base case) ─────────────────────
E     = 5.0e9     # Pa
nu    = 0.40
phi   = 0.10
alpha = 0.75
perm  = 1.0e-20   # m²
visc  = 1.0e-3    # Pa·s
Kf    = 2.2e9     # Pa
M     = Kf / phi  # Pa

# ── Derived consolidation coefficient ─────────────────────────────────────────
mu = E / (2 * (1 + nu))
S  = 1.0 / M + alpha**2 * (1 - 2*nu) / (2 * mu * (1 - nu))
cv = perm / (visc * S)

# ── Load/time controls (match PAPER-01 base case) ─────────────────────────────
L0      = 0.0
L1      = -10.0e6
T_START = 0.0
T_END   = 36000.0

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dry-run", action="store_true",
                    help="Print cases without writing files")
args = parser.parse_args()


def _make_config(label, H_over_Re):
    H    = Re * H_over_Re
    H_dr = H / 2.0

    t_end       = T_END
    end_time_tv = cv * t_end / (H_dr**2)
    dt_max_s    = 600.0
    dt_min_s    = 0.001

    return f"""\
general:
  description: "AR-sweep-sealed {label}  (H/Re={H_over_Re:.2f})"
  tags: "aspect_ratio_sweep sealed paper01_base"
  run_dir: "{RUNS_DIR / label}"
  run_id: "{label}"

mesh:
  Re: {Re}
  H: {H:.6g}
  N: 6             # fallback only
  Nr: 30            # quasi-1D: minimal r-elements
  Nz: 30           # vertical resolution
  grade_r: false
  grade_z_bottom: false

materials:
  E: {E:.4g}
  nu: {nu}
  alpha: {alpha}
  perm: {perm:.4g}
  visc: {visc:.4g}
  M: {M:.4g}

# SEALED LATERAL — matches PAPER-01 OAT-SEALED BC pattern
boundary_conditions:
  top:
    Pressure: 0.0
    U_z_rigid: 1
    periodic_load:
      L0: {L0}
      L1: {L1}
      t_start: {T_START}
      period: 72000.0
      duty_cycle: 1.0
      n_periods: -1

  bottom:
    U_z: 0.0

  right:
    U_r: 0.0

  left:
    U_r: 0.0

numerical:
  theta_cn: 0.75
  end_time_tv: {end_time_tv:.6g}   # T_v = c_v·t/(H_dr²); t_end = {t_end:.3g} s
  dt_min_s: {dt_min_s:.4g}
  dt_max_s: {dt_max_s:.4g}
  dt_factor: 1.5

output:
  results: "outputs/results.bp"
  timeseries: "outputs/fem_timeseries.nc"
"""


# ── Generate ───────────────────────────────────────────────────────────────────
print(f"\nAspect-ratio sweep — Thin Disc Consolidation (sealed lateral)")
print(f"  Re = {Re} m  |  cv = {cv:.3e} m²/s  |  t_end = {T_END:.0f} s")
print(f"  {len(H_OVER_RE_VALUES)} cases\n")

for H_over_Re in H_OVER_RE_VALUES:
    H    = Re * H_over_Re
    H_dr = H / 2.0
    t_end = T_END

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
