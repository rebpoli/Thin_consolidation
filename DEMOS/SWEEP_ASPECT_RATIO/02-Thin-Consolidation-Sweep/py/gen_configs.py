#!/usr/bin/env python3
"""Aspect-ratio × Poisson sweep — Traditional Consolidation (laterally drained).

Generates one config.yaml per (H/Re, nu) pair.  Load and drainage are at the
bottom face; the lateral boundary (r=Re) is also drained (Pressure=0).
This is the "unjacketed" configuration where fluid can escape both axially
and radially.

Each run is sized so the simulation covers T_v = c_v·t_end/H_dr² consolidation
time factors.  Because c_v depends on nu, end_time_tv is recomputed per case.

Sweep:
  H/Re : 10 values  (0.01 … 10)
  nu   :  4 values  (0.1, 0.2, 0.3, 0.4)
Labels:  ar_<H/Re>__nu_<nu>     e.g. ar_0.50__nu_0.30

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
# 30 log-spaced cases with AR ∈ [2, 50]; AR = 2/(H/Re)  →  H/Re = 2/AR
_AR_VALUES       = np.logspace(np.log10(2.0), np.log10(50.0), 30)
H_OVER_RE_VALUES = [round(2.0 / ar, 4) for ar in _AR_VALUES]
NU_VALUES        = [0.1, 0.2, 0.3, 0.4]

Re = 0.025   # fixed radius [m]

# ── Fixed material properties (match PAPER-01 base case) ─────────────────────
E     = 5.0e9     # Pa
phi   = 0.10
alpha = 0.75
perm  = 1.0e-20   # m²
visc  = 1.0e-3    # Pa·s
Kf    = 2.2e9     # Pa
M     = Kf / phi  # Pa

# ── Load/time controls (match PAPER-01 base case) ─────────────────────────────
L0      = 0.0
L1      = -10.0e6
T_START = 0.0
T_END   = 36000.0


def _cv(nu):
    mu = E / (2 * (1 + nu))
    S  = 1.0 / M + alpha**2 * (1 - 2*nu) / (2 * mu * (1 - nu))
    return perm / (visc * S)


def _label(H_over_Re, nu):
    return f"ar_{H_over_Re:.4f}__nu_{nu:.2f}"


# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dry-run", action="store_true",
                    help="Print cases without writing files")
args = parser.parse_args()


def _make_config(label, H_over_Re, nu):
    H    = Re * H_over_Re
    H_dr = H / 2.0

    cv          = _cv(nu)
    t_end       = T_END
    end_time_tv = cv * t_end / (H_dr**2)
    dt_max_s    = 600.0
    dt_min_s    = 0.001

    return f"""\
general:
  description: "AR-sweep-drained {label}  (H/Re={H_over_Re:.2f}, nu={nu:.2f})"
  tags: "aspect_ratio_sweep poisson_sweep drained paper01_base"
  run_dir: "{RUNS_DIR / label}"
  run_id: "{label}"

mesh:
  Re: {Re}
  H: {H:.6g}
  N: 6             # fallback only
  Nr: 30            # quasi-1D: minimal r-elements
  Nz: 30           # vertical resolution
  grade_r: true
  grade_z_bottom: false

materials:
  E: {E:.4g}
  nu: {nu}
  alpha: {alpha}
  perm: {perm:.4g}
  visc: {visc:.4g}
  M: {M:.4g}

# DRAINED LATERAL — matches PAPER-01 OAT-DRAINED BC pattern
boundary_conditions:
  top:
    Pressure: 0.0
    U_r: 0.0
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
    Pressure: 0.0

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
n_cases = len(H_OVER_RE_VALUES) * len(NU_VALUES)
print(f"\nAspect-ratio × Poisson sweep — Traditional Consolidation (laterally drained)")
print(f"  Re = {Re} m  |  t_end = {T_END:.0f} s")
print(f"  {len(H_OVER_RE_VALUES)} ARs × {len(NU_VALUES)} nus = {n_cases} cases\n")

for H_over_Re in H_OVER_RE_VALUES:
    H    = Re * H_over_Re
    H_dr = H / 2.0
    for nu in NU_VALUES:
        label = _label(H_over_Re, nu)
        cv    = _cv(nu)
        print(f"  {label:<22}  H={H:.4g} m  nu={nu}  cv={cv:.3e} m²/s")

        if args.dry_run:
            continue

        run_dir = RUNS_DIR / label
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(_make_config(label, H_over_Re, nu))

if args.dry_run:
    print(f"\n--dry-run: no files written.")
else:
    print(f"\nGenerated {n_cases} cases in {RUNS_DIR}/")
