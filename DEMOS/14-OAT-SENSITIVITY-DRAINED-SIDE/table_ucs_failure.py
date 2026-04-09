#!/usr/bin/env python3
"""Demo 14 — UCS failure table.

For each run, scans every timestep and finds the maximum UCS (Mohr-Coulomb,
Drucker-Prager outer-circumscribed, φ=30°) at which 5% of the sampled domain
points lie outside the failure envelope.  This is the most critical (demanding)
state across the simulation — the highest rock strength that would still be
required to keep 95% of the domain inside the envelope.

Reports per run:
  - the time [min] at which that maximum UCS occurs
  - the maximum UCS value [MPa]

Physics
-------
Envelope in p-q space:  q = C1 + C2 * (-p')
  C2 = 6·sin(φ) / (3 − sin φ)                     [slope, φ-only]
  C1 = 3·UCS·(1 − sin φ) / (3 − sin φ)            [intercept, scales with UCS]

A point (p_i, q_i) — where p_i = −p'_i ≥ 0 — is OUTSIDE the envelope when:
  q_i > C1(UCS) + C2·p_i
  ⟺  UCS < (q_i − C2·p_i) / K       where K = 3(1−sin φ)/(3−sin φ)

So the minimum UCS that keeps point i inside is:
  UCS_crit_i = (q_i − C2·p_i) / K    (negative ⟹ always inside)

For exactly 5% of points to be outside: UCS = 95th-percentile of UCS_crit.
We then minimise this across all timesteps.
"""
import argparse
import numpy as np
import xarray as xr
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
RUNS_DIR = DEMO_DIR / "runs"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Demo 14 UCS failure threshold table")
parser.add_argument("--phi",      type=float, default=30.0,
                    help="MC friction angle [deg] (default: 30)")
parser.add_argument("--fraction", type=float, default=0.05,
                    help="Failed-domain fraction threshold (default: 0.05 = 5%%)")
parser.add_argument("--run",      default=None,
                    help="Single run label (default: all runs)")
args = parser.parse_args()

phi_rad  = np.radians(args.phi)
fraction = args.fraction

# Envelope coefficients (φ-only)
C2  = 6.0 * np.sin(phi_rad) / (3.0 - np.sin(phi_rad))
K   = 3.0 * (1.0 - np.sin(phi_rad)) / (3.0 - np.sin(phi_rad))   # C1 = K * UCS
K_t = (1.0 - np.sin(phi_rad)) / (2.0 * np.sin(phi_rad))          # T0 = K_t * UCS

print(f"φ = {args.phi:.0f}°  |  C2 = {C2:.4f}  |  K = {K:.4f}  |  K_t = {K_t:.4f}")
print(f"Failed fraction threshold: {fraction*100:.0f}%")
print()

# ── Collect runs ──────────────────────────────────────────────────────────────
if args.run:
    run_labels = [args.run]
else:
    run_labels = sorted(
        p.parent.parent.name
        for p in RUNS_DIR.glob("*/outputs/invariants.nc")
    )
    if not run_labels:
        raise FileNotFoundError(f"No invariants.nc files found under {RUNS_DIR}")

# ── Process each run ──────────────────────────────────────────────────────────
results = []

for label in run_labels:
    nc_path = RUNS_DIR / label / "outputs" / "invariants.nc"
    if not nc_path.exists():
        print(f"  Skipping {label}: invariants.nc not found")
        continue

    ds     = xr.open_dataset(nc_path)
    time   = ds["time"].values            # [s], shape (n_t,)

    # Filter to post-load timesteps only (t > 50 s) — pre-load state is near-zero
    # effective stress and would produce spurious tensile p''_t values.
    T_LOAD    = 50.0
    post_mask = time > T_LOAD
    time      = time[post_mask]
    p_eff_t   = ds["p_eff_t"].values[post_mask]  # [Pa], Terzaghi (α=1); negative = compression
    q         = ds["q"].values[post_mask] / 1e6   # [MPa]
    p         = -p_eff_t / 1e6                    # [MPa], positive = compression (Cambridge sign)

    n_t, n_pt = q.shape
    print(f"\n[{label}]  n_t={n_t}  n_pt={n_pt}")
    print(f"  time  : {time[0]:.1f} … {time[-1]:.1f} s  "
          f"({time[0]/60:.2f} … {time[-1]/60:.2f} min)")
    print(f"  p_eff_t: min={p_eff_t.min():.3e}  max={p_eff_t.max():.3e}  Pa  (Terzaghi, α=1)")
    print(f"  q     : min={q.min():.4f}  max={q.max():.4f}  MPa")
    print(f"  p(cam): min={p.min():.4f}  max={p.max():.4f}  MPa"
          f"  (positive=compression)")

    # ── Shear: find worst-case UCS ────────────────────────────────────────────
    ucs_max = -np.inf
    t_ucs_s = np.nan
    for i in range(n_t):
        ucs_s    = (q[i] - C2 * p[i]) / K
        ucs_5pct = float(np.percentile(ucs_s, (1.0 - fraction) * 100))
        if ucs_5pct > ucs_max:
            ucs_max = ucs_5pct
            t_ucs_s = time[i]

    # ── Tension: find worst-case p''_t (min 5th-pct of -p'', converted to p''_t ≥ 0) ──
    # p = -p_eff_t/1e6 is Cambridge x-axis (positive = compression).
    # Worst timestep = most tensile = minimum 5th-pct of x.
    # p''_t = -x_cutoff ≥ 0  (tensile strength; compression-negative convention).
    pt_min = +np.inf
    t_pt_s = np.nan
    for i in range(n_t):
        val = float(np.percentile(p[i], fraction * 100))   # 5th pct
        if val < pt_min:
            pt_min = val
            t_pt_s = time[i]
    pt_required = max(-pt_min, 0.0)   # p''_t ≥ 0; 0 = zero tensile strength

    results.append({
        "run":        label,
        "t_ucs_s":    t_ucs_s  - T_LOAD,
        "ucs_MPa":    ucs_max,
        "t_pt_s":     t_pt_s   - T_LOAD,
        "pt_MPa":     pt_required,
    })
    ds.close()

# ── Print table ───────────────────────────────────────────────────────────────
col_w = [30, 15, 12, 15, 12]
header = (f"{'Run':<{col_w[0]}}  {'t_UCS [s]':>{col_w[1]}}"
          f"  {'UCS [MPa]':>{col_w[2]}}  {'t_p''t [s]':>{col_w[3]}}"
          f"  {'p''_t [MPa]':>{col_w[4]}}")
sep    = "-" * len(header)
print(header)
print(sep)
for r in results:
    print(f"{r['run']:<{col_w[0]}}  {r['t_ucs_s']:>{col_w[1]}.1f}"
          f"  {r['ucs_MPa']:>{col_w[2]}.2f}  {r['t_pt_s']:>{col_w[3]}.1f}"
          f"  {r['pt_MPa']:>{col_w[4]}.2f}")
print(sep)
