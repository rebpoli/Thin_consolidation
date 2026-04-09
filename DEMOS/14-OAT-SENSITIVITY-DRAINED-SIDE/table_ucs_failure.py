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
C2 = 6.0 * np.sin(phi_rad) / (3.0 - np.sin(phi_rad))
K  = 3.0 * (1.0 - np.sin(phi_rad)) / (3.0 - np.sin(phi_rad))   # C1 = K * UCS

print(f"φ = {args.phi:.0f}°  |  C2 = {C2:.4f}  |  K = {K:.4f}")
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
    time   = ds["time"].values          # [s], shape (n_t,)
    p_eff  = ds["p_eff"].values         # [Pa], shape (n_t, n_pt); negative = compression
    q      = ds["q"].values / 1e6       # [MPa]
    p      = -p_eff / 1e6               # [MPa], positive = compression (Cambridge sign)

    n_t, n_pt = q.shape
    print(f"\n[{label}]  n_t={n_t}  n_pt={n_pt}")
    print(f"  time  : {time[0]:.1f} … {time[-1]:.1f} s  "
          f"({time[0]/60:.2f} … {time[-1]/60:.2f} min)")
    print(f"  p_eff : min={p_eff.min():.3e}  max={p_eff.max():.3e}  Pa")
    print(f"  q     : min={q.min():.4f}  max={q.max():.4f}  MPa")
    print(f"  p(cam): min={p.min():.4f}  max={p.max():.4f}  MPa"
          f"  (positive=compression)")

    ucs_max      = -np.inf
    t_max_s      = np.nan

    for i in range(n_t):
        p_i = p[i]      # (n_pt,)
        q_i = q[i]      # (n_pt,)

        # UCS at which each point would be exactly on the envelope
        # Points with ucs_crit <= 0 are always inside for any UCS > 0
        ucs_crit = (q_i - C2 * p_i) / K   # (n_pt,)

        # 95th percentile = UCS that puts exactly 5% of points outside
        ucs_5pct = float(np.percentile(ucs_crit, (1.0 - fraction) * 100))

        if ucs_5pct > ucs_max:
            ucs_max = ucs_5pct
            t_max_s = time[i]

    # ── Per-run debug summary ──────────────────────────────────────────────────
    # Sample the worst timestep for inspection
    i_worst = np.where(time == t_max_s)[0]
    if len(i_worst):
        iw = i_worst[0]
        p_w = p[iw]; q_w = q[iw]
        ucs_crit_w = (q_w - C2 * p_w) / K
        n_outside = int(np.sum(ucs_crit_w > ucs_max))
        print(f"  worst timestep t={t_max_s:.1f} s  ({(t_max_s-50)/60:.2f} min after load):")
        print(f"    q     : min={q_w.min():.4f}  p50={np.median(q_w):.4f}  "
              f"max={q_w.max():.4f}  MPa")
        print(f"    p(cam): min={p_w.min():.4f}  p50={np.median(p_w):.4f}  "
              f"max={p_w.max():.4f}  MPa")
        print(f"    ucs_crit: p5={np.percentile(ucs_crit_w,5):.4f}  "
              f"p50={np.percentile(ucs_crit_w,50):.4f}  "
              f"p95={np.percentile(ucs_crit_w,95):.4f}  MPa")
        print(f"    ucs_5pct (95th pct of ucs_crit) = {ucs_max:.4f} MPa")
        print(f"    pts outside at that UCS: {n_outside}/{n_pt} "
              f"({100*n_outside/n_pt:.1f}%)")

    T_LOAD = 50.0   # [s] step load applied at this time
    t_after_s = t_max_s - T_LOAD
    results.append({
        "run":        label,
        "t_max_s":    t_max_s,
        "t_after_min": t_after_s / 60.0,
        "ucs_MPa":    ucs_max,
    })
    ds.close()

# ── Print table ───────────────────────────────────────────────────────────────
col_w = [30, 17, 10]
header = f"{'Run':<{col_w[0]}}  {'t after load [s]':>{col_w[1]}}  {'UCS [MPa]':>{col_w[2]}}"
sep    = "-" * len(header)
print(header)
print(sep)
for r in results:
    t_after_s = r["t_after_min"] * 60.0
    print(f"{r['run']:<{col_w[0]}}  {t_after_s:>{col_w[1]}.1f}  {r['ucs_MPa']:>{col_w[2]}.2f}")
print(sep)
