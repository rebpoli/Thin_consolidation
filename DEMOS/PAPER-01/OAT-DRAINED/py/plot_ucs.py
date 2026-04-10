#!/usr/bin/env python3
"""Demo 14 — tile plot of required UCS and p''_t vs sweep parameter.

Three subplots (phi | alpha | perm), each showing:
  - UCS  [MPa]  : shear failure criterion (95th-pct, post-load)
  - p''_t [MPa] : tensile strength criterion (5th-pct of Cambridge p, post-load)

USAGE:
    python plot_ucs_table.py
    python plot_ucs_table.py --phi 35 --fraction 0.10
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]   # OAT-DRAINED/
RUNS_DIR = DEMO_DIR / "runs"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Demo 14 UCS / p''_t sweep tile plot")
parser.add_argument("--phi",      type=float, default=30.0,
                    help="MC friction angle [deg] (default: 30)")
parser.add_argument("--fraction", type=float, default=0.05,
                    help="Failed fraction threshold (default: 0.05 = 5%%)")
args = parser.parse_args()

phi_rad  = np.radians(args.phi)
fraction = args.fraction
T_LOAD   = 50.0   # [s] load applied at this time

# Envelope coefficients
C2 = 6.0 * np.sin(phi_rad) / (3.0 - np.sin(phi_rad))
K  = 3.0 * (1.0 - np.sin(phi_rad)) / (3.0 - np.sin(phi_rad))

# ── Sweep definitions ─────────────────────────────────────────────────────────
PHI_VALUES   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ALPHA_VALUES = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
PERM_VALUES  = [1e-21, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16]

def _fmt_perm(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

SWEEPS = [
    {
        "key":    "phi",
        "values": PHI_VALUES,
        "labels": [f"phi_{v:.2f}"         for v in PHI_VALUES],
        "xlabel": "Porosity $\\phi$",
        "xfmt":   lambda v: v,
        "xscale": "linear",
        "title":  "$\\phi$ sweep  ($\\alpha=0.50$, $k=10^{-20}$ m²)",
    },
    {
        "key":    "alpha",
        "values": ALPHA_VALUES,
        "labels": [f"alpha_{v:.2f}"        for v in ALPHA_VALUES],
        "xlabel": "Biot coefficient $\\alpha$",
        "xfmt":   lambda v: v,
        "xscale": "linear",
        "title":  "$\\alpha$ sweep  ($\\phi=0.10$, $k=10^{-20}$ m²)",
    },
    {
        "key":    "perm",
        "values": PERM_VALUES,
        "labels": [f"perm_{_fmt_perm(v)}" for v in PERM_VALUES],
        "xlabel": "Permeability $k$ (m²)",
        "xfmt":   lambda v: v,
        "xscale": "log",
        "title":  "$k$ sweep  ($\\phi=0.10$, $\\alpha=0.50$)",
    },
]

# ── Compute UCS and p''_t for one run ────────────────────────────────────────
def _compute(label):
    nc_path = RUNS_DIR / label / "outputs" / "invariants.nc"
    if not nc_path.exists():
        return np.nan, np.nan

    ds        = xr.open_dataset(nc_path)
    time      = ds["time"].values
    post_mask = time > T_LOAD
    p_eff_t   = ds["p_eff_t"].values[post_mask]   # [Pa], compression-negative
    q_Pa      = ds["q"].values[post_mask]          # [Pa]
    ds.close()

    p = -p_eff_t / 1e6   # Cambridge pressure [MPa], positive = compression
    q =  q_Pa   / 1e6    # deviatoric [MPa]

    # UCS: 95th-pct of shear-critical UCS, maximised over timesteps
    ucs_max = -np.inf
    for i in range(q.shape[0]):
        ucs_s = (q[i] - C2 * p[i]) / K
        val   = float(np.percentile(ucs_s, (1.0 - fraction) * 100))
        if val > ucs_max:
            ucs_max = val

    # p''_t: worst (most tensile) 5th-pct of Cambridge p, converted to ≥ 0
    pt_min = +np.inf
    for i in range(p.shape[0]):
        val = float(np.percentile(p[i], fraction * 100))
        if val < pt_min:
            pt_min = val

    ucs_MPa = max(ucs_max, 0.0)
    pt_MPa  = max(-pt_min, 0.0)
    return ucs_MPa, pt_MPa

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 4))
gs  = gridspec.GridSpec(1, 3, figure=fig,
                        wspace=0.32, left=0.07, right=0.97,
                        top=0.88, bottom=0.17)

UCS_KW = dict(color="#1f77b4", marker="o", ms=5, lw=1.4, label="UCS (MPa)")
PT_KW  = dict(color="#d62728", marker="s", ms=5, lw=1.4, ls="--",
              label="$p''_t$ (MPa)")

for col, sw in enumerate(SWEEPS):
    ax = fig.add_subplot(gs[0, col])
    ax2 = ax.twinx()

    ucs_vals = []
    pt_vals  = []
    for label in sw["labels"]:
        ucs, pt = _compute(label)
        ucs_vals.append(ucs)
        pt_vals.append(pt)

    xvals = sw["values"]
    ax.plot(xvals, ucs_vals, **UCS_KW)
    ax2.plot(xvals, pt_vals, **PT_KW)

    ax.set_xscale(sw["xscale"])
    ax.set_xlabel(sw["xlabel"], fontsize=8)
    ax.set_ylabel("UCS (MPa)", fontsize=8, color="#1f77b4")
    ax2.set_ylabel("$p''_t$ (MPa)", fontsize=8, color="#d62728")
    ax.tick_params(which="both", labelsize=7)
    ax2.tick_params(which="both", labelsize=7)
    ax.tick_params(axis="y", colors="#1f77b4")
    ax2.tick_params(axis="y", colors="#d62728")
    ax.set_ylim(10, 24)
    ax2.set_ylim(0, 9)
    ax.set_title(sw["title"], fontsize=8, pad=4)
    ax.grid(True, alpha=0.25)

    leg_lines = ax.get_lines() + ax2.get_lines()
    leg_labs  = [l.get_label() for l in leg_lines]
    ax.legend(leg_lines, leg_labs,
              fontsize=7, loc="upper center", ncol=2,
              frameon=False, handlelength=1.8)

fig.suptitle(
    f"Demo 14 — Required rock strength (UCS and tensile limit $p''_t$) for each parameter setting  |  "
    f"MC $\\varphi={args.phi:.0f}^{{\\circ}}$, {fraction*100:.0f}% failure threshold  |  "
    f"post-load ($t > {T_LOAD:.0f}$ s)",
    fontsize=9, y=0.98)

png_dir = DEMO_DIR / "png"
png_dir.mkdir(exist_ok=True)
out = png_dir / "ucs_sweep.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
