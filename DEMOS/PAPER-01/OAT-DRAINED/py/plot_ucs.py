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
import matplotlib.ticker as mticker
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]   # OAT-DRAINED/
RUNS_DIR = DEMO_DIR / "runs"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Demo 14 UCS sweep plot")
parser.add_argument("--phi",      type=float, default=30.0,
                    help="MC friction angle [deg] (default: 30)")
parser.add_argument("--fraction", type=float, default=0.05,
                    help="Failed fraction threshold (default: 0.05 = 5%%)")
args = parser.parse_args()

phi_rad  = np.radians(args.phi)
fraction = args.fraction
T_LOAD   = 0.0    # [s] use post-load history from t>0+

# Envelope coefficients
C2 = 6.0 * np.sin(phi_rad) / (3.0 - np.sin(phi_rad))
K  = 3.0 * (1.0 - np.sin(phi_rad)) / (3.0 - np.sin(phi_rad))

# ── Paper style ───────────────────────────────────────────────────────────────
MM = 1 / 25.4
FIG_W_MM = 90
FIG_H_MM = 60
plt.rcParams.update({
    "font.size": 6,
    "axes.titlesize": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "legend.handlelength": 1.0,
    "lines.linewidth": 0.8,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

LEGEND_STYLE = dict(
    framealpha=1.0,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    labelspacing=0.15,
    borderpad=0.3,
    handletextpad=0.3,
)

# ── Sweep definitions ─────────────────────────────────────────────────────────
PHI_VALUES   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ALPHA_VALUES = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
PERM_VALUES  = [1e-18, 1e-19, 1e-20, 1e-21, 1e-22]

def _fmt_perm(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

PHI_LABELS = [f"phi_{v:.2f}" for v in PHI_VALUES]
ALPHA_LABELS = [f"alpha_{v:.2f}" for v in ALPHA_VALUES]
PERM_LABELS = [f"perm_{_fmt_perm(v)}" for v in PERM_VALUES]

# ── Compute UCS for one run ───────────────────────────────────────────────────
def _compute(label):
    nc_path = RUNS_DIR / label / "outputs" / "invariants.nc"
    if not nc_path.exists():
        return np.nan

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

    ucs_MPa = max(ucs_max, 0.0)
    return ucs_MPa

# ── Figures ───────────────────────────────────────────────────────────────────
png_dir = DEMO_DIR / "png"
png_dir.mkdir(exist_ok=True)

# φ + α combined figure (dual x-axis)
fig, ax = plt.subplots(figsize=(FIG_W_MM * MM, FIG_H_MM * MM))
ax_top = ax.twiny()
ax_top.xaxis.set_ticks_position("bottom")
ax_top.xaxis.set_label_position("bottom")
ax_top.spines["bottom"].set_position(("outward", 14))
ax_top.spines["top"].set_visible(False)

ucs_phi = [_compute(label) for label in PHI_LABELS]
ucs_alpha = [_compute(label) for label in ALPHA_LABELS]

line_phi, = ax.plot(PHI_VALUES, ucs_phi, color="#1f77b4", linestyle="-", label="Porosity sweep")
line_alpha, = ax_top.plot(ALPHA_VALUES, ucs_alpha, color="#d62728", linestyle="--", label="Biot coefficient sweep")

ax.set_xlabel("Porosity ($\\phi$)")
ax_top.set_xlabel("Biot coeff. ($\\alpha$)")
ax.xaxis.set_label_coords(-0.08, -0.03)
ax_top.xaxis.set_label_coords(-0.08, -0.13)
ax.xaxis.label.set_horizontalalignment("left")
ax_top.xaxis.label.set_horizontalalignment("left")

phi_ticks = PHI_VALUES[1:]
alpha_ticks = np.arange(0.6, 0.91, 0.05)
ax.set_xticks(phi_ticks)
ax_top.set_xticks(alpha_ticks)

def _fmt_bottom_primary(x, _):
    return f"{x:.2f}".rstrip("0").rstrip(".")

def _fmt_bottom_twin(x, _):
    return f"{x:.2f}".rstrip("0").rstrip(".")

ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_bottom_primary))
ax_top.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_bottom_twin))
ax.set_ylabel("Required UCS (MPa)")
ax.grid(True, which="both", alpha=0.25)
ax.margins(x=0.02)
ax_top.margins(x=0.02)

leg = ax.legend(handles=[line_phi, line_alpha], loc="best", **LEGEND_STYLE)
leg.get_frame().set_linewidth(0.3)

out = png_dir / "oat_drained_ucs_phi_alpha.png"
fig.savefig(out, dpi=500, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved {out}")

# Permeability figure
fig, ax = plt.subplots(figsize=(FIG_W_MM * MM, FIG_H_MM * MM))
ucs_perm = [_compute(label) for label in PERM_LABELS]
ax.plot(PERM_VALUES, ucs_perm, color="black", linestyle="-")
ax.set_xscale("log")
ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1]))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel("Permeability $k$, m$^2$")
ax.set_ylabel("Required UCS (MPa)")
ax.grid(True, which="both", alpha=0.25)
ax.margins(x=0.02)

out = png_dir / "oat_drained_ucs_perm.png"
fig.savefig(out, dpi=500, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved {out}")
