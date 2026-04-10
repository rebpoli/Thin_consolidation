#!/usr/bin/env python3
"""Aspect-ratio sweep — Thin Disc Consolidation (sealed lateral) — plots.

Two figures:
  1. Time series (pressure mean, uz settlement) for each case, colored by H/Re.
  2. Final-state errors vs H/Re (pressure, displacement, volume).

The sealed-lateral configuration is the exact 1D Terzaghi problem, so errors
vs the analytical solution should be small across all aspect ratios.

USAGE:
    ./py/postproc.py
    ./py/postproc.py --max-time 1e6
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = DEMO_DIR / "runs"

H_OVER_RE_VALUES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00, 10.00]

parser = argparse.ArgumentParser()
parser.add_argument("--max-time", type=float, default=None)
args = parser.parse_args()

# ── Load datasets ──────────────────────────────────────────────────────────────
datasets = {}
for ar in H_OVER_RE_VALUES:
    label = f"ar_{ar:.2f}"
    nc = RUNS_DIR / label / "outputs" / "fem_timeseries.nc"
    if nc.exists():
        try:
            datasets[ar] = xr.open_dataset(nc)
        except Exception as e:
            print(f"  Warning: {label}: {e}")

if not datasets:
    raise FileNotFoundError(f"No output files found under {RUNS_DIR}. Run make run first.")

print(f"Loaded {len(datasets)}/{len(H_OVER_RE_VALUES)} cases.")

palette = cm.plasma(np.linspace(0.1, 0.9, len(H_OVER_RE_VALUES)))
color_map = {ar: palette[i] for i, ar in enumerate(H_OVER_RE_VALUES)}

# ── Figure 1: Time series ──────────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                          gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08})
ax_p, ax_u = axes

for ar, ds in sorted(datasets.items()):
    t   = ds["time"].values
    mk  = t > 0
    if args.max_time:
        mk &= t <= args.max_time
    t_plot = t[mk]
    color  = color_map[ar]
    label  = f"H/Re={ar:.2f}"

    if "pressure_mean" in ds:
        p = np.clip(ds["pressure_mean"].values[mk] / 1e3, 1e-3, None)
        ax_p.plot(t_plot, p, color=color, linewidth=1.4, label=label)
        if "pressure_p10" in ds and "pressure_p90" in ds:
            p10 = np.clip(ds["pressure_p10"].values[mk] / 1e3, 1e-3, None)
            p90 = np.clip(ds["pressure_p90"].values[mk] / 1e3, 1e-3, None)
            ax_p.fill_between(t_plot, p10, p90, color=color, alpha=0.15, linewidth=0)

    if "uz_at_top" in ds:
        uz = -ds["uz_at_top"].values[mk] * 1e6   # m → μm, flip sign (settlement positive)
        ax_u.plot(t_plot, uz, color=color, linewidth=1.4)

ax_p.set_xscale("log"); ax_p.set_yscale("log")
ax_p.set_ylabel("Mean excess pressure [kPa]", fontsize=9)
ax_p.grid(True, which="both", alpha=0.25)
ax_p.legend(fontsize=7, loc="upper right", ncol=2, handlelength=1.2, framealpha=0.8)
ax_p.set_title("AR sweep — Thin Disc (sealed lateral / 1D Terzaghi)  |  "
               "E=14.4 GPa, α=0.78, k=1e-20 m², σ_zz=−100 kPa", fontsize=9)

ax_u.set_xscale("log")
ax_u.set_ylabel("Settlement −u_z [μm]", fontsize=9)
ax_u.set_xlabel("Time [s]", fontsize=9)
ax_u.grid(True, which="both", alpha=0.25)

png_dir = DEMO_DIR / "png"
png_dir.mkdir(exist_ok=True)
out1 = png_dir / "ar_timeseries.png"
fig1.savefig(out1, dpi=200, bbox_inches="tight")
print(f"Saved {out1}")

# ── Figure 2: Final errors vs H/Re ────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
ax_pe, ax_ue, ax_ve = axes2

ar_vals, p_errs, u_errs, v_errs = [], [], [], []
for ar, ds in sorted(datasets.items()):
    ar_vals.append(ar)
    p_errs.append(float(ds["pressure_error_percent"].values[-1]))
    u_errs.append(float(ds["uz_error_percent"].values[-1]))
    v_errs.append(float(ds["volume_error_percent"].values[-1]))

colors_final = [color_map[ar] for ar in ar_vals]

for ax, errs, title in [
    (ax_pe, p_errs, "Pressure error"),
    (ax_ue, u_errs, "Displacement error"),
    (ax_ve, v_errs, "Volume error"),
]:
    ax.scatter(ar_vals, errs, c=colors_final, s=60, zorder=3)
    ax.plot(ar_vals, errs, color="gray", linewidth=0.8, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("H/Re", fontsize=9)
    ax.set_ylabel("Final error vs Terzaghi 1D [%]", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(labelsize=8)

fig2.suptitle("AR sweep — Thin Disc (sealed / 1D Terzaghi)  |  final-state errors vs analytical",
              fontsize=9)
fig2.tight_layout()

out2 = png_dir / "ar_errors.png"
fig2.savefig(out2, dpi=200, bbox_inches="tight")
print(f"Saved {out2}")
