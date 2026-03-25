#!/usr/bin/env python
"""Plot uz_at_bottom for all Biot sweep cases on one figure.

Reads: runs/biot_<α>/outputs/fem_timeseries.nc
Saves: uz_comparison.png

Can be run while simulations are still in progress — only completed cases
(i.e. those with an output file) are included.
"""
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
RUNS_DIR = DEMO_DIR / "runs"

BIOT_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]
COLORS      = cm.viridis(np.linspace(0.15, 0.85, len(BIOT_VALUES)))

# ── load data ──────────────────────────────────────────────────────────────
datasets = {}
for alpha in BIOT_VALUES:
    nc = RUNS_DIR / f"biot_{alpha}" / "outputs" / "fem_timeseries.nc"
    if nc.exists():
        try:
            datasets[alpha] = xr.open_dataset(nc)
        except Exception as e:
            print(f"  Warning: could not read α={alpha}: {e}")

if not datasets:
    raise FileNotFoundError(
        f"No output files found under {RUNS_DIR}. "
        "Run run_all.py first."
    )

print(f"Found {len(datasets)}/{len(BIOT_VALUES)} completed cases: "
      f"α = {list(datasets)}")

# ── plot ───────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 5))

# Secondary axis for applied load (same for every case)
ax2 = ax1.twinx()

# Plot load from the first available dataset
first_ds = next(iter(datasets.values()))
if "sig_zz_applied" in first_ds:
    t_load = first_ds["time"].values
    sig_zz = first_ds["sig_zz_applied"].values / 1e6   # Pa → MPa
    ax2.step(t_load, sig_zz, color="crimson", linewidth=0.9,
             alpha=0.5, where="post", label="σ_zz applied", zorder=1)
    ax2.set_ylabel("Applied σ_zz [MPa]", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

# Plot uz_at_bottom for each case
for (alpha, ds), color in zip(datasets.items(), COLORS):
    t   = ds["time"].values
    uz  = ds["uz_at_bottom"].values * 1e6   # m → μm
    ax1.plot(t, uz, color=color, linewidth=1.6,
             label=f"α = {alpha}", zorder=2)

ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Vertical displacement uz_bottom [μm]")
ax1.set_title("Z-displacement at bottom — Biot coefficient sweep\n"
              "(periodic load: L0=0, L1=1 MPa, L2=5 MPa, period=100 s, t_start=50 s)")
ax1.grid(True, alpha=0.3, zorder=0)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

plt.tight_layout()
out = DEMO_DIR / "uz_comparison.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
