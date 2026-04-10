#!/usr/bin/env python
"""Plot uz_at_bottom for all M sweep cases on one figure.

Reads: runs/M_<value>/outputs/fem_timeseries.nc
Saves: uz_comparison.png

Can be run while simulations are still in progress.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
RUNS_DIR = DEMO_DIR / "runs"

M_VALUES = np.logspace(np.log10(1e8), np.log10(1.5e10), 5)

def _fmt_M(M: float) -> str:
    s = f"{M:.2e}"
    mantissa, exp = s.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"

COLORS = cm.plasma(np.linspace(0.1, 0.9, len(M_VALUES)))

# ── load data ──────────────────────────────────────────────────────────────
datasets = {}
for M in M_VALUES:
    nc = RUNS_DIR / f"M_{_fmt_M(M)}" / "outputs" / "fem_timeseries.nc"
    if nc.exists():
        try:
            datasets[M] = xr.open_dataset(nc)
        except Exception as e:
            print(f"  Warning: could not read M={_fmt_M(M)}: {e}")

if not datasets:
    raise FileNotFoundError(
        f"No output files found under {RUNS_DIR}. Run run_all.py first."
    )

print(f"Found {len(datasets)}/{len(M_VALUES)} completed cases: "
      f"M = {[_fmt_M(M) for M in datasets]}")

# ── plot ───────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

# Applied load (same for all cases — use first available dataset)
first_ds = next(iter(datasets.values()))
if "sig_zz_applied" in first_ds:
    t_load = first_ds["time"].values
    sig_zz = first_ds["sig_zz_applied"].values / 1e6   # Pa → MPa
    ax2.step(t_load, sig_zz, color="crimson", linewidth=0.9,
             alpha=0.4, where="post", label="σ_zz applied", zorder=1)
    ax2.set_ylabel("Applied σ_zz [MPa]", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

# uz_at_bottom per case
for (M, ds), color in zip(datasets.items(), COLORS):
    t  = ds["time"].values
    uz = ds["uz_at_bottom"].values * 1e6   # m → μm
    ax1.plot(t, uz, color=color, linewidth=1.6,
             label=f"M = {_fmt_M(M)} Pa", zorder=2)

ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Vertical displacement uz_bottom [μm]")
ax1.set_title("Z-displacement at bottom — Biot modulus sweep\n"
              "(α=0.75, periodic load: L0=0, L1=1 MPa, L2=5 MPa, period=100 s)")
ax1.grid(True, alpha=0.3, zorder=0)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

plt.tight_layout()
out = DEMO_DIR / "uz_comparison.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
