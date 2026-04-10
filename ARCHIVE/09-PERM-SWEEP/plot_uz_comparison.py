#!/usr/bin/env python
"""Plot uz_at_bottom for all permeability sweep cases on one figure.

Reads: runs/perm_<value>/outputs/fem_timeseries.nc
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

PERM_VALUES = np.logspace(-20, -16, 5)

def _fmt_perm(perm: float) -> str:
    s = f"{perm:.2e}"
    mantissa, exp = s.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"

COLORS = cm.cividis(np.linspace(0.1, 0.9, len(PERM_VALUES)))

# ── load data ──────────────────────────────────────────────────────────────
datasets = {}
for perm in PERM_VALUES:
    nc = RUNS_DIR / f"perm_{_fmt_perm(perm)}" / "outputs" / "fem_timeseries.nc"
    if nc.exists():
        try:
            datasets[perm] = xr.open_dataset(nc)
        except Exception as e:
            print(f"  Warning: could not read perm={_fmt_perm(perm)}: {e}")

if not datasets:
    raise FileNotFoundError(
        f"No output files found under {RUNS_DIR}. Run run_all.py first."
    )

print(f"Found {len(datasets)}/{len(PERM_VALUES)} completed cases: "
      f"perm = {[_fmt_perm(p) for p in datasets]}")

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
for (perm, ds), color in zip(datasets.items(), COLORS):
    t  = ds["time"].values
    uz = ds["uz_at_bottom"].values * 1e6   # m → μm
    ax1.plot(t, uz, color=color, linewidth=1.6,
             label=f"k = {_fmt_perm(perm)} m²", zorder=2)

ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Vertical displacement uz_bottom [μm]")
ax1.set_title("Z-displacement at bottom — Permeability sweep\n"
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
