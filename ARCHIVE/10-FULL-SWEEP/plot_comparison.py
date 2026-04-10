#!/usr/bin/env python
"""4-row × 2-column parameter-sweep comparison plot.

Left column  : uz_at_bottom per case (α, M, k sweeps) + load track
Right column : pressure_mean ± P10/P90 shaded band per case + load track

Reads: runs/<group>_<value>/outputs/fem_timeseries.nc
Saves: png/comparison.png  (500 dpi)

Can be run while simulations are still in progress.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
RUNS_DIR = DEMO_DIR / "runs"

# ── Sub-batch definitions ─────────────────────────────────────────────────────

def _fmt_M(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

_fmt_perm = _fmt_M

BIOT_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]
M_VALUES    = np.logspace(np.log10(1e8), np.log10(1.5e10), 5)
PERM_VALUES = np.logspace(-20, -16, 5)

BATCHES = [
    {
        "label":   "α sweep",
        "keys":    BIOT_VALUES,
        "labels":  [f"biot_{v}" for v in BIOT_VALUES],
        "legends": [f"α = {v}" for v in BIOT_VALUES],
        "cmap":    cm.viridis,
    },
    {
        "label":   "M sweep",
        "keys":    M_VALUES,
        "labels":  [f"M_{_fmt_M(v)}" for v in M_VALUES],
        "legends": [f"M = {_fmt_M(v)} Pa" for v in M_VALUES],
        "cmap":    cm.plasma,
    },
    {
        "label":   "k sweep",
        "keys":    PERM_VALUES,
        "labels":  [f"perm_{_fmt_perm(v)}" for v in PERM_VALUES],
        "legends": [f"k = {_fmt_perm(v)} m²" for v in PERM_VALUES],
        "cmap":    cm.cividis,
    },
]

# ── Load data ─────────────────────────────────────────────────────────────────

for batch in BATCHES:
    datasets = {}
    for key, label in zip(batch["keys"], batch["labels"]):
        nc = RUNS_DIR / label / "outputs" / "fem_timeseries.nc"
        if nc.exists():
            try:
                datasets[label] = xr.open_dataset(nc)
            except Exception as e:
                print(f"  Warning: could not read {label}: {e}")
    batch["datasets"] = datasets
    print(f"{batch['label']}: {len(datasets)}/{len(batch['keys'])} cases available")

any_data = any(len(b["datasets"]) > 0 for b in BATCHES)
if not any_data:
    raise FileNotFoundError(f"No output files found under {RUNS_DIR}. Run run_all.py first.")

# ── Figure layout ─────────────────────────────────────────────────────────────
#   4 rows × 2 cols; rows 0-2 are data panels, row 3 is the thin load track
#   Within each row, left and right panels share x.

fig = plt.figure(figsize=(18, 11))
gs  = gridspec.GridSpec(4, 2, figure=fig,
                        height_ratios=[3, 3, 3, 1],
                        hspace=0.06, wspace=0.32,
                        top=0.91, bottom=0.06,
                        left=0.07, right=0.96)

ax_uz   = [fig.add_subplot(gs[i, 0]) for i in range(3)]
ax_pr   = [fig.add_subplot(gs[i, 1]) for i in range(3)]
ax_load = [fig.add_subplot(gs[3, c]) for c in range(2)]

# Share x within each row
for i in range(3):
    ax_pr[i].sharex(ax_uz[i])

# ── Helper: load track ────────────────────────────────────────────────────────

def _draw_load(ax, batch):
    ds = next(iter(batch["datasets"].values()), None) if batch["datasets"] else None
    if ds is not None and "sig_zz_applied" in ds:
        t   = ds["time"].values
        sig = ds["sig_zz_applied"].values / 1e6
        ax.step(t, sig, color="crimson", linewidth=1.0, where="post", zorder=2)
        ax.fill_between(t, sig, step="post", color="crimson", alpha=0.15, zorder=1)
    ax.set_ylabel("σ_zz\n[MPa]", fontsize=8, labelpad=2)
    ax.set_xlabel("Time [s]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

# ── Left column: uz panels ───────────────────────────────────────────────────

for i, (ax, batch) in enumerate(zip(ax_uz, BATCHES)):
    colors = batch["cmap"](np.linspace(0.1, 0.9, len(batch["keys"])))
    for label, legend, color in zip(batch["labels"], batch["legends"], colors):
        if label not in batch["datasets"]:
            continue
        ds = batch["datasets"][label]
        ax.plot(ds["time"].values, ds["uz_at_bottom"].values * 1e6,
                color=color, linewidth=1.5, label=legend, zorder=2)

    ax.text(0.01, 0.97, batch["label"], transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")
    ax.set_ylabel("uz_bottom [μm]", fontsize=9)
    ax.tick_params(labelsize=8, labelbottom=False)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.legend(loc="lower left", fontsize=7, ncol=2)

# ── Right column: pressure panels ────────────────────────────────────────────

for i, (ax, batch) in enumerate(zip(ax_pr, BATCHES)):
    colors = batch["cmap"](np.linspace(0.1, 0.9, len(batch["keys"])))
    for label, legend, color in zip(batch["labels"], batch["legends"], colors):
        if label not in batch["datasets"]:
            continue
        ds = batch["datasets"][label]
        t   = ds["time"].values

        has_stats = ("pressure_mean" in ds and
                     "pressure_p10"  in ds and
                     "pressure_p90"  in ds)
        if has_stats:
            p_mean = ds["pressure_mean"].values / 1e3   # Pa → kPa
            p_p10  = ds["pressure_p10"].values  / 1e3
            p_p90  = ds["pressure_p90"].values  / 1e3
            ax.plot(t, p_mean, color=color, linewidth=1.5, label=legend, zorder=3)
            ax.fill_between(t, p_p10, p_p90,
                            color=color, alpha=0.20, linewidth=0, zorder=2)
        else:
            # Fallback: pressure_at_top only (old files without statistics)
            ax.plot(t, ds["pressure_at_top"].values / 1e3,
                    color=color, linewidth=1.5, label=legend,
                    linestyle="--", zorder=3)

    ax.text(0.01, 0.97, batch["label"], transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")
    ax.set_ylabel("Pressure [kPa]", fontsize=9)
    ax.tick_params(labelsize=8, labelbottom=False)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.legend(loc="upper right", fontsize=7, ncol=2)

# ── Bottom row: load tracks (both columns) ────────────────────────────────────

# Use whichever batch has data; all share the same load signal
_load_batch = next((b for b in BATCHES if b["datasets"]), None)
for ax in ax_load:
    if _load_batch:
        _draw_load(ax, _load_batch)

# Share x: load tracks with their column's data panels.
# ax_load axes are fresh (._sharex = None), so they must be the caller.
ax_load[0].sharex(ax_uz[2])
ax_load[1].sharex(ax_pr[2])
for ax in ax_uz[:2] + ax_pr[:2]:
    ax.tick_params(labelbottom=False)

# ── Titles ────────────────────────────────────────────────────────────────────

ax_uz[0].set_title("Z-displacement at bottom", fontsize=10, pad=4)
ax_pr[0].set_title("Pressure  (mean + P10/P90 band)", fontsize=10, pad=4)

fig.suptitle("Parameter sweeps  (α, M, k)   |   Re=10 cm, H=2 cm   |   "
             "duty cycle=90 %,  5 cycles,  period=100 s,  t_start=50 s",
             fontsize=11)

# ── Save ─────────────────────────────────────────────────────────────────────

png_dir = DEMO_DIR / "png"
png_dir.mkdir(exist_ok=True)
out = png_dir / "comparison.png"
plt.savefig(out, dpi=500)
print(f"\nSaved {out}")
plt.show()
