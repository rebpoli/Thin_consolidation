#!/usr/bin/env python
"""Load-strategy comparison plot.

(N_PERM+1) rows × 2 columns:
  Rows 0..N-1 : one row per permeability — pressure mean + P10/P90 band
  Row N (thin): load track
  Col 0       : Setup A  (500 s at −5 MPa → 1000 s at −1 MPa)
  Col 1       : Setup B  (1500 s at −1 MPa)

Reads: runs/<setup>_perm_<value>/outputs/fem_timeseries.nc
Saves: png/comparison.png  (500 dpi)

USAGE:
    ./plot_comparison.py                    # Plot full time range
    ./plot_comparison.py --max-time 100     # Limit to first 100 seconds
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from pathlib import Path
import argparse

DEMO_DIR = Path(__file__).resolve().parent
RUNS_DIR = DEMO_DIR / "runs"

# ── Parse command-line arguments ──────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Plot load-strategy comparison across permeability sweep",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
EXAMPLES:
  # Plot full time range
  ./plot_comparison.py
  
  # Limit to first 100 seconds
  ./plot_comparison.py --max-time 100
  
  # Limit to first 500 seconds
  ./plot_comparison.py --max-time 500
    """
)
parser.add_argument('--max-time', type=float, default=None,
                    help='Maximum time to plot (seconds). If not specified, use full range.')

args = parser.parse_args()
max_time = args.max_time
if max_time is not None:
    print(f"Limiting plot to max-time: {max_time} s")


# ── Case definitions ──────────────────────────────────────────────────────────

def _fmt_perm(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

PERM_VALUES = np.array([1e-22, 1e-21, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16])
PERM_LABELS = [_fmt_perm(v) for v in PERM_VALUES]
COLORS      = cm.cividis(np.linspace(0.1, 0.9, len(PERM_VALUES)))

# Load datasets: data[setup][perm_label]
data = {"A": {}, "B": {}}
for setup in ("A", "B"):
    for label in PERM_LABELS:
        nc = RUNS_DIR / f"{setup}_perm_{label}" / "outputs" / "fem_timeseries.nc"
        if nc.exists():
            try:
                data[setup][label] = xr.open_dataset(nc)
            except Exception as e:
                print(f"  Warning: could not read {setup}_perm_{label}: {e}")

n_A = len(data["A"]); n_B = len(data["B"])
print(f"Setup A: {n_A}/{len(PERM_LABELS)} cases available")
print(f"Setup B: {n_B}/{len(PERM_LABELS)} cases available")
if n_A + n_B == 0:
    raise FileNotFoundError(f"No output files found under {RUNS_DIR}. Run run_all.py first.")

# ── Figure layout ─────────────────────────────────────────────────────────────
#   4 cols: [uz_A | p_A | uz_B | p_B]
#   N_PERM data rows + 1 thin load row

N_PERM = len(PERM_VALUES)
fig = plt.figure(figsize=(14, 2.2 * N_PERM + 1.5))
gs  = gridspec.GridSpec(N_PERM + 1, 2, figure=fig,
                        height_ratios=[3] * N_PERM + [1],
                        hspace=0.05, wspace=0.28,
                        top=0.93, bottom=0.06,
                        left=0.08, right=0.97)

ax_p_A  = [fig.add_subplot(gs[i, 0]) for i in range(N_PERM)]
ax_p_B  = [fig.add_subplot(gs[i, 1]) for i in range(N_PERM)]
ax_load = [fig.add_subplot(gs[N_PERM, c]) for c in range(2)]

# Share x within each row (A and B same perm → same _T_PHYS)
for i in range(N_PERM):
    ax_p_B[i].sharex(ax_p_A[i])   # fresh → OK

# Load tracks (fresh) share with row-0 root — all cases have same _T_PHYS
for ax in ax_load:
    ax.sharex(ax_p_A[0])

# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_pressure(ax, ds, color, max_time=None):
    if ds is None:
        return
    t = ds["time"].values

    # Filter by max_time if specified
    if max_time is not None:
        mask = t <= max_time
        t = t[mask]

    has_stats = ("pressure_mean" in ds and
                 "pressure_p10"  in ds and
                 "pressure_p90"  in ds)
    if has_stats:
        p_mean = ds["pressure_mean"].values / 1e3
        p_p10  = ds["pressure_p10"].values  / 1e3
        p_p90  = ds["pressure_p90"].values  / 1e3

        # Filter by max_time
        if max_time is not None:
            p_mean = p_mean[mask]
            p_p10  = p_p10[mask]
            p_p90  = p_p90[mask]

        ax.plot(t, p_mean, color=color, linewidth=1.5, zorder=3)
        ax.fill_between(t, p_p10, p_p90,
                        color=color, alpha=0.25, linewidth=0, zorder=2)

        # Annotate with P90-P10 range at final plotted timestep
        final_range = float(p_p90[-1] - p_p10[-1])
        ax.text(0.99, 0.97, f"ΔP={final_range:.1f} kPa",
                transform=ax.transAxes, fontsize=7,
                va="top", ha="right", color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    elif "pressure_at_top" in ds:
        p_data = ds["pressure_at_top"].values / 1e3

        # Filter by max_time
        if max_time is not None:
            p_data = p_data[mask]

        ax.plot(t, p_data,
                color=color, linewidth=1.5, linestyle="--", zorder=3)

# ── Draw data rows ────────────────────────────────────────────────────────────

p_vals = []

for i, (label, color) in enumerate(zip(PERM_LABELS, COLORS)):
    legend = f"k = {label} m²"
    ds_A = data["A"].get(label)
    ds_B = data["B"].get(label)

    _plot_pressure(ax_p_A[i], ds_A, color, max_time=max_time)
    _plot_pressure(ax_p_B[i], ds_B, color, max_time=max_time)

    # Collect range for uniform y-limits
    for ds in (ds_A, ds_B):
        if ds is not None:
            # Filter by max_time if specified
            if max_time is not None:
                t = ds["time"].values
                mask = t <= max_time
            else:
                mask = slice(None)  # No filtering

            if "pressure_mean" in ds:
                p_vals.extend(ds["pressure_mean"].values[mask] / 1e3)
            elif "pressure_at_top" in ds:
                p_vals.extend(ds["pressure_at_top"].values[mask] / 1e3)

    # Per-row label annotation
    for ax in (ax_p_A[i], ax_p_B[i]):
        ax.text(0.01, 0.97, legend, transform=ax.transAxes,
                fontsize=7, va="top", ha="left",
                color=color, fontweight="bold")
        ax.set_ylabel("P [kPa]", fontsize=8)
        ax.tick_params(labelsize=7, labelbottom=False)
        ax.grid(True, alpha=0.25, zorder=0)

    # Add text with P10-P90 range at final timestep
    if ds_A is not None and "pressure_p10" in ds_A and "pressure_p90" in ds_A:
        final_time = ds_A["time"].values[-1]
        p10_final = ds_A["pressure_p10"].values[-1] / 1e3  # Convert to kPa
        p90_final = ds_A["pressure_p90"].values[-1] / 1e3  # Convert to kPa
        range_final = p90_final - p10_final
        ax_p_A[i].text(0.99, 0.03, f"Range: {range_final:.1f} kPa",
                      transform=ax_p_A[i].transAxes, fontsize=6, va="bottom", ha="right",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    if ds_B is not None and "pressure_p10" in ds_B and "pressure_p90" in ds_B:
        final_time = ds_B["time"].values[-1]
        p10_final = ds_B["pressure_p10"].values[-1] / 1e3  # Convert to kPa
        p90_final = ds_B["pressure_p90"].values[-1] / 1e3  # Convert to kPa
        range_final = p90_final - p10_final
        ax_p_B[i].text(0.99, 0.03, f"Range: {range_final:.1f} kPa",
                      transform=ax_p_B[i].transAxes, fontsize=6, va="bottom", ha="right",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

p_vals = []

for i, (label, color) in enumerate(zip(PERM_LABELS, COLORS)):
    legend = f"k = {label} m²"
    ds_A = data["A"].get(label)
    ds_B = data["B"].get(label)

    _plot_pressure(ax_p_A[i], ds_A, color, max_time=max_time)
    _plot_pressure(ax_p_B[i], ds_B, color, max_time=max_time)

    # Collect range for uniform y-limits
    for ds in (ds_A, ds_B):
        if ds is not None:
            # Filter by max_time if specified
            if max_time is not None:
                t = ds["time"].values
                mask = t <= max_time
            else:
                mask = slice(None)  # No filtering
            
            if "pressure_mean" in ds:
                p_vals.extend(ds["pressure_mean"].values[mask] / 1e3)
            elif "pressure_at_top" in ds:
                p_vals.extend(ds["pressure_at_top"].values[mask] / 1e3)

    # Per-row label annotation
    for ax in (ax_p_A[i], ax_p_B[i]):
        ax.text(0.01, 0.97, legend, transform=ax.transAxes,
                fontsize=7, va="top", ha="left",
                color=color, fontweight="bold")
        ax.set_ylabel("P [kPa]", fontsize=8)
        ax.tick_params(labelsize=7, labelbottom=False)
        ax.grid(True, alpha=0.25, zorder=0)

# Uniform y-limits
if p_vals:
    lo, hi = np.nanmin(p_vals), np.nanmax(p_vals)
    pad = 0.06 * max(abs(hi - lo), 1.0)
    for ax in ax_p_A + ax_p_B:
        ax.set_ylim(lo - pad, hi + pad)

# ── Load track panels ─────────────────────────────────────────────────────────

def _draw_load(ax, setup, max_time=None):
    ds = next(iter(data[setup].values()), None)
    if ds is not None and "sig_zz_applied" in ds:
        t   = ds["time"].values
        sig = ds["sig_zz_applied"].values / 1e6
        
        # Filter by max_time if specified
        if max_time is not None:
            mask = t <= max_time
            t = t[mask]
            sig = sig[mask]
        
        ax.step(t, sig, color="crimson", linewidth=1.0, where="post", zorder=2)
        ax.fill_between(t, sig, step="post", color="crimson", alpha=0.15, zorder=1)
    ax.set_ylabel("σ_zz\n[MPa]", fontsize=7, labelpad=2)
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

_draw_load(ax_load[0], "A", max_time=max_time)
_draw_load(ax_load[1], "B", max_time=max_time)

# ── Titles ────────────────────────────────────────────────────────────────────

ax_p_A[0].set_title("Setup A  (500 s at −5 MPa → 1000 s at −1 MPa)", fontsize=9, pad=4)
ax_p_B[0].set_title("Setup B  (1500 s at −1 MPa)",                    fontsize=9, pad=4)

fig.suptitle(
    "Load-strategy comparison  |  permeability sweep  |  "
    "Re=10 cm, H=2 cm  |  α=0.75, M=1.35×10¹⁰ Pa\n"
    "Setup A: 500 s at −5 MPa → 1000 s at −1 MPa      "
    "Setup B: 1500 s at −1 MPa",
    fontsize=9)

# ── Save ──────────────────────────────────────────────────────────────────────

png_dir = DEMO_DIR / "png"
png_dir.mkdir(exist_ok=True)
out = png_dir / "comparison.png"
plt.savefig(out, dpi=500)
print(f"\nSaved {out}")
plt.show()
