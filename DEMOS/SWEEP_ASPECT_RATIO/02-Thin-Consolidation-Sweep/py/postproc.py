#!/usr/bin/env python3
"""Aspect-ratio sweep post-processing for conventional consolidation.

Produces one plot per figure with paper-style formatting.
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = DEMO_DIR / "runs"

H_OVER_RE_VALUES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00, 10.00]

MM = 1 / 25.4
FIG_W_MM = 90
FIG_H_MM = 60

mpl.rcParams.update({
    "font.size": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
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
    columnspacing=0.5,
)

parser = argparse.ArgumentParser()
parser.add_argument("--max-time", type=float, default=None)
args = parser.parse_args()


def _new_fig():
    return plt.subplots(figsize=(FIG_W_MM * MM, FIG_H_MM * MM))


def _save(fig, path):
    fig.savefig(path, dpi=500, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {path}")


def _fmt_log_decimal_axis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(plt.LogLocator(base=10, subs=[1]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


def _aspect_ratio(h_over_re):
    return 2.0 / h_over_re


# Load datasets
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

palette = cm.viridis(np.linspace(0.1, 0.9, len(H_OVER_RE_VALUES)))
color_map = {ar: palette[i] for i, ar in enumerate(H_OVER_RE_VALUES)}

png_dir = DEMO_DIR / "png"
png_dir.mkdir(exist_ok=True)

# Figure: pressure time series
fig_p, ax_p = _new_fig()
for ar, ds in sorted(datasets.items()):
    t = ds["time"].values
    mk = t > 0
    if args.max_time is not None:
        mk &= t <= args.max_time
    t_plot = t[mk]
    if t_plot.size == 0 or "pressure_mean" not in ds:
        continue
    p = np.clip(ds["pressure_mean"].values[mk] / 1e3, 1e-3, None)
    ax_p.plot(t_plot, p, color=color_map[ar], label=f"AR={_aspect_ratio(ar):g}")
    if "pressure_p10" in ds and "pressure_p90" in ds:
        p10 = np.clip(ds["pressure_p10"].values[mk] / 1e3, 1e-3, None)
        p90 = np.clip(ds["pressure_p90"].values[mk] / 1e3, 1e-3, None)
        ax_p.fill_between(t_plot, p10, p90, color=color_map[ar], alpha=0.15, linewidth=0)

_fmt_log_decimal_axis(ax_p)
ax_p.set_yscale("log")
ax_p.set_xlabel("Time (s)")
ax_p.set_ylabel("Mean excess pressure (kPa)")
ax_p.grid(True, which="both", alpha=0.25, zorder=0)
leg_p = ax_p.legend(loc="upper right", ncol=2, **LEGEND_STYLE)
leg_p.get_frame().set_linewidth(0.3)
_save(fig_p, png_dir / "ar_pressure_timeseries.png")

# Figure: settlement time series
fig_u, ax_u = _new_fig()
for ar, ds in sorted(datasets.items()):
    t = ds["time"].values
    mk = t > 0
    if args.max_time is not None:
        mk &= t <= args.max_time
    t_plot = t[mk]
    if t_plot.size == 0 or "uz_at_top" not in ds:
        continue
    uz = -ds["uz_at_top"].values[mk] * 1e6
    ax_u.plot(t_plot, uz, color=color_map[ar], label=f"AR={_aspect_ratio(ar):g}")

_fmt_log_decimal_axis(ax_u)
ax_u.set_xlabel("Time (s)")
ax_u.set_ylabel("Settlement (-u_z) (um)")
ax_u.grid(True, which="both", alpha=0.25, zorder=0)
leg_u = ax_u.legend(loc="best", ncol=2, **LEGEND_STYLE)
leg_u.get_frame().set_linewidth(0.3)
_save(fig_u, png_dir / "ar_settlement_timeseries.png")

# Final errors vs aspect ratio (one plot per figure)
ar_vals = []
p_errs = []
u_errs = []
v_errs = []
for h_over_re, ds in sorted(datasets.items()):
    ar_vals.append(_aspect_ratio(h_over_re))
    p_errs.append(float(ds["pressure_error_percent"].values[-1]))
    u_errs.append(float(ds["uz_error_percent"].values[-1]))
    v_errs.append(float(ds["volume_error_percent"].values[-1]))


def _plot_error(values, ylabel, out_name):
    fig, ax = _new_fig()
    ax.plot(ar_vals, values, color="black", linestyle="-")
    _fmt_log_decimal_axis(ax)
    ax.set_xlim(2, 100)
    ax.margins(x=0)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Aspect Ratio (diameter/height)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25, zorder=0)

    # Set specific x-axis ticks and labels
    ax.set_xticks([2, 4, 10, 50, 100])
    ax.set_xticklabels(["2", "4", "10", "50", "100"])

    # Mark regime boundaries
    ax.axvspan(2, 4, color="gray", alpha=0.25, zorder=0)
    ax.axvline(4, color="k", linestyle="--", linewidth=0.5, alpha=1, zorder=1)
    ax.annotate("1D consolidation", xy=(3.95, 97.5), xytext=(0, -5), textcoords="offset points",
                fontsize=5, ha="right", va="top", style="italic", alpha=1, rotation=90)
    ax.text(4.5, 97.5, "Thin disc test", fontsize=5, ha="left", va="top", style="italic", alpha=1)

    _save(fig, png_dir / out_name)


_plot_error(p_errs, "Pressure error (%)", "ar_pressure_error.png")
_plot_error(u_errs, "Displacement error (%)", "ar_displacement_error.png")
_plot_error(v_errs, "Volume error (%)", "ar_volume_error.png")
