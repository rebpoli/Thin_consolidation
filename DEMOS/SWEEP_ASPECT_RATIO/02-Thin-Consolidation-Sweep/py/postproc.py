#!/usr/bin/env python3
"""Aspect-ratio × Poisson sweep post-processing.

Each plot draws one line per Poisson ratio.
  - Error figures: x = aspect ratio, one curve per nu.
  - Timeseries figures: small-multiples grid with one subplot per AR,
    four curves (one per nu) inside each subplot.
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

import re

_LABEL_RE = re.compile(r"^ar_(?P<ar>[0-9.]+)__nu_(?P<nu>[0-9.]+)$")


def _discover_cases():
    """Discover (h_over_re, nu) → dir-name from runs/ with completed output."""
    found = {}
    if RUNS_DIR.exists():
        for d in RUNS_DIR.iterdir():
            if not d.is_dir():
                continue
            m = _LABEL_RE.match(d.name)
            if not m or not (d / "outputs" / "fem_timeseries.nc").exists():
                continue
            found[(float(m["ar"]), float(m["nu"]))] = d.name
    h_vals  = sorted({h for h, _ in found})
    nu_vals = sorted({n for _, n in found})
    return h_vals, nu_vals, found


H_OVER_RE_VALUES, NU_VALUES, _CASE_DIRS = _discover_cases()
if not H_OVER_RE_VALUES:
    raise FileNotFoundError(
        f"No completed runs found in {RUNS_DIR}. Run `make run` first."
    )

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


def _label(h_over_re, nu):
    return _CASE_DIRS.get((h_over_re, nu), f"ar_{h_over_re:.4f}__nu_{nu:.2f}")


# ── Load datasets: datasets[(ar, nu)] = xr.Dataset ────────────────────────────
datasets = {}
for ar in H_OVER_RE_VALUES:
    for nu in NU_VALUES:
        nc = RUNS_DIR / _label(ar, nu) / "outputs" / "fem_timeseries.nc"
        if nc.exists():
            try:
                datasets[(ar, nu)] = xr.open_dataset(nc)
            except Exception as e:
                print(f"  Warning: {_label(ar, nu)}: {e}")

if not datasets:
    raise FileNotFoundError(f"No output files found under {RUNS_DIR}. Run make run first.")

print(f"Loaded {len(datasets)}/{len(H_OVER_RE_VALUES)*len(NU_VALUES)} cases.")

nu_palette = cm.coolwarm(np.linspace(0.0, 1.0, len(NU_VALUES)))
nu_color   = {nu: nu_palette[i] for i, nu in enumerate(NU_VALUES)}

png_dir = DEMO_DIR / "png"
png_dir.mkdir(exist_ok=True)


# ── Timeseries: small multiples (one subplot per AR, 4 nu curves each) ────────
def _plot_timeseries_grid(var, scale, ylabel, out_name, *, ylog=False, neg=False):
    n_ar = len(H_OVER_RE_VALUES)
    ncol = 5
    nrow = (n_ar + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(FIG_W_MM * MM * 1.8, FIG_H_MM * MM * nrow * 0.9),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()

    for i, ar in enumerate(H_OVER_RE_VALUES):
        ax = axes[i]
        for nu in NU_VALUES:
            ds = datasets.get((ar, nu))
            if ds is None or var not in ds:
                continue
            t = ds["time"].values
            mk = t > 0
            if args.max_time is not None:
                mk &= t <= args.max_time
            t_plot = t[mk]
            if t_plot.size == 0:
                continue
            y = ds[var].values[mk] * scale
            if neg:
                y = -y
            if ylog:
                y = np.clip(y, 1e-3, None)
            ax.plot(t_plot, y, color=nu_color[nu], label=f"ν={nu:g}")
        _fmt_log_decimal_axis(ax)
        if ylog:
            ax.set_yscale("log")
        ax.set_title(f"AR={_aspect_ratio(ar):g}", fontsize=6)
        ax.grid(True, which="both", alpha=0.25, zorder=0)

    for j in range(len(H_OVER_RE_VALUES), len(axes)):
        axes[j].set_visible(False)

    for ax in axes[-ncol:]:
        ax.set_xlabel("Time (s)")
    for r in range(nrow):
        axes[r * ncol].set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        leg = fig.legend(handles, labels, loc="upper center",
                         ncol=len(NU_VALUES), bbox_to_anchor=(0.5, 1.02), **LEGEND_STYLE)
        leg.get_frame().set_linewidth(0.3)

    fig.tight_layout()
    _save(fig, png_dir / out_name)


_plot_timeseries_grid("pressure_mean", 1 / 1e3,
                      "Mean p (kPa)", "ar_pressure_timeseries.png", ylog=True)
_plot_timeseries_grid("uz_at_top", 1e6,
                      "Settlement (-u_z) (μm)", "ar_settlement_timeseries.png", neg=True)


# ── Error vs AR: one line per nu ──────────────────────────────────────────────
def _plot_error(var, ylabel, out_name):
    fig, ax = _new_fig()
    for nu in NU_VALUES:
        ar_vals, errs = [], []
        for h_over_re in H_OVER_RE_VALUES:
            ds = datasets.get((h_over_re, nu))
            if ds is None or var not in ds:
                continue
            ar_vals.append(_aspect_ratio(h_over_re))
            errs.append(float(ds[var].values[-1]))
        if not ar_vals:
            continue
        order = np.argsort(ar_vals)
        ar_vals = np.asarray(ar_vals)[order]
        errs    = np.asarray(errs)[order]
        ax.plot(ar_vals, errs, color=nu_color[nu], label=f"ν={nu:g}")

    _fmt_log_decimal_axis(ax)
    ax.set_xlim(4, 50)
    ax.margins(x=0)
    ax.set_ylim(0, 40)
    ax.set_xlabel("Aspect Ratio (diameter/height)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25, zorder=0)
    ax.set_xticks([4, 10, 50])
    ax.set_xticklabels(["4", "10", "50"])

    leg = ax.legend(loc="upper right", **LEGEND_STYLE)
    leg.get_frame().set_linewidth(0.3)
    _save(fig, png_dir / out_name)


_plot_error("pressure_error_percent",   "Pressure error (%)",     "ar_pressure_error.png")
_plot_error("uz_error_percent",         "Displacement error (%)", "ar_displacement_error.png")
_plot_error("volume_error_percent",     "Volume error (%)",       "ar_volume_error.png")
