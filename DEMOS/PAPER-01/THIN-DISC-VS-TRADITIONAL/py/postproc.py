#!/usr/bin/env python3
"""THIN-DISC-VS-TRADITIONAL — split paper-style plots.

Outputs per family (perm / alpha):
    png/thin_disc_vs_trad_{family}_p.png
    png/thin_disc_vs_trad_{family}_uz.png
    png/thin_disc_vs_trad_{family}_eps.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
from pathlib import Path

MM = 1 / 25.4
FIG_W_MM = 90
FIG_H_MM = 60
XMIN_MIN = 0.01

mpl.rcParams.update({
    "font.size": 6,
    "axes.titlesize": 6,
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

DEMO_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = DEMO_DIR / "runs"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--max-time", type=float, default=30 * 86400.0)
args = parser.parse_args()
max_time = args.max_time

PERM_VALUES = [1e-20, 1e-21, 1e-22]
ALPHA_VALUES = [0.70, 0.80, 0.90]
H_THIN = 0.010
H_TRAD = 2 * 0.0254


def _fmt_perm(v):
    s = f"{v:.2e}"
    m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"


def _load(label):
    nc = RUNS_DIR / label / "outputs" / "fem_timeseries.nc"
    if not nc.exists():
        return None
    try:
        return xr.open_dataset(nc)
    except Exception as e:
        print(f"  Warning: {label}: {e}")
        return None


def _new_fig():
    return plt.subplots(figsize=(FIG_W_MM * MM, FIG_H_MM * MM))


def _save(fig, path):
    fig.savefig(path, dpi=500, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {path}")


def _fmt_xaxis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(plt.LogLocator(base=10, subs=[1]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(XMIN_MIN, max_time / 60.0)
    ax.margins(x=0)
    ax.set_xlabel("Time (min)")


def _series(ds):
    t = ds["time"].values
    mk = t > 0
    return (t[mk] / 60.0), mk


def _two_legends(ax, entries, colors_thin, colors_trad, ncol):
    gray_palette = cm.gray(np.linspace(0.7, 0.1, len(entries)))
    sweep_handles = [
        mlines.Line2D([], [], color=gray_palette[len(entries)-1-i], linestyle="-", label=e["legend"])
        for i, e in enumerate(reversed(entries))
    ]
    style_handles = [
        mlines.Line2D([], [], color="black", linestyle="-", label="Thin-disc", linewidth=0.6),
        mlines.Line2D([], [], color="gray", linestyle="--", label="Triaxial", linewidth=0.6),
    ]
    leg1 = ax.legend(handles=sweep_handles, loc="lower left", ncol=1, **LEGEND_STYLE)
    leg1.get_frame().set_linewidth(0.3)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=style_handles, loc="upper right", ncol=1, handlelength=2.0, **LEGEND_STYLE)
    leg2.get_frame().set_linewidth(0.3)


def _plot_family(family_key, entries, colors_thin, colors_trad, ncol=2):
    png_dir = DEMO_DIR / "png"
    png_dir.mkdir(exist_ok=True)

    # Pressure
    fig_p, ax_p = _new_fig()
    for i, e in enumerate(entries):
        d_thin, d_trad = e["thin_ds"], e["trad_ds"]
        for ds, ls, proto, colors in [(d_thin, "-", "Thin-disc", colors_thin), (d_trad, "--", "Traditional", colors_trad)]:
            if ds is None:
                continue
            t, mk = _series(ds)
            pos = t > 0
            if not np.any(pos):
                continue
            label = f"{e['legend']} ({proto})"
            if "pressure_mean" in ds and "pressure_p10" in ds and "pressure_p90" in ds:
                pm = np.clip(ds["pressure_mean"].values[mk][pos] / 1e3, 1e-1, None)
                p10 = np.clip(ds["pressure_p10"].values[mk][pos] / 1e3, 1e-1, None)
                p90 = np.clip(ds["pressure_p90"].values[mk][pos] / 1e3, 1e-1, None)
                ax_p.plot(t[pos], pm, color=colors[i], linestyle=ls, zorder=3)
                if ls == "-":
                    ax_p.fill_between(t[pos], p10, p90, color=colors[i], alpha=0.15, linewidth=0, zorder=2)
    ax_p.set_yscale("log")
    ax_p.set_ylim(bottom=1e-1)
    ax_p.set_ylabel("$P$ (kPa)")
    ax_p.grid(True, which="both", alpha=0.25, zorder=0)
    _two_legends(ax_p, entries, colors_thin, colors_trad, ncol)
    _fmt_xaxis(ax_p)
    _save(fig_p, png_dir / f"thin_disc_vs_trad_{family_key}_p.png")

    # uz (focus)
    fig_u, ax_u = _new_fig()
    for i, e in enumerate(entries):
        d_thin, d_trad = e["thin_ds"], e["trad_ds"]
        for ds, ls, proto, colors in [(d_thin, "-", "Thin-disc", colors_thin), (d_trad, "--", "Traditional", colors_trad)]:
            if ds is None or "uz_at_top" not in ds:
                continue
            t, mk = _series(ds)
            pos = t > 0
            if not np.any(pos):
                continue
            uz = ds["uz_at_top"].values[mk][pos] * 2e6
            ax_u.plot(t[pos], uz, color=colors[i], linestyle=ls, zorder=3)
    ax_u.set_ylabel("$u_z$ (μm)")
    ax_u.grid(True, which="both", alpha=0.25, zorder=0)
    _two_legends(ax_u, entries, colors_thin, colors_trad, ncol)
    _fmt_xaxis(ax_u)
    _save(fig_u, png_dir / f"thin_disc_vs_trad_{family_key}_uz.png")

    # strain (kept, separate)
    fig_e, ax_e = _new_fig()
    for i, e in enumerate(entries):
        for ds, ls, Hf, proto, colors in [
            (e["thin_ds"], "-", H_THIN, "Thin-disc", colors_thin),
            (e["trad_ds"], "--", H_TRAD, "Traditional", colors_trad),
        ]:
            if ds is None or "uz_at_top" not in ds:
                continue
            t, mk = _series(ds)
            pos = t > 0
            if not np.any(pos):
                continue
            eps = ds["uz_at_top"].values[mk][pos] * 2.0 / Hf * 1e3
            ax_e.plot(t[pos], eps, color=colors[i], linestyle=ls, zorder=3)
    ax_e.set_ylabel("$\\varepsilon_{zz}$ (‰)")
    ax_e.grid(True, which="both", alpha=0.25, zorder=0)
    _two_legends(ax_e, entries, colors_thin, colors_trad, ncol)
    _fmt_xaxis(ax_e)
    _save(fig_e, png_dir / f"thin_disc_vs_trad_{family_key}_eps.png")


# Build families
perm_entries = []
for p in PERM_VALUES:
    tag = _fmt_perm(p)
    perm_entries.append({
        "legend": f"k={tag}",
        "thin_ds": _load(f"thin_disc_{tag}"),
        "trad_ds": _load(f"traditional_{tag}"),
    })

alpha_entries = []
for a in ALPHA_VALUES:
    tag = f"{a:.2f}"
    alpha_entries.append({
        "legend": f"$\\alpha={tag}$",
        "thin_ds": _load(f"thin_disc_alpha_{tag}"),
        "trad_ds": _load(f"traditional_alpha_{tag}"),
    })

if sum(1 for e in perm_entries + alpha_entries for k in ("thin_ds", "trad_ds") if e[k] is not None) == 0:
    raise FileNotFoundError(f"No outputs under {RUNS_DIR}. Run make run first.")

_plot_family("perm", perm_entries,
             cm.gray(np.linspace(0.7, 0.1, len(perm_entries))),
             cm.gray(np.linspace(0.7, 0.1, len(perm_entries))),
             ncol=2)
_plot_family("alpha", alpha_entries,
             cm.gray(np.linspace(0.7, 0.1, len(alpha_entries))),
             cm.gray(np.linspace(0.7, 0.1, len(alpha_entries))),
             ncol=2)

md = DEMO_DIR / "png" / "thin_disc_vs_traditional.md"
md.write_text(
    "# Thin-disc vs Traditional figures\n\n"
    "Split figures (one panel each, 90 x 60 mm): pressure, displacement, strain.\n"
    "Two families are generated: permeability sweep (3 values) and alpha sweep (3 values).\n"
    "Within each plot: solid = Thin-disc, dashed = Traditional.\n"
)
print(f"Saved {md}")
