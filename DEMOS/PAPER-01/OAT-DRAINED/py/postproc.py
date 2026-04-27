#!/usr/bin/env python3
"""OAT-DRAINED — sensitivity plots (paper style, one 90×60mm PNG per panel).

Outputs:
    png/oat_drained_{sw}_p.png
    png/oat_drained_{sw}_uz.png (with strain secondary axis)
    png/oat_drained_sensitivity.md

USAGE:
    ./py/postproc.py
    ./py/postproc.py --max-time 500
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

# ── Paper style ───────────────────────────────────────────────────────────────
MM = 1 / 25.4
FIG_W_MM = 90
FIG_H_MM = 60
XMIN_MIN = 1e-3
mpl.rcParams.update({
    'font.size':         6,   'font.weight':       'normal',
    'axes.titlesize':    7,   'axes.titleweight':  'normal',
    'axes.labelsize':    6,   'axes.labelweight':  'normal',
    'xtick.labelsize':   6,   'ytick.labelsize':   6,
    'legend.fontsize':   5,   'legend.handlelength': 1.2,
    'lines.linewidth':   0.8,
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size':  2.5, 'ytick.major.size':  2.5,
    'xtick.minor.size':  1.5, 'ytick.minor.size':  1.5,
    'grid.linewidth':    0.4,
    'pdf.fonttype': 42,       'ps.fonttype':  42,
})

DEMO_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = DEMO_DIR / "runs"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--min-time", type=float, default=0.0)
parser.add_argument("--max-time", type=float, default=20000.0)
args     = parser.parse_args()
max_time = args.max_time

H_OAT = 0.01

# ── Sweep definitions ─────────────────────────────────────────────────────────
PHI_VALUES   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ALPHA_VALUES = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
PERM_VALUES  = [1e-18, 1e-19, 1e-20, 1e-21, 1e-22]

def _fmt_perm(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

SWEEPS = {
    "phi":   {"labels": [f"phi_{v:.2f}"        for v in PHI_VALUES],
              "values": PHI_VALUES,
              "fmt":    lambda v: f"$\\phi={v*100:.0f}\\%$",
              "ncol":   2},
    "alpha": {"labels": [f"alpha_{v:.2f}"       for v in ALPHA_VALUES],
              "values": ALPHA_VALUES,
              "fmt":    lambda v: f"$\\alpha={v:.2f}$",
              "ncol":   2},
    "perm":  {"labels": [f"perm_{_fmt_perm(v)}" for v in PERM_VALUES],
              "values": PERM_VALUES,
              "fmt":    lambda v: f"$k={_fmt_perm(v)}$ (m$^2$)",
              "ncol":   1},
}

PALETTES = {
    "phi":   cm.gray(np.linspace(0.7, 0.1, len(PHI_VALUES))),
    "alpha": cm.gray(np.linspace(0.7, 0.1, len(ALPHA_VALUES))),
    "perm":  cm.gray(np.linspace(0.7, 0.1, len(PERM_VALUES))),
}

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

# ── Load datasets ─────────────────────────────────────────────────────────────
data = {sw: {} for sw in SWEEPS}
for sw, spec in SWEEPS.items():
    for label in spec["labels"]:
        nc = RUNS_DIR / label / "outputs" / "fem_timeseries.nc"
        if nc.exists():
            try:
                data[sw][label] = xr.open_dataset(nc)
            except Exception as e:
                print(f"  Warning: {label}: {e}")

for sw, spec in SWEEPS.items():
    print(f"{sw}: {len(data[sw])}/{len(spec['labels'])} cases")

if sum(len(v) for v in data.values()) == 0:
    raise FileNotFoundError(f"No outputs under {RUNS_DIR}. Run make run first.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _mask(ds):
    t = ds["time"].values / 60.0
    mk = np.ones(len(t), bool)
    if max_time is not None: mk &= (t <= max_time / 60.0)
    return t, mk

def _plot_pressure(ax, ds, color, label):
    if ds is None: return
    t, mk = _mask(ds)
    t_s = t[mk]
    pos = t_s > 0
    if not np.any(pos):
        return
    if "pressure_mean" in ds and "pressure_p10" in ds and "pressure_p90" in ds:
        pm   = np.clip(ds["pressure_mean"].values[mk][pos] / 1e3, 1e-1, None)
        pp10 = np.clip(ds["pressure_p10"].values[mk][pos]  / 1e3, 1e-1, None)
        pp90 = np.clip(ds["pressure_p90"].values[mk][pos]  / 1e3, 1e-1, None)
        ax.plot(t_s[pos], pm, color=color, label=label, zorder=3)
        ax.fill_between(t_s[pos], pp10, pp90, color=color, alpha=0.15,
                        linewidth=0, zorder=2)
    elif "pressure_at_base" in ds:
        p = np.clip(ds["pressure_at_base"].values[mk][pos] / 1e3, 1e-1, None)
        ax.plot(t_s[pos], p, color=color, linestyle="--", label=label, zorder=3)

def _plot_uz(ax, ds, color, label):
    if ds is None or "uz_at_top" not in ds: return
    t, mk = _mask(ds)
    t_s = t[mk]
    pos = t_s > 0
    if not np.any(pos):
        return
    uz = ds["uz_at_top"].values[mk][pos] * 2e6
    ax.plot(t_s[pos], uz, color=color, label=label, zorder=3)

def _fmt_xaxis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(plt.LogLocator(base=10, subs=[1]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(XMIN_MIN, max_time / 60.0)
    ax.margins(x=0)
    ax.set_xlabel("Time (min)")

def _new_fig():
    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM, FIG_H_MM * MM))
    return fig, ax

def _save(fig, path):
    fig.savefig(path, dpi=500, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {path}")

# ── One figure per sweep and variable ─────────────────────────────────────────
png_dir = DEMO_DIR / "png"
png_dir.mkdir(exist_ok=True)

for sw, spec in SWEEPS.items():
    palette = PALETTES[sw]
    ncol = spec["ncol"]

    fig_p, ax_p = _new_fig()
    for i, (label, val) in enumerate(zip(spec["labels"], spec["values"])):
        ds    = data[sw].get(label)
        color = palette[i]
        fmt   = spec["fmt"](val)
        _plot_pressure(ax_p, ds, color, fmt)
    ax_p.set_yscale("log")
    ax_p.set_ylim(bottom=1e-1)
    ax_p.set_ylabel("$P$ (kPa)")
    ax_p.grid(True, which="both", alpha=0.25, zorder=0)
    p_legend_loc = "lower left" if sw in {"perm", "alpha"} else "upper right"
    leg_p = ax_p.legend(loc=p_legend_loc, ncol=ncol, **LEGEND_STYLE)
    leg_p.get_frame().set_linewidth(0.3)
    _fmt_xaxis(ax_p)
    _save(fig_p, png_dir / f"oat_drained_{sw}_p.png")

    fig_u, ax_u = _new_fig()
    for i, (label, val) in enumerate(zip(spec["labels"], spec["values"])):
        ds    = data[sw].get(label)
        color = palette[i]
        fmt   = spec["fmt"](val)
        _plot_uz(ax_u, ds, color, fmt)
    ax_u.set_ylabel("$u_z$ (μm)")
    ax_u.grid(True, which="both", alpha=0.25, zorder=0)

    # Add secondary y-axis for strain
    ax_u_eps = ax_u.twinx()
    ax_u_eps.set_ylabel("$\\varepsilon_{zz}$ (‰)")
    y_min, y_max = ax_u.get_ylim()
    ax_u_eps.set_ylim(y_min / 10, y_max / 10)

    leg_u = ax_u.legend(loc="best", ncol=ncol, **LEGEND_STYLE)
    leg_u.get_frame().set_linewidth(0.3)
    _fmt_xaxis(ax_u)
    _save(fig_u, png_dir / f"oat_drained_{sw}_uz.png")

# ── Figure descriptions ───────────────────────────────────────────────────────
lines = ["# OAT-DRAINED sensitivity figures\n",
         "Drained lateral boundary (P = 0, mechanically free on right). "
         "E = 5 GPa, ν = 0.40, Kf = 2.2 GPa, μ = 10⁻³ Pa·s. "
         "H = 1 cm (drainage H/2 = 5 mm), Re = 2.5 cm. "
         "Load: −10 MPa step at t = 0.\n",
         "Each figure: 1-column (90 × 60 mm), one panel, 500 dpi.\n"]
descs = {
    "phi":   ("oat_drained_phi_p.png / oat_drained_phi_uz.png",
              "φ sweep: φ ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}; "
              "fixed α = 0.75, k = 1×10⁻²⁰ m²."),
    "alpha": ("oat_drained_alpha_p.png / oat_drained_alpha_uz.png",
              "α sweep: α ∈ {0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90}; "
              "fixed φ = 0.10, k = 1×10⁻²⁰ m²."),
    "perm":  ("oat_drained_perm_p.png / oat_drained_perm_uz.png",
              "k sweep: k ∈ {10⁻¹⁸, 10⁻¹⁹, 10⁻²⁰, 10⁻²¹, 10⁻²²} m²; "
              "fixed φ = 0.10, α = 0.75."),
}
for sw, (fname, desc) in descs.items():
    lines.append(f"## {fname}\n{desc}\n")
(png_dir / "oat_drained_sensitivity.md").write_text("\n".join(lines))
print(f"Saved {png_dir / 'oat_drained_sensitivity.md'}")
