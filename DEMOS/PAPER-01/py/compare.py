#!/usr/bin/env python3
"""OAT-SEALED vs OAT-DRAINED — lateral BC comparison (paper style, one figure per sweep).

Outputs:
    png/compare_phi.png
    png/compare_alpha.png
    png/compare_perm.png
    png/compare_sealed_vs_drained.md

USAGE:
    ./py/compare.py
    ./py/compare.py --max-time 500
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from pathlib import Path

# ── Paper style ───────────────────────────────────────────────────────────────
MM = 1 / 25.4
mpl.rcParams.update({
    'font.size':         6,   'font.weight':       'normal',
    'axes.titlesize':    7,   'axes.titleweight':  'normal',
    'axes.labelsize':    6,   'axes.labelweight':  'normal',
    'xtick.labelsize':   6,   'ytick.labelsize':   6,
    'legend.fontsize':   5,   'legend.handlelength': 1.4,
    'lines.linewidth':   0.8,
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size':  2.5, 'ytick.major.size':  2.5,
    'xtick.minor.size':  1.5, 'ytick.minor.size':  1.5,
    'grid.linewidth':    0.4,
    'pdf.fonttype': 42,       'ps.fonttype':  42,
})

PAPER01_DIR = Path(__file__).resolve().parents[1]
D13_RUNS    = PAPER01_DIR / "OAT-SEALED"  / "runs"
D14_RUNS    = PAPER01_DIR / "OAT-DRAINED" / "runs"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--min-time", type=float, default=0.0)
parser.add_argument("--max-time", type=float, default=20000.0)
args     = parser.parse_args()
min_time = args.min_time
max_time = args.max_time

# ── Sweep definitions ─────────────────────────────────────────────────────────
PHI_VALUES   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ALPHA_VALUES = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
PERM_VALUES  = [1e-18, 1e-19, 1e-20, 1e-21, 1e-22]

def _fmt_perm(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

SWEEPS = {
    "phi":   {"labels": [f"phi_{v:.2f}"        for v in PHI_VALUES],
              "title":  "$\\phi$ sweep",
              "fname":  "compare_phi.png"},
    "alpha": {"labels": [f"alpha_{v:.2f}"       for v in ALPHA_VALUES],
              "title":  "$\\alpha$ sweep",
              "fname":  "compare_alpha.png"},
    "perm":  {"labels": [f"perm_{_fmt_perm(v)}" for v in PERM_VALUES],
              "title":  "$k$ sweep",
              "fname":  "compare_perm.png"},
}
SWEEP_ORDER = ["phi", "alpha", "perm"]

D13_KW = dict(color="gray",  lw=1.2, ls="-",  zorder=2, alpha=0.85)
D14_KW = dict(color="black", lw=0.8, ls="--", zorder=3)

LEG_HANDLES = [
    mlines.Line2D([], [], color="gray",  lw=1.2, ls="-",  label="Sealed ($U_r=0$)"),
    mlines.Line2D([], [], color="black", lw=0.8, ls="--", label="Drained ($P=0$)"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _load(runs_dir, label):
    nc = runs_dir / label / "outputs" / "fem_timeseries.nc"
    if nc.exists():
        try:
            return xr.open_dataset(nc)
        except Exception as e:
            print(f"  Warning: {label}: {e}")
    return None

def _mask(ds):
    t = ds["time"].values
    mk = np.ones(len(t), bool)
    if min_time is not None: mk &= (t >= min_time)
    if max_time is not None: mk &= (t <= max_time)
    return t, mk

def _plot_pressure(ax, ds, **kw):
    if ds is None: return
    t, mk = _mask(ds)
    t_min = t[mk] / 60.0
    pos = t_min > 0
    if "pressure_mean" in ds:
        p = np.clip(ds["pressure_mean"].values[mk][pos] / 1e3, 1e-1, None)
    elif "pressure_at_base" in ds:
        p = np.clip(ds["pressure_at_base"].values[mk][pos] / 1e3, 1e-1, None)
    else:
        return
    ax.plot(t_min[pos], p, **kw)

def _plot_uz(ax, ds, **kw):
    if ds is None or "uz_at_top" not in ds: return
    t, mk = _mask(ds)
    t_min = t[mk] / 60.0
    pos = t_min > 0
    uz = ds["uz_at_top"].values[mk][pos] * 2e6
    ax.plot(t_min[pos], uz, **kw)

def _fmt_xaxis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(plt.LogLocator(base=10, subs=[1, 2, 5]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Time [min]")

# ── One figure per sweep ──────────────────────────────────────────────────────
png_dir = PAPER01_DIR / "png"
png_dir.mkdir(exist_ok=True)

for sw in SWEEP_ORDER:
    spec = SWEEPS[sw]
    loaded_any = False

    fig = plt.figure(figsize=(90*MM, 120*MM))
    gs  = gridspec.GridSpec(2, 1, figure=fig,
                            height_ratios=[1.2, 1],
                            hspace=0.30,
                            top=0.90, bottom=0.13,
                            left=0.20, right=0.97)
    ax_p  = fig.add_subplot(gs[0])
    ax_uz = fig.add_subplot(gs[1])

    for label in spec["labels"]:
        d13 = _load(D13_RUNS, label)
        d14 = _load(D14_RUNS, label)
        if d13 is not None:
            loaded_any = True
        _plot_pressure(ax_p,  d13, **D13_KW)
        _plot_pressure(ax_p,  d14, **D14_KW)
        _plot_uz(ax_uz,       d13, **D13_KW)
        _plot_uz(ax_uz,       d14, **D14_KW)

    if not loaded_any:
        print(f"  Warning: no data for {sw} sweep — skipping")
        plt.close(fig)
        continue

    # Pressure
    ax_p.set_yscale("log")
    ax_p.set_ylim(bottom=1e-1)
    ax_p.set_ylabel("$P$ [kPa]")
    ax_p.grid(True, which="both", alpha=0.25, zorder=0)
    ax_p.set_title(spec["title"], pad=3)
    ax_p.legend(handles=LEG_HANDLES, loc="upper right", framealpha=0.85,
                labelspacing=0.2, borderpad=0.4, handletextpad=0.4)
    _fmt_xaxis(ax_p)

    # uz
    ax_uz.set_ylabel("$u_z$ [μm]")
    ax_uz.grid(True, which="both", alpha=0.25, zorder=0)
    _fmt_xaxis(ax_uz)

    out = png_dir / spec["fname"]
    fig.savefig(out, dpi=500)
    plt.close(fig)
    print(f"Saved {out}")

# ── Figure descriptions ───────────────────────────────────────────────────────
lines = ["# Sealed vs Drained comparison figures\n",
         "OAT sweeps comparing sealed (U_r = 0) vs drained (P = 0, free) "
         "lateral boundary conditions. All sweep curves overlaid per figure.\n"
         "E = 5 GPa, ν = 0.40, Kf = 2.2 GPa, μ = 10⁻³ Pa·s. "
         "H = 1 cm, Re = 2.5 cm. Load: −10 MPa step at t = 0.\n",
         "Each figure: 1-column (90 × 120 mm), two rows — P [kPa] (log–log) and u_z [μm].\n",
         "Gray solid — sealed (OAT-SEALED). Black dashed — drained (OAT-DRAINED).\n"]
descs = {
    "phi":   ("compare_phi.png",
              "φ sweep: φ ∈ {0.05 … 0.30}; fixed α = 0.75, k = 1×10⁻²⁰ m²."),
    "alpha": ("compare_alpha.png",
              "α sweep: α ∈ {0.50 … 0.90}; fixed φ = 0.10, k = 1×10⁻²⁰ m²."),
    "perm":  ("compare_perm.png",
              "k sweep: k ∈ {10⁻¹⁸ … 10⁻²²} m²; fixed φ = 0.10, α = 0.75."),
}
for sw, (fname, desc) in descs.items():
    lines.append(f"## {fname}\n{desc}\n")
(png_dir / "compare_sealed_vs_drained.md").write_text("\n".join(lines))
print(f"Saved {png_dir / 'compare_sealed_vs_drained.md'}")
