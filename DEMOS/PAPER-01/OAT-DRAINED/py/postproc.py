#!/usr/bin/env python3
"""OAT-DRAINED — sensitivity plot (drained lateral boundary).

Three columns, one per sweep (phi | alpha | perm).
Rows: pressure (log y) | uz | load track.

USAGE:
    ./py/postproc.py
    ./py/postproc.py --max-time 500
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]   # OAT-DRAINED/
RUNS_DIR = DEMO_DIR / "runs"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Demo 14 OAT sensitivity plot (drained side)")
parser.add_argument("--min-time", type=float, default=0.0,
                    help="Minimum time to plot [s] (default: 0)")
parser.add_argument("--max-time", type=float, default=20000.0,
                    help="Maximum time to plot [s] (default: 20000)")
args = parser.parse_args()
min_time = args.min_time
max_time = args.max_time

# ── Sweep definitions (must match run_all.py) ─────────────────────────────────
PHI_VALUES   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ALPHA_VALUES = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
PERM_VALUES  = [1e-18, 1e-19, 1e-20, 1e-21, 1e-22]

def _fmt_perm(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

SWEEPS = {
    "phi":   {"labels": [f"phi_{v:.2f}"   for v in PHI_VALUES],
              "values": PHI_VALUES,
              "fmt":    lambda v: f"$\\phi={v*100:.0f}$%",
              "title":  "$\\phi$ sweep  ($\\alpha=0.75$, $k=10^{-20}$ m²)"},
    "alpha": {"labels": [f"alpha_{v:.2f}" for v in ALPHA_VALUES],
              "values": ALPHA_VALUES,
              "fmt":    lambda v: f"$\\alpha={v:.2f}$",
              "title":  "$\\alpha$ sweep  ($\\phi=0.10$, $k=10^{-20}$ m²)"},
    "perm":  {"labels": [f"perm_{_fmt_perm(v)}" for v in PERM_VALUES],
              "values": PERM_VALUES,
              "fmt":    lambda v: f"$k={_fmt_perm(v)}$ m²",
              "title":  "$k$ sweep  ($\\phi=0.10$, $\\alpha=0.75$)"},
}
SWEEP_ORDER = ["phi", "alpha", "perm"]
N_COLS = 3

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
    n = len(data[sw])
    print(f"{sw} sweep: {n}/{len(spec['labels'])} cases available")

total = sum(len(v) for v in data.values())
if total == 0:
    raise FileNotFoundError(f"No output files found under {RUNS_DIR}. Run run_all.py first.")

# ── Figure layout ─────────────────────────────────────────────────────────────
# 3 cols × 3 rows: pressure (log) | uz | load
fig = plt.figure(figsize=(15, 9))
gs  = gridspec.GridSpec(3, N_COLS, figure=fig,
                        height_ratios=[4, 3, 1],
                        hspace=0.10, wspace=0.30,
                        top=0.91, bottom=0.07,
                        left=0.07, right=0.97)

ax_p    = [fig.add_subplot(gs[0, c]) for c in range(N_COLS)]
ax_uz   = [fig.add_subplot(gs[1, c]) for c in range(N_COLS)]
ax_load = [fig.add_subplot(gs[2, c]) for c in range(N_COLS)]

# Share x within each column
for c in range(N_COLS):
    ax_uz[c].sharex(ax_p[c])
    ax_load[c].sharex(ax_p[c])

# ── Color palettes ────────────────────────────────────────────────────────────
PALETTES = {
    "phi":   cm.Blues(np.linspace(0.35, 0.95, len(PHI_VALUES))),
    "alpha": cm.Oranges(np.linspace(0.35, 0.95, len(ALPHA_VALUES))),
    "perm":  cm.Purples(np.linspace(0.35, 0.95, len(PERM_VALUES))),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _mask(ds, min_time, max_time):
    t  = ds["time"].values
    mk = np.ones(len(t), bool)
    if min_time is not None:
        mk &= (t >= min_time)
    if max_time is not None:
        mk &= (t <= max_time)
    return t, mk


def _plot_pressure(ax, ds, color, label, max_time=None):
    if ds is None:
        return
    t, mk = _mask(ds, min_time, max_time)
    t = t[mk] / 60.0   # s → min
    pos = t >= 0
    if "pressure_mean" in ds and "pressure_p10" in ds and "pressure_p90" in ds:
        p_mean = np.clip(ds["pressure_mean"].values[mk][pos] / 1e3, 1e-1, None)
        p_p10  = np.clip(ds["pressure_p10"].values[mk][pos]  / 1e3, 1e-1, None)
        p_p90  = np.clip(ds["pressure_p90"].values[mk][pos]  / 1e3, 1e-1, None)
        ax.plot(t[pos], p_mean, color=color, linewidth=1.5, label=label, zorder=3)
        ax.fill_between(t[pos], p_p10, p_p90, color=color, alpha=0.20, linewidth=0, zorder=2)
    elif "pressure_at_base" in ds:
        p = np.clip(ds["pressure_at_base"].values[mk][pos] / 1e3, 1e-1, None)
        ax.plot(t[pos], p, color=color, linewidth=1.5, linestyle="--", label=label, zorder=3)


def _plot_uz(ax, ds, color, label, max_time=None):
    if ds is None or "uz_at_top" not in ds:
        return
    t, mk = _mask(ds, min_time, max_time)
    t_plot = t[mk] / 60.0   # s → min
    pos = t_plot >= 0
    uz  = ds["uz_at_top"].values[mk][pos] * 2e6   # m → μm, ×2 for full specimen
    ax.plot(t_plot[pos], uz, color=color, linewidth=1.5, label=label, zorder=3)


def _draw_load(ax, ds, max_time=None):
    if ds is None or "sig_zz_applied" not in ds:
        return
    t, mk = _mask(ds, min_time, max_time)
    t_plot = t[mk] / 60.0   # s → min
    sig    = ds["sig_zz_applied"].values[mk] / 1e6
    pos = t_plot >= 0
    ax.step(t_plot[pos], sig[pos], color="crimson", linewidth=1.0, where="post", zorder=2)
    ax.fill_between(t_plot[pos], sig[pos], step="post", color="crimson", alpha=0.15, zorder=1)
    ax.set_xscale("log")
    ax.set_ylabel("$\\sigma_{zz}$ [MPa]", fontsize=7, labelpad=2)
    ax.set_xlabel("Time [min]", fontsize=8)
    ax.tick_params(which="both", labelsize=7)
    ax.xaxis.set_major_locator(plt.LogLocator(base=10, subs=[1, 2, 5]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.grid(True, which="both", alpha=0.25, zorder=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))


# ── Draw ──────────────────────────────────────────────────────────────────────

for c, sw in enumerate(SWEEP_ORDER):
    spec    = SWEEPS[sw]
    palette = PALETTES[sw]

    for i, (label, val) in enumerate(zip(spec["labels"], spec["values"])):
        ds    = data[sw].get(label)
        color = palette[i]
        fmt   = spec["fmt"](val)
        _plot_pressure(ax_p[c],  ds, color, fmt, max_time=max_time)
        _plot_uz(ax_uz[c],       ds, color, fmt, max_time=max_time)

    # Pressure axis — log-log
    ax_p[c].set_yscale("log")
    ax_p[c].set_xscale("log")
    ax_p[c].set_ylim(bottom=1e-1)
    ax_p[c].set_ylabel("Pressure [kPa]", fontsize=8)
    ax_p[c].tick_params(which="both", labelsize=7)
    plt.setp(ax_p[c].get_xticklabels(which="both"), visible=False)
    ax_p[c].grid(True, which="both", alpha=0.25, zorder=0)
    ax_p[c].set_title(spec["title"], fontsize=9, pad=4)
    ax_p[c].legend(fontsize=6, loc="upper right", framealpha=0.7,
                   ncol=1, handlelength=1.2, labelspacing=0.3)

    # uz axis — log time
    ax_uz[c].set_xscale("log")
    ax_uz[c].set_ylabel("$u_z$ (total displacement) [μm]", fontsize=8)
    ax_uz[c].tick_params(which="both", labelsize=7)
    plt.setp(ax_uz[c].get_xticklabels(which="both"), visible=False)
    ax_uz[c].grid(True, which="both", alpha=0.25, zorder=0)
    if c == 0:
        ax_uz[c].legend(fontsize=6, loc="lower right", framealpha=0.7,
                        ncol=1, handlelength=1.2, labelspacing=0.3)

    # Load track
    first_ds = next(iter(data[sw].values()), None)
    _draw_load(ax_load[c], first_ds, max_time=max_time)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "OAT-DRAINED — Sensitivity, drained lateral boundary (right: P=0, free)  |  "
    "E=5 GPa, ν=0.40, M=Kf/φ (Kf=2.2 GPa), μ_fluid=1e-3 Pa·s\n"
    "H=1.0 cm (H/2=0.5 cm drainage path), Re=2.5 cm  |  Load: −10 MPa step at t=50 s  |  "
    "base: φ=0.10, α=0.75, k=1e-20 m²",
    fontsize=9)

# ── Save ──────────────────────────────────────────────────────────────────────
png_dir = DEMO_DIR / "png"
png_dir.mkdir(exist_ok=True)
out = png_dir / "oat_sensitivity_drained_side.png"
plt.savefig(out, dpi=500)
print(f"\nSaved {out}")
