#!/usr/bin/env python3
"""Compare OAT-SEALED vs OAT-DRAINED lateral boundary conditions.

OAT-SEALED — thick gray solid line  (sealed side: U_r=0 on right)
OAT-DRAINED — dashed black line     (drained side: P=0 on right, mechanically free)

Three columns: φ sweep | α sweep | k sweep
Two data rows:  pressure P50 | u_z at bottom
One load row.

USAGE:
    ./py/compare.py
    ./py/compare.py --max-time 500
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from pathlib import Path

PAPER01_DIR = Path(__file__).resolve().parents[1]   # PAPER-01/
D13_RUNS    = PAPER01_DIR / "OAT-SEALED" / "runs"
D14_RUNS    = PAPER01_DIR / "OAT-DRAINED" / "runs"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Compare OAT-SEALED vs OAT-DRAINED")
parser.add_argument("--min-time", type=float, default=0.0,
                    help="Minimum time to plot [s] (default: 0)")
parser.add_argument("--max-time", type=float, default=20000.0,
                    help="Maximum time to plot [s] (default: 20000)")
args = parser.parse_args()
min_time = args.min_time
max_time = args.max_time

# ── Sweep definitions (must match run_all.py in both demos) ───────────────────
PHI_VALUES   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ALPHA_VALUES = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
PERM_VALUES  = [1e-21, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16]

def _fmt_perm(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

SWEEPS = {
    "phi":   {"labels": [f"phi_{v:.2f}"          for v in PHI_VALUES],
              "title":  "$\\phi$ sweep  ($\\alpha=0.50$, $k=10^{-20}$ m²)"},
    "alpha": {"labels": [f"alpha_{v:.2f}"         for v in ALPHA_VALUES],
              "title":  "$\\alpha$ sweep  ($\\phi=0.10$, $k=10^{-20}$ m²)"},
    "perm":  {"labels": [f"perm_{_fmt_perm(v)}"   for v in PERM_VALUES],
              "title":  "$k$ sweep  ($\\phi=0.10$, $\\alpha=0.50$)"},
}
SWEEP_ORDER = ["phi", "alpha", "perm"]
N_COLS = 3

# ── Styles ────────────────────────────────────────────────────────────────────
D13_KW = dict(color="gray",  lw=2.5, ls="-",  zorder=2, alpha=0.85)
D14_KW = dict(color="black", lw=1.5, ls="--", zorder=3)

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
    t  = ds["time"].values
    mk = np.ones(len(t), bool)
    if min_time is not None: mk &= (t >= min_time)
    if max_time is not None: mk &= (t <= max_time)
    return t, mk


def _plot_pressure(ax, ds, **kw):
    if ds is None:
        return
    t, mk = _mask(ds)
    t = t[mk] / 60.0; pos = t >= 0   # s → min
    if "pressure_mean" in ds:
        p = np.clip(ds["pressure_mean"].values[mk][pos] / 1e3, 1e-1, None)
    elif "pressure_at_base" in ds:
        p = np.clip(ds["pressure_at_base"].values[mk][pos] / 1e3, 1e-1, None)
    else:
        return
    ax.plot(t[pos], p, **kw)


def _plot_uz(ax, ds, **kw):
    if ds is None or "uz_at_top" not in ds:
        return
    t, mk = _mask(ds)
    t = t[mk] / 60.0; pos = t >= 0   # s → min
    uz = ds["uz_at_top"].values[mk][pos] * 2e6   # m → μm, ×2 for full specimen
    ax.plot(t[pos], uz, **kw)


def _draw_load(ax, ds):
    if ds is None or "sig_zz_applied" not in ds:
        return
    t, mk = _mask(ds)
    t = t[mk] / 60.0; sig = ds["sig_zz_applied"].values[mk] / 1e6   # s → min
    pos = t >= 0
    ax.step(t[pos], sig[pos], color="crimson", lw=1.0, where="post", zorder=2)
    ax.fill_between(t[pos], sig[pos], step="post", color="crimson", alpha=0.15, zorder=1)
    ax.set_xscale("log")
    ax.set_ylabel("$\\sigma_{zz}$ [MPa]", fontsize=7, labelpad=2)
    ax.set_xlabel("Time [min]", fontsize=8)
    ax.tick_params(which="both", labelsize=7)
    ax.xaxis.set_major_locator(plt.LogLocator(base=10, subs=[1, 2, 5]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.grid(True, which="both", alpha=0.25, zorder=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))


# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9))
gs  = gridspec.GridSpec(3, N_COLS, figure=fig,
                        height_ratios=[4, 3, 1],
                        hspace=0.10, wspace=0.30,
                        top=0.91, bottom=0.07,
                        left=0.07, right=0.97)

ax_p    = [fig.add_subplot(gs[0, c]) for c in range(N_COLS)]
ax_uz   = [fig.add_subplot(gs[1, c]) for c in range(N_COLS)]
ax_load = [fig.add_subplot(gs[2, c]) for c in range(N_COLS)]

for c in range(N_COLS):
    ax_uz[c].sharex(ax_p[c])
    ax_load[c].sharex(ax_p[c])

# ── Draw ──────────────────────────────────────────────────────────────────────
loaded_any = False

for c, sw in enumerate(SWEEP_ORDER):
    spec      = SWEEPS[sw]
    first_ds  = None

    for label in spec["labels"]:
        d13 = _load(D13_RUNS, label)
        d14 = _load(D14_RUNS, label)

        if d13 is not None:
            loaded_any = True
            if first_ds is None:
                first_ds = d13

        _plot_pressure(ax_p[c],  d13, **D13_KW)
        _plot_pressure(ax_p[c],  d14, **D14_KW)
        _plot_uz(ax_uz[c],       d13, **D13_KW)
        _plot_uz(ax_uz[c],       d14, **D14_KW)

    if not loaded_any:
        raise FileNotFoundError(
            f"No output files found under {D13_RUNS} or {D14_RUNS}. "
            "Run run_all.py in both demos first.")

    # Pressure axis — log-log
    ax_p[c].set_yscale("log")
    ax_p[c].set_xscale("log")
    ax_p[c].set_ylim(bottom=1e-1)
    ax_p[c].set_ylabel("Pressure [kPa]", fontsize=8)
    ax_p[c].tick_params(which="both", labelsize=7)
    plt.setp(ax_p[c].get_xticklabels(which="both"), visible=False)
    ax_p[c].grid(True, which="both", alpha=0.25, zorder=0)
    ax_p[c].set_title(spec["title"], fontsize=9, pad=4)

    # uz axis — log time
    ax_uz[c].set_xscale("log")
    ax_uz[c].set_ylabel("$u_z$ (total displacement) [μm]", fontsize=8)
    ax_uz[c].tick_params(which="both", labelsize=7)
    plt.setp(ax_uz[c].get_xticklabels(which="both"), visible=False)
    ax_uz[c].grid(True, which="both", alpha=0.25, zorder=0)

    _draw_load(ax_load[c], first_ds)

# ── Shared legend ─────────────────────────────────────────────────────────────
leg_handles = [
    mlines.Line2D([], [], color="gray",  lw=2.5, ls="-",
                  label="OAT-SEALED — sealed  (right: U_r=0)"),
    mlines.Line2D([], [], color="black", lw=1.5, ls="--",
                  label="OAT-DRAINED — drained side  (right: P=0, free)"),
]
ax_p[0].legend(handles=leg_handles, fontsize=7, loc="upper right",
               framealpha=0.85, handlelength=2.2)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "OAT-SEALED vs OAT-DRAINED — Sealed vs Drained Lateral Boundary  |  "
    "E=10 GPa, ν=0.35, M=Kf/φ (Kf=2.2 GPa), μ_fluid=1e-3 Pa·s\n"
    "H=1.0 cm (H/2=0.5 cm drainage path), Re=2.5 cm  |  Load: −10 MPa step at t=50 s  |  "
    "base: φ=0.10, α=0.50, k=1e-20 m²  |  each family: all sweep values overlaid",
    fontsize=9)

# ── Save ──────────────────────────────────────────────────────────────────────
png_dir = PAPER01_DIR / "png"
png_dir.mkdir(exist_ok=True)
out = png_dir / "compare_sealed_vs_drained.png"
plt.savefig(out, dpi=500)
print(f"\nSaved {out}")
