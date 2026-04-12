#!/usr/bin/env python3
"""THIN-DISC-VS-TRADITIONAL — comparison plot.

Three columns, one per permeability (1e-20, 1e-21, 1e-22 m²).
Rows: pressure (log y) | uz | vertical strain | load track.

Each panel overlays thin-disc (drained side) vs traditional (sealed side).

USAGE:
    ./py/postproc.py
    ./py/postproc.py --max-time 2592000
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = DEMO_DIR / "runs"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--min-time", type=float, default=49.0,
                    help="Minimum time to plot [s] (default: 49)")
parser.add_argument("--max-time", type=float, default=30*86400.0,
                    help="Maximum time to plot [s] (default: 30 days)")
args = parser.parse_args()
min_time = args.min_time
max_time = args.max_time

# ── Case definitions ──────────────────────────────────────────────────────────
PERM_VALUES = [1e-20, 1e-21, 1e-22]

def _fmt_perm(v):
    s = f"{v:.2e}"; m, e = s.split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"

CASES = []
for perm in PERM_VALUES:
    p = _fmt_perm(perm)
    CASES.append({
        "perm": perm,
        "thin_label":  f"thin_disc_{p}",
        "trad_label":  f"traditional_{p}",
        "col_title":   f"$k = {p}$ m²",
    })

# Full specimen heights (used to compute vertical strain)
H_THIN = 0.010          # 1 cm
H_TRAD = 2 * 0.0254     # 2" = 50.8 mm

N_COLS = len(CASES)

# ── Load datasets ─────────────────────────────────────────────────────────────
def _load(label):
    nc = RUNS_DIR / label / "outputs" / "fem_timeseries.nc"
    if nc.exists():
        try:
            return xr.open_dataset(nc)
        except Exception as e:
            print(f"  Warning: {label}: {e}")
    return None

for case in CASES:
    case["thin_ds"] = _load(case["thin_label"])
    case["trad_ds"] = _load(case["trad_label"])
    thin_ok = case["thin_ds"] is not None
    trad_ok = case["trad_ds"] is not None
    p = _fmt_perm(case["perm"])
    print(f"k={p}: thin_disc={'OK' if thin_ok else 'MISSING'}, "
          f"traditional={'OK' if trad_ok else 'MISSING'}")

total = sum(1 for c in CASES for k in ("thin_ds","trad_ds") if c[k] is not None)
if total == 0:
    raise FileNotFoundError(f"No output files found under {RUNS_DIR}. Run make run first.")

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(5 * N_COLS, 11))
gs  = gridspec.GridSpec(4, N_COLS, figure=fig,
                        height_ratios=[4, 3, 3, 1],
                        hspace=0.10, wspace=0.30,
                        top=0.91, bottom=0.06,
                        left=0.08, right=0.97)

ax_p      = [fig.add_subplot(gs[0, c]) for c in range(N_COLS)]
ax_uz     = [fig.add_subplot(gs[1, c]) for c in range(N_COLS)]
ax_eps    = [fig.add_subplot(gs[2, c]) for c in range(N_COLS)]
ax_load   = [fig.add_subplot(gs[3, c]) for c in range(N_COLS)]

for c in range(N_COLS):
    ax_uz[c].sharex(ax_p[c])
    ax_eps[c].sharex(ax_p[c])
    ax_load[c].sharex(ax_p[c])

# ── Style ─────────────────────────────────────────────────────────────────────
STYLE = {
    "thin": dict(color="#1f77b4", linestyle="-",  linewidth=1.8, label="Thin-disc (drained side)"),
    "trad": dict(color="#d62728", linestyle="--", linewidth=1.8, label="Traditional (sealed)"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _mask(ds):
    t  = ds["time"].values
    mk = np.ones(len(t), bool)
    if min_time is not None:
        mk &= (t >= min_time)
    if max_time is not None:
        mk &= (t <= max_time)
    return t, mk


def _plot_pressure(ax, ds, style):
    if ds is None:
        return
    t, mk = _mask(ds)
    t_plot = t[mk] / 3600.0   # s → hours
    pos = t_plot > 0
    if "pressure_mean" in ds and "pressure_p10" in ds and "pressure_p90" in ds:
        p_mean = np.clip(ds["pressure_mean"].values[mk][pos] / 1e3, 1e-1, None)
        p_p10  = np.clip(ds["pressure_p10"].values[mk][pos]  / 1e3, 1e-1, None)
        p_p90  = np.clip(ds["pressure_p90"].values[mk][pos]  / 1e3, 1e-1, None)
        ax.plot(t_plot[pos], p_mean, **style, zorder=3)
        ax.fill_between(t_plot[pos], p_p10, p_p90,
                        color=style["color"], alpha=0.15, linewidth=0, zorder=2)
    elif "pressure_at_base" in ds:
        p = np.clip(ds["pressure_at_base"].values[mk][pos] / 1e3, 1e-1, None)
        ax.plot(t_plot[pos], p, **{**style, "linestyle": ":"}, zorder=3)


def _plot_uz(ax, ds, style):
    if ds is None or "uz_at_top" not in ds:
        return
    t, mk = _mask(ds)
    t_plot = t[mk] / 3600.0   # s → hours
    pos = t_plot > 0
    uz = ds["uz_at_top"].values[mk][pos] * 2e6   # m → μm, ×2 full specimen
    ax.plot(t_plot[pos], uz, **style, zorder=3)


def _plot_eps(ax, ds, H_full, style):
    """Vertical strain ε_zz = 2·uz_at_top / H_full  [×10⁻³ = ‰]."""
    if ds is None or "uz_at_top" not in ds:
        return
    t, mk = _mask(ds)
    t_plot = t[mk] / 3600.0
    pos = t_plot > 0
    eps = ds["uz_at_top"].values[mk][pos] * 2.0 / H_full * 1e3   # → ‰
    ax.plot(t_plot[pos], eps, **style, zorder=3)


def _draw_load(ax, ds):
    if ds is None or "sig_zz_applied" not in ds:
        return
    t, mk = _mask(ds)
    t_plot = t[mk] / 3600.0
    sig    = ds["sig_zz_applied"].values[mk] / 1e6
    pos = t_plot > 0
    ax.step(t_plot[pos], sig[pos], color="crimson", linewidth=1.0, where="post", zorder=2)
    ax.fill_between(t_plot[pos], sig[pos], step="post", color="crimson", alpha=0.15, zorder=1)
    ax.set_xscale("log")
    ax.set_ylabel("$\\sigma_{zz}$ [MPa]", fontsize=7, labelpad=2)
    ax.set_xlabel("Time [h]", fontsize=8)
    ax.tick_params(which="both", labelsize=7)
    ax.xaxis.set_major_locator(plt.LogLocator(base=10, subs=[1, 2, 5]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.grid(True, which="both", alpha=0.25, zorder=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))


# ── Draw ──────────────────────────────────────────────────────────────────────
for c, case in enumerate(CASES):
    thin_ds = case["thin_ds"]
    trad_ds = case["trad_ds"]

    _plot_pressure(ax_p[c],  thin_ds, {**STYLE["thin"]})
    _plot_pressure(ax_p[c],  trad_ds, {**STYLE["trad"]})
    _plot_uz(ax_uz[c],       thin_ds, {**STYLE["thin"]})
    _plot_uz(ax_uz[c],       trad_ds, {**STYLE["trad"]})
    _plot_eps(ax_eps[c],     thin_ds, H_THIN, {**STYLE["thin"]})
    _plot_eps(ax_eps[c],     trad_ds, H_TRAD, {**STYLE["trad"]})

    # Pressure axis — log-log
    ax_p[c].set_yscale("log")
    ax_p[c].set_xscale("log")
    ax_p[c].set_ylim(bottom=1e-1)
    ax_p[c].set_ylabel("Pressure [kPa]", fontsize=8)
    ax_p[c].tick_params(which="both", labelsize=7)
    plt.setp(ax_p[c].get_xticklabels(which="both"), visible=False)
    ax_p[c].grid(True, which="both", alpha=0.25, zorder=0)
    ax_p[c].set_title(case["col_title"], fontsize=10, pad=4)
    ax_p[c].legend(fontsize=7, loc="upper right", framealpha=0.7,
                   handlelength=1.4, labelspacing=0.3)

    # uz axis
    ax_uz[c].set_xscale("log")
    ax_uz[c].set_ylabel("$u_z$ (total) [μm]", fontsize=8)
    ax_uz[c].tick_params(which="both", labelsize=7)
    plt.setp(ax_uz[c].get_xticklabels(which="both"), visible=False)
    ax_uz[c].grid(True, which="both", alpha=0.25, zorder=0)

    # Vertical strain axis
    ax_eps[c].set_xscale("log")
    ax_eps[c].set_ylabel("$\\varepsilon_{zz}$ [‰]", fontsize=8)
    ax_eps[c].tick_params(which="both", labelsize=7)
    plt.setp(ax_eps[c].get_xticklabels(which="both"), visible=False)
    ax_eps[c].grid(True, which="both", alpha=0.25, zorder=0)

    # Load track — use whichever dataset is available
    ref_ds = thin_ds if thin_ds is not None else trad_ds
    _draw_load(ax_load[c], ref_ds)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "THIN-DISC vs TRADITIONAL — Consolidation comparison  |  "
    "E=5 GPa, ν=0.40, α=0.75, φ=0.10, M=22 GPa, μ=1e-3 Pa·s  |  Load: −10 MPa step\n"
    "Thin-disc: H=1 cm, Re=2.5 cm, right boundary drained (P=0, free)  |  "
    "Traditional: H=2\", Re=0.5\", right boundary sealed (U_r=0)",
    fontsize=8.5)

# ── Save ──────────────────────────────────────────────────────────────────────
png_dir = DEMO_DIR / "png"
png_dir.mkdir(exist_ok=True)
out = png_dir / "thin_disc_vs_traditional.png"
plt.savefig(out, dpi=500)
print(f"\nSaved {out}")
