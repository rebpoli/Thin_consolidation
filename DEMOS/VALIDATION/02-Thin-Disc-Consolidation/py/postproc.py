#!/usr/bin/env python3
"""02-Thin-Disc-Consolidation — FEM vs Analytical comparison plot.

Unjacketed thin-disc test: right boundary is drained (P=0) and mechanically
free, allowing lateral drainage and radial expansion.  The consolidation
response should converge to the 1D Terzaghi analytical solution as H/Re → 0.

Loads fem_timeseries.nc and analytical_timeseries.nc from outputs/ and
produces a comparison of pressure dissipation and axial displacement.

USAGE:
    ./py/postproc.py
    ./py/postproc.py --outputs-dir outputs
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr

DEMO_DIR = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-dir", default="outputs",
                        help="Directory containing NetCDF outputs (default: outputs/)")
    args = parser.parse_args()

    out_dir = DEMO_DIR / args.outputs_dir
    fem_nc  = out_dir / "fem_timeseries.nc"
    ana_nc  = out_dir / "analytical_timeseries.nc"

    if not fem_nc.exists():
        raise FileNotFoundError(f"FEM output not found: {fem_nc}\nRun the simulation first.")

    fem = xr.open_dataset(fem_nc)
    ana = xr.open_dataset(ana_nc) if ana_nc.exists() else None

    t = fem["time"].values
    mask = t > 0

    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30,
                            top=0.88, bottom=0.09, left=0.09, right=0.97)

    ax_p   = fig.add_subplot(gs[0, 0])
    ax_uz  = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, :])

    # ── Pressure ──────────────────────────────────────────────────────────────
    if "pressure_mean" in fem:
        p_fem = np.clip(fem["pressure_mean"].values[mask] / 1e3, 1e-2, None)
        ax_p.plot(t[mask] / 60, p_fem, color="steelblue", lw=1.8, label="FEM (mean)")
        if "pressure_p10" in fem and "pressure_p90" in fem:
            ax_p.fill_between(t[mask] / 60,
                              np.clip(fem["pressure_p10"].values[mask] / 1e3, 1e-2, None),
                              np.clip(fem["pressure_p90"].values[mask] / 1e3, 1e-2, None),
                              color="steelblue", alpha=0.2)
    if ana is not None and "pressure_mean" in ana:
        p_ana = np.clip(ana["pressure_mean"].values[mask] / 1e3, 1e-2, None)
        ax_p.plot(t[mask] / 60, p_ana, "k--", lw=1.2, label="Analytical (1D Terzaghi)")
    ax_p.set_yscale("log"); ax_p.set_xscale("log")
    ax_p.set_xlabel("Time [min]", fontsize=8)
    ax_p.set_ylabel("Mean pressure [kPa]", fontsize=8)
    ax_p.set_title("Pressure dissipation", fontsize=9)
    ax_p.legend(fontsize=7); ax_p.grid(True, which="both", alpha=0.25)
    ax_p.tick_params(labelsize=7)

    # ── Displacement ───────────────────────────────────────────────────────────
    if "uz_at_top" in fem:
        uz_fem = fem["uz_at_top"].values[mask] * 2e6   # m → μm, ×2 full specimen
        ax_uz.plot(t[mask] / 60, uz_fem, color="darkorange", lw=1.8, label="FEM")
    if ana is not None and "uz_at_top" in ana:
        uz_ana = ana["uz_at_top"].values[mask] * 2e6
        ax_uz.plot(t[mask] / 60, uz_ana, "k--", lw=1.2, label="Analytical")
    ax_uz.set_xscale("log")
    ax_uz.set_xlabel("Time [min]", fontsize=8)
    ax_uz.set_ylabel("Axial displacement [μm]", fontsize=8)
    ax_uz.set_title("Axial displacement (full specimen)", fontsize=9)
    ax_uz.legend(fontsize=7); ax_uz.grid(True, which="both", alpha=0.25)
    ax_uz.tick_params(labelsize=7)

    # ── L2 error ──────────────────────────────────────────────────────────────
    if "pressure_l2_error" in fem:
        err = fem["pressure_l2_error"].values[mask]
        ax_err.plot(t[mask] / 60, err, color="firebrick", lw=1.5,
                    label="Pressure L2 error vs 1D analytical")
        ax_err.set_xscale("log"); ax_err.set_yscale("log")
        ax_err.set_xlabel("Time [min]", fontsize=8)
        ax_err.set_ylabel("L2 error [Pa]", fontsize=8)
        ax_err.set_title("Convergence to Analytical Solution", fontsize=9)
        ax_err.legend(fontsize=7); ax_err.grid(True, which="both", alpha=0.25)
        ax_err.tick_params(labelsize=7)
    else:
        ax_err.set_visible(False)

    fig.suptitle("02-Thin-Disc-Consolidation — FEM vs Terzaghi Analytical\n"
                 "Unjacketed test (drained side P=0, mechanically free), "
                 "drained top, step axial load",
                 fontsize=10)

    png_dir = DEMO_DIR / "png"
    png_dir.mkdir(exist_ok=True)
    out = png_dir / "fem_vs_analytical.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")
    fem.close()
    if ana: ana.close()


if __name__ == "__main__":
    main()
