#!/usr/bin/env python3
"""Demo 14 — P-Q stress-invariant scatter plot.

Reads invariants.nc from every run under runs/ and saves one PNG per run.

Convention (Cambridge / geomechanics):
    p  = I1 / 3          (mean stress, negative = compression)
    q  = sqrt(3 * J2)    (deviatoric stress, always ≥ 0)

Mohr-Coulomb envelope (Drucker-Prager outer-circumscribed):
    q  = C1 + C2 * |p|
    C1 = 6·c·cos(φ) / (3 − sin φ)
    C2 = 6·sin(φ)   / (3 − sin φ)

Default material: carbonate (φ=40°, c=10 MPa)

USAGE:
    python plot_pq.py                       # all runs
    python plot_pq.py --run phi_0.20        # single run
    python plot_pq.py --phi 35 --coh 8      # override MC envelope
"""
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── LaTeX-like fonts ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex":         False,
    "font.family":         "serif",
    "font.serif":          ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "mathtext.fontset":    "dejavuserif",
    "axes.titlesize":      9,
    "axes.labelsize":      8,
    "axes.labelpad":       2,
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
    "legend.fontsize":     7,
})

DEMO_DIR = Path(__file__).resolve().parents[1]   # OAT-DRAINED/
RUNS_DIR = DEMO_DIR / "runs"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Demo 14 P-Q invariant plot")
parser.add_argument("--run",  default=None,
                    help="Single run label to plot (default: all runs)")
parser.add_argument("--phi",  type=float, default=30.0,
                    help="MC friction angle [deg] (default: 30)")
parser.add_argument("--ucs",  type=float, default=None,
                    help="UCS [MPa] for MC envelope (default: computed from data — "
                         "max UCS at which 5%% of domain is outside)")
parser.add_argument("--p-max", type=float, default=None,
                    help="Max -p' for MC envelope [MPa] (default: auto)")
parser.add_argument("--snap-time", type=float, default=120.0,
                    help="Physical time [s] for spatial snapshot (default: 120 s ≈ 2 min)")
args = parser.parse_args()

phi_deg = args.phi
phi_rad = np.radians(phi_deg)

# Envelope coefficients (φ-only)
_C2_env = 6.0 * np.sin(phi_rad) / (3.0 - np.sin(phi_rad))
_K_env  = 3.0 * (1.0 - np.sin(phi_rad)) / (3.0 - np.sin(phi_rad))
# Tension cutoff: T0 = UCS * K_t  (T0 = c/tan(φ))
_K_t    = (1.0 - np.sin(phi_rad)) / (2.0 * np.sin(phi_rad))


def _ucs_crit_combined(p_MPa, q_MPa):
    """Per-point UCS at which each point reaches failure (shear OR tension).

    p_MPa : Cambridge pressure -p'' [MPa], positive = compression
    q_MPa : deviatoric stress [MPa]

    Shear  failure: q > C1(UCS) + C2·p  →  UCS_s = (q - C2·p) / K
    Tension failure: p < -T0(UCS)        →  UCS_t = |p| / K_t   (only for p < 0)
    Combined: max(UCS_s, UCS_t)
    """
    ucs_s = (q_MPa - _C2_env * p_MPa) / _K_env          # shear
    ucs_t = np.where(p_MPa < 0, -p_MPa / _K_t, 0.0)     # tension (only tensile points)
    return np.maximum(ucs_s, ucs_t)


def _compute_required_ucs(p_eff_t_Pa, q_Pa, fraction=0.05):
    """Required UCS (shear only) so that at most `fraction` of points fail.

    Returns max over timesteps of the 95th-percentile of ucs_crit_shear.
    """
    p = -p_eff_t_Pa / 1e6
    q =  q_Pa / 1e6
    pct = (1.0 - fraction) * 100
    ucs_max = -np.inf
    for i in range(q.shape[0]):
        ucs_s = (q[i] - _C2_env * p[i]) / _K_env
        val   = float(np.percentile(ucs_s, pct))
        if val > ucs_max:
            ucs_max = val
    return max(ucs_max, 0.0)


def _compute_required_pt(p_eff_t_Pa, fraction=0.05):
    """Required tensile strength p''_t (≥ 0) so that at most `fraction` fail in tension.

    Sign convention: p'' is compression-negative.
      Tensile failure when p'' > p''_t  (stress exceeds tensile strength).
      p''_t ≥ 0; p''_t = 0 means zero tensile strength.

    On the Cambridge plot (x = -p''), the tension cutoff is at x = -p''_t ≤ 0.

    Worst-case timestep = minimum 5th-percentile of x = most tensile.
    Required p''_t = -min_timestep(5th_pct(x)), capped at 0 from below.
    """
    p = -p_eff_t_Pa / 1e6   # x-axis variable: -p'' [MPa], positive = compression
    pct = fraction * 100     # 5th percentile
    pt_min = +np.inf         # find the most tensile timestep (minimum x-5th-pct)
    for i in range(p.shape[0]):
        val = float(np.percentile(p[i], pct))
        if val < pt_min:
            pt_min = val
    # p''_t = -x_cutoff; cap at 0 (zero tensile strength lower bound)
    return max(-pt_min, 0.0)

# ── Collect run labels ────────────────────────────────────────────────────────
if args.run:
    run_labels = [args.run]
else:
    run_labels = sorted(
        p.parent.parent.name
        for p in RUNS_DIR.glob("*/outputs/invariants.nc")
    )
    if not run_labels:
        raise FileNotFoundError(
            f"No invariants.nc files found under {RUNS_DIR}\n"
            "Run run_all.py first.")

print(f"Plotting {len(run_labels)} run(s): {run_labels}")

# ── Mohr-Coulomb envelope ─────────────────────────────────────────────────────
def _mc_envelope(phi_rad, coh_MPa, p_max_MPa, p_min_MPa=0.0, npoints=300):
    C1 = 6.0 * coh_MPa  * np.cos(phi_rad) / (3.0 - np.sin(phi_rad))
    C2 = 6.0 * np.sin(phi_rad)             / (3.0 - np.sin(phi_rad))
    p_pos = np.linspace(p_min_MPa, p_max_MPa, npoints)
    q_env = C1 + C2 * p_pos
    print(f"MC envelope: φ={np.degrees(phi_rad):.1f}°, c={coh_MPa:.2f} MPa "
          f"→ C1={C1:.3f} MPa, C2={C2:.4f}")
    return p_pos, q_env

# ── Plot one run ──────────────────────────────────────────────────────────────
def _plot_run(run_label):
    nc_path = RUNS_DIR / run_label / "outputs" / "invariants.nc"
    if not nc_path.exists():
        print(f"  Skipping {run_label}: invariants.nc not found")
        return

    ds     = xr.open_dataset(nc_path)
    Re     = float(ds.attrs.get("Re", 0.025))
    H      = float(ds.attrs.get("H",  0.010))   # full specimen height [m]
    time   = ds["time"].values            # [s]
    r      = ds["r"].values               # [m]
    z      = ds["z"].values               # [m]
    p      = -ds["p_eff_t"].values / 1e6  # -p'_T   [MPa], Terzaghi (α=1), positive = compression
    q      =  ds["q"].values     / 1e6    #  q      [MPa]
    p_pore =  ds["p_pore"].values / 1e6   #  p_pore [MPa]
    u_r    =  ds["u_r"].values * 1e6      #  u_r    [µm]

    # Filter to post-load timesteps only (t > 50 s)
    T_LOAD    = 50.0
    post_mask = time > T_LOAD
    time_post = time[post_mask]
    p_post    = p[post_mask]
    q_post    = q[post_mask]

    # Flatten for scatter
    n_t, n_pt = p.shape
    time_mat  = np.repeat(time_post[:, np.newaxis], n_pt, axis=1)
    time_flat = time_mat.ravel() / 60.0   # minutes
    p_flat    = p_post.ravel()
    q_flat    = q_post.ravel()

    # UCS (shear) and p''_t (tension) found independently — post-load only
    p_eff_t_post = ds["p_eff_t"].values[post_mask]
    q_post_Pa    = ds["q"].values[post_mask]
    if args.ucs is not None:
        ucs_MPa = args.ucs
    else:
        ucs_MPa = _compute_required_ucs(p_eff_t_post, q_post_Pa)
    cohesion_MPa = ucs_MPa * (1.0 - np.sin(phi_rad)) / (2.0 * np.cos(phi_rad))
    pt_MPa       = _compute_required_pt(p_eff_t_post)

    # Axis limits: cover all data with 5% margin (both sides)
    p_data_min = float(np.nanmin(p_flat))
    p_data_max = float(np.nanmax(p_flat))
    q_data_max = float(np.nanmax(q_flat)) if q_flat.size > 0 else 30.0
    margin     = 0.05
    p_ax_min   = p_data_min - abs(p_data_min) * margin
    p_ax_max   = p_data_max * (1 + margin)
    q_ax_max   = q_data_max * (1 + margin)

    # Figure
    fig = plt.figure(figsize=(27 / 2.54, 9 / 2.54))
    gs  = gridspec.GridSpec(2, 8,
                            width_ratios=[2.5, 0.07, 0.60, 2, 0.12, 0.35, 2, 0.12],
                            height_ratios=[1, 1],
                            hspace=0.55, wspace=0.10,
                            left=0.07, right=0.95,
                            top=0.80, bottom=0.13)

    ax      = fig.add_subplot(gs[0:2, 0])
    ax_cbar = fig.add_subplot(gs[0:2, 1])
    ax_p    = fig.add_subplot(gs[0, 3])
    ax_q    = fig.add_subplot(gs[1, 3])
    ax_cp   = fig.add_subplot(gs[0, 4])
    ax_cq   = fig.add_subplot(gs[1, 4])
    ax_pp   = fig.add_subplot(gs[0, 6])
    ax_ur   = fig.add_subplot(gs[1, 6])
    ax_cpp  = fig.add_subplot(gs[0, 7])
    ax_cur  = fig.add_subplot(gs[1, 7])

    # P-Q scatter
    vmin  = 0.0
    vmax  = time_flat.max()
    vctr  = (vmin + vmax) / 2.0
    norm  = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vctr, vmax=vmax)
    sc = ax.scatter(p_flat, q_flat,
                    c=time_flat, cmap="jet", norm=norm,
                    alpha=0.12, s=8, edgecolors="none", rasterized=True)

    # x-axis position of tension cutoff: x = -p''_t ≤ 0
    x_cutoff = -pt_MPa   # ≤ 0

    # Ensure x-axis covers tension cutoff with at least 1 MPa margin
    p_ax_min = min(p_ax_min, x_cutoff - 1.0)

    # MC envelope starts at the tension cutoff so the shaded region meets it
    p_max_env = args.p_max if args.p_max else p_ax_max
    p_env, q_env = _mc_envelope(phi_rad, cohesion_MPa, p_max_env,
                                 p_min_MPa=x_cutoff)

    ax.plot(p_env, q_env, color="k", ls="--", lw=0.8,
            label=f"MC: $\\varphi={phi_deg:.0f}^{{\\circ}}$, UCS$={ucs_MPa:.1f}$ MPa")
    # Shade shear failure zone (above MC line, starting from tension cutoff)
    ax.fill_between(p_env, q_env, 1e9, color="k", alpha=0.10)
    # Tension cutoff vertical line
    ax.axvline(0.0, color="k", lw=0.5, zorder=4, solid_capstyle="butt")  # -p''=0 axis
    ax.axhline(0.0, color="k", lw=0.5, zorder=4, solid_capstyle="butt")  # q=0 axis
    ax.axvline(x_cutoff, color="k", ls=":", lw=0.8,
               label=f"$p''_t = {pt_MPa:.2f}$ MPa")
    # Shade tensile failure zone (left of cutoff) — same alpha, meets shear zone
    ax.axvspan(p_ax_min, x_cutoff, color="k", alpha=0.10)

    ax.set_xlim(p_ax_min, p_ax_max)
    ax.set_ylim(0.0,      q_ax_max)
    ax.set_xlabel("$-p'' = -(I_1/3 + p_{\\rm pore})$ (MPa)")
    ax.set_ylabel("$q = \\sqrt{3 J_2}$ (MPa)")
    ax.grid(True, alpha=0.25)
    ax.tick_params(which="both", width=0.5, length=3)
    ax.legend(loc="upper left", framealpha=0.7, handlelength=1.5, fontsize=6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(sc, cax=ax_cbar, drawedges=False)
    cbar.solids.set_alpha(1.0)
    cbar.set_label("$t$ (min)", rotation=270, labelpad=10)
    cbar.outline.set_linewidth(0.3)

    # Spatial snapshot
    snap_idx = int(np.argmin(np.abs(time - args.snap_time)))
    t_snap_m = float(time[snap_idx]) / 60.0

    p_snap  = p[snap_idx, :]
    q_snap  = q[snap_idx, :]
    pp_snap = p_pore[snap_idx, :]
    ur_snap = u_r[snap_idx, :]

    r_unique = np.unique(np.round(r, 12))
    z_unique = np.unique(np.round(z, 12))
    n_r, n_z = len(r_unique), len(z_unique)
    R_mm = r_unique * 1000
    Z_mm = z_unique * 1000
    p_2d  = p_snap.reshape(n_z, n_r)
    q_2d  = q_snap.reshape(n_z, n_r)
    pp_2d = pp_snap.reshape(n_z, n_r)
    ur_2d = ur_snap.reshape(n_z, n_r)

    P_MIN = 1e-4   # 0.1 kPa in MPa — lower limit for pressure colormaps
    pm_p  = ax_p.pcolormesh(R_mm, Z_mm, p_2d,  cmap="coolwarm",
                             vmin=P_MIN, vmax=6.0,
                             shading="gouraud", rasterized=True)
    pm_q  = ax_q.pcolormesh(R_mm, Z_mm, q_2d,  cmap="coolwarm",
                             vmin=0, vmax=10.0,
                             shading="gouraud", rasterized=True)
    pm_pp = ax_pp.pcolormesh(R_mm, Z_mm, pp_2d, cmap="coolwarm",
                              vmin=P_MIN,
                              shading="gouraud", rasterized=True)
    pm_ur = ax_ur.pcolormesh(R_mm, Z_mm, ur_2d, cmap="coolwarm",
                              vmin=ur_2d.min(), vmax=ur_2d.max(),
                              shading="gouraud", rasterized=True)

    for pm, cax in ((pm_p, ax_cp), (pm_q, ax_cq), (pm_pp, ax_cpp), (pm_ur, ax_cur)):
        plt.colorbar(pm, cax=cax, drawedges=False).ax.tick_params()

    for _ax, _title, _show_x, _show_y in (
            (ax_p,  f"$-p'$ (MPa)  [$t={t_snap_m:.1f}$ min]",               False, True),
            (ax_q,  f"$q = \\sqrt{{3J_2}}$ (MPa)  [$t={t_snap_m:.1f}$ min]", True,  True),
            (ax_pp, f"$p_{{\\rm pore}}$ (MPa)  [$t={t_snap_m:.1f}$ min]",    False, False),
            (ax_ur, f"$u_r$ ($\\mu$m)  [$t={t_snap_m:.1f}$ min]",            True,  False)):
        _ax.set_xlabel("$r$ (mm)" if _show_x else "")
        _ax.set_ylabel("$z$ (mm)" if _show_y else "")
        _ax.set_title(_title, pad=6)

    fig.suptitle(
        f"Demo 14 — $p$-$q$ space  |  run: {run_label}  |  "
        f"MC: $\\varphi={phi_deg:.0f}^{{\\circ}}$, UCS$={ucs_MPa:.1f}$ MPa → $c={cohesion_MPa:.1f}$ MPa  |  $-p'' = -(I_1/3 + p_{{\\rm pore}})$\n"
        f"$R_e={Re*1000:.1f}$ mm, $H={H*1000:.1f}$ mm ($H/2={H*500:.1f}$ mm drainage path)  |  "
        f"{n_pt} pts $\\times$ {n_t} steps  |  $t_{{\\rm end}}={vmax:.0f}$ min",
        fontsize=9, y=0.98)

    png_dir = DEMO_DIR / "png"
    png_dir.mkdir(exist_ok=True)
    out = png_dir / f"pq_{run_label}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Main loop (parallel) ──────────────────────────────────────────────────────
import multiprocessing as mp

if __name__ == "__main__":
    n_workers = min(len(run_labels), mp.cpu_count())
    print(f"Using {n_workers} worker(s) for {len(run_labels)} run(s)")
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            pool.map(_plot_run, run_labels)
    else:
        _plot_run(run_labels[0])
