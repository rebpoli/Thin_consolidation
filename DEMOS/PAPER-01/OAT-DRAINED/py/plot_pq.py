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
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paper style ───────────────────────────────────────────────────────────────
MM = 1 / 25.4
plt.rcParams.update({
    "text.usetex":         False,
    "font.size":           6,
    "axes.titlesize":      6,
    "axes.labelsize":      6,
    "axes.labelpad":       2,
    "xtick.labelsize":     5,
    "ytick.labelsize":     5,
    "legend.fontsize":     5,
    "legend.handlelength": 1.0,
    "lines.linewidth":     0.8,
    "axes.linewidth":      0.5,
    "xtick.major.width":   0.5,
    "ytick.major.width":   0.5,
    "xtick.minor.width":   0.4,
    "ytick.minor.width":   0.4,
    "xtick.major.size":    2.5,
    "ytick.major.size":    2.5,
    "xtick.minor.size":    1.5,
    "ytick.minor.size":    1.5,
    "grid.linewidth":      0.4,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})

DEMO_DIR = Path(__file__).resolve().parents[1]   # OAT-DRAINED/
RUNS_DIR = DEMO_DIR / "runs"
MAX_PQ_TIME_MIN = 150.0
MAX_PQ_TIME_MIN_ALPHA = 30.0
MIN_PQ_TIME_MIN = 1e-3

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
parser.add_argument("--steady-rel-tol", type=float, default=5e-3,
                    help="Relative change threshold for steady-stress detection (default: 5e-3)")
parser.add_argument("--steady-window", type=int, default=5,
                    help="Consecutive timesteps required below threshold (default: 5)")
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


def _detect_steady_time(time_s, p_MPa, q_MPa, rel_tol=5e-3, window=5):
    """Find first time where stress evolution becomes small and stays small.

    Uses mean stress trajectories over space and a normalized timestep-change metric.
    Returns the cutoff time (s) to retain in the dataset.
    """
    if len(time_s) <= window + 1:
        return float(time_s[-1])

    p_mean = np.nanmean(p_MPa, axis=1)
    q_mean = np.nanmean(q_MPa, axis=1)

    dp = np.abs(np.diff(p_mean))
    dq = np.abs(np.diff(q_mean))

    p_scale = max(float(np.nanmax(np.abs(p_mean)) - np.nanmin(np.abs(p_mean))), 1e-8)
    q_scale = max(float(np.nanmax(np.abs(q_mean)) - np.nanmin(np.abs(q_mean))), 1e-8)
    metric = np.maximum(dp / p_scale, dq / q_scale)

    ok = metric < rel_tol
    run = np.convolve(ok.astype(int), np.ones(window, dtype=int), mode="valid")
    idx = np.where(run == window)[0]
    if idx.size == 0:
        return float(time_s[-1])

    cutoff_step = int(idx[0] + window)
    cutoff_step = min(cutoff_step, len(time_s) - 1)
    return float(time_s[cutoff_step])


def _nice_integer_ticks(vmin, vmax, target=4):
    """Choose evenly spaced integer ticks inside [vmin, vmax].

    Rule:
    - Start at first integer in range: ceil(vmin)
    - End at last integer in range OR next-last (last-1)
    - Pick an integer step that gives at least 3 ticks, evenly spaced
    """
    first = int(np.ceil(vmin))
    last = int(np.floor(vmax))

    # If the integer span is too small, expand minimally to get 4 ticks.
    if last - first < 3:
        base = int(np.floor(vmin))
        return np.array([base, base + 1, base + 2, base + 3], dtype=int)

    best = None
    for end in (last, last - 1):
        if end - first < 2:
            continue
        diff = end - first

        # Candidate integer steps that divide the span exactly.
        step_candidates = [s for s in range(1, diff + 1) if diff % s == 0]
        for step in step_candidates:
            n_ticks = diff // step + 1
            if n_ticks not in (4, 5):
                continue
            # Prefer around target ticks, then larger end (last first), then larger step.
            score = (abs(n_ticks - target), last - end, -step)
            if best is None or score < best[0]:
                ticks = np.arange(first, end + 1, step, dtype=int)
                best = (score, ticks)

    if best is not None:
        return best[1]

    # Fallback: force 4 evenly spaced integers.
    end = last
    span = max(end - first, 3)
    step = max(1, span // 3)
    return np.array([first, first + step, first + 2 * step, first + 3 * step], dtype=int)

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

    # Use full time history (no data filtering).
    all_mask = np.ones_like(time, dtype=bool)
    time_all = time[all_mask]
    p_all = p[all_mask]
    q_all = q[all_mask]

    # Detect setting-specific steady-stress time and limit plotted/processed dataset.
    t_cutoff = _detect_steady_time(
        time_all, p_all, q_all,
        rel_tol=args.steady_rel_tol,
        window=args.steady_window,
    )
    print(f"  {run_label}: steady-change threshold reached at t={t_cutoff/60:.2f} min ({t_cutoff:.1f} s)")

    # Keep only early-time window for P-Q visualization/analysis.
    max_time_min = MAX_PQ_TIME_MIN_ALPHA if run_label.startswith("alpha_") else MAX_PQ_TIME_MIN
    pq_mask = (time_all >= (MIN_PQ_TIME_MIN * 60.0)) & (time_all <= (max_time_min * 60.0))
    time_post = time_all[pq_mask]
    p_post = p_all[pq_mask]
    q_post = q_all[pq_mask]

    # Flatten for scatter
    n_t, n_pt = p.shape
    time_mat  = np.repeat(time_post[:, np.newaxis], n_pt, axis=1)
    time_flat = time_mat.ravel() / 60.0   # minutes
    p_flat    = p_post.ravel()
    q_flat    = q_post.ravel()
    if not (len(time_flat) == len(p_flat) == len(q_flat)):
        raise RuntimeError("Time-to-color mapping size mismatch in P-Q scatter data")

    # UCS (shear) and p''_t (tension) found independently — post-load only
    p_eff_t_post = ds["p_eff_t"].values[all_mask][pq_mask]
    q_post_Pa    = ds["q"].values[all_mask][pq_mask]
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
    fig = plt.figure(figsize=(175 * MM, 56 * MM))
    gs  = gridspec.GridSpec(2, 8,
                            width_ratios=[2.5, 0.07, 0.60, 2, 0.08, 0.35, 2, 0.08],
                            height_ratios=[1, 1],
                            hspace=0.58, wspace=0.12,
                            left=0.05, right=0.97,
                            top=0.92, bottom=0.15)

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
    time_color = np.maximum(time_flat, 1e-6)
    vmin = max(float(np.nanmin(time_color)), 1e-6)
    vmax = float(np.nanmax(time_color))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin * 1.001
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    sc = ax.scatter(p_flat, q_flat,
                    c=time_color, cmap="jet", norm=norm,
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

    # Add line for UCS=0, φ=30°
    _C2_ucs0 = 6.0 * np.sin(phi_rad) / (3.0 - np.sin(phi_rad))
    p_ucs0 = np.array([0.0, p_ax_max])
    q_ucs0 = _C2_ucs0 * p_ucs0
    ax.plot(p_ucs0, q_ucs0, color="gray", ls=":", lw=0.8,
            label=f"UCS=0, $\\varphi={phi_deg:.0f}^{{\\circ}}$")

    # Shade shear failure zone (above MC line, starting from tension cutoff)
    ax.fill_between(p_env, q_env, 1e9, color="k", alpha=0.10)
    # Tension cutoff vertical line
    ax.axvline(0.0, color="k", lw=0.5, zorder=4, solid_capstyle="butt")  # -p''=0 axis
    ax.axhline(0.0, color="k", lw=0.5, zorder=4, solid_capstyle="butt")  # q=0 axis
    # p''_t cutoff line intentionally omitted (keep shaded tensile zone only)
    # Shade tensile failure zone (left of cutoff) — same alpha, meets shear zone
    ax.axvspan(p_ax_min, x_cutoff, color="k", alpha=0.10)

    ax.set_xlim(-2.0, 13.0)
    ax.set_ylim(0.0, 15.0)
    ax.set_xlabel("$-p'' = -(I_1/3 + p_{\\rm pore})$ (MPa)")
    ax.set_ylabel("$q = \\sqrt{3 J_2}$ (MPa)")
    ax.set_title("Stress path evolution", pad=4)
    ax.grid(True, alpha=0.25)
    ax.tick_params(which="both", width=0.5, length=3)
    leg = ax.legend(loc="upper left", framealpha=1.0, handlelength=1.0,
                    fancybox=False, edgecolor="black", facecolor="white")
    leg.get_frame().set_linewidth(0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(sc, cax=ax_cbar, drawedges=False)
    cbar.solids.set_alpha(1.0)
    cbar.ax.set_title("$t$ (min)", pad=2)
    cbar.ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1]))
    cbar.ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    cbar.ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    def _cbar_fmt(x, _):
        if x > 0 and x < 0.1:
            exp = np.log10(x)
            if abs(exp - round(exp)) < 1e-8:
                return f"$10^{{{int(round(exp))}}}$"
        return f"{x:g}"
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(_cbar_fmt))
    cbar.outline.set_linewidth(0.3)

    # Spatial snapshot
    snap_idx = int(np.argmin(np.abs(time - args.snap_time)))
    t_snap_m = float(time[snap_idx]) / 60.0
    t_snap_m_rounded = int(round(t_snap_m))

    p_snap  = p[snap_idx, :]
    q_snap  = q[snap_idx, :]
    pp_snap = p_pore[snap_idx, :]
    ur_snap = u_r[snap_idx, :]

    r_unique = np.unique(np.round(r, 12))
    z_unique = np.unique(np.round(z, 12))
    n_r, n_z = len(r_unique), len(z_unique)
    R_mm = r_unique * 1000
    Z_mm = z_unique * 1000
    # Mohr-Coulomb coefficients (depend on UCS and φ)
    _C1_mob = 6.0 * cohesion_MPa * np.cos(phi_rad) / (3.0 - np.sin(phi_rad))
    _C2_mob = 6.0 * np.sin(phi_rad) / (3.0 - np.sin(phi_rad))
    p_2d  = p_snap.reshape(n_z, n_r)
    # Mobilization ratio q / (C1 + C2·|p|): take MAX across all timesteps per spatial point
    mob_ratio_all_t = q / np.maximum(_C1_mob + _C2_mob * np.abs(p), 1e-6)  # (n_t, n_pt)
    mob_ratio_max = np.nanmax(mob_ratio_all_t, axis=0)                      # (n_pt,)
    q_2d  = mob_ratio_max.reshape(n_z, n_r)
    pp_2d = pp_snap.reshape(n_z, n_r)
    ur_2d = ur_snap.reshape(n_z, n_r)

    # Create deformed mesh coordinates (0.3:1 scaling for u_r only)
    R_grid, Z_grid = np.meshgrid(R_mm, Z_mm)
    DEFORM_SCALE = 0.3
    R_deformed = R_grid + ur_2d * DEFORM_SCALE
    Z_deformed = Z_grid

    P_MIN = 1e-4   # 0.1 kPa in MPa — lower limit for pressure colormaps
    pm_p  = ax_p.pcolormesh(R_deformed, Z_deformed, p_2d,  cmap="coolwarm",
                             vmin=P_MIN, vmax=6.0,
                             shading="gouraud", rasterized=True)
    pm_q  = ax_q.pcolormesh(R_deformed, Z_deformed, q_2d,  cmap="coolwarm",
                             vmin=0.5, vmax=1.5,
                             shading="gouraud", rasterized=True)

    # Mark points above MC envelope across ALL timesteps
    C1_calc = 6.0 * cohesion_MPa * np.cos(phi_rad) / (3.0 - np.sin(phi_rad))
    C2_calc = 6.0 * np.sin(phi_rad) / (3.0 - np.sin(phi_rad))

    # Evaluate failure at all timesteps using mobilization ratio > 1
    ever_failed = np.zeros(p.shape[1], dtype=bool)  # one flag per spatial point
    for t_idx in range(p.shape[0]):
        p_t = p[t_idx, :]
        q_t = q[t_idx, :]
        p_abs_t = np.abs(p_t)
        mob_ratio = q_t / np.maximum(C1_calc + C2_calc * p_abs_t, 1e-6)
        failed_t = mob_ratio > 1.0
        ever_failed |= failed_t

    # Reshape to 2D and mark on current snapshot
    ever_failed_2d = ever_failed.reshape(n_z, n_r)
    n_failure = np.sum(ever_failed)
    print(f"\n  Failure region (q/p'' > MC criterion) across ALL timesteps:")
    print(f"  Points that ever failed: {n_failure} / {ever_failed.size} ({100*n_failure/ever_failed.size:.1f}%)")

    # Check distribution of failures by z level
    failure_by_z = np.sum(ever_failed_2d, axis=1)
    print(f"  Failures by z level: {failure_by_z}")
    print(f"  Z values: {Z_mm}")
    print(f"  Max failures at z index: {np.argmax(failure_by_z)}, z = {Z_mm[np.argmax(failure_by_z)]:.4f} mm")

    # Contour of failure region boundary (dashed, thin) on deformed mesh.
    # Use level=1e-6 (nearly at safe=0 side) so contour fully encloses all failed points.
    if n_failure > 0:
        ax_q.contour(R_deformed, Z_deformed, ever_failed_2d.astype(float), levels=[1e-6],
                    colors="black", linestyles="--", linewidths=0.6, zorder=15)

    # Add contour line for calculated UCS (Mohr-Coulomb envelope: q = C1 + C2*p)
    C1_calc = 6.0 * cohesion_MPa * np.cos(phi_rad) / (3.0 - np.sin(phi_rad))
    C2_calc = 6.0 * np.sin(phi_rad) / (3.0 - np.sin(phi_rad))
    p_abs_2d = np.abs(p_2d)
    # MC criterion in q/p'' form: q/p'' = C1/p + C2
    q_over_p_mc_criterion = C1_calc / np.maximum(p_abs_2d, 1e-6) + C2_calc
    # Contour where actual q/p'' equals the MC criterion
    criterion_diff = q_2d - q_over_p_mc_criterion
    if np.nanmin(criterion_diff) <= 0 <= np.nanmax(criterion_diff):
        ax_q.contour(R_deformed, Z_deformed, criterion_diff, levels=[0],
                     colors="k", linestyles="--", linewidths=1.5, zorder=10)

    # Print debug table
    print(f"\n  Parameters: UCS={ucs_MPa:.2f} MPa, φ={phi_deg:.1f}°")
    print(f"  C1_calc={C1_calc:.3f} MPa, C2_calc={C2_calc:.3f}")
    print(f"\n  Table: q (MPa), p'' (MPa), q/p''")
    print(f"  Min:  {q_snap.min():.3f}, {p_snap.min():.3f}, {(q_snap/np.maximum(np.abs(p_snap), 1e-6)).min():.3f}")
    print(f"  Max:  {q_snap.max():.3f}, {p_snap.max():.3f}, {(q_snap/np.maximum(np.abs(p_snap), 1e-6)).max():.3f}")
    print(f"  Mean: {q_snap.mean():.3f}, {p_snap.mean():.3f}, {(q_snap/np.maximum(np.abs(p_snap), 1e-6)).mean():.3f}")
    print(f"  MC criterion q/p'' range: {q_over_p_mc_criterion.min():.3f} to {q_over_p_mc_criterion.max():.3f}")
    pm_pp = ax_pp.pcolormesh(R_deformed, Z_deformed, pp_2d, cmap="coolwarm",
                              vmin=P_MIN,
                              shading="gouraud", rasterized=True)
    pm_ur = ax_ur.pcolormesh(R_deformed, Z_deformed, ur_2d, cmap="coolwarm",
                              vmin=ur_2d.min(), vmax=ur_2d.max(),
                              shading="gouraud", rasterized=True)

    for pm, cax in ((pm_p, ax_cp), (pm_q, ax_cq), (pm_pp, ax_cpp), (pm_ur, ax_cur)):
        cb = plt.colorbar(pm, cax=cax, drawedges=False)
        cb.outline.set_linewidth(0.3)
        v0, v1 = pm.get_clim()
        if pm is pm_q:
            # Mobilization ratio: dimensionless, often fractional — use auto ticks/format
            cb.ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
            cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        else:
            if np.isfinite(v0) and np.isfinite(v1) and v1 > v0:
                cb.set_ticks(_nice_integer_ticks(v0, v1, target=4))
            cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        cb.ax.tick_params(width=0.3, length=2.0)
        for spine in cax.spines.values():
            spine.set_linewidth(0.3)

    for _ax, _title, _show_x, _show_y in (
            (ax_p,  f"$-p''$ (MPa)  [$t={t_snap_m_rounded}$ min]",            False, True),
            (ax_q,  f"$\\max_t \\, q / (C_1 + C_2 \\cdot |p''|)$", True,  True),
            (ax_pp, f"$p_{{\\rm pore}}$ (MPa)  [$t={t_snap_m_rounded}$ min]",  False, False),
            (ax_ur, f"$u_r$ ($\\mu$m)  [$t={t_snap_m_rounded}$ min]",          True,  False)):
        _ax.set_xlabel("$r$ (mm)" if _show_x else "")
        _ax.set_ylabel("$z$ (mm)" if _show_y else "")
        _ax.set_title(_title, pad=5)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        for spine in ["bottom", "left"]:
            _ax.spines[spine].set_linewidth(0.3)

    png_dir = DEMO_DIR / "png"
    png_dir.mkdir(exist_ok=True)
    out = png_dir / f"pq_{run_label}.png"
    plt.savefig(out, dpi=500)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Main loop (parallel) ──────────────────────────────────────────────────────
import multiprocessing as mp

if __name__ == "__main__":
    n_workers = min(len(run_labels), 6)
    print(f"Using {n_workers} worker(s) for {len(run_labels)} run(s)")
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            pool.map(_plot_run, run_labels)
    else:
        _plot_run(run_labels[0])

    png_dir = DEMO_DIR / "png"
    md = png_dir / "pq_figures.md"
    lines = [
        "# OAT-DRAINED P-Q figures",
        "",
        "Each figure shows P-Q scatter with Mohr-Coulomb envelope and tensile/shear shaded failure regions,",
        "plus spatial snapshots of $-p''$, $q$, pore pressure, and radial displacement.",
        "Each run is auto-trimmed at its detected steady-stress cutoff time.",
        "",
    ]
    for run_label in run_labels:
        lines.append(f"## pq_{run_label}.png")
        lines.append(f"Run: `{run_label}`.")
        lines.append("")
    md.write_text("\n".join(lines))
    print(f"Saved {md}")
