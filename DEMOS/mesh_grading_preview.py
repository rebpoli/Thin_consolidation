#!/usr/bin/env python3
"""Preview geometric graded mesh for different N values.

X: cumulative element index (from fine end)
Y: cumulative distance from fine end

Grading rules:
  h_min  = L / (N * 10)
  n_ref  = N // 2   elements in refined zone  (0 … L/4)
  n_crs  = N - n_ref elements in coarse zone  (L/4 … L)
  r      = geometric ratio, solved so sum of n_ref elements = L/4
  h_max  = h_min * r^(n_ref-1)
  h_crs  = (3L/4) / n_crs   (exact uniform tiling)
"""
import numpy as np
import matplotlib.pyplot as plt

# ── Parameters to sweep ───────────────────────────────────────────────────────
L = 0.025          # domain length [m]  (e.g. Re or H_mesh)
N_VALUES = [15, 20, 30, 50, 80]
COLORS   = plt.cm.viridis(np.linspace(0.15, 0.90, len(N_VALUES)))

# ── Core solver ───────────────────────────────────────────────────────────────

def geometric_ratio(h_min, n_ref, L_ref, tol=1e-12):
    """Bisect for r such that h_min*(r^n_ref - 1)/(r-1) = L_ref."""
    # r=1 limit: sum = n_ref * h_min
    if n_ref * h_min >= L_ref:
        raise ValueError(f"n_ref*h_min={n_ref*h_min:.3e} >= L_ref={L_ref:.3e}; "
                         "reduce h_min or increase L_ref")
    lo, hi = 1.0 + 1e-10, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2.0
        s   = h_min * (mid**n_ref - 1.0) / (mid - 1.0)
        if s < L_ref:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


def make_mesh(L, N):
    n_ref  = N // 2
    n_crs  = N - n_ref
    h_min  = L / (N * 10)
    L_ref  = L / 6.0
    L_crs  = L - L_ref

    r      = geometric_ratio(h_min, n_ref, L_ref)
    h_max  = h_min * r ** (n_ref - 1)
    h_crs  = L_crs / n_crs

    # Build element sizes from fine end
    sizes_ref = [h_min * r**i for i in range(n_ref)]
    sizes_crs = [h_crs] * n_crs
    sizes     = sizes_ref + sizes_crs

    cumulative_dist = np.concatenate([[0.0], np.cumsum(sizes)])
    cumulative_idx  = np.arange(len(cumulative_dist))

    return cumulative_idx, cumulative_dist, r, h_max, h_crs


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for N, color in zip(N_VALUES, COLORS):
    idx, dist, r, h_max, h_crs = make_mesh(L, N)
    n_ref = N // 2

    ax.plot(idx, dist * 1e3, color=color, lw=1.8,
            label=f"N={N}  r={r:.3f}  h_min={L/(N*10)*1e6:.1f} µm  "
                  f"h_max={h_max*1e3:.3f} mm")

    # Mark transition refined → coarse
    ax.axvline(n_ref, color=color, lw=0.6, ls=":")

# Reference lines
ax.axhline(L / 6 * 1e3, color="gray", lw=0.8, ls="--", label="L/6 (zone boundary)")
ax.axhline(L       * 1e3, color="k",    lw=0.8, ls="-",  alpha=0.4)

ax.set_xlabel("Cumulative element count (from fine end)")
ax.set_ylabel("Cumulative distance from fine end  [mm]")
ax.set_title(f"Geometric graded mesh  |  L = {L*1e3:.1f} mm\n"
             f"h_min = L/(N·10),  n_ref = N/2 in [0, L/6],  "
             f"n_crs = N/2 uniform in [L/6, L]")
ax.legend(fontsize=7, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mesh_grading_preview.png", dpi=200)
print("Saved mesh_grading_preview.png")
plt.show()
