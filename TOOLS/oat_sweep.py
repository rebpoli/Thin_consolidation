#!/usr/bin/env python3
"""
OAT (One-At-a-Time) sweep of poroelastic derived parameters.

Fixed:
  E   = 5 GPa
  NU  = 0.35
  Kf  = 2.20 GPa  (constant)

Swept independently:
  PHI   ∈ (0.01, 0.30)   — alpha held at nominal
  ALPHA ∈ (0.40, 1.00)   — phi held at nominal

Derived quantities plotted:
  G, K, Ks, M, B   (same definitions as param_distributions.py)

Validity limits drawn as horizontal reference lines:
  Ks = 10 GPa  (lower bound)
  Ks = 100 GPa (upper bound)
  B  = 1       (upper bound)
  K  = 50 GPa  (upper bound)

Usage
-----
  ./oat_sweep.py [--out OUT]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
E_FIXED  = 5.00e9
NU_FIXED = 0.40
KF       = 2.20e9

# Nominals (held constant during the other parameter's sweep)
PHI_NOM   = 0.10
ALPHA_NOM = 0.75

# Sweep ranges
PHI_RANGE   = (0.05, 0.30)
ALPHA_RANGE = (0.50, 0.90)
E_RANGE     = (0.1e9, 10e9)

N_PTS = 400

# Validity limits
LIM = dict(Ks_lo=10e9, Ks_hi=100e9, B_hi=1.0, K_hi=50e9)

# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def derive(phi, alpha, E=E_FIXED, nu=NU_FIXED, Kf=KF):
    """
    All inputs may be scalars or arrays.
    Returns dict of derived arrays.

    G  = E / (2*(1+nu))
    K  = E / (3*(1-2*nu))
    Ks = K / (1 - alpha)          [Biot-Willis: alpha = 1 - K/Ks]
    M  = Kf / phi                 [Biot modulus, incompressible grains]
    B  = alpha*M / (K + alpha²*M) [Skempton coefficient]
    """
    phi   = np.asarray(phi,   dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    E     = np.asarray(E,     dtype=float)
    shape = np.broadcast(phi, alpha, E).shape
    G  = np.broadcast_to(E / (2.0 * (1.0 + nu)),         shape).copy()
    K  = np.broadcast_to(E / (3.0 * (1.0 - 2.0 * nu)),   shape).copy()
    Ks = K / np.clip(1.0 - alpha, 1e-12, None)
    M  = Kf / np.clip(phi, 1e-12, None)
    B  = (alpha * M) / (K + alpha**2 * M)
    return dict(G=G, K=K, Ks=Ks, M=M, B=B)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", default="oat_sweep.png")
    args = parser.parse_args()

    phi_sweep   = np.linspace(*PHI_RANGE,   N_PTS)
    alpha_sweep = np.linspace(*ALPHA_RANGE, N_PTS)
    e_sweep     = np.linspace(*E_RANGE,     N_PTS)

    d_phi   = derive(phi_sweep,                          np.full_like(phi_sweep,   ALPHA_NOM))
    d_alpha = derive(np.full_like(alpha_sweep, PHI_NOM), alpha_sweep)
    d_e     = derive(np.full(N_PTS, PHI_NOM),            np.full(N_PTS, ALPHA_NOM), E=e_sweep)

    # Nominal derived values (single point)
    d_nom    = derive(PHI_NOM, ALPHA_NOM, E=E_FIXED)

    # Quantities to plot: (key, label, unit, y-limits-for-reference-lines)
    quantities = [
        ("K",  "K",  "GPa", 1e9, [LIM["K_hi"]/1e9]),
        ("Ks", "Ks", "GPa", 1e9, [LIM["Ks_lo"]/1e9, LIM["Ks_hi"]/1e9]),
        ("B",  "B",  "—",   1.0, [LIM["B_hi"]]),
    ]

    sweeps = [
        (phi_sweep,   d_phi,   "PHI",   "—",   PHI_NOM,      PHI_RANGE),
        (alpha_sweep, d_alpha, "ALPHA", "—",   ALPHA_NOM,    ALPHA_RANGE),
        (e_sweep/1e9, d_e,     "E",     "GPa", E_FIXED/1e9,  (E_RANGE[0]/1e9, E_RANGE[1]/1e9)),
    ]

    C_LINE   = ["#4C72B0", "#DD8452", "#55A868"]  # blue, orange, green
    C_LIMIT  = "crimson"
    C_NOM_X  = "dimgray"

    fig, axes = plt.subplots(3, 3, figsize=(10, 8.5), sharex=False)
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.55, wspace=0.40)
    fig.suptitle(
        f"OAT Sweep  —  NU = {NU_FIXED},  Kf = {KF/1e9:.2f} GPa  |  "
        f"Reference: PHI = {PHI_NOM},  ALPHA = {ALPHA_NOM},  E = {E_FIXED/1e9:.0f} GPa\n"
        f"Row 1: PHI sweep    Row 2: ALPHA sweep    Row 3: E sweep",
        fontsize=10, fontweight="bold"
    )

    for row, (x, d, xlabel, xunit, x_nom, x_range) in enumerate(sweeps):
        color = C_LINE[row]
        for col, (key, ylabel, yunit, yscale, hlims) in enumerate(quantities):
            ax = axes[row, col]
            y  = d[key] / yscale

            ax.plot(x, y, color=color, lw=2)

            # Shade invalid region(s) in light red
            for hlim in hlims:
                if key == "Ks" and hlim == LIM["Ks_lo"]/1e9:
                    ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] < hlim else 0,
                               hlim, alpha=0.08, color=C_LIMIT, zorder=0)
                ax.axhline(hlim, color=C_LIMIT, lw=1.2, ls="--", zorder=3)

            # Nominal x marker
            ax.axvline(x_nom, color=C_NOM_X, lw=1.2, ls=":", zorder=3)

            # Nominal y value
            y_nom = d_nom[key] / yscale
            ax.scatter([x_nom], [y_nom], color=C_NOM_X, s=40, zorder=5)

            ax.set_xlim(x_range)
            ax.set_xlabel(f"{xlabel} [{xunit}]" if xunit != "—" else xlabel, fontsize=8.5)
            ax.set_ylabel(f"{ylabel} [{yunit}]" if yunit != "—" else ylabel, fontsize=8.5)
            ax.tick_params(labelsize=8)

            # Column title only on top row
            if row == 0:
                ax.set_title(ylabel, fontsize=10, fontweight="bold", pad=3)

    # Row labels on left
    for row, (_, _, xlabel, _, _, _) in enumerate(sweeps):
        axes[row, 0].annotate(
            f"sweep: {xlabel}", xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=8.5, rotation=90, va="center", ha="center",
            color=C_LINE[row], fontweight="bold"
        )

    # Footer
    fig.text(
        0.5, 0.01,
        f"Dashed red = validity limits (Ks∈[10,100] GPa, B<1, K<50 GPa).  "
        f"Dotted gray = nominal (PHI={PHI_NOM}, ALPHA={ALPHA_NOM}).  "
        f"Dot = nominal derived value.",
        ha="center", fontsize=7.5, color="dimgray", style="italic"
    )

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
