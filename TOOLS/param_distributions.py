#!/usr/bin/env python3
"""
Poroelastic Parameter Distribution Plotter
===========================================

Generates Monte Carlo samples of primary poroelastic input parameters using
truncated normal (or log-normal) distributions, propagates them through
Biot poroelastic theory to derive secondary parameters, and plots histograms
for both input and quality-metric quantities.

FORMULAS
--------

Primary inputs
  PHI   : porosity                          [—]
  Kf    : fluid bulk modulus                [Pa]
  ALPHA : Biot–Willis coefficient           [—]
  E     : drained Young's modulus           [Pa]
  NU    : drained Poisson's ratio           [—]

Elastic moduli (drained)
  G  = E / (2*(1 + NU))                                    shear modulus
  K  = E / (3*(1 − 2*NU))                                  drained bulk modulus

Grain modulus — from Biot–Willis definition α = 1 − K/Ks:
  Ks = K / (1 − ALPHA)

Biot modulus — simplified form (incompressible solid grains, exact when α→1):
  M  = Kf / PHI
  (general form: 1/M = PHI/Kf + (ALPHA − PHI)/Ks)

Skempton's B coefficient — pressure induced by undrained isotropic loading:
  B  = ALPHA * M / (K + ALPHA² * M)

Undrained Poisson's ratio:
  NU_U = (3*NU + ALPHA*B*(1 − 2*NU)) / (3 − ALPHA*B*(1 − 2*NU))

Loading coupling coefficient:
  ETA = ALPHA*(1 − 2*NU) / (2*(1 − NU))

Storage coefficient at constant stress (compressibility):
  S = PHI/Kf + (ALPHA − PHI)/Ks

Initial pore pressure response (Skempton, isotropic→uniaxial correction):
  P0 = B * (1 + NU_U) / (3*(1 − NU_U)) * SIGMA

SAMPLING STRATEGY
-----------------
  PHI   : truncated normal,  range (0.10, 0.40),  central 0.20
  Kf    : constant                                central 2.20e9 Pa
  ALPHA : truncated normal,  range (0.50, 1.00),  central 0.70
  E     : truncated log-normal, range (1e9, 1e10), central 5e9 Pa
  NU    : truncated normal,  range (0.20, 0.49),  central 0.35

σ for truncated-normal parameters = (max − min) / 6  (3-sigma rule).

Usage
-----
  ./param_distributions.py [--n N] [--sigma SIGMA] [--out OUT]

  --n      number of Monte Carlo samples (default 20000)
  --sigma  range-to-sigma multiplier denominator (default 6)
  --out    output PNG path (default param_distributions.png)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ---------------------------------------------------------------------------
# Central (nominal) values
# ---------------------------------------------------------------------------
NOMINAL = dict(
    PHI=0.06,
    Kf=2.20e9,
    ALPHA=0.65,
    E=25.00e9,
    NU=0.25,
    SIGMA=1.00e7,
)

# Parameter ranges (min, max)
RANGES = dict(
    PHI=(0.02, 0.30),
    ALPHA=(0.40, 1.00),
    E=(1.00e9, 60.00e9),
    NU=(0.20, 0.45),
)

E_SOFT_THRESHOLD = 10e9   # E < this → "soft" regime, highlighted in histograms

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_truncated_normal(center, lo, hi, sigma, n):
    """Truncated normal distribution clipped to [lo, hi]."""
    a = (lo - center) / sigma
    b = (hi - center) / sigma
    return stats.truncnorm.rvs(a, b, loc=center, scale=sigma, size=n)


def sample_truncated_lognormal(center, lo, hi, sigma_log, n):
    """
    Truncated log-normal: normal in log-space, clipped to [lo, hi].
    sigma_log is the std of ln(E).
    """
    mu_log = np.log(center)
    a = (np.log(lo) - mu_log) / sigma_log
    b = (np.log(hi) - mu_log) / sigma_log
    log_samples = stats.truncnorm.rvs(a, b, loc=mu_log, scale=sigma_log, size=n)
    return np.exp(log_samples)


# ---------------------------------------------------------------------------
# Poroelastic derived quantities
# ---------------------------------------------------------------------------

def derive(PHI, Kf, ALPHA, E, NU):
    """
    Compute derived poroelastic parameters from primary inputs.
    Returns a dict of arrays.
    """
    G  = E / (2.0 * (1.0 + NU))
    K  = E / (3.0 * (1.0 - 2.0 * NU))
    Ks = K / np.clip(1.0 - ALPHA, 1e-9, None)       # Biot-Willis: α = 1 - K/Ks
    M  = Kf / PHI                                     # simplified (incompressible grains)
    B  = (ALPHA * M) / (K + ALPHA**2 * M)            # Skempton coefficient
    B  = np.clip(B, 0.0, 1.0)
    nu_u = ((3.0*NU + ALPHA*B*(1.0 - 2.0*NU))
            / (3.0 - ALPHA*B*(1.0 - 2.0*NU)))
    ETA = ALPHA * (1.0 - 2.0*NU) / (2.0 * (1.0 - NU))
    S   = PHI / Kf + (ALPHA - PHI) / np.clip(Ks, 1e-9, None)

    return dict(G=G, K=K, Ks=Ks, M=M, B=B, nu_u=nu_u, ETA=ETA, S=S)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--n",     type=int,   default=20_000, help="MC samples")
    parser.add_argument("--sigma", type=float, default=6.0,
                        help="range/sigma divisor (default 6 → 3-sigma = half-range)")
    parser.add_argument("--out",   default="param_distributions.png")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    N   = args.n
    sd  = args.sigma

    # --- Sample primary parameters (uniform before validity filters) ---
    rng2  = np.random.default_rng(42)
    PHI   = rng2.uniform(*RANGES["PHI"],   size=N)
    ALPHA = rng2.uniform(*RANGES["ALPHA"], size=N)
    NU    = rng2.uniform(*RANGES["NU"],    size=N)

    lo_E, hi_E = RANGES["E"]
    E     = rng2.uniform(lo_E, hi_E, size=N)

    Kf    = np.full(N, NOMINAL["Kf"])   # constant

    # --- Derive secondary parameters ---
    d = derive(PHI, Kf, ALPHA, E, NU)
    G, K, Ks, M, B, nu_u, ETA, S = (d[k] for k in ("G","K","Ks","M","B","nu_u","ETA","S"))


    # --- Discard unphysical / out-of-range samples ---
    # ALPHA bounds: PHI <= ALPHA <= 1 (PHI lower bound + physical upper bound)
    mask  = (B < 1.0) & (Ks <= 100e9) & (K <= 50e9) & (PHI <= ALPHA) & (ALPHA <= 1.0) \
          & (nu_u < 0.5) & (Ks >= 10e9)
    n_discarded = int((~mask).sum())
    print(f"Discarded {n_discarded}/{N} samples ({100*n_discarded/N:.1f}%):")
    print(f"  B >= 1        : {int((B >= 1.0).sum())}")
    print(f"  Ks > 100 GPa  : {int((Ks > 100e9).sum())}")
    print(f"  Ks < 10  GPa  : {int((Ks < 10e9).sum())}")
    print(f"  K  > 50  GPa  : {int((K  > 50e9).sum())}")
    print(f"  ALPHA < PHI   : {int((ALPHA < PHI).sum())}")
    print(f"  ALPHA > 1     : {int((ALPHA > 1.0).sum())}")
    print(f"  nu_u >= 0.5   : {int((nu_u >= 0.5).sum())}")
    # Save pre-filter data for background distribution in plots
    pre = dict(PHI=PHI, ALPHA=ALPHA, E=E, NU=NU,
               **{k: d[k] for k in ("G", "K", "Ks", "M", "B")})

    PHI, Kf, ALPHA, E, NU = PHI[mask], Kf[mask], ALPHA[mask], E[mask], NU[mask]
    G, K, Ks, M, B, nu_u, ETA, S = (d[k][mask] for k in ("G","K","Ks","M","B","nu_u","ETA","S"))

    # Soft subset: E < E_SOFT_THRESHOLD (highlighted differently in histograms)
    soft = E < E_SOFT_THRESHOLD

    # --- Nominal check (print to console) ---
    d0 = derive(
        np.array([NOMINAL["PHI"]]),
        np.array([NOMINAL["Kf"]]),
        np.array([NOMINAL["ALPHA"]]),
        np.array([NOMINAL["E"]]),
        np.array([NOMINAL["NU"]]),
    )
    P0 = d0["B"] * (1 + d0["nu_u"]) / (3*(1 - d0["nu_u"])) * NOMINAL["SIGMA"]
    print("=== Nominal values ===")
    for k, v in d0.items():
        print(f"  {k:6s} = {v[0]:.4g}")
    print(f"  {'P0':6s} = {P0[0]:.4g}   (SIGMA = {NOMINAL['SIGMA']:.2g})")

    # -------------------------------------------------------------------
    # Plot layout: 3 rows
    #   Row 0 (5 cols): free inputs   PHI | ALPHA | E | NU | Kf
    #   Row 1 (5 cols): quality/deriv Ks  | B     | M | G  | K
    #   Row 2 (1 col, full-width): summary statistics table
    # -------------------------------------------------------------------
    n_kept  = int(mask.sum())
    n_soft  = int(soft.sum())
    n_stiff = n_kept - n_soft

    from matplotlib.ticker import LogLocator

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    fig.subplots_adjust(top=0.91, bottom=0.13, hspace=0.42, wspace=0.38)
    fig.suptitle(
        f"Poroelastic Parameter Distributions  "
        f"(n = {n_kept:,} kept, {n_discarded:,} discarded: "
        f"B≥1 | Ks∉[10,100] GPa | K>50 GPa | α∉[φ,1] | νᵤ≥0.5)",
        fontsize=10, fontweight="bold"
    )

    C_INPUT  = "#4C72B0"
    C_METRIC = "#DD8452"
    C_DERIV  = "#55A868"
    C_SOFT   = "#E84545"   # E < 10 GPa portion of each bar
    C_NOM    = "crimson"
    BINS     = 60

    def make_bins(data, log_x):
        if log_x:
            return np.logspace(np.log10(data.min()), np.log10(data.max()), BINS + 1)
        else:
            return np.linspace(data.min(), data.max(), BINS + 1)

    def hist_panel(ax, data, data_pre, nominal, label, unit, color, xrange=None, log_x=False, show_pre=True):
        """
        Background (gray): pre-filter distribution.
        Foreground: kept samples (main color) with E<10 GPa subset (red) overlaid.
        All three share the same density normalisation (pre-filter total).
        """
        bins  = make_bins(data_pre, log_x)
        widths = np.diff(bins)
        total_pre = len(data_pre)

        counts_pre,  _ = np.histogram(data_pre,   bins=bins)
        counts_all,  _ = np.histogram(data,        bins=bins)
        counts_soft, _ = np.histogram(data[soft],  bins=bins)

        dens_pre  = counts_pre  / (total_pre * widths)
        dens_all  = counts_all  / (total_pre * widths)
        dens_soft = counts_soft / (total_pre * widths)

        if show_pre:
            ax.bar(bins[:-1], dens_pre, width=widths, align="edge",
                   color="gray", alpha=0.25, label=f"pre-filter (n={len(data_pre):,})")
        ax.bar(bins[:-1], dens_all,  width=widths, align="edge",
               color=color,  alpha=0.70, label=f"kept  (n={n_kept:,})")
        ax.bar(bins[:-1], dens_soft, width=widths, align="edge",
               color=C_SOFT, alpha=0.85, label=f"E<10 GPa  (n={n_soft:,})")

        ax.axvline(nominal, color=C_NOM, lw=1.8, ls="--", label=f"nom = {nominal:.3g}")
        ax.set_xlabel(f"{label} [{unit}]" if unit else label, fontsize=8.5)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=6.5, handlelength=1.2, borderpad=0.3)
        if xrange:
            ax.set_xlim(xrange)
        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.4g}"))
            ax.tick_params(axis="x", which="minor", labelsize=0)
        ax.set_title(label, fontsize=10, fontweight="bold", pad=3)

    # ---- Row 0: free inputs ----
    hist_panel(axes[0, 0], PHI,   pre["PHI"],   NOMINAL["PHI"],   "PHI",   "—",   C_INPUT, RANGES["PHI"])
    hist_panel(axes[0, 1], ALPHA, pre["ALPHA"], NOMINAL["ALPHA"], "ALPHA", "—",   C_INPUT, RANGES["ALPHA"])
    hist_panel(axes[0, 2], E/1e9, pre["E"]/1e9, NOMINAL["E"]/1e9, "E",    "GPa", C_INPUT,
               xrange=(lo_E/1e9, hi_E/1e9))
    hist_panel(axes[0, 3], NU,    pre["NU"],    NOMINAL["NU"],    "NU",    "—",   C_INPUT, RANGES["NU"])

    # Kf constant — single value, no stacking needed; just annotate
    ax_kf = axes[0, 4]
    ax_kf.set_xlim(0, 5)
    ax_kf.set_ylim(0, 1)
    ax_kf.axis("off")
    ax_kf.text(0.5, 0.65, "Kf  (constant)",  ha="center", va="center",
               fontsize=10, fontweight="bold", transform=ax_kf.transAxes)
    ax_kf.text(0.5, 0.45, f"Kf = {NOMINAL['Kf']/1e9:.2f} GPa",
               ha="center", va="center", fontsize=12, transform=ax_kf.transAxes,
               color=C_NOM)
    ax_kf.text(0.5, 0.28, f"all {n_kept:,} samples",
               ha="center", va="center", fontsize=9, transform=ax_kf.transAxes,
               color="dimgray")

    # ---- Row 1: quality metrics + derived ----
    hist_panel(axes[1, 0], Ks/1e9, np.clip(pre["Ks"], 0, 100e9)/1e9, d0["Ks"][0]/1e9, "Ks", "GPa", C_METRIC, xrange=(0, 100), show_pre=False)
    hist_panel(axes[1, 1], B,      pre["B"],       d0["B"][0],      "B (Skempton)", "—", C_METRIC, (0, 1), show_pre=False)
    hist_panel(axes[1, 2], M/1e9,  pre["M"]/1e9,  d0["M"][0]/1e9,  "M",  "GPa", C_DERIV, show_pre=False)
    hist_panel(axes[1, 3], G/1e9,  pre["G"]/1e9,  d0["G"][0]/1e9,  "G",  "GPa", C_DERIV, show_pre=False)
    hist_panel(axes[1, 4], K/1e9,  pre["K"]/1e9,  d0["K"][0]/1e9,  "K",  "GPa", C_DERIV, show_pre=False)

    # ---- Summary statistics table → terminal ----
    stat_params = [
        ("PHI",      PHI,    NOMINAL["PHI"],        "—"),
        ("ALPHA",    ALPHA,  NOMINAL["ALPHA"],       "—"),
        ("E",        E/1e9,  NOMINAL["E"]/1e9,       "GPa"),
        ("NU",       NU,     NOMINAL["NU"],          "—"),
        ("Ks  *",    Ks/1e9, d0["Ks"][0]/1e9,        "GPa"),
        ("B   *",    B,      d0["B"][0],              "—"),
        ("M",        M/1e9,  d0["M"][0]/1e9,          "GPa"),
        ("G",        G/1e9,  d0["G"][0]/1e9,          "GPa"),
        ("K",        K/1e9,  d0["K"][0]/1e9,          "GPa"),
    ]
    hdr = f"{'Param':<8} {'Unit':<5} {'Nominal':>9} {'Mean':>9} {'Std':>9} "  \
          f"{'P5':>9} {'P25':>9} {'P50':>9} {'P75':>9} {'P95':>9}"
    sep = "─" * len(hdr)
    print(f"\n=== Summary statistics (n={n_kept:,} kept samples) ===")
    print(f"    * quality metrics    E<10 GPa: {n_soft:,} samples ({100*n_soft/n_kept:.1f}%)")
    print(sep)
    print(hdr)
    print(sep)
    for name, arr, nom, unit in stat_params:
        p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
        fmt = ".4f" if arr.max() < 2 else ".3f"
        print(f"{name:<8} {unit:<5} {nom:>9{fmt}} {arr.mean():>9{fmt}} {arr.std():>9{fmt}} "
              f"{p5:>9{fmt}} {p25:>9{fmt}} {p50:>9{fmt}} {p75:>9{fmt}} {p95:>9{fmt}}")
    print(sep)

    # ---- Footer ----
    range_txt = (
        "Input ranges:  PHI∈(0.02, 0.30)  |  ALPHA∈(0.40, 1.00)  |  "
        "Kf = 2.20 GPa (const)  |  E∈(1–60 GPa, uniform)  |  NU∈(0.20, 0.45)\n"
        f"Red (bottom) = E<10 GPa ({n_soft:,}, {100*n_soft/n_kept:.1f}%).  "
        "Shaded band = 16th–84th percentile.  Dashed line = nominal.  "
        "Orange = quality metrics (Ks, B).  Green = derived moduli."
    )
    fig.text(0.5, 0.01, range_txt, ha="center", fontsize=7.5,
             color="dimgray", style="italic")

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
