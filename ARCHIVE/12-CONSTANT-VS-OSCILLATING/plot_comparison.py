#!/usr/bin/env python3
"""
DEMO 12: Constant vs Oscillating Load Comparison

Validates theta-weighting of stress boundary conditions by comparing:
1. Constant stress case (5 MPa) - monotonic consolidation
2. Oscillating load case (0-10 MPa, 50% duty cycle, 10 cycles) - dynamic response

Both cases use the same material properties and mesh to isolate the effect
of load time-dependence on the consolidation response.

USAGE:
    # Run constant stress case
    cd runs/constant_5mpa && python ../../SRC/run.py
    
    # Run oscillating load case
    cd runs/oscillating_0_10mpa && python ../../SRC/run.py
    
    # Then plot comparison
    ./plot_comparison.py [--max-time SECONDS]
    
    # Example: Show only first 1000 seconds
    ./plot_comparison.py --max-time 1000
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse

DEMO_DIR = Path(__file__).resolve().parent
RUNS_DIR = DEMO_DIR / "runs"

# ── Command-line arguments ────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Plot DEMO 12 comparison")
parser.add_argument('--max-time', type=float, default=None,
                    help='Maximum time in seconds to display')
args = parser.parse_args()

# ── Load datasets ─────────────────────────────────────────────────────────────

def load_case(case_name):
    """Load FEM timeseries for a case."""
    nc_file = RUNS_DIR / case_name / "outputs" / "fem_timeseries.nc"
    if not nc_file.exists():
        print(f"ERROR: {nc_file} not found. Run simulation first.")
        return None
    return xr.open_dataset(nc_file)

ds_constant = load_case("constant_5mpa")
ds_oscillating = load_case("oscillating_0_10mpa")

if ds_constant is None or ds_oscillating is None:
    raise FileNotFoundError("Missing output files. Run both simulations first.")

print("✓ Loaded constant stress case")
print("✓ Loaded oscillating load case")
if args.max_time is not None:
    print(f"✓ Displaying data up to t={args.max_time} s")

# ── Figure setup ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 2, figure=fig,
                       hspace=0.35, wspace=0.3,
                       top=0.95, bottom=0.08,
                       left=0.08, right=0.97)

ax_load = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
ax_pressure = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
ax_displacement = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

# ── Helper functions ──────────────────────────────────────────────────────────

def plot_load(ax, ds, title, max_time=None):
    """Plot applied stress over time."""
    t = ds['time'].values
    sig = ds['sig_zz_applied'].values / 1e6
    
    if max_time is not None:
        mask = t <= max_time
        t = t[mask]
        sig = sig[mask]
    
    ax.step(t, sig, color='crimson', linewidth=2, where='post', label='Applied σ_zz')
    ax.fill_between(t, sig, step='post', color='crimson', alpha=0.15)
    
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel('Stress [MPa]', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

def plot_pressure(ax, ds, title, y_lim=None, max_time=None):
    """Plot mean pore pressure over time."""
    t = ds['time'].values
    p_mean = ds['pressure_mean'].values / 1e3  # Convert to kPa
    
    if max_time is not None:
        mask = t <= max_time
        t = t[mask]
        p_mean = p_mean[mask]
    
    ax.plot(t, p_mean, color='blue', linewidth=2, label='FEM (mean)')
    
    if 'analytical_pressure' in ds:
        p_analytical = ds['analytical_pressure'].values / 1e3
        if max_time is not None:
            mask = ds['time'].values <= max_time
            p_analytical = p_analytical[mask]
        ax.plot(t, p_analytical, color='orange', linewidth=1.5, 
               linestyle='--', label='Analytical', alpha=0.7)
    
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel('Mean Pressure [kPa]', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    if y_lim is not None:
        ax.set_ylim(y_lim)

def plot_displacement(ax, ds, title, y_lim=None, max_time=None):
    """Plot vertical displacement at bottom."""
    t = ds['time'].values
    uz = ds['uz_at_bottom'].values * 1e3  # Convert to mm
    
    if max_time is not None:
        mask = t <= max_time
        t = t[mask]
        uz = uz[mask]
    
    ax.plot(t, uz, color='green', linewidth=2, label='FEM')
    
    if 'analytical_uz' in ds:
        uz_analytical = ds['analytical_uz'].values * 1e3
        if max_time is not None:
            mask = ds['time'].values <= max_time
            uz_analytical = uz_analytical[mask]
        ax.plot(t, uz_analytical, color='orange', linewidth=1.5,
               linestyle='--', label='Analytical', alpha=0.7)
    
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel('Displacement u_z [mm]', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    if y_lim is not None:
        ax.set_ylim(y_lim)

# ── Compute shared y-axis ranges ──────────────────────────────────────────────

# Filter by max_time if specified
t_const = ds_constant['time'].values
t_osc = ds_oscillating['time'].values

if args.max_time is not None:
    mask_const = t_const <= args.max_time
    mask_osc = t_osc <= args.max_time
    p_const = ds_constant['pressure_mean'].values[mask_const] / 1e3
    p_osc = ds_oscillating['pressure_mean'].values[mask_osc] / 1e3
    uz_const = ds_constant['uz_at_bottom'].values[mask_const] * 1e3
    uz_osc = ds_oscillating['uz_at_bottom'].values[mask_osc] * 1e3
else:
    p_const = ds_constant['pressure_mean'].values / 1e3
    p_osc = ds_oscillating['pressure_mean'].values / 1e3
    uz_const = ds_constant['uz_at_bottom'].values * 1e3
    uz_osc = ds_oscillating['uz_at_bottom'].values * 1e3

# Pressure range
p_min = min(np.min(p_const), np.min(p_osc))
p_max = max(np.max(p_const), np.max(p_osc))
p_lim = (p_min, p_max)

# Displacement range
uz_min = min(np.min(uz_const), np.min(uz_osc))
uz_max = max(np.max(uz_const), np.max(uz_osc))
uz_lim = (uz_min, uz_max)

# ── Plot data with shared scales ───────────────────────────────────────────────

plot_load(ax_load[0], ds_constant, 'Constant Stress (5 MPa)', max_time=args.max_time)
plot_load(ax_load[1], ds_oscillating, 'Oscillating Load (0-10 MPa, 50% duty)', max_time=args.max_time)

plot_pressure(ax_pressure[0], ds_constant, 'Mean Pore Pressure - Constant', y_lim=p_lim, max_time=args.max_time)
plot_pressure(ax_pressure[1], ds_oscillating, 'Mean Pore Pressure - Oscillating', y_lim=p_lim, max_time=args.max_time)

plot_displacement(ax_displacement[0], ds_constant, 'Vertical Displacement - Constant', y_lim=uz_lim, max_time=args.max_time)
plot_displacement(ax_displacement[1], ds_oscillating, 'Vertical Displacement - Oscillating', y_lim=uz_lim, max_time=args.max_time)

# ── Title ─────────────────────────────────────────────────────────────────────

fig.suptitle(
    'Theta-Weighting Validation: Constant vs Oscillating Load\n'
    'Re=10 cm, H=2 cm | E=14.4 GPa, ν=0.2, α=0.75, k=1e-17 m², M=13.5 GPa | θ_CN=0.75',
    fontsize=11, fontweight='bold'
)

# ── Save ──────────────────────────────────────────────────────────────────────

png_dir = DEMO_DIR / "png"
png_dir.mkdir(exist_ok=True)
out = png_dir / "comparison.png"
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved {out}")

plt.show()
