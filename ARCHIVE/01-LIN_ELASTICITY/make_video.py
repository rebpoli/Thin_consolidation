#!/usr/bin/env python
"""
Direct HDF5 reading version - Most reliable approach

Reads the HDF5 files directly using h5py and matplotlib only
No DOLFINx, no PyVista - just pure Python!
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import h5py
import xml.etree.ElementTree as ET
import os
import sys


def parse_xdmf_structure(xdmf_path):
    """Parse XDMF to get times and HDF5 dataset paths"""
    print(f"Parsing {xdmf_path}...")
    
    tree = ET.parse(xdmf_path)
    root = tree.getroot()
    
    h5_file = xdmf_path.replace('.xdmf', '.h5')
    
    times = []
    h5_paths = []
    
    # Find temporal collection
    for grid in root.findall('.//Grid[@CollectionType="Temporal"]'):
        for time_grid in grid.findall('.//Grid'):
            # Get time
            time_elem = time_grid.find('.//Time')
            if time_elem is not None:
                times.append(float(time_elem.get('Value')))
            
            # Get HDF5 path
            attr_elem = time_grid.find('.//Attribute')
            if attr_elem is not None:
                data_item = attr_elem.find('.//DataItem')
                if data_item is not None:
                    text = data_item.text.strip()
                    if ':/' in text:
                        path = text.split(':/')[1]
                        h5_paths.append(path)
    
    print(f"  Found {len(times)} timesteps")
    if len(times) > 1:
        print(f"  Time range: {times[0]:.4f} to {times[-1]:.4f} s")
    
    return times, h5_file, h5_paths


def get_mesh_coordinates(h5_file):
    """Read mesh coordinates from HDF5"""
    print(f"Reading mesh from {os.path.basename(h5_file)}...")
    
    with h5py.File(h5_file, 'r') as f:
        # Try common paths
        for path in ['/Mesh/mesh/geometry', '/mesh/geometry', 
                     '/Mesh/coordinates', '/Mesh/mesh/coordinates']:
            if path in f:
                coords = f[path][:]
                print(f"  Found {coords.shape[0]} nodes")
                return coords
        
        raise ValueError(f"Could not find coordinates in {h5_file}")


def read_pressure_field(h5_file, h5_path):
    """Read pressure data from HDF5"""
    with h5py.File(h5_file, 'r') as f:
        if h5_path in f:
            return f[h5_path][:]
        return None


def extract_vertical_line(coords, pressure, z_points, r_pos=0.0, tol=0.1):
    """
    Extract pressure along vertical line at r=r_pos
    
    Uses nearest neighbor approach with tolerance
    """
    # Ensure pressure is 1D
    if pressure.ndim > 1:
        pressure = pressure.flatten()
    
    r_coords = coords[:, 0]
    z_coords = coords[:, 1]
    
    # Find nodes near the vertical line
    near_line = np.abs(r_coords - r_pos) < tol
    
    if np.sum(near_line) < 2:
        # Expand tolerance
        tol = max(tol * 2, 0.2)
        near_line = np.abs(r_coords - r_pos) < tol
    
    z_on_line = z_coords[near_line]
    p_on_line = pressure[near_line]
    
    if len(z_on_line) < 2:
        return np.zeros_like(z_points)
    
    # Sort by z coordinate
    idx = np.argsort(z_on_line)
    z_sorted = z_on_line[idx]
    p_sorted = p_on_line[idx]
    
    # Ensure 1D arrays for interpolation
    z_sorted = np.asarray(z_sorted).flatten()
    p_sorted = np.asarray(p_sorted).flatten()
    z_points_flat = np.asarray(z_points).flatten()
    
    # Interpolate
    try:
        p_interp = np.interp(z_points_flat, z_sorted, p_sorted)
    except Exception as e:
        print(f"    Warning: Interpolation failed: {e}")
        print(f"    z_sorted shape: {z_sorted.shape}, p_sorted shape: {p_sorted.shape}")
        return np.zeros_like(z_points)
    
    return p_interp


def create_video_from_h5(
    fem_xdmf="results/pressure.xdmf",
    analytical_xdmf="consolidation_analytical/pressure_analytical.xdmf",
    output="pressure_video.mp4",
    H=1.0,
    p0=106667.0,
    fps=10,
    dpi=150,
    num_points=100
):
    """Create pressure profile video from HDF5 files"""
    
    print("="*70)
    print("PRESSURE PROFILE VIDEO - HDF5 DIRECT READING")
    print("="*70)
    
    # Parse XDMF files
    times_fem, h5_fem, paths_fem = parse_xdmf_structure(fem_xdmf)
    times_ana, h5_ana, paths_ana = parse_xdmf_structure(analytical_xdmf)
    
    # Read mesh coordinates
    coords_fem = get_mesh_coordinates(h5_fem)
    coords_ana = get_mesh_coordinates(h5_ana)
    
    # Setup z points
    z_min = min(coords_fem[:, 1].min(), coords_ana[:, 1].min())
    z_max = max(coords_fem[:, 1].max(), coords_ana[:, 1].max())
    z_points = np.linspace(z_min, z_max, num_points)
    
    # Use minimum timesteps
    num_steps = min(len(times_fem), len(times_ana), len(paths_fem), len(paths_ana))
    times = times_fem[:num_steps]
    
    print(f"\nVideo settings:")
    print(f"  Timesteps: {num_steps}")
    print(f"  Duration: {num_steps/fps:.1f} seconds at {fps} FPS")
    print(f"  Resolution: {int(10*dpi)}x{int(6*dpi)} pixels")
    
    # Setup figure
    print("\nInitializing plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line_fem, = ax.plot([], [], 'b-', linewidth=2.5, label='FEM')
    line_ana, = ax.plot([], [], 'r--', linewidth=2.5, label='Analytical')
    
    ax.set_xlabel('Position z [m]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pressure [kPa]', fontsize=14, fontweight='bold')
    ax.set_xlim(0, H)
    ax.set_ylim(0, 1.05 * p0 / 1000)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title('Pressure Profile Evolution During Consolidation', 
                 fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    def init():
        line_fem.set_data([], [])
        line_ana.set_data([], [])
        time_text.set_text('')
        return line_fem, line_ana, time_text
    
    def animate(frame):
        # Read pressure data
        p_fem = read_pressure_field(h5_fem, paths_fem[frame])
        p_ana = read_pressure_field(h5_ana, paths_ana[frame])
        
        # Debug first frame
        if frame == 0:
            if p_fem is not None:
                print(f"  FEM pressure shape: {p_fem.shape}, range: [{p_fem.min():.1f}, {p_fem.max():.1f}]")
            if p_ana is not None:
                print(f"  Analytical pressure shape: {p_ana.shape}, range: [{p_ana.min():.1f}, {p_ana.max():.1f}]")
        
        # Extract along line
        if p_fem is not None and len(p_fem) > 0:
            vals_fem = extract_vertical_line(coords_fem, p_fem, z_points)
        else:
            vals_fem = np.zeros_like(z_points)
        
        if p_ana is not None and len(p_ana) > 0:
            vals_ana = extract_vertical_line(coords_ana, p_ana, z_points)
        else:
            vals_ana = np.zeros_like(z_points)
        
        # Update plot
        line_fem.set_data(z_points, vals_fem / 1000)
        line_ana.set_data(z_points, vals_ana / 1000)
        time_text.set_text(f't = {times[frame]:.4f} s\nFrame {frame+1}/{num_steps}')
        
        # Progress
        if frame % max(1, num_steps // 20) == 0:
            print(f"  Progress: {100*(frame+1)/num_steps:5.1f}%")
        
        return line_fem, line_ana, time_text
    
    # Create animation
    print("\nGenerating animation...")
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=num_steps,
        interval=1000/fps, blit=True
    )
    
    # Save
    print(f"\nSaving to {output}...")
    writer = FFMpegWriter(
        fps=fps,
        metadata={'title': 'Consolidation'},
        codec='libx264',
        bitrate=5000,
        extra_args=['-pix_fmt', 'yuv420p', '-crf', '18']
    )
    
    anim.save(output, writer=writer, dpi=dpi)
    plt.close()
    
    size_mb = os.path.getsize(output) / 1024 / 1024
    print(f"\n{'='*70}")
    print(f"✓ SUCCESS! Video created: {output}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Duration: {num_steps/fps:.1f} seconds")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create pressure video from HDF5')
    parser.add_argument('--fem', default='results/pressure.xdmf')
    parser.add_argument('--analytical', default='consolidation_analytical/pressure_analytical.xdmf')
    parser.add_argument('--output', default='pressure_video.mp4')
    parser.add_argument('--H', type=float, default=1.0)
    parser.add_argument('--p0', type=float, default=106667.0)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--dpi', type=int, default=150)
    
    args = parser.parse_args()
    
    # Check files
    for f in [args.fem, args.analytical]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found")
            sys.exit(1)
    
    create_video_from_h5(
        fem_xdmf=args.fem,
        analytical_xdmf=args.analytical,
        output=args.output,
        H=args.H,
        p0=args.p0,
        fps=args.fps,
        dpi=args.dpi
    )
