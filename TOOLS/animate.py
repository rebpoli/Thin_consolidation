#!/usr/bin/env python3
"""
Vertical Line Pressure Animator

Visualizes pore pressure evolution along the vertical centerline (r=0)
using animation. Creates interactive plots with play/pause controls.

USAGE
=====

Basic interactive animation:
    python animate_vertical_line.py outputs/pressure_profile.nc

Use default location (outputs/pressure_profile.nc):
    python animate_vertical_line.py

Save animation as GIF (takes time):
    python animate_vertical_line.py outputs/pressure_profile.nc --save-gif animation.gif

Save individual frames as PNG images:
    python animate_vertical_line.py outputs/pressure_profile.nc --save-frames frames/

Control animation speed (interval in milliseconds):
    python animate_vertical_line.py outputs/pressure_profile.nc --interval 50

INTERACTIVE CONTROLS
====================

When viewing the animation:
- Play/Pause: Click the play button
- Speed: Use the slider below the plot
- Zoom: Scroll or click-drag to zoom
- Pan: Right-click-drag to pan

REQUIREMENTS
============

pip install matplotlib netCDF4 numpy
"""

import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import argparse
import os
from pathlib import Path


class VerticalLinePressureAnimator:
    """
    Animates pressure evolution along the vertical centerline.
    """
    
    def __init__(self, nc_file):
        """
        Initialize animator from NetCDF file.
        
        Parameters
        ----------
        nc_file : str
            Path to pressure_profile.nc
        """
        self.nc_file = nc_file
        self.ds = nc4.Dataset(nc_file, 'r')
        
        # Load data
        self.z_coord = self.ds['z_coord'][:].copy()
        self.time = self.ds['time'][:].copy()
        self.pressure = self.ds['pressure'][:].copy()  # shape: (time, z_point)
        
        # Load material properties
        self.E = getattr(self.ds, 'E', None)
        self.H = getattr(self.ds, 'H', None)
        self.perm = getattr(self.ds, 'perm', None)
        
        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.setup_figure()
    
    def setup_figure(self):
        """Configure the matplotlib figure."""
        self.ax.set_xlabel('Pore Pressure (Pa)', fontsize=12)
        self.ax.set_ylabel('Height z (m)', fontsize=12)
        self.ax.set_title('Pressure Profile Along Vertical Centerline (r=0)', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.invert_yaxis()  # z increases downward in typical plot
        
        # Pressure statistics text
        self.text_info = self.ax.text(0.98, 0.97, '', transform=self.ax.transAxes,
                                       fontsize=10, verticalalignment='top',
                                       horizontalalignment='right',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Line object for pressure profile
        self.line, = self.ax.plot([], [], 'b-', linewidth=2, label='Pressure')
        self.ax.legend(loc='upper left', fontsize=10)
    
    def _get_axes_limits(self):
        """Compute appropriate axis limits from data."""
        p_min, p_max = np.nanmin(self.pressure), np.nanmax(self.pressure)
        p_range = p_max - p_min
        p_pad = 0.1 * p_range if p_range > 0 else 1e5
        
        z_min, z_max = self.z_coord.min(), self.z_coord.max()
        
        return (p_min - p_pad, p_max + p_pad), (z_min, z_max)
    
    def update(self, frame):
        """Update function for animation."""
        p_range, z_range = self._get_axes_limits()
        self.ax.set_xlim(p_range)
        self.ax.set_ylim(z_range)
        
        # Update line with already-interpolated data
        pressure_at_t = self.pressure[frame, :]
        self.line.set_data(pressure_at_t, self.z_coord)
        
        # Update info text
        t_current = self.time[frame]
        p_max = np.max(pressure_at_t)
        p_mean = np.mean(pressure_at_t)
        p_min = np.min(pressure_at_t)
        
        info_text = f"Time: {t_current:.4f} s\n"
        info_text += f"P_max: {p_max:.3e} Pa\n"
        info_text += f"P_mean: {p_mean:.3e} Pa\n"
        info_text += f"P_min: {p_min:.3e} Pa\n"
        info_text += f"Frame: {frame + 1}/{len(self.time)}"
        
        self.text_info.set_text(info_text)
        
        return self.line, self.text_info
    
    def animate(self, interval=100, save_video=None):
        """
        Create and display animation.
        
        Parameters
        ----------
        interval : int
            Delay between frames in milliseconds (default: 100)
        save_video : str, optional
            If provided, save animation to this file (PNG/GIF)
        """
        n_frames = len(self.time)
        
        anim = FuncAnimation(self.fig, self.update, frames=n_frames,
                            interval=interval, blit=True, repeat=True)
        
        if save_video:
            # Save as GIF
            print(f"Saving animation to {save_video}...")
            writer = PillowWriter(fps=10)
            anim.save(save_video, writer=writer)
            print(f"✓ Animation saved: {save_video}")
        else:
            # Show interactive window
            print("✓ Animation ready. Use matplotlib controls to interact.")
            plt.show()
    
    def save_frames(self, output_dir='frames'):
        """
        Save each frame as a PNG image.
        
        Parameters
        ----------
        output_dir : str
            Directory to save frames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(len(self.time)):
            self.update(i)
            filename = os.path.join(output_dir, f'frame_{i:04d}.png')
            self.fig.savefig(filename, dpi=100, bbox_inches='tight')
            if (i + 1) % max(1, len(self.time) // 10) == 0:
                print(f"  Saved {i + 1}/{len(self.time)} frames")
        
        print(f"✓ Frames saved to {output_dir}/")
    
    def close(self):
        """Close NetCDF file and matplotlib figure."""
        self.ds.close()
        plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(
        description="Animate vertical line pressure profile from NetCDF output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Interactive animation (default location)
  python animate_vertical_line.py
  
  # Interactive animation (custom file)
  python animate_vertical_line.py outputs/pressure_profile.nc
  
  # Save as GIF
  python animate_vertical_line.py outputs/pressure_profile.nc --save-gif animation.gif
  
  # Save individual frames
  python animate_vertical_line.py outputs/pressure_profile.nc --save-frames frames/
  
  # Slower animation (200ms between frames)
  python animate_vertical_line.py --interval 200
        """
    )
    parser.add_argument('ncfile', nargs='?', default='outputs/pressure_profile.nc',
                       help='Path to pressure_profile.nc (default: outputs/pressure_profile.nc)')
    parser.add_argument('--save-gif', type=str, default=None,
                       help='Save animation to GIF file (slow)')
    parser.add_argument('--save-frames', type=str, default=None,
                       help='Save individual frames to directory')
    parser.add_argument('--interval', type=int, default=33,
                       help='Delay between frames in ms (default: 33 = 3x faster)')
    
    args = parser.parse_args()
    
    # Check file exists
    if not os.path.exists(args.ncfile):
        print(f"ERROR: File not found: {args.ncfile}")
        print(f"Looking in: {os.path.abspath(args.ncfile)}")
        print("\nMake sure you're in the demo directory with pressure_profile.nc")
        return 1
    
    print(f"Loading: {args.ncfile}")
    
    try:
        animator = VerticalLinePressureAnimator(args.ncfile)
        print(f"✓ Loaded {len(animator.time)} timesteps, {len(animator.z_coord)} z-points")
        print(f"  Time: {animator.time[0]:.3f} - {animator.time[-1]:.3f} s")
        print(f"  Height: {animator.z_coord[0]:.6f} - {animator.z_coord[-1]:.6f} m")
        
        if args.save_frames:
            print(f"\nSaving frames to {args.save_frames}...")
            animator.save_frames(args.save_frames)
        
        if args.save_gif:
            print(f"\nSaving animation to {args.save_gif} (this may take a minute)...")
            animator.animate(interval=args.interval, save_video=args.save_gif)
            animator.close()
        else:
            print("\nStarting interactive animation...")
            animator.animate(interval=args.interval)
            animator.close()
        
        return 0
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
