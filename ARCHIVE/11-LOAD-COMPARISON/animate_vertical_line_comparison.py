#!/usr/bin/env python3
"""
Dual Pressure Profile Animator for Load Comparison

Compares vertical line pressure profiles between two configurations (e.g., A vs B)
in the same plot. Used in DEMO 11 for load comparison studies.

USAGE
=====

Compare A and B configurations:
    ./animate_vertical_line_comparison.py runs/A_perm_1e-20 runs/B_perm_1e-20

Save as GIF:
    ./animate_vertical_line_comparison.py runs/A_perm_1e-20 runs/B_perm_1e-20 --save-gif comparison.gif

Save as MP4 (with automatic frame skipping for speed):
    ./animate_vertical_line_comparison.py runs/A_perm_1e-20 runs/B_perm_1e-20 --save-mp4 comparison.mp4

Save as MP4 with manual frame skipping (every 3rd frame):
    ./animate_vertical_line_comparison.py runs/A_perm_1e-20 runs/B_perm_1e-20 --save-mp4 comparison.mp4 --frame-skip 3

Limit animation to first 100 seconds:
    ./animate_vertical_line_comparison.py runs/A_perm_1e-20 runs/B_perm_1e-20 --max-time 100

Save frames:
    ./animate_vertical_line_comparison.py runs/A_perm_1e-20 runs/B_perm_1e-20 --save-frames frames/

INTERACTIVE CONTROLS
====================

When viewing the animation:
- Play/Pause: Click the play button
- Speed: Use the slider below the plot
- Zoom: Scroll or click-drag to zoom
- Pan: Right-click-drag to pan
"""

import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import argparse
import os
from pathlib import Path


class DualPressureProfileAnimator:
    """
    Compares pressure evolution along vertical centerline for two configurations.
    """
    
    def __init__(self, nc_file_a, nc_file_b, label_a='A', label_b='B', max_time=None, frame_skip=1):
        """
        Initialize animator with two NetCDF files.
        
        Parameters
        ----------
        nc_file_a : str
            Path to first pressure_profile.nc
        nc_file_b : str
            Path to second pressure_profile.nc
        label_a : str
            Label for first configuration
        label_b : str
            Label for second configuration
        max_time : float, optional
            Maximum time to animate (seconds). If None, use full range.
        frame_skip : int, optional
            Skip every N frames (default: 1, no skipping). 
            Use frame_skip=2 to render every 2nd frame, etc.
        """
        self.label_a = label_a
        self.label_b = label_b
        self.max_time = max_time
        self.frame_skip = frame_skip
        
        # Load first dataset
        self.ds_a = nc4.Dataset(nc_file_a, 'r')
        self.z_a = np.array(self.ds_a['z_coord'][:])  # Convert to regular array
        self.time_a = np.array(self.ds_a['time'][:])
        self.pressure_a = np.array(self.ds_a['pressure'][:])
        
        # Load second dataset
        self.ds_b = nc4.Dataset(nc_file_b, 'r')
        self.z_b = np.array(self.ds_b['z_coord'][:])  # Convert to regular array
        self.time_b = np.array(self.ds_b['time'][:])
        self.pressure_b = np.array(self.ds_b['pressure'][:])
        
        # Trim to max_time if specified
        if max_time is not None:
            mask_a = self.time_a <= max_time
            mask_b = self.time_b <= max_time
            
            self.time_a = self.time_a[mask_a]
            self.pressure_a = self.pressure_a[mask_a, :]
            
            self.time_b = self.time_b[mask_b]
            self.pressure_b = self.pressure_b[mask_b, :]
            
            print(f"Trimmed to max_time={max_time}s: A has {len(self.time_a)} steps, B has {len(self.time_b)} steps")
        
        # Handle different time grids - use shorter one
        if len(self.time_a) != len(self.time_b):
            print(f"Note: Different timesteps - A has {len(self.time_a)}, B has {len(self.time_b)}")
            
            # Use the shorter time grid
            if len(self.time_a) < len(self.time_b):
                print(f"Using A's time grid ({len(self.time_a)} steps)")
                self.time = self.time_a.copy()
                # Interpolate B to match A's time grid
                from scipy.interpolate import interp1d
                interp_b = interp1d(self.time_b, self.pressure_b, axis=0, 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
                self.pressure_b = interp_b(self.time_a)
                self.time_b = self.time_a.copy()
            else:
                print(f"Using B's time grid ({len(self.time_b)} steps)")
                self.time = self.time_b.copy()
                # Interpolate A to match B's time grid
                from scipy.interpolate import interp1d
                interp_a = interp1d(self.time_a, self.pressure_a, axis=0,
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
                self.pressure_a = interp_a(self.time_b)
                self.time_a = self.time_b.copy()
        else:
            self.time = self.time_a.copy()
        
        # Setup figure with two subplots side by side, or shared x-axis
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.setup_figure()
    
    def setup_figure(self):
        """Configure the matplotlib figure."""
        self.ax.set_xlabel('Pore Pressure (Pa)', fontsize=12)
        self.ax.set_ylabel('Height z (m)', fontsize=12)
        self.ax.set_title('Pressure Profile Comparison Along Vertical Centerline', 
                         fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.invert_yaxis()
        
        # Info text box
        self.text_info = self.ax.text(0.98, 0.97, '', transform=self.ax.transAxes,
                                       fontsize=10, verticalalignment='top',
                                       horizontalalignment='right',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Line objects
        self.line_a, = self.ax.plot([], [], 'b-', linewidth=2.5, label=f'{self.label_a}', alpha=0.8)
        self.line_b, = self.ax.plot([], [], 'r--', linewidth=2.5, label=f'{self.label_b}', alpha=0.8)
        
        self.ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    
    def _get_axes_limits(self):
        """Compute appropriate axis limits from both datasets."""
        p_min = min(np.nanmin(self.pressure_a), np.nanmin(self.pressure_b))
        p_max = max(np.nanmax(self.pressure_a), np.nanmax(self.pressure_b))
        p_range = p_max - p_min
        p_pad = 0.1 * p_range if p_range > 0 else 1e5
        
        z_min = min(self.z_a[0], self.z_b[0])
        z_max = max(self.z_a[-1], self.z_b[-1])
        
        return (p_min - p_pad, p_max + p_pad), (z_min, z_max)
    
    def update(self, frame):
        """Update function for animation."""
        p_range, z_range = self._get_axes_limits()
        self.ax.set_xlim(p_range)
        self.ax.set_ylim(z_range)
        
        # Update both lines with data from current timestep
        pressure_a_t = self.pressure_a[frame, :]
        pressure_b_t = self.pressure_b[frame, :]
        
        self.line_a.set_data(pressure_a_t, self.z_a)
        self.line_b.set_data(pressure_b_t, self.z_b)
        
        # Update info text
        t_current = self.time[frame]
        
        p_max_a = np.max(pressure_a_t)
        p_mean_a = np.mean(pressure_a_t)
        p_max_b = np.max(pressure_b_t)
        p_mean_b = np.mean(pressure_b_t)
        
        info_text = f"Time: {t_current:.4f} s\n"
        info_text += f"\n{self.label_a}: P_max={p_max_a:.3e} Pa, P_mean={p_mean_a:.3e} Pa\n"
        info_text += f"{self.label_b}: P_max={p_max_b:.3e} Pa, P_mean={p_mean_b:.3e} Pa\n"
        info_text += f"\nFrame: {frame + 1}/{len(self.time)}"
        
        self.text_info.set_text(info_text)
        
        return self.line_a, self.line_b, self.text_info
    
    def animate(self, interval=33, save_video=None):
        """
        Create and display animation.
        
        Parameters
        ----------
        interval : int
            Delay between frames in milliseconds (default: 33)
        save_video : str, optional
            If provided, save animation to this file (GIF or MP4)
        """
        n_frames = len(self.time)
        
        # For MP4 with large frame counts, apply frame skipping to speed up encoding
        frame_step = self.frame_skip
        if save_video and save_video.lower().endswith('.mp4') and n_frames > 500:
            # Auto-skip frames if >500 frames for MP4
            # Target ~300-400 frames for reasonable encoding speed
            frame_step = max(1, n_frames // 300)
            if frame_step > 1:
                print(f"  Skipping every {frame_step}th frame to speed up encoding ({n_frames} -> {n_frames // frame_step} frames)")
        
        frame_indices = list(range(0, n_frames, frame_step))
        
        anim = FuncAnimation(self.fig, self.update, frames=frame_indices,
                            interval=interval, blit=True, repeat=True)
        
        if save_video:
            print(f"Saving animation to {save_video}...")
            
            # Detect file format from extension
            if save_video.lower().endswith('.mp4'):
                # Use FFMpegWriter for MP4 output with optimizations
                fps = 1000 // interval  # Convert interval (ms) to fps
                # Adjust fps based on frame skipping
                fps = fps // frame_step if frame_step > 1 else fps
                
                # Optimize encoding: lower bitrate and use faster codec preset
                writer = FFMpegWriter(fps=fps, bitrate=1200, codec='libx264')
                anim.save(save_video, writer=writer, dpi=100)
            else:
                # Default to GIF with PillowWriter
                writer = PillowWriter(fps=10)
                anim.save(save_video, writer=writer)
            
            print(f"✓ Animation saved: {save_video}")
        else:
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
        """Close NetCDF files and matplotlib figure."""
        self.ds_a.close()
        self.ds_b.close()
        plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(
        description="Animate dual vertical line pressure profiles for comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Compare A and B configurations
  ./animate_vertical_line_comparison.py runs/A_perm_1e-20 runs/B_perm_1e-20
  
  # With custom labels
  ./animate_vertical_line_comparison.py runs/A_perm_1e-20 runs/B_perm_1e-20 -a "Config A" -b "Config B"
  
  # Save as GIF
  ./animate_vertical_line_comparison.py runs/A_perm_1e-20 runs/B_perm_1e-20 --save-gif comparison.gif
  
  # Save as MP4 (auto-skips frames if >500 for faster encoding)
  ./animate_vertical_line_comparison.py runs/A_perm_1e-20 runs/B_perm_1e-20 --save-mp4 comparison.mp4
  
  # Save as MP4 with manual frame skipping (every 3rd frame)
  ./animate_vertical_line_comparison.py runs/A_perm_1e-20 runs/B_perm_1e-20 --save-mp4 comparison.mp4 --frame-skip 3
  
  # Limit to first 100 seconds
  ./animate_vertical_line_comparison.py runs/A_perm_1e-20 runs/B_perm_1e-20 --max-time 100
  
  # Save frames
  ./animate_vertical_line_comparison.py runs/A_perm_1e-20 runs/B_perm_1e-20 --save-frames frames/
        """
    )
    parser.add_argument('dir_a', help='Directory containing first run (with pressure_profile.nc)')
    parser.add_argument('dir_b', help='Directory containing second run (with pressure_profile.nc)')
    parser.add_argument('-a', '--label-a', default='A', help='Label for first configuration (default: A)')
    parser.add_argument('-b', '--label-b', default='B', help='Label for second configuration (default: B)')
    parser.add_argument('--save-gif', type=str, default=None, help='Save animation to GIF file')
    parser.add_argument('--save-mp4', type=str, default=None, help='Save animation to MP4 file')
    parser.add_argument('--save-frames', type=str, default=None, help='Save individual frames to directory')
    parser.add_argument('--interval', type=int, default=33, help='Delay between frames in ms (default: 33)')
    parser.add_argument('--max-time', type=float, default=None, help='Maximum time to animate (seconds). If not specified, use full range.')
    parser.add_argument('--frame-skip', type=int, default=1, help='Skip every N frames (default: 1, no skipping). Use 2 for every 2nd frame, 3 for every 3rd, etc.')
    
    args = parser.parse_args()
    
    # Construct paths to pressure_profile.nc
    nc_file_a = os.path.join(args.dir_a, 'outputs', 'pressure_profile.nc')
    nc_file_b = os.path.join(args.dir_b, 'outputs', 'pressure_profile.nc')
    
    # Check files exist
    if not os.path.exists(nc_file_a):
        print(f"ERROR: File not found: {nc_file_a}")
        return 1
    if not os.path.exists(nc_file_b):
        print(f"ERROR: File not found: {nc_file_b}")
        return 1
    
    print(f"Loading: {nc_file_a}")
    print(f"Loading: {nc_file_b}")
    
    try:
        animator = DualPressureProfileAnimator(nc_file_a, nc_file_b, args.label_a, args.label_b, 
                                             max_time=args.max_time, frame_skip=args.frame_skip)
        print(f"✓ Loaded {len(animator.time)} timesteps")
        print(f"  {args.label_a}: {len(animator.z_a)} z-points, time {animator.time[0]:.3f}-{animator.time[-1]:.3f} s")
        print(f"  {args.label_b}: {len(animator.z_b)} z-points, time {animator.time[0]:.3f}-{animator.time[-1]:.3f} s")
        
        if args.save_frames:
            print(f"\nSaving frames to {args.save_frames}...")
            animator.save_frames(args.save_frames)
        
        # Handle video output (GIF or MP4)
        if args.save_gif:
            print(f"\nSaving animation to {args.save_gif}...")
            animator.animate(interval=args.interval, save_video=args.save_gif)
            animator.close()
        elif args.save_mp4:
            print(f"\nSaving animation to {args.save_mp4}...")
            animator.animate(interval=args.interval, save_video=args.save_mp4)
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
