#!/usr/bin/env -S python -u

import pyvista as pv
import h5py
import numpy as np
from pathlib import Path

class XDMFViewer:
    """Simple PyVista interface for XDMF files"""

    def __init__(self, filename):
        """
        Initialize viewer with XDMF file

        Parameters:
        -----------
        filename : str
            Path to .xdmf file
        """
        self.filename = filename
        self.reader = pv.get_reader(filename)
        self.mesh = None
        self.plotter = None

    def load(self, time_step=0):
        """Load mesh at specific time step"""
        self.reader.set_active_time_point(time_step)
        self.mesh = self.reader.read()
        return self.mesh

    def plot(self, field_name=None, time_step=0, **kwargs):
        """
        Plot mesh and field

        Parameters:
        -----------
        field_name : str, optional
            Name of field to visualize
        time_step : int
            Time step to display
        **kwargs : dict
            Additional arguments for pyvista plotter
        """
        self.load(time_step)

        # Create plotter
        pl = pv.Plotter()

        # Default plot settings
        plot_kwargs = {
            'show_edges': True,
            'edge_color': 'black',
            'line_width': 0.5,
            'cmap': 'viridis',
        }
        plot_kwargs.update(kwargs)

        # Add mesh
        if field_name and field_name in self.mesh.array_names:
            pl.add_mesh(self.mesh, scalars=field_name, **plot_kwargs)
            pl.add_scalar_bar(title=field_name)
        else:
            pl.add_mesh(self.mesh, **plot_kwargs)

        # Add info
        pl.add_text(f"Time step: {time_step}", position='upper_left')

        # Show
        pl.show()

    def get_available_fields(self):
        """Get list of available field names"""
        if self.mesh is None:
            self.load(0)
        return self.mesh.array_names

    def get_time_steps(self):
        """Get number of time steps"""
        return self.reader.number_time_points

    def animate(self, field_name=None, output_file=None, **kwargs):
        """
        Create animation through time steps

        Parameters:
        -----------
        field_name : str, optional
            Field to animate
        output_file : str, optional
            Save animation to file (e.g., 'animation.gif')
        """
        n_steps = self.get_time_steps()

        # Setup plotter
        pl = pv.Plotter(off_screen=output_file is not None)

        # Animation loop
        pl.open_gif(output_file) if output_file else None

        for i in range(n_steps):
            pl.clear()
            self.load(i)

            plot_kwargs = {
                'show_edges': True,
                'cmap': 'viridis',
            }
            plot_kwargs.update(kwargs)

            if field_name:
                pl.add_mesh(self.mesh, scalars=field_name, **plot_kwargs)
                pl.add_scalar_bar(title=field_name)
            else:
                pl.add_mesh(self.mesh, **plot_kwargs)

            pl.add_text(f"Time step: {i}/{n_steps-1}", position='upper_left')

            if output_file:
                pl.write_frame()
            else:
                pl.show(auto_close=False)
                pl.update()

        if output_file:
            pl.close()

    def plot_deformed(self, displacement_field, scale=1.0, time_step=0, **kwargs):
        """
        Plot deformed mesh

        Parameters:
        -----------
        displacement_field : str
            Name of displacement field
        scale : float
            Deformation scale factor
        """
        self.load(time_step)

        # Get displacement
        if displacement_field not in self.mesh.array_names:
            raise ValueError(f"Field '{displacement_field}' not found")

        # Warp mesh
        warped = self.mesh.warp_by_vector(displacement_field, factor=scale)

        # Plot
        pl = pv.Plotter()

        plot_kwargs = {
            'show_edges': True,
            'cmap': 'viridis',
        }
        plot_kwargs.update(kwargs)

        # Original mesh (wireframe)
        pl.add_mesh(self.mesh, style='wireframe', color='gray',
                    opacity=0.3, label='Original')

        # Deformed mesh
        pl.add_mesh(warped, scalars=displacement_field, **plot_kwargs,
                    label='Deformed')

        pl.add_scalar_bar(title=f"{displacement_field} (scale={scale})")
        pl.add_text(f"Time step: {time_step}", position='upper_left')
        pl.add_legend()
        pl.show()


### MAIN
h5_file = "outputs/test.h5"
try:
    with h5py.File(h5_file, 'r') as f:
        print(f"✓ HDF5 file accessible")
        print(f"  Keys: {list(f.keys())}")
except Exception as e:
    print(f"✗ Cannot access HDF5: {e}")

# Create viewer
viewer = XDMFViewer("outputs/test.xdmf")

# Check available fields
print("Available fields:", viewer.get_available_fields())
print("Time steps:", viewer.get_time_steps())

# Plot mesh
viewer.plot()

# Plot specific field
viewer.plot(field_name="P", time_step=0)
