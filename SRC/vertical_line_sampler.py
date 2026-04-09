"""
Vertical Line Pressure Sampler

Extracts pressure along the centerline (r=0, z: 0 → H) at specified points
for all timesteps. Uses cubic spline interpolation for smooth spatial representation.
"""

import numpy as np
from dolfinx import fem
from dolfinx.fem import FunctionSpace
from scipy.interpolate import interp1d


class VerticalLineSampler:
    """
    Samples pressure along the vertical centerline (r=0).
    
    The centerline is parameterized as:
      Points: (r=0, z) where z ∈ [0, H]
    
    Pressure is evaluated at num_points evenly-spaced z-coordinates using
    cubic spline interpolation for smooth spatial representation.
    """
    
    def __init__(self, W, domain, cfg, num_points=None):
        """
        Initialize sampler.
        
        Parameters
        ----------
        W : FunctionSpace
            Mixed function space (P2 displacement + P1 pressure)
        domain : Mesh
            DOLFINx mesh
        cfg : Config
            Configuration object (contains H)
        num_points : int, optional
            Number of sample points along centerline (default: cfg.output.n_profile_points)
        """
        self.domain = domain
        self.cfg = cfg
        
        if num_points is None:
            num_points = cfg.output.n_profile_points
        self.num_points = num_points
        
        # Get pressure space (sub(1) of mixed space)
        self.P_space, self.p_map = W.sub(1).collapse()
        
        # Create z-coordinates: evenly spaced from 0 to H
        H = cfg.mesh.H
        self.z_coords = np.linspace(0, H, num_points)
        
        # Find all unique z-values in the mesh at r≈0 for interpolation basis
        self.z_mesh, self.dof_indices = self._get_mesh_centerline_dofs()
    
    def _get_mesh_centerline_dofs(self):
        """
        Get all DOFs along centerline (r≈0) sorted by z-coordinate.
        
        Returns
        -------
        tuple
            (z_values, dof_indices) - sorted z-coordinates and corresponding DOF indices
        """
        # Get coordinates of all pressure DOFs
        dof_coords = self.P_space.tabulate_dof_coordinates()  # shape: (n_dofs, 2)
        r_coords = dof_coords[:, 0]
        z_coords = dof_coords[:, 1]
        
        # Find DOFs exactly on the axis r=0.
        # Tolerance = half the smallest non-zero r so only the r=0 column is
        # captured regardless of mesh refinement level.
        r_positive = r_coords[r_coords > 0]
        if len(r_positive) > 0:
            r_tolerance = r_positive.min() / 2.0
        else:
            r_tolerance = 1e-10
        r_mask = r_coords < r_tolerance
        
        # Get z-coordinates and indices for centerline DOFs
        centerline_z = z_coords[r_mask]
        centerline_indices = np.where(r_mask)[0]
        
        # Sort by z
        sort_idx = np.argsort(centerline_z)
        z_mesh = centerline_z[sort_idx]
        dof_idx_mesh = centerline_indices[sort_idx]
        
        return z_mesh, dof_idx_mesh
    
    def sample(self, solution_vector):
        """
        Sample pressure at all centerline points using cubic spline interpolation.
        
        Parameters
        ----------
        solution_vector : array-like
            Pressure DOF values (from wh.x.array[p_map])
        
        Returns
        -------
        np.ndarray
            Interpolated pressure values at sample z-coordinates (shape: num_points)
        """
        # Extract pressure at mesh centerline DOFs
        pressure_mesh = solution_vector[self.dof_indices]
        
        # Check if we have enough points for cubic interpolation
        if len(self.z_mesh) < 4:
            # Fall back to linear interpolation
            kind = 'linear'
        else:
            kind = 'cubic'
        
        # Create interpolation function
        interp_func = interp1d(self.z_mesh, pressure_mesh, kind=kind, 
                              bounds_error=False, fill_value='extrapolate')
        
        # Evaluate at requested sample points
        pressure_values = interp_func(self.z_coords)
        
        return pressure_values
    
    def get_z_coordinates(self):
        """
        Get z-coordinates of sample points.
        
        Returns
        -------
        np.ndarray
            z-coordinates (shape: num_points)
        """
        return self.z_coords.copy()
