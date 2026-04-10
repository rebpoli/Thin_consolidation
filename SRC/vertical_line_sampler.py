"""Vertical centerline pressure sampler for NetCDF profile output.

Extracts the pressure field along the axis of symmetry (r=0, z: 0 → H/2)
at a fixed set of evenly-spaced z-coordinates.  DOF values are read from
the pressure subspace, then interpolated with a cubic spline (falling back
to linear if fewer than 4 mesh nodes lie on the axis).
"""

import numpy as np
from scipy.interpolate import interp1d

from config import CONFIG


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class VerticalLineSampler:
    """Samples pressure along the vertical centerline (r=0) at each timestep.

    Construction locates all pressure DOFs on the axis (r < r_tol) and sorts
    them by z.  Each call to sample() interpolates the current pressure solution
    onto the fixed z_coords grid stored in the NetCDF file.
    """

    def __init__(self, W, domain, num_points=None):
        self.domain = domain

        if num_points is None:
            num_points = CONFIG.output.n_profile_points
        self.num_points = num_points

        self.P_space, self.p_map = W.sub(1).collapse()
        self.z_coords = np.linspace(0, CONFIG.mesh.H / 2, num_points)

        self.z_mesh, self.dof_indices = self._get_mesh_centerline_dofs()

    #
    #
    def _get_mesh_centerline_dofs(self):
        """Find pressure DOFs on the axis r=0 and return them sorted by z.

        The r=0 tolerance is set to half the smallest non-zero r value in the
        mesh so it adapts to any refinement level without hard-coded epsilons.

        Returns (z_values, dof_indices) — arrays of the same length.
        """
        dof_coords = self.P_space.tabulate_dof_coordinates()
        r_coords   = dof_coords[:, 0]
        z_coords   = dof_coords[:, 1]

        r_positive = r_coords[r_coords > 0]
        r_tol      = r_positive.min() / 2.0 if len(r_positive) > 0 else 1e-10
        r_mask     = r_coords < r_tol

        centerline_z       = z_coords[r_mask]
        centerline_indices = np.where(r_mask)[0]

        sort_idx     = np.argsort(centerline_z)
        z_mesh       = centerline_z[sort_idx]
        dof_idx_mesh = centerline_indices[sort_idx]

        return z_mesh, dof_idx_mesh

    #
    #
    def sample(self, solution_vector):
        """Interpolate pressure from mesh DOFs onto the fixed z_coords grid.

        Uses cubic spline interpolation; falls back to linear if fewer than
        4 axis nodes are available.

        Parameters
        ----------
        solution_vector : array
            Pressure DOF values — wh.x.array[p_map].

        Returns
        -------
        np.ndarray, shape (num_points,)
        """
        pressure_mesh = solution_vector[self.dof_indices]
        kind          = 'cubic' if len(self.z_mesh) >= 4 else 'linear'
        interp_func   = interp1d(self.z_mesh, pressure_mesh, kind=kind,
                                 bounds_error=False, fill_value='extrapolate')
        return interp_func(self.z_coords)

    #
    #
    def get_z_coordinates(self):
        """Return the fixed z-coordinate array written to the NetCDF file."""
        return self.z_coords.copy()
