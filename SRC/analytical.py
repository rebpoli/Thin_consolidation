"""Terzaghi 1D consolidation analytical solution (Fourier series).

Provides reference pressure, displacement, and drained-volume histories
for a single-step load applied at t=0 on a half-specimen (drainage at
z=H/2, symmetry at z=0).  Used during the FEM time loop to compute
relative errors at every timestep.

Reference: Cheng (2016) — Poroelasticity, §3.4.
"""

import numpy as np

from config import CONFIG


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------

class Analytical1DConsolidation:
    """Terzaghi 1D consolidation — pressure, displacement, and volume histories.

    The specimen is modelled as half-height: drainage at z = H/2, symmetry
    (no-flux, no-displacement) at z = 0.  The Fourier series is truncated at
    _N_TERMS (pressure/displacement) or _N_TERMS_VOLUME (volume integral).
    """

    _N_TERMS        = 50    # terms for pressure and displacement series
    _N_TERMS_VOLUME = 500   # more terms needed for volume convergence

    def __init__(self, n_spatial_points: int = 50):
        self.material_cfg = CONFIG.materials
        self.mesh_cfg     = CONFIG.mesh
        self.times        = CONFIG.timestepper.expected_time_list()

        self.n_z      = n_spatial_points
        self.z_coords = np.linspace(0, self.mesh_cfg.H / 2, self.n_z)

        # Initial excess pore pressure from undrained loading (Skempton B · σ_applied)
        mu   = self.material_cfg.mu
        S    = self.material_cfg.S
        eta  = self.material_cfg.eta
        sig0 = -1e5          # reference step load [Pa] — kept internal
        self.sig0 = sig0
        self.p0   = -sig0 * eta / mu / S

    # ------------------------------------------------------------------
    # Public API — one method per quantity, called each FEM timestep
    # ------------------------------------------------------------------

    def pressure_at_base(self, t: float) -> float:
        """Analytical excess pore pressure at the undrained base (maximum pressure).

        In the _pressure formula z is measured from the drainage face, so z=H_dr
        gives the no-flux (symmetry) face — the point of maximum excess pressure.
        This corresponds to FEM z=0 (bottom face).
        """
        H_dr = self.mesh_cfg.H / 2
        return self._pressure(t, H_dr)

    #
    #
    def uz_at_top(self, t: float) -> float:
        """Settlement at the drainage/loaded face.

        In the _uz formula the coordinate z is measured from the drainage face
        (free end), so z=0 gives the maximum settlement there.  This corresponds
        to the FEM top face (z = H/2 in FEM coordinates).
        """
        return self._uz(t, 0.0)

    #
    #
    def volume_drained(self, t: float) -> float:
        """Cumulative volume of fluid expelled from the specimen [m³]."""
        return self._V_drained(t)

    # ------------------------------------------------------------------
    # Fourier core expression and helpers
    # ------------------------------------------------------------------

    def _modes(self, n_terms: int = None):
        """Yield mode values m = (2n+1)π/2 for n = 0 … n_terms-1.

        These are the eigenvalues of the 1D consolidation equation with
        drainage at one end and no-flux at the other.
        """
        N = n_terms if n_terms is not None else self._N_TERMS
        return [(2 * n + 1) * np.pi / 2 for n in range(N)]

    #
    #
    def _T_v(self, t: float) -> float:
        """Dimensionless time factor T_v = c_v · t / (4 · H_dr²)."""
        H_dr = self.mesh_cfg.H / 2
        return self.material_cfg.c_v * t / (H_dr ** 2) / 4

    #
    #
    def _pressure(self, t: float, z: float) -> float:
        """Excess pore pressure at height z and time t."""
        if t == 0:
            return self.p0
        H_dr = self.mesh_cfg.H / 2
        T_v  = self._T_v(t)
        F1   = sum(
            (2.0 / m) * np.sin(m * z / H_dr) * np.exp(-4 * m**2 * T_v)
            for m in self._modes()
        )
        return self.p0 * F1

    #
    #
    def _uz(self, t: float, z: float) -> float:
        """Axial displacement at height z and time t.

        Composed of an instantaneous undrained term and a consolidation term
        that grows from zero to its drained value as T_v → ∞.
        """
        mat  = self.material_cfg
        H_dr = self.mesh_cfg.H / 2
        T_v  = self._T_v(t)
        F2   = sum(
            (2.0 / m**2) * np.cos(m * z / H_dr) * (1 - np.exp(-4 * m**2 * T_v))
            for m in self._modes()
        )
        uz  = -self.sig0 * H_dr / 2 / mat.mu * (1 - 2 * mat.nu_u) / (1 - mat.nu_u) * (1 - z / H_dr)
        uz += -self.sig0 * H_dr * F2 * (mat.nu_u - mat.nu) / (2 * mat.mu) / (1 - mat.nu_u) / (1 - mat.nu)
        return uz

    #
    #
    def _V_drained(self, t: float) -> float:
        """Cumulative drained volume using a flux-integral Fourier series.

        Uses a different mode formula: m = (2n+1)π (not π/2) and requires
        many more terms (_N_TERMS_VOLUME) for accurate integration near t=0.
        """
        if t == 0:
            return 0.0
        mat  = self.material_cfg
        H_dr = self.mesh_cfg.H / 2
        Re   = self.mesh_cfg.Re
        t_s  = mat.c_v * t / (4 * H_dr**2)
        modes = [(2 * n + 1) * np.pi for n in range(self._N_TERMS_VOLUME)]
        F    = sum((1.0 / m**2) * (1.0 - np.exp(-m**2 * t_s)) for m in modes)
        return np.pi * Re**2 * (mat.perm / mat.visc) * (2 * self.p0 / H_dr) * (4 * H_dr**2 / mat.c_v) * F
