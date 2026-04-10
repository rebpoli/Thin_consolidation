"""UFL weak form and boundary condition setup for axisymmetric Biot poroelasticity.

Implements the mixed P2/P1 (displacement/pressure) formulation with
Crank-Nicolson time integration.  All integrals carry the r·dΩ
axisymmetric weight.

Effective stress sign convention:  σ_total = σ_effective − α·p·I
  (compression negative, tension positive — geomechanics convention)
"""

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

from config import CONFIG, SURFACES


# ---------------------------------------------------------------------------
# Formulation
# ---------------------------------------------------------------------------

class PoroelasticityFormulation:
    """Assembles the weak form and manages boundary conditions.

    Responsibilities:
      - Define axisymmetric strain / volumetric strain / effective-stress operators.
      - Build the bilinear form a(u,p; v,q) and linear form L at each timestep.
      - Apply Dirichlet BCs (displacement, pressure) and Neumann tractions.
      - Enforce optional rigid-body coupling via a penalty method.
      - Project stress components from the solution field.
    """

    # Marker IDs assigned by CylinderMesh — must stay in sync
    _MARKERS = {'bottom': 1, 'right': 2, 'top': 3, 'left': 4}

    def __init__(self, domain, function_space, facets):
        self.domain  = domain
        self.W       = function_space
        self.facets  = facets
        self.bc_cfg  = CONFIG.boundary_conditions

        self._define_kinematics()
        self._create_fem_constants(CONFIG)

    #
    #
    def _define_kinematics(self):
        """Create axisymmetric kinematic operators as closures over the spatial coordinate.

        eps(w)      — full 3×3 axisymmetric strain tensor (rr, tt, zz, rz components)
        eps_v(w)    — volumetric strain = tr(eps)
        sigma_eff(w) — linear elastic effective stress (Lamé)

        The θθ component w[0]/r accounts for the hoop strain in axisymmetry.
        """
        x = ufl.SpatialCoordinate(self.domain)

        def eps(w):
            return ufl.sym(ufl.as_tensor([
                [w[0].dx(0),      0,    0.5 * (w[0].dx(1) + w[1].dx(0))],
                [0,           w[0] / x[0],  0                           ],
                [0.5 * (w[0].dx(1) + w[1].dx(0)),  0,   w[1].dx(1)     ],
            ]))

        def eps_v(w):
            return w[0].dx(0) + w[0] / x[0] + w[1].dx(1)

        def sigma_eff(w):
            return self.lmbda * ufl.tr(eps(w)) * ufl.Identity(3) + 2.0 * self.mu * eps(w)

        self.eps        = eps
        self.eps_v      = eps_v
        self.sigma_eff  = sigma_eff
        self.x          = x

    #
    #
    def _create_fem_constants(self, cfg):
        """Wrap scalar material parameters as PETSc/UFL constants for the weak form."""
        m = cfg.materials
        self.alpha          = fem.Constant(self.domain, m.alpha)
        self.perm           = fem.Constant(self.domain, m.perm)
        self.visc           = fem.Constant(self.domain, m.visc)
        self.M              = fem.Constant(self.domain, m.M)
        self.mu             = fem.Constant(self.domain, m.mu)
        self.lmbda          = fem.Constant(self.domain, m.lmbda)
        self.theta          = fem.Constant(self.domain, cfg.timestepper.theta_cn)
        self._penalty_rigid = cfg.timestepper.penalty_rigid
        print(f"✓ Time integration: Crank-Nicolson (θ={self.theta.value})")

    #
    #
    def weak_form(self, dt: float, t: float, wh_old):
        """Return (a, L) — the bilinear and linear forms for one timestep.

        Crank-Nicolson weighting θ is applied to the flow term:
          θ·k/μ·∇p·∇q  (implicit)  +  (1−θ)·k/μ·∇p_old·∇q  (explicit)

        All terms are multiplied by r for the axisymmetric volume element r·dΩ.
        """
        u, p         = ufl.TrialFunctions(self.W)
        v, q         = ufl.TestFunctions(self.W)
        u_old, p_old = ufl.split(wh_old)

        dt_c  = fem.Constant(self.domain, dt)
        r     = self.x[0]
        theta = self.theta

        dx = ufl.Measure("dx", domain=self.domain)
        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facets)

        a = (
            ufl.inner(self.sigma_eff(u), self.eps(v)) * r * dx
            - self.alpha * p * self.eps_v(v) * r * dx
            + self.alpha / dt_c * self.eps_v(u) * q * r * dx
            + theta * (self.perm / self.visc) * ufl.inner(ufl.grad(p), ufl.grad(q)) * r * dx
            + (1.0 / self.M) * p / dt_c * q * r * dx
        )

        L = (
            self.alpha / dt_c * self.eps_v(u_old) * q * r * dx
            - (1 - theta) * (self.perm / self.visc) * ufl.inner(ufl.grad(p_old), ufl.grad(q)) * r * dx
            + (1.0 / self.M) * p_old / dt_c * q * r * dx
            + self._neumann_terms(v, q, r, dx, ds, t)
        )

        return a, L

    #
    #
    def _neumann_terms(self, v, q, r, dx, ds, t: float):
        """Accumulate traction boundary terms for all surfaces with non-zero loads."""
        terms = [fem.Constant(self.domain, 0.0) * q * r * dx]
        n = ufl.FacetNormal(self.domain)

        for surface in SURFACES:
            bc_spec = getattr(self.bc_cfg, surface)
            marker  = self._MARKERS[surface]
            sig_rr  = bc_spec.sig_rr or 0.0
            sig_zz  = (bc_spec.periodic_load.eval(t) if bc_spec.periodic_load is not None
                       else (bc_spec.sig_zz or 0.0))
            if sig_rr == 0.0 and sig_zz == 0.0:
                continue
            traction = ufl.as_vector([sig_rr * n[0], sig_zz * n[1]])
            terms.append(ufl.inner(traction, v) * r * ds(marker))

        return sum(terms)

    #
    #
    def setup_boundary_conditions(self):
        """Build and return the list of Dirichlet BCs from the config."""
        bcs = []
        for surface in SURFACES:
            bc_spec = getattr(self.bc_cfg, surface)
            marker  = self._MARKERS[surface]
            if bc_spec.U_r is not None:
                bcs.append(self._dirichlet(self.W.sub(0).sub(0), float(bc_spec.U_r), marker))
            if bc_spec.U_z is not None:
                bcs.append(self._dirichlet(self.W.sub(0).sub(1), float(bc_spec.U_z), marker))
            if bc_spec.Pressure is not None:
                bcs.append(self._dirichlet(self.W.sub(1), float(bc_spec.Pressure), marker))
        return bcs

    #
    #
    def set_rigid_constraints(self, A):
        """Enforce rigid-body coupling on surfaces flagged U_r_rigid / U_z_rigid.

        Adds penalty entries to the stiffness matrix so all DOFs on the surface
        move together (equal displacement).  The penalty value is taken from
        cfg.timestepper.penalty_rigid (default 1e10).
        """
        penalty = self._penalty_rigid
        for surface in SURFACES:
            bc_spec   = getattr(self.bc_cfg, surface)
            marker_id = self._MARKERS[surface]
            components = []
            if bc_spec.U_r_rigid:
                components.append(0)
            if bc_spec.U_z_rigid:
                components.append(1)
            for comp in components:
                facet_idx = self.facets.find(marker_id)
                dofs = np.sort(fem.locate_dofs_topological(
                    self.W.sub(0).sub(comp), self.domain.topology.dim - 1, facet_idx))
                ref = dofs[0]
                A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
                A.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, False)
                for dof in dofs[1:]:
                    A.setValue(dof,  dof,  penalty, addv=PETSc.InsertMode.ADD_VALUES)
                    A.setValue(ref,  ref,  penalty, addv=PETSc.InsertMode.ADD_VALUES)
                    A.setValue(dof,  ref, -penalty, addv=PETSc.InsertMode.ADD_VALUES)
                    A.setValue(ref,  dof, -penalty, addv=PETSc.InsertMode.ADD_VALUES)
                A.assemble()

    #
    #
    def compute_stress_components(self, wh):
        """Return UFL expressions for the four stress components and von Mises stress.

        Returns (sigma_rr, sigma_tt, sigma_zz, sigma_rz, von_mises) as UFL scalars
        suitable for L2 projection onto a DG0 or Lagrange function space.
        """
        u, p = ufl.split(wh)
        sigma_eff   = self.sigma_eff(u)
        sigma_total = sigma_eff - self.alpha * p * ufl.Identity(3)
        sigma_mean  = ufl.tr(sigma_total) / 3.0
        s           = sigma_total - sigma_mean * ufl.Identity(3)
        von_mises   = ufl.sqrt((3.0 / 2.0) * ufl.inner(s, s))
        return (sigma_total[0, 0], sigma_total[1, 1],
                sigma_total[2, 2], sigma_total[0, 2], von_mises)

    #
    #
    def _dirichlet(self, subspace, value, marker):
        """Create a Dirichlet BC on a named surface for the given subspace."""
        dofs = fem.locate_dofs_topological(
            subspace, self.domain.topology.dim - 1, self.facets.find(marker))
        return fem.dirichletbc(value, dofs, subspace)
