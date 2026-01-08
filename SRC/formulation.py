from dolfinx import fem
import ufl
import numpy as np

import config


class PoroelasticityFormulation:
    def __init__(self, domain, function_space, facets):
        self.domain = domain
        self.W      = function_space
        self.facets = facets
        
        cfg = config.get()
        self.material_cfg = cfg.materials
        self.bc_cfg       = cfg.boundary_conditions
        
        self._define_kinematics()
        self._create_fem_constants()
    
    def _define_kinematics(self):
        x = ufl.SpatialCoordinate(self.domain)
        
        def eps(w):
            e_rr = w[0].dx(0)
            e_tt = w[0] / x[0]
            e_zz = w[1].dx(1)
            e_rz = 0.5 * (w[0].dx(1) + w[1].dx(0))
            
            return ufl.sym(ufl.as_tensor([
                [e_rr, 0,    e_rz],
                [0,    e_tt, 0   ],
                [e_rz, 0,    e_zz]
            ]))
        
        def eps_v(w):
            return w[0].dx(0) + w[0] / x[0] + w[1].dx(1)
        
        def sigma_eff(w):
            return self.lmbda * ufl.tr(eps(w)) * ufl.Identity(3) + 2.0 * self.mu * eps(w)
        
        self.eps       = eps
        self.eps_v     = eps_v
        self.sigma_eff = sigma_eff
        self.x         = x
    
    def _create_fem_constants(self):
        self.alpha = fem.Constant(self.domain, self.material_cfg.alpha)
        self.perm  = fem.Constant(self.domain, self.material_cfg.perm)
        self.visc  = fem.Constant(self.domain, self.material_cfg.visc)
        self.M     = fem.Constant(self.domain, self.material_cfg.M)
        self.mu    = fem.Constant(self.domain, self.material_cfg.mu)
        self.lmbda = fem.Constant(self.domain, self.material_cfg.lmbda)
    
    def weak_form(self, dt, wh_old):
        u, p = ufl.TrialFunctions(self.W)
        v, q = ufl.TestFunctions(self.W)
        
        u_old, p_old = ufl.split(wh_old)
        
        dt_const = fem.Constant(self.domain, dt)
        r = self.x[0]
        
        dx = ufl.Measure("dx", domain=self.domain)
        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facets)
        
        # BILINEAR FORM (matching reference exactly)
        a = (
            # Mechanical equilibrium
            ufl.inner(self.sigma_eff(u), self.eps(v)) * r * dx
            - self.alpha * p * self.eps_v(v) * r * dx
            
            # Flow equation with time derivative
            + self.alpha / dt_const * self.eps_v(u) * q * r * dx
            + (self.perm / self.visc) * ufl.inner(ufl.grad(p), ufl.grad(q)) * r * dx
            + (1.0 / self.M) * p / dt_const * q * r * dx
        )
        
        # LINEAR FORM (RHS)
        L_terms = []
        
        # Old volumetric strain terms
        L_terms.append(self.alpha / dt_const * self.eps_v(u_old) * q * r * dx)
        L_terms.append((1.0 / self.M) * p_old / dt_const * q * r * dx)
        
        # Neumann boundary conditions
        L_neumann = self._apply_neumann_bcs(v, q, r, dx, ds)
        
        # Combine all terms
        if L_terms:
            L = sum(L_terms) + L_neumann
        else:
            L = L_neumann
        
        return a, L
    
    def _apply_neumann_bcs(self, v, q, r, dx, ds):
        # Collect all Neumann BC terms
        terms = []
        
        for surface in ['bottom', 'right', 'top', 'left']:
            bc_spec = getattr(self.bc_cfg, surface)
            marker = self._get_boundary_marker(surface)
            
            # Build traction vector for this surface
            traction_r = bc_spec.sig_rr if bc_spec.sig_rr is not None else 0.0
            traction_z = bc_spec.sig_zz if bc_spec.sig_zz is not None else 0.0
            
            # Only add term if at least one traction component is non-zero
            if traction_r != 0.0 or traction_z != 0.0:
                traction = ufl.as_vector([traction_r, traction_z])
                terms.append(ufl.inner(traction, v) * r * ds(marker))
        
        # Return sum of terms, or zero if no Neumann BCs
        if terms:
            return sum(terms)
        else:
            # No Neumann BCs - return zero constant
            return fem.Constant(self.domain, 0.0) * q * r * dx
    
    def setup_boundary_conditions(self):
        bcs = []
        
        for surface in ['bottom', 'right', 'top', 'left']:
            bc_spec = getattr(self.bc_cfg, surface)
            marker  = self._get_boundary_marker(surface)
            
            if bc_spec.U_r is not None:
                bc_ur = self._create_displacement_bc(0, float(bc_spec.U_r), marker)
                bcs.append(bc_ur)
            
            if bc_spec.U_z is not None:
                bc_uz = self._create_displacement_bc(1, float(bc_spec.U_z), marker)
                bcs.append(bc_uz)
            
            if bc_spec.Pressure is not None:
                bc_p = self._create_pressure_bc(float(bc_spec.Pressure), marker)
                bcs.append(bc_p)
        
        return bcs
    
    def _create_displacement_bc(self, component, value, marker):
        facet_indices = self.facets.find(marker)
        facet_dofs    = fem.locate_dofs_topological(
            self.W.sub(0).sub(component), 
            self.domain.topology.dim - 1, 
            facet_indices
        )
        return fem.dirichletbc(value, facet_dofs, self.W.sub(0).sub(component))
    
    def _create_pressure_bc(self, value, marker):
        facet_indices = self.facets.find(marker)
        facet_dofs    = fem.locate_dofs_topological(
            self.W.sub(1), 
            self.domain.topology.dim - 1, 
            facet_indices
        )
        return fem.dirichletbc(value, facet_dofs, self.W.sub(1))
    
    def _get_boundary_marker(self, surface):
        marker_map = {"bottom": 1, "right": 2, "top": 3, "left": 4}
        return marker_map[surface]
