from dolfinx import fem
import ufl
import numpy as np

from petsc4py import PETSc
import config


class PoroelasticityFormulation:
    #
    #
    def __init__(self, domain, function_space, facets):
        self.domain = domain
        self.W      = function_space
        self.facets = facets
        
        cfg = config.get()
        self.material_cfg = cfg.materials
        self.numerical_cfg = cfg.numerical
        self.bc_cfg       = cfg.boundary_conditions
        
        self._define_kinematics()
        self._create_fem_constants()
    
    #
    #
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
    
    #
    #
    def _create_fem_constants(self):
        self.alpha = fem.Constant(self.domain, self.material_cfg.alpha)
        self.perm  = fem.Constant(self.domain, self.material_cfg.perm)
        self.visc  = fem.Constant(self.domain, self.material_cfg.visc)
        self.M     = fem.Constant(self.domain, self.material_cfg.M)
        self.mu    = fem.Constant(self.domain, self.material_cfg.mu)
        self.lmbda = fem.Constant(self.domain, self.material_cfg.lmbda)
        self.theta = fem.Constant(self.domain, self.numerical_cfg.theta_cn) 
        
        print(f"✓ Time integration: Crank-Nicolson (θ={self.theta.value})")
    
    #
    #
    def weak_form(self, dt, wh_old):
        u, p = ufl.TrialFunctions(self.W)
        v, q = ufl.TestFunctions(self.W)
        
        u_old, p_old = ufl.split(wh_old)
        
        dt_const = fem.Constant(self.domain, dt)
        r = self.x[0]
        theta = self.theta
        
        dx = ufl.Measure("dx", domain=self.domain)
        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facets)

        # For rigid BC
        bottom_marker = self._get_boundary_marker('bottom')

        # BILINEAR FORM (LHS)
        a = (
            # Mechanical equilibrium: quasi-static, NO theta-weighting
            # (instantaneous equilibrium at current timestep)
            +  ufl.inner(self.sigma_eff(u), self.eps(v)) * r * dx
            -  self.alpha * p * self.eps_v(v) * r * dx

            # Flow equation: time derivatives + theta-weighted diffusion
            + self.alpha / dt_const * self.eps_v(u) * q * r * dx
            + theta * (self.perm / self.visc) * ufl.inner(ufl.grad(p), ufl.grad(q)) * r * dx
            + (1.0 / self.M) * p / dt_const * q * r * dx
        )
        
        # LINEAR FORM (RHS)
        L = (
            # Flow equation: time derivatives + (1-theta)-weighted diffusion
            + self.alpha / dt_const * self.eps_v(u_old) * q * r * dx
            - (1 - theta) * (self.perm / self.visc) * ufl.inner(ufl.grad(p_old), ufl.grad(q)) * r * dx
            + (1.0 / self.M) * p_old / dt_const * q * r * dx

            # Neumann boundary conditions
            +  self._apply_neumann_bcs(v, q, r, dx, ds)
        )
        
        return a, L

    #
    # Collect all Neumann BC terms
    def _apply_neumann_bcs(self, v, q, r, dx, ds):
        
        terms = [fem.Constant(self.domain, 0.0) * q * r * dx] ## Dummy to keep type consistency
        n = ufl.FacetNormal(self.domain)
        
        for surface in ['bottom', 'right', 'top', 'left']:
            bc_spec = getattr(self.bc_cfg, surface)
            marker = self._get_boundary_marker(surface)

            sig_rr = bc_spec.sig_rr if bc_spec.sig_rr is not None else 0.0
            sig_zz = bc_spec.sig_zz if bc_spec.sig_zz is not None else 0.0
            if sig_rr == 0.0 and sig_zz == 0.0: continue

            # Apply normal
            traction_r = sig_rr * n[0]
            traction_z = sig_zz * n[1]

            traction = ufl.as_vector([traction_r, traction_z])
            terms.append(ufl.inner(traction, v) * r * ds(marker))

        return sum(terms)
    
    #
    #
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

    #
    # Set rigid constraints using the penalty method
    #
    def set_rigid_constraints( self, A ) :
        for surface in ['bottom', 'right', 'top', 'left']:
            ## Set rigid constraints
            marker_id  = self._get_boundary_marker(surface)
            bc_spec = getattr(self.bc_cfg, surface)

            components = []
            if bc_spec.U_r_rigid : components.append(0)
            if bc_spec.U_z_rigid : components.append(1)

            for component in components :
                facet_indices = self.facets.find(marker_id)
                dofs = fem.locate_dofs_topological( self.W.sub(0).sub(component), self.domain.topology.dim - 1, facet_indices)
                dofs = np.sort(dofs)
                ref_dof = dofs[0]
                penalty = self.numerical_cfg.penalty_rigid

                indptr, indices, data = A.getValuesCSR()
                A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
                A.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, False)

                ## Assign a penalty between the refernce DOF and every other DOF in the boundary
                for dof in dofs[1:]:
                    A.setValue(dof, dof, penalty, addv=PETSc.InsertMode.ADD_VALUES)
                    A.setValue(ref_dof, ref_dof, penalty, addv=PETSc.InsertMode.ADD_VALUES)
                    A.setValue(dof, ref_dof, -penalty, addv=PETSc.InsertMode.ADD_VALUES)
                    A.setValue(ref_dof, dof, -penalty, addv=PETSc.InsertMode.ADD_VALUES)
                
                A.assemble()


    
    #
    #
    def _create_displacement_bc(self, component, value, marker):
        facet_indices = self.facets.find(marker)
        facet_dofs    = fem.locate_dofs_topological(
            self.W.sub(0).sub(component), 
            self.domain.topology.dim - 1, 
            facet_indices
        )
        return fem.dirichletbc(value, facet_dofs, self.W.sub(0).sub(component))
    
    #
    #
    def _create_pressure_bc(self, value, marker):
        facet_indices = self.facets.find(marker)
        facet_dofs    = fem.locate_dofs_topological(
            self.W.sub(1), 
            self.domain.topology.dim - 1, 
            facet_indices
        )
        return fem.dirichletbc(value, facet_dofs, self.W.sub(1))
    
    #
    #
    def _get_boundary_marker(self, surface):
        marker_map = {"bottom": 1, "right": 2, "top": 3, "left": 4}
        return marker_map[surface]
    
    #
    #
    def compute_stress_components(self, wh):
        u, p = ufl.split(wh)
        
        # Effective stress: σ' = λ*tr(ε)*I + 2μ*ε
        sigma_eff = self.sigma_eff(u)
        
        # Total stress: σ = σ' - α*p*I
        sigma_total = sigma_eff - self.alpha * p * ufl.Identity(3)
        
        # Extract components
        sigma_rr = sigma_total[0, 0]
        sigma_zz = sigma_total[2, 2]
        sigma_rz = sigma_total[0, 2]
        
        # von Mises: √(3/2 * s:s) where s is deviatoric stress
        sigma_mean = ufl.tr(sigma_total) / 3.0
        s = sigma_total - sigma_mean * ufl.Identity(3)
        von_mises = ufl.sqrt((3.0/2.0) * ufl.inner(s, s))
        
        return sigma_rr, sigma_zz, sigma_rz, von_mises
