import os
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element, mixed_element
from ufl import *
import ufl
import xarray as xr
from dolfinx.io import VTXWriter

        
import config
from formulation import PoroelasticityFormulation
from analytical import Analytical1DConsolidation
from export_pvd import PVDExporter


class PoroelasticitySolver:
    #
    #
    def __init__(self, mesh):
        self.mesh   = mesh
        self.domain = mesh.domain
        self.facets = mesh.facets
        self.comm   = mesh.comm
        self.gdim   = self.domain.geometry.dim
        
        cfg = config.get()
        self.cfg = cfg
        
        self.analytical = Analytical1DConsolidation()
        
        # Function spaces setup
        # P2 for displacement ; P1 for pressure
        cell_name = self.domain.topology.cell_name()
        P2 = element("Lagrange", cell_name, 2, shape=(self.gdim,))
        P1 = element("Lagrange", cell_name, 1)

        mixed_elem = mixed_element([P2, P1])
        self.W = fem.functionspace(self.domain, mixed_elem)

        
        # Formulation
        self.formulation = PoroelasticityFormulation(self.domain, self.W, self.facets)
        
        # History tracking
        self.wh     = fem.Function(self.W, name="Solution")
        self.wh_old = fem.Function(self.W, name="Previous")
        self.bcs = self.formulation.setup_boundary_conditions()

        self._setup_vtk_output()
        
        self.history = None

        # Create xdmf output
        with io.XDMFFile(self.domain.comm, self.cfg.output.results, "w") as xdmf :
            xdmf.write_mesh(self.domain)


    
    
    #
    #
    def _setup_vtk_output(self):
        # Scalar P2 space for u_r
        cell_name = self.domain.topology.cell_name()
        DG0_elem = element("DG", cell_name, 0)
        DG0_space = fem.functionspace(self.domain, DG0_elem)
        
        # DG0 vector space for stress vector [σ_rr, σ_zz, σ_rz]
        DG0_vector_elem = element("DG", cell_name, 1, shape=(3,))
        DG0_vector_space = fem.functionspace(self.domain, DG0_vector_elem)
        
        # Displacement output
        u_space, _ = self.W.sub(0).collapse()
        self.u_out = fem.Function(u_space, name="Displacement")
        # Pressure output
        p_space, _ = self.W.sub(1).collapse()
        self.p_out = fem.Function(p_space, name="Pressure")
        
        # Stress outputs
        self.stress_vector_out = fem.Function(DG0_vector_space, name="Stress")
        self.von_mises_out = fem.Function(DG0_space, name="von_Mises")
        
        # Get output directory from config
        output_dir = os.path.dirname(self.cfg.output.results)
        if self.comm.rank == 0: print(f"✓ Results output configured.")
    
    #
    #
    def solve(self):
        if self.comm.rank == 0:
            print("\n" + "="*70)
            print("STARTING TIME INTEGRATION")
            print("="*70)
        
        history = self._initialize_history()
        
        #
        # Timestep loop
        #
        for i_ts, dt, time in self.cfg.numerical:
            self._solve_timestep(dt)
            self._compute_and_project_stresses()
            self._collect_history(history, i_ts, time)
            self._write_timestep(time)
            self._print_progress(i_ts, time, dt, history)
            self.wh_old.x.array[:] = self.wh.x.array[:]
        
        self.history = history
        
        if self.comm.rank == 0:
            print("="*70)
            print("TIME INTEGRATION COMPLETED")
            print("="*70)
        
        self._save_fem_timeseries()


        self._save_run_summary()

    #
    #
    def _solve_timestep(self, dt):
        a, L = self.formulation.weak_form(dt, self.wh_old)
        
        # Assemble the system
        A = fem.petsc.assemble_matrix(fem.form(a), bcs=self.bcs)

        b = fem.petsc.assemble_vector(fem.form(L))
        fem.petsc.apply_lifting(b, [fem.form(a)], [self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, self.bcs)

        A.assemble()

        self.formulation.set_rigid_constraints( A )

        A.assemble()

        # Create solver
        solver = PETSc.KSP().create(self.mesh.comm)
        solver.setOptionsPrefix("my_ksp_")
        solver.setOperators(A)

        #  Access the PETSc options database
        opts = PETSc.Options()
        prefix = solver.getOptionsPrefix()
#         opts[f"{prefix}ksp_monitor"] = None
        opts[f"{prefix}ksp_type"] = "preonly"
        opts[f"{prefix}pc_type"] = "lu"
        opts[f"{prefix}pc_factor_mat_solver_type"] = "mumps"

        # Solve
        solver.setFromOptions()
        solver.solve(b, self.wh.x.petsc_vec)

        # Sync all processors
        self.wh.x.scatter_forward()


# #             # Iterative solver
# #             petsc_options={
# #                 "ksp_type": "gmres",
# #                 "pc_type": "hypre",
# #                 "ksp_rtol": 1e-20,
# #                 "ksp_atol": 1e-20,
# # #                 "ksp_monitor": None,  
# #                 "ksp_converged_reason": None,  
# # #                 "ksp_view": None,     # Show solver configuration
# #                 "ksp_max_it" : 10000,
# #                 "pc_hypre_type":"boomeramg",
# #                 "pc_hypre_boomeramg_strong_threshold": 0.7,
# #                 "pc_hypre_boomeramg_agg_nl": 4,
# #                 "pc_hypre_boomeramg_agg_num_paths": 5,
# #                 "pc_hypre_boomeramg_max_levels": 5, 
# #                 "pc_hypre_boomeramg_coarsen_type": "HMIS", 
# #                 "pc_hypre_boomeramg_interp_type": "ext+i",
# #                 "pc_hypre_boomeramg_P_max": 2, 
# #                 "pc_hypre_boomeramg_truncfactor": 0.3 
# #             },

    #
    #
    def _compute_and_project_stresses(self):
        """Project stress components to DG0 space as vector [σ_rr, σ_zz, σ_rz] + scalar von_Mises"""
        # Get stress expressions from formulation
        sigma_rr, sigma_zz, sigma_rz, von_mises = self.formulation.compute_stress_components(self.wh)
        
        # Project stress vector [σ_rr, σ_zz, σ_rz]
        self._project_stress_vector_to_dg0(sigma_rr, sigma_zz, sigma_rz, self.stress_vector_out)
        
        # Project von Mises scalar
        self._project_to_dg0(von_mises, self.von_mises_out)
        
        # Ensure ghost values are updated
        self.stress_vector_out.x.scatter_forward()
        self.von_mises_out.x.scatter_forward()
    
    def _project_stress_vector_to_dg0(self, sigma_rr, sigma_zz, sigma_rz, target_func):
        """Project stress components to DG0 vector function [σ_rr, σ_zz, σ_rz]"""
        V = target_func.function_space
        tau = ufl.TrialFunction(V)
        w = ufl.TestFunction(V)
        
        # Axisymmetric: need r * dx weighting
        x = ufl.SpatialCoordinate(self.domain)
        r = x[0]
        dx = ufl.Measure("dx", domain=self.domain)
        
        # Stress vector from UFL expressions
        stress_vector = ufl.as_vector([sigma_rr, sigma_zz, sigma_rz])
        
        # L2 projection: ∫ τ·w r dx = ∫ σ·w r dx
        a = ufl.inner(tau, w) * r * dx
        L = ufl.inner(stress_vector, w) * r * dx
        
        # Solve projection
        problem = LinearProblem(a, L, u=target_func,
                               petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                               petsc_options_prefix="consolid")
        problem.solve()
    
    def _project_to_dg0(self, expr, target_func):
        """Project expression to DG0 function using L2 projection with axisymmetric weighting"""
        V = target_func.function_space
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Axisymmetric: need r * dx weighting
        x = ufl.SpatialCoordinate(self.domain)
        r = x[0]
        
        dx = ufl.Measure("dx", domain=self.domain)
        a = ufl.inner(u, v) * r * dx
        L = ufl.inner(expr, v) * r * dx
        
        # Solve projection
        problem = LinearProblem(a, L, u=target_func, 
                               petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                               petsc_options_prefix="consolid"
                                )
        problem.solve()
    
    #
    #
    def _initialize_history(self):
        return {
            'times':                  [],
            'pressure_at_top':        [],
            'uz_at_bottom':           [],
            'volume_drained':         [],
            'analytical_pressure':    [],
            'analytical_uz':          [],
            'analytical_volume':      [],
            'pressure_error_percent': [],
            'uz_error_percent':       [],
            'volume_error_percent':   [],
        }
    
    #
    #
    def _collect_history(self, history, i_ts, time):
        fem_pressure = self._compute_pressure_at_top()
        fem_uz       = self._compute_uz_at_bottom()
        fem_volume   = self._compute_volume_drained()
        
        analytical_pressure = self.analytical.pressure_at_top(time)
        analytical_uz       = self.analytical.uz_at_bottom(time)
        analytical_volume   = self.analytical.volume_drained(time)
        
        pressure_error = 100 * abs(fem_pressure - analytical_pressure) / abs(analytical_pressure) if analytical_pressure != 0 else 0
        uz_error       = 100 * abs(fem_uz - analytical_uz) / abs(analytical_uz) if analytical_uz != 0 else 0
        volume_error   = 100 * abs(fem_volume - analytical_volume) / abs(analytical_volume) if analytical_volume != 0 else 0

#         print(f"FEM Pressure: {fem_pressure:.6e}, Analytical Pressure: {analytical_pressure:.6e}, Error: {pressure_error:.2f}%")

        history['times'].append(time)
        history['pressure_at_top'].append(fem_pressure)
        history['uz_at_bottom'].append(fem_uz)
        history['volume_drained'].append(fem_volume)
        history['analytical_pressure'].append(analytical_pressure)
        history['analytical_uz'].append(analytical_uz)
        history['analytical_volume'].append(analytical_volume)
        history['pressure_error_percent'].append(pressure_error)
        history['uz_error_percent'].append(uz_error)
        history['volume_error_percent'].append(volume_error)
    
    #
    #
    def _compute_pressure_at_top(self):
        H = self.cfg.mesh.H

        # Get pressure subspace and collapse
        p_space, p_map = self.W.sub(1).collapse()
        
        # Extract pressure values
        p_values = self.wh.x.array[p_map]
        
        # Get DOF coordinates for pressure space
        dof_coords = p_space.tabulate_dof_coordinates()
        z_coords = dof_coords[:, 1]
        
        # Find nodes at top (z ≈ 0)
        top_mask = np.abs(z_coords-H) < 1e-10
        
        if np.any(top_mask):
            return np.max(p_values[top_mask])
        return 0.0
    
    #
    #
    def _compute_uz_at_bottom(self):
        
        # Get displacement z-component subspace and collapse
        uz_space, uz_map = self.W.sub(0).sub(1).collapse()
        
        # Extract uz values
        uz_values = self.wh.x.array[uz_map]
        
        # Get DOF coordinates for uz space
        dof_coords = uz_space.tabulate_dof_coordinates()
        z_coords = dof_coords[:, 1]
        
        # Find nodes at top (z ≈ H)
        bottom_mask = np.abs(z_coords) < 1e-10
        
        if np.any(bottom_mask):
            return np.mean(uz_values[bottom_mask])
        return 0.0
    
    #
    #
    def _compute_volume_drained(self):
        """
        Volume variation is calculated as:
        ΔV = - 2π ∫_Ω (α*ε_v + p/M) * r dΩ
        """

        # Extract displacement and pressure from solution
        u, p = ufl.split(self.wh)
        u_old, p_old = ufl.split(self.wh_old)
        r = ufl.SpatialCoordinate(self.domain)[0]  # Radial coordinate

        cfg = self.cfg
        mu    = cfg.materials.mu
        S     = cfg.materials.S
        eta   = cfg.materials.eta
        sig0 = -1e5
        p0  = -sig0 * eta / mu / S
        Re = cfg.mesh.Re

        dx = ufl.Measure("dx", domain=self.domain, metadata={"quadrature_degree": 10})
        
        i1 = -(self.formulation.alpha * self.formulation.eps_v(u) + p / self.formulation.M) * 2 * np.pi * r
        form = fem.form(i1 * dx)
        l1 = fem.assemble_scalar(form)
        delta_V = self.comm.allreduce(l1, op=MPI.SUM)

        return delta_V
    
    #
    #
    def _write_timestep(self, time):
        mesh_degree = self.domain.geometry.cmap.degree
        cell_name = self.domain.topology.cell_name()

        # Interpolation spaces P2 => P1
        V_out = fem.functionspace(self.domain, element("Lagrange", cell_name, mesh_degree,shape=(self.gdim,)))
        Q_out = fem.functionspace(self.domain, element("Lagrange", cell_name, mesh_degree))

        # Fetch data
        wh = self.wh
        uh, ph = wh.split()
        urh, uzh = uh.split()
        sigma_rr_dg0, sigma_zz_dg0, sigma_rz_dg0 = self.stress_vector_out.split()


        # Setup export
        WRITE = { 
                 'U(r,z)' : { 'space': V_out, 'data': uh  },
                 'Ur'     : { 'space': Q_out, 'data': urh },
                 'Uz'     : { 'space': Q_out, 'data': uzh },
                 'P'      : { 'space': Q_out, 'data': ph },
                 'sigma_rr': {'space': Q_out, 'data': sigma_rr_dg0},
                 'sigma_zz': {'space': Q_out, 'data': sigma_zz_dg0},
                 'sigma_rz': {'space': Q_out, 'data': sigma_rz_dg0},
                }

        # Do the writing
        with io.XDMFFile(self.domain.comm, self.cfg.output.results, "a") as xdmf :
            for vname in WRITE :
                reg = WRITE[vname]
                _out = fem.Function(reg['space'], name=vname)
                _out.interpolate(reg['data'])  
                xdmf.write_function(_out, time)

    
    #
    #
    def _print_progress(self, i_ts, time, dt, history):
        if self.comm.rank == 0:
            print(f"Step {i_ts:4d} | t={time:8.3f}s (dt:{dt:6.3}s) | "
                  f"p_err={history['pressure_error_percent'][-1]:6.2f}% | "
                  f"u_err={history['uz_error_percent'][-1]:6.2f}% | "
                  f"vol_err={history['volume_error_percent'][-1]:6.2f}%")
    
    def _save_fem_timeseries(self):
        # Ensure output directory exists
        output_dir = os.path.dirname(self.cfg.output.timeseries)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        ds = xr.Dataset(
            {
                'pressure_at_top':        (['time'], self.history['pressure_at_top']),
                'uz_at_bottom':           (['time'], self.history['uz_at_bottom']),
                'volume_drained':         (['time'], self.history['volume_drained']),
                'analytical_pressure':    (['time'], self.history['analytical_pressure']),
                'analytical_uz':          (['time'], self.history['analytical_uz']),
                'analytical_volume':      (['time'], self.history['analytical_volume']),
                'pressure_error_percent': (['time'], self.history['pressure_error_percent']),
                'uz_error_percent':       (['time'], self.history['uz_error_percent']),
                'volume_error_percent':   (['time'], self.history['volume_error_percent']),
            },
            coords={'time': self.history['times']},
            attrs={
                'E':     self.cfg.materials.E,
                'nu':    self.cfg.materials.nu,
                'alpha': self.cfg.materials.alpha,
                'perm':  self.cfg.materials.perm,
                'visc':  self.cfg.materials.visc,
                'M':     self.cfg.materials.M,
                'Re':    self.cfg.mesh.Re,
                'H':     self.cfg.mesh.H,
            }
        )
        
        if self.comm.rank == 0:
            ds.to_netcdf(self.cfg.output.timeseries)
            print(f"\n✓ FEM timeseries saved: {self.cfg.output.timeseries}")
    
    #
    #
    def summary(self):
        if self.history is None:
            return "No results available. Run solve() first."
        
        lines = []
        lines.append("\n" + "="*70)
        lines.append("SIMULATION RESULTS")
        lines.append("="*70)
        lines.append(f"\nTimeseries: {self.cfg.output.timeseries}")
        lines.append(f"\nFinal State (t={self.history['times'][-1]:.3f}s):")
        lines.append(f"  Pressure at top:  {self.history['pressure_at_top'][-1]:.6e} Pa")
        lines.append(f"  Displacement at bottom: {self.history['uz_at_bottom'][-1]:.6e} m")
        lines.append(f"  Volume drained:      {self.history['volume_drained'][-1]:.6e} m³")
        lines.append(f"\nErrors vs Analytical:")
        lines.append(f"  Pressure error:      {self.history['pressure_error_percent'][-1]:.2f}%")
        lines.append(f"  Displacement error:  {self.history['uz_error_percent'][-1]:.2f}%")
        lines.append(f"  Volume error:        {self.history['volume_error_percent'][-1]:.2f}%")
        lines.append("="*70)
        return "\n".join(lines)

    #
    #
    def _save_run_summary(self) :
        data = {
            'run_id': self.cfg.general.run_id,
            'run_dir': self.cfg.general.run_dir,
            'description': self.cfg.general.description,
            'tags': self.cfg.general.tags,
            'E': self.cfg.materials.E,
            'nu': self.cfg.materials.nu,
            'alpha': self.cfg.materials.alpha,
            'perm': self.cfg.materials.perm,
            'visc': self.cfg.materials.visc,
            'M': self.cfg.materials.M,
            'Re': self.cfg.mesh.Re,
            'H': self.cfg.mesh.H,
            'Re/H': self.cfg.mesh.Re/self.cfg.mesh.H,
            'final_time': self.history['times'][-1],
            'pressure_at_top': self.history['pressure_at_top'][-1],
            'uz_at_bottom': self.history['uz_at_bottom'][-1],
            'volume_drained': self.history['volume_drained'][-1],
            'pressure_error_percent': self.history['pressure_error_percent'][-1],
            'uz_error_percent': self.history['uz_error_percent'][-1],
            'volume_error_percent': self.history['volume_error_percent'][-1],
        }
        import pandas as pd
        df = pd.DataFrame([data])

        output_dir = os.path.dirname(self.cfg.output.results)
        df.to_pickle(f'{output_dir}/summary.pkl')        
