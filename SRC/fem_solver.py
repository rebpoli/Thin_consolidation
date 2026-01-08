import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element, mixed_element
import xarray as xr

import config
from formulation import PoroelasticityFormulation
from analytical import Analytical1DConsolidation


class PoroelasticitySolver:
    def __init__(self, mesh):
        self.mesh   = mesh
        self.domain = mesh.domain
        self.facets = mesh.facets
        self.comm   = mesh.comm
        
        cfg = config.get()
        self.cfg = cfg
        
        self.analytical = Analytical1DConsolidation()
        
        self._setup_function_spaces()
        
        self.formulation = PoroelasticityFormulation(
            self.domain, 
            self.W, 
            self.facets
        )
        
        self._setup_problem()
        self._setup_vtk_output()
        
        self.history = None
    
    def _setup_function_spaces(self):
        gdim      = self.domain.geometry.dim
        cell_name = self.domain.topology.cell_name()
        
        V_elem = element("Lagrange", cell_name, 2, shape=(gdim,))
        Q_elem = element("Lagrange", cell_name, 1)
        
        mixed_elem = mixed_element([V_elem, Q_elem])
        
        self.W    = fem.functionspace(self.domain, mixed_elem)
        self.gdim = gdim
        
        if self.comm.rank == 0:
            ndofs = self.W.dofmap.index_map.size_global
            print(f"✓ Function space created: {ndofs} total DOFs")
    
    def _setup_problem(self):
        self.wh     = fem.Function(self.W, name="Solution")
        self.wh_old = fem.Function(self.W, name="Previous")
        
        self.bcs = self.formulation.setup_boundary_conditions()
        
        if self.comm.rank == 0:
            print(f"✓ Boundary conditions created: {len(self.bcs)} BCs applied")
    
    def _setup_vtk_output(self):
        u_space, _ = self.W.sub(0).collapse()
        p_space, _ = self.W.sub(1).collapse()
        
        self.u_out = fem.Function(u_space, name="Displacement")
        self.p_out = fem.Function(p_space, name="Pressure")
        
        vtk_base = self.cfg.output.vtk_file.replace(".pvd", "")
        
        u_file = vtk_base + "_displacement.bp"
        p_file = vtk_base + "_pressure.bp"
        
        self.u_writer = io.VTXWriter(self.comm, u_file, [self.u_out], engine="BP4")
        self.p_writer = io.VTXWriter(self.comm, p_file, [self.p_out], engine="BP4")
        
        if self.comm.rank == 0:
            print(f"✓ VTK output configured:")
            print(f"  Displacement (P2): {u_file}")
            print(f"  Pressure (P1):     {p_file}")
    
    def solve(self):
        if self.comm.rank == 0:
            print("\n" + "="*70)
            print("STARTING TIME INTEGRATION")
            print("="*70)
        
        history = self._initialize_history()
        
        for i_ts, dt, time in self.cfg.numerical:
            self._solve_timestep(dt)
            self._collect_history(history, i_ts, time)
            self._write_vtk(time)
            self._print_progress(i_ts, time, history)
            
            self.wh_old.x.array[:] = self.wh.x.array[:]
        
        self.u_writer.close()
        self.p_writer.close()
        
        self.history = history
        
        if self.comm.rank == 0:
            print("="*70)
            print("TIME INTEGRATION COMPLETED")
            print("="*70)
        
        self._save_fem_timeseries()
    
    def _solve_timestep(self, dt):
        a, L = self.formulation.weak_form(dt, self.wh_old)
        
        problem = LinearProblem(
            a, L,
            bcs=self.bcs,
            u=self.wh,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="consolidation_"
        )
        
        problem.solve()
    
    def _initialize_history(self):
        return {
            'times':                  [],
            'pressure_at_bottom':     [],
            'uz_at_top':              [],
            'volume_drained':         [],
            'analytical_pressure':    [],
            'analytical_uz':          [],
            'analytical_volume':      [],
            'pressure_error_percent': [],
            'uz_error_percent':       [],
            'volume_error_percent':   [],
        }
    
    def _collect_history(self, history, i_ts, time):
        fem_pressure = self._compute_pressure_at_bottom()
        fem_uz       = self._compute_uz_at_top()
        fem_volume   = self._compute_volume_drained()
        
        analytical_pressure = self.analytical.pressure_at_bottom(time)
        analytical_uz       = self.analytical.uz_at_top(time)
        analytical_volume   = self.analytical.volume_drained(time)
        
        pressure_error = 100 * abs(fem_pressure - analytical_pressure) / abs(analytical_pressure) if analytical_pressure != 0 else 0
        uz_error       = 100 * abs(fem_uz - analytical_uz) / abs(analytical_uz) if analytical_uz != 0 else 0
        volume_error   = 100 * abs(fem_volume - analytical_volume) / abs(analytical_volume) if analytical_volume != 0 else 0
        
        history['times'].append(time)
        history['pressure_at_bottom'].append(fem_pressure)
        history['uz_at_top'].append(fem_uz)
        history['volume_drained'].append(fem_volume)
        history['analytical_pressure'].append(analytical_pressure)
        history['analytical_uz'].append(analytical_uz)
        history['analytical_volume'].append(analytical_volume)
        history['pressure_error_percent'].append(pressure_error)
        history['uz_error_percent'].append(uz_error)
        history['volume_error_percent'].append(volume_error)
    
    def _compute_pressure_at_bottom(self):
        # Get pressure subspace and collapse
        p_space, p_map = self.W.sub(1).collapse()
        
        # Extract pressure values
        p_values = self.wh.x.array[p_map]
        
        # Get DOF coordinates for pressure space
        dof_coords = p_space.tabulate_dof_coordinates()
        z_coords = dof_coords[:, 1]
        
        # Find nodes at bottom (z ≈ 0)
        bottom_mask = np.abs(z_coords) < 1e-10
        
        if np.any(bottom_mask):
            return np.mean(p_values[bottom_mask])
        return 0.0
    
    def _compute_uz_at_top(self):
        H = self.cfg.mesh.H
        
        # Get displacement z-component subspace and collapse
        uz_space, uz_map = self.W.sub(0).sub(1).collapse()
        
        # Extract uz values
        uz_values = self.wh.x.array[uz_map]
        
        # Get DOF coordinates for uz space
        dof_coords = uz_space.tabulate_dof_coordinates()
        z_coords = dof_coords[:, 1]
        
        # Find nodes at top (z ≈ H)
        top_mask = np.abs(z_coords - H) < 1e-10
        
        if np.any(top_mask):
            return np.mean(uz_values[top_mask])
        return 0.0
    
    def _compute_volume_drained(self):
        return 0.0
    
    def _write_vtk(self, time):
        u_dofs = self.W.sub(0).collapse()[1]
        p_dofs = self.W.sub(1).collapse()[1]
        
        self.u_out.x.array[:] = self.wh.x.array[u_dofs]
        self.p_out.x.array[:] = self.wh.x.array[p_dofs]
        
        self.u_writer.write(time)
        self.p_writer.write(time)
    
    def _print_progress(self, i_ts, time, history):
        if self.comm.rank == 0:
            print(f"Step {i_ts:4d} | t={time:8.3f}s | "
                  f"p_err={history['pressure_error_percent'][-1]:6.2f}% | "
                  f"u_err={history['uz_error_percent'][-1]:6.2f}%")
    
    def _save_fem_timeseries(self):
        ds = xr.Dataset(
            {
                'pressure_at_bottom':     (['time'], self.history['pressure_at_bottom']),
                'uz_at_top':              (['time'], self.history['uz_at_top']),
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
                'sig0':  self.cfg.materials.sig0,
                'Re':    self.cfg.mesh.Re,
                'H':     self.cfg.mesh.H,
            }
        )
        
        if self.comm.rank == 0:
            ds.to_netcdf(self.cfg.output.fem_timeseries_file)
            print(f"\n✓ FEM timeseries saved: {self.cfg.output.fem_timeseries_file}")
    
    def summary(self):
        if self.history is None:
            return "No results available. Run solve() first."
        
        lines = []
        lines.append("\n" + "="*70)
        lines.append("SIMULATION RESULTS")
        lines.append("="*70)
        lines.append(f"\nOutputs:")
        vtk_base = self.cfg.output.vtk_file.replace(".pvd", "")
        lines.append(f"  Displacement (P2): {vtk_base}_displacement.bp/")
        lines.append(f"  Pressure (P1):     {vtk_base}_pressure.bp/")
        lines.append(f"  Timeseries:        {self.cfg.output.fem_timeseries_file}")
        lines.append(f"\nFinal State (t={self.history['times'][-1]:.3f}s):")
        lines.append(f"  Pressure at bottom:  {self.history['pressure_at_bottom'][-1]:.6e} Pa")
        lines.append(f"  Displacement at top: {self.history['uz_at_top'][-1]:.6e} m")
        lines.append(f"  Volume drained:      {self.history['volume_drained'][-1]:.6e} m³")
        lines.append(f"\nErrors vs Analytical:")
        lines.append(f"  Pressure error:      {self.history['pressure_error_percent'][-1]:.2f}%")
        lines.append(f"  Displacement error:  {self.history['uz_error_percent'][-1]:.2f}%")
        lines.append(f"  Volume error:        {self.history['volume_error_percent'][-1]:.2f}%")
        lines.append(f"\nOpen in ParaView:")
        lines.append(f"  {vtk_base}_displacement.bp/  (or)")
        lines.append(f"  {vtk_base}_pressure.bp/")
        lines.append("="*70)
        return "\n".join(lines)
