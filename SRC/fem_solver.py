import os
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element, mixed_element
from ufl import *
import ufl
import netCDF4 as nc4
from dolfinx.io import VTXWriter

        
import config
from formulation import PoroelasticityFormulation
from analytical import Analytical1DConsolidation
from export_pvd import PVDExporter
from vertical_line_sampler import VerticalLineSampler


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
        self._setup_vertical_line_sampler()

        self.history = None


    
    
    #
    #
    def _setup_vtk_output(self):
        # All output functions use Lagrange degree 1 so VTXWriter accepts them together.
        cell_name = self.domain.topology.cell_name()
        V_out = fem.functionspace(self.domain, element("Lagrange", cell_name, 1, shape=(self.gdim,)))
        Q_out = fem.functionspace(self.domain, element("Lagrange", cell_name, 1))

        self.u_out         = fem.Function(V_out, name="Displacement")
        self.p_out         = fem.Function(Q_out, name="Pressure")
        self.sigma_rr_out  = fem.Function(Q_out, name="sigma_rr")
        self.sigma_zz_out  = fem.Function(Q_out, name="sigma_zz")
        self.sigma_rz_out  = fem.Function(Q_out, name="sigma_rz")
        self.von_mises_out = fem.Function(Q_out, name="von_Mises")

        output_path = self.cfg.output.results
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.vtx_writer = io.VTXWriter(
            self.domain.comm,
            output_path,
            [self.u_out, self.p_out,
             self.sigma_rr_out, self.sigma_zz_out, self.sigma_rz_out,
             self.von_mises_out],
            engine="BP4",
        )

        if self.comm.rank == 0:
            print(f"✓ Results output configured: {output_path}")
    
    def _setup_vertical_line_sampler(self):
        """Initialize vertical line pressure sampler for centerline (r=0)."""
        num_points = self.cfg.output.n_profile_points
        self.v_sampler = VerticalLineSampler(self.W, self.domain, self.cfg, num_points=num_points)
        
        if self.comm.rank == 0:
            print(f"✓ Vertical line sampler configured: {num_points} points along centerline")
    
    #
    #
    def solve(self):
        if self.comm.rank == 0:
            print("\n" + "="*70)
            print("STARTING TIME INTEGRATION")
            print("="*70)
        
        history = self._initialize_history()
        prev_loads = {}
        self._open_timeseries_nc()
        self._open_pressure_profile_nc()

        #
        # Timestep loop
        #
        for i_ts, dt, time in self.cfg.numerical:
            prev_loads = self._print_load_changes(time, prev_loads)
            self._solve_timestep(dt, time)
            self._compute_and_project_stresses()
            self._collect_history(history, i_ts, time)
            self._write_timestep(time)
            self._append_timeseries_record(history, time)
            self._append_pressure_profile_record(time)
            self._print_progress(i_ts, time, dt, history)
            self.wh_old.x.array[:] = self.wh.x.array[:]

        self.history = history

        if self.comm.rank == 0:
            print("="*70)
            print("TIME INTEGRATION COMPLETED")
            print("="*70)

        self.vtx_writer.close()
        self._close_timeseries_nc()
        self._close_pressure_profile_nc()

        self._save_run_summary()

    #
    #
    def _solve_timestep(self, dt, t: float = 0.0):
        a, L = self.formulation.weak_form(dt, t, self.wh_old)
        
        # Assemble the system
        A = fem.petsc.assemble_matrix(fem.form(a), bcs=self.bcs)

        b = fem.petsc.assemble_vector(fem.form(L))
        fem.petsc.apply_lifting(b, [fem.form(a)], [self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, self.bcs)

        A.assemble()

        self.formulation.set_rigid_constraints( A )

        A.assemble()

        # Create solver: GMRES with LU (MUMPS) preconditioner
        solver = PETSc.KSP().create(self.mesh.comm)
        solver.setOptionsPrefix("main_ksp_")
        solver.setOperators(A)

        opts   = PETSc.Options()
        prefix = solver.getOptionsPrefix()
        opts[f"{prefix}ksp_type"]                  = "gmres"
        opts[f"{prefix}pc_type"]                   = "lu"
        opts[f"{prefix}pc_factor_mat_solver_type"] = "mumps"
        opts[f"{prefix}ksp_rtol"]                  = 1e-10
        opts[f"{prefix}ksp_atol"]                  = 1e-12
        opts[f"{prefix}ksp_max_it"]                = 1000
        solver.setFromOptions()
        solver.solve(b, self.wh.x.petsc_vec)

        # Sync all processors
        self.wh.x.scatter_forward()

    #
    #
    def _compute_and_project_stresses(self):
        sigma_rr, sigma_zz, sigma_rz, von_mises = self.formulation.compute_stress_components(self.wh)
        self._project_scalar(sigma_rr,  self.sigma_rr_out)
        self._project_scalar(sigma_zz,  self.sigma_zz_out)
        self._project_scalar(sigma_rz,  self.sigma_rz_out)
        self._project_scalar(von_mises, self.von_mises_out)
        for f in [self.sigma_rr_out, self.sigma_zz_out, self.sigma_rz_out, self.von_mises_out]:
            f.x.scatter_forward()

    def _project_scalar(self, expr, target_func):
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
            'sig_zz_applied':         [],
            'pressure_at_top':        [],
            'pressure_mean':          [],
            'pressure_p10':           [],
            'pressure_p90':           [],
            'uz_at_bottom':           [],
            'uz_at_top':              [],
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
        fem_pressure  = self._compute_pressure_at_top()
        fem_uz        = self._compute_uz_at_bottom()
        fem_uz_top    = self._compute_uz_at_top()
        fem_volume    = self._compute_volume_drained()
        p_mean, p_p10, p_p90 = self._compute_pressure_statistics()
        sig_zz_applied = self._get_applied_sig_zz(time)
        
        analytical_pressure = self.analytical.pressure_at_top(time)
        analytical_uz       = self.analytical.uz_at_bottom(time)
        analytical_volume   = self.analytical.volume_drained(time)
        
        _p_ref  = abs(analytical_pressure)
        _u_ref  = abs(analytical_uz)
        _v_ref  = abs(analytical_volume)
        _tol    = 1e-30   # treat reference as zero below this magnitude
        pressure_error = 100 * abs(fem_pressure - analytical_pressure) / _p_ref if _p_ref > _tol else float('nan')
        uz_error       = 100 * abs(fem_uz       - analytical_uz)       / _u_ref if _u_ref > _tol else float('nan')
        volume_error   = 100 * abs(fem_volume   - analytical_volume)   / _v_ref if _v_ref > _tol else float('nan')

#         print(f"FEM Pressure: {fem_pressure:.6e}, Analytical Pressure: {analytical_pressure:.6e}, Error: {pressure_error:.2f}%")

        history['times'].append(time)
        history['sig_zz_applied'].append(sig_zz_applied)
        history['pressure_at_top'].append(fem_pressure)
        history['pressure_mean'].append(p_mean)
        history['pressure_p10'].append(p_p10)
        history['pressure_p90'].append(p_p90)
        history['uz_at_bottom'].append(fem_uz)
        history['uz_at_top'].append(fem_uz_top)
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

    def _compute_pressure_statistics(self):
        """Return (mean, P10, P90) of pressure over interior DOFs only.
        Excludes the drainage boundary (z≈0, P=0 Dirichlet) so that
        P10 reflects the true low-pressure region, not the forced boundary."""
        p_space, p_map = self.W.sub(1).collapse()
        p_values       = self.wh.x.array[p_map]
        dof_coords     = p_space.tabulate_dof_coordinates()
        interior_mask  = dof_coords[:, 1] > 1e-10   # exclude z=0 drainage face
        p_int = p_values[interior_mask]
        return (float(np.mean(p_int)),
                float(np.percentile(p_int, 10)),
                float(np.percentile(p_int, 90)))

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

    def _compute_uz_at_top(self):
        H = self.cfg.mesh.H
        uz_space, uz_map = self.W.sub(0).sub(1).collapse()
        uz_values = self.wh.x.array[uz_map]
        dof_coords = uz_space.tabulate_dof_coordinates()
        z_coords = dof_coords[:, 1]
        top_mask = np.abs(z_coords - H) < 1e-10
        if np.any(top_mask):
            return np.mean(uz_values[top_mask])
        return 0.0

    def _get_applied_sig_zz(self, time: float) -> float:
        """Return the net sig_zz currently applied across all loaded surfaces."""
        total = 0.0
        for surface in ['bottom', 'right', 'top', 'left']:
            bc_spec = getattr(self.cfg.boundary_conditions, surface)
            if bc_spec.periodic_load is not None:
                total += bc_spec.periodic_load.eval(time)
            elif bc_spec.sig_zz is not None:
                total += bc_spec.sig_zz
        return total

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

        dx = ufl.Measure("dx", domain=self.domain, metadata={"quadrature_degree": 10})
        
        i1 = -(self.formulation.alpha * self.formulation.eps_v(u) + p / self.formulation.M) * 2 * np.pi * r
        form = fem.form(i1 * dx)
        l1 = fem.assemble_scalar(form)
        delta_V = self.comm.allreduce(l1, op=MPI.SUM)

        return delta_V
    
    #
    #
    def _write_timestep(self, time):
        uh, ph = self.wh.split()
        self.u_out.interpolate(uh)
        self.p_out.interpolate(ph)
        # stress_vector_out / von_mises_out already updated by _compute_and_project_stresses()
        self.vtx_writer.write(time)

    
    #
    #
    def _print_load_changes(self, time, prev_loads):
        """Print a line whenever a periodic load switches level. Returns updated load dict."""
        current_loads = {}
        first_call = len(prev_loads) == 0
        for surface in ['bottom', 'right', 'top', 'left']:
            bc_spec = getattr(self.cfg.boundary_conditions, surface)
            if bc_spec.periodic_load is None:
                continue
            val = bc_spec.periodic_load.eval(time)
            current_loads[surface] = val
            if not first_call and val != prev_loads.get(surface):
                if self.comm.rank == 0:
                    prev = prev_loads.get(surface)
                    prev_str = f"{prev:.3e} Pa" if prev is not None else "none"
                    print(f"  >> LOAD CHANGE at t={time:.4f}s | {surface} sig_zz: {prev_str} → {val:.3e} Pa")
        return current_loads

    def _print_progress(self, i_ts, time, dt, history):
        if self.comm.rank == 0:
            def _fmt(v):
                return f"{v:6.2f}%" if v == v else "   n/a"   # nan != nan
            print(f"Step {i_ts:4d} | t={time:8.3f}s (dt:{dt:6.3}s) | "
                  f"p_err={_fmt(history['pressure_error_percent'][-1])} | "
                  f"u_err={_fmt(history['uz_error_percent'][-1])} | "
                  f"vol_err={_fmt(history['volume_error_percent'][-1])}")
    
    _NC_VARS = [
        'sig_zz_applied',
        'pressure_at_top', 'pressure_mean', 'pressure_p10', 'pressure_p90',
        'uz_at_bottom', 'uz_at_top',
        'volume_drained', 'analytical_pressure', 'analytical_uz',
        'analytical_volume', 'pressure_error_percent', 'uz_error_percent',
        'volume_error_percent',
    ]

    def _open_timeseries_nc(self):
        if self.comm.rank != 0:
            return
        path = self.cfg.output.timeseries
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ds = nc4.Dataset(path, 'w', format='NETCDF3_64BIT_OFFSET')
        ds.createDimension('time', None)   # unlimited — grows each step
        ds.createVariable('time', 'f8', ('time',))
        for name in self._NC_VARS:
            ds.createVariable(name, 'f8', ('time',))
        # Global attributes
        for k, v in [('E', self.cfg.materials.E), ('nu', self.cfg.materials.nu),
                     ('alpha', self.cfg.materials.alpha), ('perm', self.cfg.materials.perm),
                     ('visc', self.cfg.materials.visc), ('M', self.cfg.materials.M),
                     ('Re', self.cfg.mesh.Re), ('H', self.cfg.mesh.H)]:
            setattr(ds, k, v)
        self._nc_ds  = ds
        self._nc_idx = 0
        print(f"✓ Timeseries NC opened: {path}")

    def _append_timeseries_record(self, history, time):
        if self.comm.rank != 0:
            return
        i = self._nc_idx
        self._nc_ds['time'][i] = time
        for name in self._NC_VARS:
            self._nc_ds[name][i] = history[name][-1]
        self._nc_ds.sync()   # flush to disk immediately
        self._nc_idx += 1

    def _close_timeseries_nc(self):
        if self.comm.rank != 0:
            return
        self._nc_ds.close()
        print(f"\n✓ FEM timeseries saved: {self.cfg.output.timeseries}")
    
    def _open_pressure_profile_nc(self):
        """Open NetCDF file for vertical line pressure profile."""
        if self.comm.rank != 0:
            return
        path = self.cfg.output.pressure_profile
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ds = nc4.Dataset(path, 'w', format='NETCDF3_64BIT_OFFSET')
        
        # Create dimensions
        num_points = self.v_sampler.num_points
        ds.createDimension('z_point', num_points)
        ds.createDimension('time', None)  # unlimited
        
        # Create variables
        z_var = ds.createVariable('z_coord', 'f8', ('z_point',))
        time_var = ds.createVariable('time', 'f8', ('time',))
        pressure_var = ds.createVariable('pressure', 'f8', ('time', 'z_point'))
        
        # Store z-coordinates (constant across all timesteps)
        z_coord = self.v_sampler.get_z_coordinates()
        z_var[:] = z_coord
        
        # Add attributes
        z_var.units = 'm'
        z_var.long_name = 'Height along centerline (r=0)'
        time_var.units = 's'
        time_var.long_name = 'Time'
        pressure_var.units = 'Pa'
        pressure_var.long_name = 'Pore pressure along vertical centerline'
        
        # Store material parameters as global attributes
        for k, v in [('E', self.cfg.materials.E), ('nu', self.cfg.materials.nu),
                     ('alpha', self.cfg.materials.alpha), ('perm', self.cfg.materials.perm),
                     ('visc', self.cfg.materials.visc), ('M', self.cfg.materials.M),
                     ('Re', self.cfg.mesh.Re), ('H', self.cfg.mesh.H)]:
            setattr(ds, k, v)
        
        self._pp_ds = ds
        self._pp_idx = 0
        print(f"✓ Pressure profile NC opened: {path}")
    
    def _append_pressure_profile_record(self, time):
        """Append pressure profile at current timestep."""
        if self.comm.rank != 0:
            return
        
        # Get pressure DOFs and map
        _, p_map = self.W.sub(1).collapse()
        p_values = self.wh.x.array[p_map]
        
        # Sample pressure along centerline
        pressure_profile = self.v_sampler.sample(p_values)
        
        # Append to NetCDF
        i = self._pp_idx
        self._pp_ds['time'][i] = time
        self._pp_ds['pressure'][i, :] = pressure_profile
        self._pp_ds.sync()
        self._pp_idx += 1
    
    def _close_pressure_profile_nc(self):
        """Close pressure profile NetCDF file."""
        if self.comm.rank != 0:
            return
        self._pp_ds.close()
        print(f"✓ Pressure profile saved: {self.cfg.output.pressure_profile}")
    
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
