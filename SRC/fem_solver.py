"""Poroelastic consolidation FEM solver — time integration and diagnostics.

Orchestrates the full simulation:
  1. Builds P2/P1 mixed function spaces.
  2. Delegates weak-form assembly to PoroelasticityFormulation.
  3. Solves the coupled system with PETSc GMRES + MUMPS at each timestep.
  4. Projects stresses to Lagrange-1 functions for VTX output.
  5. Tracks scalar diagnostics (pressure, displacement, volume) and compares
     against the Terzaghi 1D analytical solution.
  6. Streams results to VTX (ParaView), NetCDF timeseries, pressure profile,
     and stress-invariant files via the output_writers module.
"""

import os
import numpy as np
import pandas as pd

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element, mixed_element
import ufl

from config import CONFIG, SURFACES
from formulation import PoroelasticityFormulation
from analytical import Analytical1DConsolidation
from vertical_line_sampler import VerticalLineSampler
from output_writers import OutputManager


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class PoroelasticitySolver:
    """Time-integration driver for the coupled poroelastic system.

    Attributes exposed after solve():
        history  — dict of scalar time-series arrays (pressures, displacements,
                   volumes, errors).  Keys match _HISTORY_KEYS.
    """

    # Keys shared between _initialize_history, _collect_history, _record_timestep
    _HISTORY_KEYS = [
        'times',
        'sig_zz_applied',
        'pressure_at_base', 'pressure_mean', 'pressure_p10', 'pressure_p90',
        'uz_at_bottom', 'uz_at_top',
        'volume_drained',
        'analytical_pressure', 'analytical_uz', 'analytical_volume',
        'pressure_error_percent', 'uz_error_percent', 'volume_error_percent',
    ]

    def __init__(self, mesh):
        self.mesh   = mesh
        self.domain = mesh.domain
        self.facets = mesh.facets
        self.comm   = mesh.comm
        self.gdim   = self.domain.geometry.dim

        self.analytical = Analytical1DConsolidation()

        # Mixed function space: P2 for displacement, P1 for pressure
        cell_name = self.domain.topology.cell_name()
        P2        = element("Lagrange", cell_name, 2, shape=(self.gdim,))
        P1        = element("Lagrange", cell_name, 1)
        self.W    = fem.functionspace(self.domain, mixed_element([P2, P1]))

        self.formulation = PoroelasticityFormulation(self.domain, self.W, self.facets)

        self.wh     = fem.Function(self.W, name="Solution")
        self.wh_old = fem.Function(self.W, name="Previous")
        self.bcs    = self.formulation.setup_boundary_conditions()

        v_sampler    = VerticalLineSampler(self.W, self.domain)
        self._output = OutputManager(self.comm, self.domain, v_sampler)

        self._setup_output_functions()

        self._prev_loads = {}   # tracks load values for change-detection in console output
        self.history     = None

    # ------------------------------------------------------------------
    # Output setup
    # ------------------------------------------------------------------

    def _setup_output_functions(self):
        """Create Lagrange-1 output functions and open the VTX writer."""
        cell_name = self.domain.topology.cell_name()
        V_out = fem.functionspace(self.domain, element("Lagrange", cell_name, 1, shape=(self.gdim,)))
        Q_out = fem.functionspace(self.domain, element("Lagrange", cell_name, 1))

        self.u_out              = fem.Function(V_out, name="Displacement")
        self.p_out              = fem.Function(Q_out, name="Pressure")
        self.sigma_rr_out       = fem.Function(Q_out, name="sigma_rr")
        self.sigma_tt_out       = fem.Function(Q_out, name="sigma_tt")
        self.sigma_zz_out       = fem.Function(Q_out, name="sigma_zz")
        self.sigma_rz_out       = fem.Function(Q_out, name="sigma_rz")
        self.von_mises_out      = fem.Function(Q_out, name="von_Mises")
        self.sig_rr_eff_terz_out = fem.Function(Q_out, name="sig_rr_eff_terz")
        self.sig_zz_eff_terz_out = fem.Function(Q_out, name="sig_zz_eff_terz")

        self._output.configure_vtx(
            self.domain.comm,
            [self.u_out, self.p_out,
             self.sigma_rr_out, self.sigma_tt_out, self.sigma_zz_out,
             self.sigma_rz_out, self.von_mises_out,
             self.sig_rr_eff_terz_out, self.sig_zz_eff_terz_out],
            self.sigma_rr_out, self.sigma_tt_out, self.sigma_zz_out,
            self.sigma_rz_out, self.p_out, self.u_out,
        )

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(self):
        """Run the full time integration and write all output files."""
        if self.comm.rank == 0:
            print("\n" + "=" * 70)
            print("STARTING TIME INTEGRATION")
            print("=" * 70)

        history = self._initialize_history()

        self._output.open()

        for i_ts, dt, time in CONFIG.timestepper:
            self._print_load_changes(time)
            self._solve_timestep(dt, time)
            self._compute_and_project_stresses()
            self._collect_history(history, time)
            self._record_timestep(time, history)
            self._print_progress(i_ts, time, dt, history)
            self.wh_old.x.array[:] = self.wh.x.array[:]

        self.history = history

        if self.comm.rank == 0:
            print("=" * 70)
            print("TIME INTEGRATION COMPLETED")
            print("=" * 70)

        self._output.close()
        self._save_run_summary()

    # ------------------------------------------------------------------
    # Linear system assembly and solve
    # ------------------------------------------------------------------

    def _solve_timestep(self, dt, t: float = 0.0):
        """Assemble and solve the coupled poroelastic system for one timestep.

        Uses PETSc GMRES preconditioned with LU (MUMPS) via the
        'main_ksp_' options prefix.  Rigid constraints are applied as
        penalty entries on the assembled matrix before the solve.
        """
        a, L = self.formulation.weak_form(dt, t, self.wh_old)

        A = fem.petsc.assemble_matrix(fem.form(a), bcs=self.bcs)
        b = fem.petsc.assemble_vector(fem.form(L))
        fem.petsc.apply_lifting(b, [fem.form(a)], [self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, self.bcs)
        A.assemble()

        self.formulation.set_rigid_constraints(A)
        A.assemble()

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
        self.wh.x.scatter_forward()

    # ------------------------------------------------------------------
    # Stress projection
    # ------------------------------------------------------------------

    def _compute_and_project_stresses(self):
        """Project UFL stress expressions to Lagrange-1 and interpolate u, p."""
        sigma_rr, sigma_tt, sigma_zz, sigma_rz, von_mises = \
            self.formulation.compute_stress_components(self.wh)
        _, p_ufl = ufl.split(self.wh)
        for expr, func in [
            (sigma_rr,          self.sigma_rr_out),
            (sigma_tt,          self.sigma_tt_out),
            (sigma_zz,          self.sigma_zz_out),
            (sigma_rz,          self.sigma_rz_out),
            (von_mises,         self.von_mises_out),
            (sigma_rr + p_ufl,  self.sig_rr_eff_terz_out),
            (sigma_zz + p_ufl,  self.sig_zz_eff_terz_out),
        ]:
            self._project_scalar(expr, func)
            func.x.scatter_forward()

        uh, ph = self.wh.split()
        self.u_out.interpolate(uh)
        self.p_out.interpolate(ph)

    #
    #
    def _project_scalar(self, expr, target_func):
        """L2-project a UFL scalar expression onto target_func's function space.

        Uses r·dΩ axisymmetric weighting and a direct LU solve (preonly/lu).
        """
        V  = target_func.function_space
        u  = ufl.TrialFunction(V)
        v  = ufl.TestFunction(V)
        r  = ufl.SpatialCoordinate(self.domain)[0]
        dx = ufl.Measure("dx", domain=self.domain)
        problem = LinearProblem(
            ufl.inner(u, v) * r * dx,
            ufl.inner(expr, v) * r * dx,
            u=target_func,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="consolid",
        )
        problem.solve()

    # ------------------------------------------------------------------
    # History / diagnostics
    # ------------------------------------------------------------------

    #
    #
    def _initialize_history(self):
        return {k: [] for k in self._HISTORY_KEYS}

    def _drainage_z(self) -> float:
        """Return the z-coordinate of the drainage face (where Pressure BC = 0).

        Checks top (z=H/2) then bottom (z=0).  Returns None if no drainage face found.
        """
        H_half = CONFIG.mesh.H / 2
        bc = CONFIG.boundary_conditions
        if bc.top.Pressure    is not None:
            return H_half
        if bc.bottom.Pressure is not None:
            return 0.0
        return None

    def _collect_history(self, history, time):
        """Evaluate FEM and analytical quantities and append to history dict."""
        H_half    = CONFIG.mesh.H / 2
        drain_z   = self._drainage_z()
        nondrain_z = 0.0 if drain_z == H_half else H_half

        # Pressure: maximum at the no-flux (non-drainage) face
        fem_pressure = self._field_at_z(self.W.sub(1),        nondrain_z, 'max')
        fem_uz_bot   = self._field_at_z(self.W.sub(0).sub(1), 0.0,        'mean')
        fem_uz_top   = self._field_at_z(self.W.sub(0).sub(1), H_half,     'mean')
        fem_volume   = self._compute_volume_drained()
        p_mean, p_p10, p_p90 = self._pressure_statistics()
        sig_zz_applied = self._applied_sig_zz(time)

        an_p = self.analytical.pressure_at_base(time)
        an_u = self.analytical.uz_at_top(time)
        an_v = self.analytical.volume_drained(time)

        # Settlement: uz at the drainage/loaded face is negative (compression-negative);
        # the analytical formula gives a positive magnitude.
        fem_uz_drain   = fem_uz_top if drain_z == H_half else fem_uz_bot
        fem_settlement = -fem_uz_drain

        tol = 1e-30
        p_err = 100 * abs(fem_pressure   - an_p) / abs(an_p) if abs(an_p) > tol else float('nan')
        u_err = 100 * abs(fem_settlement - an_u) / abs(an_u) if abs(an_u) > tol else float('nan')
        v_err = 100 * abs(fem_volume     - an_v) / abs(an_v) if abs(an_v) > tol else float('nan')

        history['times'].append(time)
        history['sig_zz_applied'].append(sig_zz_applied)
        history['pressure_at_base'].append(fem_pressure)
        history['pressure_mean'].append(p_mean)
        history['pressure_p10'].append(p_p10)
        history['pressure_p90'].append(p_p90)
        history['uz_at_bottom'].append(fem_uz_bot)
        history['uz_at_top'].append(fem_uz_top)
        history['volume_drained'].append(fem_volume)
        history['analytical_pressure'].append(an_p)
        history['analytical_uz'].append(an_u)
        history['analytical_volume'].append(an_v)
        history['pressure_error_percent'].append(p_err)
        history['uz_error_percent'].append(u_err)
        history['volume_error_percent'].append(v_err)

    #
    #
    def _field_at_z(self, subspace, z_target: float, reduce: str, tol: float = 1e-10):
        """Return a scalar summary of DOF values in subspace at z ≈ z_target.

        reduce: 'max' or 'mean'.  Returns 0.0 if no DOFs are found at that height.
        """
        space, dof_map = subspace.collapse()
        values         = self.wh.x.array[dof_map]
        z_coords       = space.tabulate_dof_coordinates()[:, 1]
        mask           = np.abs(z_coords - z_target) < tol
        if not np.any(mask):
            return 0.0
        v = values[mask]
        return float(np.max(v) if reduce == 'max' else np.mean(v))

    #
    #
    def _pressure_statistics(self):
        """Return (mean, P10, P90) of pressure over interior DOFs (excludes drainage face)."""
        drain_z        = self._drainage_z()
        p_space, p_map = self.W.sub(1).collapse()
        p_values       = self.wh.x.array[p_map]
        z_coords       = p_space.tabulate_dof_coordinates()[:, 1]
        interior       = np.abs(z_coords - drain_z) > 1e-10
        p_int          = p_values[interior]
        return (float(np.mean(p_int)),
                float(np.percentile(p_int, 10)),
                float(np.percentile(p_int, 90)))

    #
    #
    def _applied_sig_zz(self, time: float) -> float:
        """Sum the currently applied sig_zz across all loaded surfaces."""
        total = 0.0
        for surface in SURFACES:
            bc = getattr(CONFIG.boundary_conditions, surface)
            if bc.periodic_load is not None:
                total += bc.periodic_load.eval(time)
            elif bc.sig_zz is not None:
                total += bc.sig_zz
        return total

    #
    #
    def _compute_volume_drained(self):
        """Compute cumulative drained volume via domain integral of fluid content change.

        ΔV = −2π ∫_Ω (α·ε_v + p/M) · r dΩ
        """
        u, p = ufl.split(self.wh)
        r    = ufl.SpatialCoordinate(self.domain)[0]
        dx   = ufl.Measure("dx", domain=self.domain, metadata={"quadrature_degree": 10})
        integrand = -(self.formulation.alpha * self.formulation.eps_v(u)
                      + p / self.formulation.M) * 2 * np.pi * r
        local = fem.assemble_scalar(fem.form(integrand * dx))
        return float(self.comm.allreduce(local, op=MPI.SUM))

    # ------------------------------------------------------------------
    # Timestep recording
    # ------------------------------------------------------------------

    def _record_timestep(self, time, history):
        """Write the current timestep to all output files via OutputManager."""
        _, p_map = self.W.sub(1).collapse()
        p_values = self.wh.x.array[p_map]
        record   = {k: history[k][-1] for k in self._HISTORY_KEYS if k != 'times'}
        self._output.write(time, record, p_values)

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def _print_load_changes(self, time):
        """Print a message whenever a periodic load switches level."""
        current_loads = {}
        first_call    = len(self._prev_loads) == 0
        for surface in SURFACES:
            bc = getattr(CONFIG.boundary_conditions, surface)
            if bc.periodic_load is None:
                continue
            val = bc.periodic_load.eval(time)
            current_loads[surface] = val
            if not first_call and val != self._prev_loads.get(surface):
                if self.comm.rank == 0:
                    prev     = self._prev_loads.get(surface)
                    prev_str = f"{prev:.3e} Pa" if prev is not None else "none"
                    print(f"  >> LOAD CHANGE at t={time:.4f}s | "
                          f"{surface} sig_zz: {prev_str} → {val:.3e} Pa")
        self._prev_loads = current_loads

    #
    #
    def _print_progress(self, i_ts, time, dt, history):
        if self.comm.rank != 0:
            return
        def _fmt(v):
            return f"{v:6.2f}%" if v == v else "   n/a"
        print(f"Step {i_ts:4d} | t={time:8.3f}s (dt:{dt:6.3}s) | "
              f"p_err={_fmt(history['pressure_error_percent'][-1])} | "
              f"u_err={_fmt(history['uz_error_percent'][-1])} | "
              f"vol_err={_fmt(history['volume_error_percent'][-1])}")

    # ------------------------------------------------------------------
    # Summary and persistence
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable results summary (call after solve())."""
        if self.history is None:
            return "No results available. Run solve() first."
        h = self.history
        lines = [
            "\n" + "=" * 70,
            "SIMULATION RESULTS",
            "=" * 70,
            f"\nTimeseries: {CONFIG.output.timeseries}",
            f"\nFinal State (t={h['times'][-1]:.3f}s):",
            f"  Pressure at base:       {h['pressure_at_base'][-1]:.6e} Pa",
            f"  Displacement at bottom: {h['uz_at_bottom'][-1]:.6e} m",
            f"  Volume drained:         {h['volume_drained'][-1]:.6e} m³",
            f"\nErrors vs Analytical:",
            f"  Pressure error:         {h['pressure_error_percent'][-1]:.2f}%",
            f"  Displacement error:     {h['uz_error_percent'][-1]:.2f}%",
            f"  Volume error:           {h['volume_error_percent'][-1]:.2f}%",
            "=" * 70,
        ]
        return "\n".join(lines)

    #
    #
    def _save_run_summary(self):
        """Persist key config + final-state values to a pickle for sweep post-processing."""
        h = self.history
        data = {
            'run_id':                 CONFIG.general.run_id,
            'run_dir':                CONFIG.general.run_dir,
            'description':            CONFIG.general.description,
            'tags':                   CONFIG.general.tags,
            'E':                      CONFIG.materials.E,
            'nu':                     CONFIG.materials.nu,
            'alpha':                  CONFIG.materials.alpha,
            'perm':                   CONFIG.materials.perm,
            'visc':                   CONFIG.materials.visc,
            'M':                      CONFIG.materials.M,
            'Re':                     CONFIG.mesh.Re,
            'H':                      CONFIG.mesh.H,
            'Re/H':                   CONFIG.mesh.Re / CONFIG.mesh.H,
            'final_time':             h['times'][-1],
            'pressure_at_base':       h['pressure_at_base'][-1],
            'uz_at_bottom':           h['uz_at_bottom'][-1],
            'volume_drained':         h['volume_drained'][-1],
            'pressure_error_percent': h['pressure_error_percent'][-1],
            'uz_error_percent':       h['uz_error_percent'][-1],
            'volume_error_percent':   h['volume_error_percent'][-1],
        }
        output_dir = os.path.dirname(CONFIG.output.results)
        pd.DataFrame([data]).to_pickle(f"{output_dir}/summary.pkl")
