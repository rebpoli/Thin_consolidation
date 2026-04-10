"""Configuration system for the poroelastic consolidation solver.

Pydantic models map directly to config.yaml sections.  MaterialCfg derives
secondary elastic/consolidation constants from the five primary inputs.
TimeStepper builds the full timestep schedule (uniform or geometric) and
acts as the iteration source for the time loop.

Usage (in every module):
    from config import CONFIG          # proxy — safe to import before load()
    from config import CONFIG, SURFACES

    # In run.py only:
    import config
    cfg = config.load("config.yaml")   # populates CONFIG
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import yaml

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared constant
# ---------------------------------------------------------------------------

# Boundary surface names in physical-group marker order (bottom=1 … left=4).
# Imported by formulation.py and fem_solver.py to avoid repeated literals.
SURFACES = ('bottom', 'right', 'top', 'left')


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class GeneralCfg(BaseModel):
    """Run metadata — written verbatim to NetCDF global attributes and summary pickle."""
    description: str = "[none]"
    tags:        str = "[none]"
    run_dir:     str = "[none]"
    run_id:      str = "[none]"


#
#
class MeshCfg(BaseModel):
    """Cylinder geometry and mesh density.

    N sets the element count for both directions.  Nr / Nz override per-direction.
    grade_r / grade_z_bottom control whether geometric grading is applied in r and
    which end of z is refined (False = near z=H, True = near z=0).
    """
    Re:             float         = Field(gt=0, description="Cylinder radius [m]")
    H:              float         = Field(gt=0, description="Cylinder height [m]")
    N:              int           = Field(gt=1, description="Default element count per direction")
    Nr:             Optional[int] = Field(default=None, gt=0, description="Elements in r (overrides N)")
    Nz:             Optional[int] = Field(default=None, gt=0, description="Elements in z (overrides N)")
    grade_r:        bool          = True    # geometric grading in r (fine near r=Re)
    grade_z_bottom: bool          = False   # True → refine near z=0 instead of z=H


#
#
class MaterialCfg(BaseModel):
    """Poroelastic material parameters.

    Primary inputs (E, nu, alpha, perm, visc, M) are validated by Pydantic.
    All secondary constants (Lamé, consolidation, Skempton) are derived
    automatically by the model_validator and stored as plain float fields.
    """

    # --- Primary inputs ---
    E:     float = Field(gt=0,         description="Young's modulus [Pa]")
    nu:    float = Field(gt=0, lt=0.5, description="Poisson's ratio")
    alpha: float = Field(gt=0, le=1,   description="Biot coefficient")
    perm:  float = Field(gt=0,         description="Permeability [m²]")
    visc:  float = Field(gt=0,         description="Viscosity [Pa·s]")
    M:     float = Field(gt=0,         description="Biot modulus [Pa]")

    # --- Derived (populated by validator) ---
    mu:    float = None   # shear modulus
    lmbda: float = None   # Lamé first parameter
    nu_u:  float = None   # undrained Poisson's ratio
    K_u:   float = None   # undrained bulk modulus
    S:     float = None   # storage coefficient
    c_v:   float = None   # consolidation coefficient
    eta:   float = None   # loading efficiency (Skempton-like)

    @model_validator(mode='after')
    def _compute_derived(self):
        """Compute all secondary poroelastic constants from primary inputs."""
        mu          = self.E / (2 * (1 + self.nu))
        lmbda       = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        K           = 2 * mu * (1 + self.nu) / (3 * (1 - 2 * self.nu))
        K_u         = K + self.alpha**2 * self.M
        nu_u        = (3 * K_u - 2 * mu) / (2 * (3 * K_u + mu))
        S           = 1 / self.M + self.alpha**2 * (1 - 2 * self.nu) / (2 * mu * (1 - self.nu))
        self.mu     = mu
        self.lmbda  = lmbda
        self.K_u    = K_u
        self.nu_u   = nu_u
        self.S      = S
        self.c_v    = self.perm / self.visc / S
        self.eta    = self.alpha * (1 - 2 * self.nu) / (2 * (1 - self.nu))
        return self


#
#
class PeriodicLoad(BaseModel):
    """Time-varying load: square-wave with optional ramp transitions.

    The load switches between L0 (off) and L1 (on) with a given duty cycle.
    After n_periods cycles it holds at L_after (defaults to the time-averaged
    value).  Transitions between levels can be stepped over transition_steps
    intervals of transition_step_dur seconds each.
    """

    L0:                  float           = 0.0
    L1:                  float
    t_start:             float           = 0.0
    period:              float
    duty_cycle:          float           = 0.5
    n_periods:           int             = -1      # -1 = run indefinitely
    L_after:             Optional[float] = None
    transition_steps:    int             = 0
    transition_step_dur: float           = 100.0

    #
    #
    def _l_after(self) -> float:
        """Final load level after all cycles complete."""
        return self.L_after if self.L_after is not None else (self.L1 - self.L0) * self.duty_cycle

    #
    #
    def _transition(self, L_src: float, L_dst: float, t_since: float) -> float:
        """Step-wise linear ramp from L_src to L_dst starting at t_since=0."""
        n = self.transition_steps
        if n <= 0 or L_src == L_dst:
            return L_dst
        total = n * self.transition_step_dur
        if t_since >= total:
            return L_dst
        k = int(t_since / self.transition_step_dur)
        return L_src + (k + 1) / n * (L_dst - L_src)

    #
    #
    def eval(self, t: float) -> float:
        """Return the load value at time t."""
        if t < self.t_start:
            return self.L0
        t_rel   = t - self.t_start
        l_after = self._l_after()
        if self.n_periods >= 0 and t_rel >= self.n_periods * self.period:
            L_src = self.L1 if self.duty_cycle >= 1.0 else self.L0
            return self._transition(L_src, l_after, t_rel - self.n_periods * self.period)
        phase = (t_rel % self.period) / self.period
        if phase < self.duty_cycle:
            return self.L1
        t_since_fall = (t_rel % self.period) - self.duty_cycle * self.period
        return self._transition(self.L1, self.L0, t_since_fall)

    def switch_times(self, t_end: float) -> list:
        """Return sorted list of all load-switch times in (0, t_end).

        Used by TimeStepper to insert checkpoints so the solver lands exactly
        on every transition, then resets dt to dt_min after each one.
        """
        times   = set()
        t_on    = self.period * self.duty_cycle
        l_after = self._l_after()
        n_max   = (self.n_periods if self.n_periods >= 0
                   else int((t_end - self.t_start) / self.period) + 2)

        def _add(t):
            if 1e-12 < t < t_end - 1e-12:
                times.add(t)

        def _add_trans(t0, L_src, L_dst):
            if L_src == L_dst:
                return
            n = self.transition_steps
            if n > 0:
                for s in range(n):
                    _add(t0 + s * self.transition_step_dur)
            else:
                _add(t0)

        for k in range(n_max):
            t_rise = self.t_start + k * self.period
            t_fall = t_rise + t_on
            if t_rise > t_end:
                break
            if t_rise > 1e-12:
                _add(t_rise)
            if self.duty_cycle < 1.0:
                _add_trans(t_fall, self.L1, self.L0)

        if self.n_periods >= 0:
            t_fin = self.t_start + self.n_periods * self.period
            L_src = self.L1 if self.duty_cycle >= 1.0 else self.L0
            _add_trans(t_fin, L_src, l_after)

        return sorted(times)


#
#
class BoundaryCondition(BaseModel):
    """Boundary conditions for one surface.

    Dirichlet: U_r, U_z, Pressure.
    Neumann:   sig_rr, sig_zz (static), or periodic_load (time-varying sig_zz).
    Rigid coupling: U_r_rigid / U_z_rigid activate penalty-based DOF coupling
    (all DOFs on the surface move together as a rigid body).
    """
    U_r:           Optional[float]        = None
    U_z:           Optional[float]        = None
    U_r_rigid:     int                    = 0
    U_z_rigid:     int                    = 0
    sig_rr:        Optional[float]        = None
    sig_zz:        Optional[float]        = None
    Pressure:      Optional[float]        = None
    periodic_load: Optional[PeriodicLoad] = None

    model_config = {"extra": "forbid"}


#
#
class BCCfg(BaseModel):
    """Per-surface boundary conditions for the four cylinder faces."""
    bottom: BoundaryCondition = Field(default_factory=BoundaryCondition)
    right:  BoundaryCondition = Field(default_factory=BoundaryCondition)
    top:    BoundaryCondition = Field(default_factory=BoundaryCondition)
    left:   BoundaryCondition = Field(default_factory=BoundaryCondition)


#
#
class NumericalCfg(BaseModel):
    """Time integration and solver settings."""
    num_steps:     Optional[int]   = Field(default=None, description="Steps for uniform fallback")
    theta_cn:      float           = Field(default=0.5,  gt=0, le=1, description="Crank-Nicolson θ")
    end_time_tv:   float           = Field(default=0.25, gt=0, description="Dimensionless end time T_v = c_v·t/H_dr²")
    dt_min_s:      Optional[float] = Field(default=None, description="Minimum timestep [s]")
    dt_max_s:      Optional[float] = Field(default=None, description="Maximum timestep [s]")
    dt_factor:     float           = Field(default=1.5,  description="Geometric growth factor")
    penalty_rigid: float           = 1e10


#
#
class OutputCfg(BaseModel):
    """Output file paths and spatial sampling resolution."""
    results:            str = "outputs/results.bp"
    timeseries:         str = "outputs/fem_timeseries.nc"
    pressure_profile:   str = "outputs/pressure_profile.nc"
    invariants_nc:      str = "outputs/invariants.nc"
    n_profile_points:   int = 1000
    n_invariant_points: int = 500

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# TimeStepper
# ---------------------------------------------------------------------------

class TimeStepper:
    """Builds the timestep schedule and acts as the time-loop iterator.

    Separated from NumericalCfg so the Pydantic model stays stateless.

    Two scheduling modes:
      - Uniform: linspace(0, t_end, N+1) when dt_min_s / dt_max_s are absent.
      - Geometric: starts at dt_min, grows by dt_factor each step, resets to
        dt_min after each periodic-load switch. Checkpoints are inserted at
        every switch time so the solver lands exactly on them.

    Iterates as:  for i, dt, time in timestepper: ...
    """

    def __init__(self, num_cfg: NumericalCfg, mat_cfg: MaterialCfg,
                 mesh_cfg: MeshCfg, bc_cfg: BCCfg):
        self.theta_cn      = num_cfg.theta_cn
        self.penalty_rigid = num_cfg.penalty_rigid

        H_dr  = mesh_cfg.H / 2
        t_end = H_dr**2 / mat_cfg.c_v * num_cfg.end_time_tv

        if num_cfg.dt_min_s is not None and num_cfg.dt_max_s is not None:
            times = self._geometric(num_cfg, bc_cfg, t_end)
        else:
            N     = num_cfg.num_steps if num_cfg.num_steps is not None else 1000
            times = np.linspace(0.0, t_end, N + 1)

        self._times   = times
        self._dts     = np.empty_like(times)
        self._dts[0]  = 0.0
        self._dts[1:] = np.diff(times)
        self._step    = 0

    def _geometric(self, num_cfg: NumericalCfg, bc_cfg: BCCfg, t_end: float) -> np.ndarray:
        """Build geometric schedule with checkpoints at every load-switch time."""
        dt_min = num_cfg.dt_min_s

        # Collect switch times from all periodic loads
        switch_times: set[float] = set()
        for face in SURFACES:
            bc = getattr(bc_cfg, face, None)
            if bc is not None and bc.periodic_load is not None:
                switch_times.update(bc.periodic_load.switch_times(t_end))

        # Insert a checkpoint just before each switch so dt_min resets in advance
        checkpoints: set[float] = set()
        for t_sw in switch_times:
            if 0.0 < t_sw < t_end - 1e-12:
                checkpoints.add(t_sw)
            t_pre = t_sw - dt_min
            if t_pre > 1e-12:
                checkpoints.add(t_pre)
        checkpoints.add(t_end)
        checkpoints_sorted = sorted(checkpoints)

        times  = [0.0]
        t      = 0.0
        dt     = dt_min
        cp_idx = 0

        while t < t_end - 1e-12:
            next_cp = checkpoints_sorted[cp_idx]
            step    = min(dt, next_cp - t)
            if step < 1e-14:
                cp_idx += 1
                dt = dt_min
                continue
            t += step
            times.append(t)
            if abs(t - next_cp) < 1e-10:
                cp_idx += 1
                dt = dt_min
            else:
                dt = min(dt * num_cfg.dt_factor, num_cfg.dt_max_s)

        return np.array(times)

    #
    #
    def expected_time_list(self) -> np.ndarray:
        """Return a copy of the full time array (used by Analytical to pre-allocate)."""
        return self._times.copy()

    #
    #
    def __iter__(self):
        self._step = 0
        return self

    def __next__(self):
        i = self._step + 1
        if i >= len(self._times):
            raise StopIteration
        self._step = i
        return i, float(self._dts[i]), float(self._times[i])


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

class Config(BaseModel):
    """Root configuration object — constructed from config.yaml.

    Pydantic validates all fields on construction.  TimeStepper is built
    post-init (it is stateful and not a Pydantic field).
    """

    general:             GeneralCfg
    mesh:                MeshCfg
    materials:           MaterialCfg
    boundary_conditions: BCCfg
    numerical:           NumericalCfg
    output:              OutputCfg = Field(default_factory=OutputCfg)

    # Stateful iterator — excluded from Pydantic serialization
    timestepper: TimeStepper = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, config_file: str = "config.yaml", **data):
        if not data:
            with open(config_file) as f:
                data = yaml.safe_load(f)
        super().__init__(**data)
        self.timestepper = TimeStepper(
            self.numerical, self.materials, self.mesh, self.boundary_conditions
        )

    def summary(self) -> str:
        """Return a multi-line human-readable configuration summary."""
        m = self.materials
        lines = [
            "=" * 70,
            "CONFIGURATION SUMMARY",
            "=" * 70,
            f"\nMesh:",
            f"  Re = {self.mesh.Re} m   H = {self.mesh.H} m   N = {self.mesh.N}",
            f"\nMaterials:",
            f"  E={m.E:.3e} Pa   nu={m.nu}   alpha={m.alpha}",
            f"  perm={m.perm:.3e} m²   visc={m.visc:.3e} Pa·s   M={m.M:.3e} Pa",
            f"  mu={m.mu:.3e}   lambda={m.lmbda:.3e}   K_u={m.K_u:.3e}",
            f"  nu_u={m.nu_u:.3e}   S={m.S:.3e}   c_v={m.c_v:.3e}   eta={m.eta:.3e}",
            f"\nNumerical:",
            f"  theta_cn={self.numerical.theta_cn}   end_time_tv={self.numerical.end_time_tv}",
            f"  steps={len(self.timestepper._times) - 1}"
            + (f"   dt=[{self.timestepper._dts[1]:.4g}, {self.timestepper._dts[1:].max():.4g}] s"
               if len(self.timestepper._dts) > 1 else ""),
            f"\nOutput:",
            f"  results={self.output.results}",
            f"  timeseries={self.output.timeseries}",
            "=" * 70,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

class _ConfigProxy:
    """Transparent proxy to the loaded Config instance.

    Stays fixed as a module-level object so ``from config import CONFIG``
    works at import time.  Attribute access is forwarded to the real Config
    after ``config.load()`` is called.
    """

    _instance: Optional[Config] = None

    def __getattr__(self, name: str):
        if self._instance is None:
            raise RuntimeError("Config not loaded — call config.load() first.")
        return getattr(self._instance, name)

    def __repr__(self):
        return repr(self._instance)


CONFIG = _ConfigProxy()


def load(config_file: str = "config.yaml") -> Config:
    """Load config.yaml, populate CONFIG, and return the Config instance."""
    CONFIG._instance = Config(config_file)
    return CONFIG._instance
