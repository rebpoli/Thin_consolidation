from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
import yaml
import numpy as np

_instance = None

class GeneralCfg(BaseModel):
    description: str = "[none]"
    tags:        str = "[none]"
    run_dir:     str = "[none]"
    run_id:      str = "[none]"

class MeshCfg(BaseModel):
    Re: float = Field(gt=0, description="Cylinder radius [m]")
    H:  float = Field(gt=0, description="Cylinder height [m]")
    N:  int   = Field(gt=5, description="Number of elements")


class MaterialCfg(BaseModel):
    E:     float = Field(gt=0,         description="Young's modulus [Pa]")
    nu:    float = Field(gt=0, lt=0.5, description="Poisson's ratio")
    alpha: float = Field(gt=0, le=1,   description="Biot coefficient")
    perm:  float = Field(gt=0,         description="Permeability [m²]")
    visc:  float = Field(gt=0,         description="Viscosity [Pa·s]")
    M:     float = Field(gt=0,         description="Biot modulus [Pa]")
    
    mu:    float = None
    lmbda: float = None
    nu_u:  float = None
    K_u:   float = None
    K:     float = None
    S:     float = None
    c_v:   float = None
    eta:   float = None
    
    def __init__(self, **data):
        super().__init__(**data)
        self.mu    = self.E / (2 * (1 + self.nu))
        self.lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.K     = ( 2 * self.mu ) * ( 1 + self.nu ) / (3 * ( 1 - 2 * self.nu ) ) 
        self.K_u   = self.K + self.alpha**2 * self.M
        self.nu_u  = ( 3 * self.K_u - 2*self.mu ) / ( 2 * (3*self.K_u + self.mu) )
        self.S     = 1/self.M + (self.alpha**2) * (1 - 2*self.nu) / (2 * self.mu * (1 - self.nu))
        self.c_v   = self.perm / self.visc / self.S
        self.eta   = self.alpha * (1 - 2*self.nu) / (2 * (1 - self.nu))
    
    @field_validator('nu')
    @classmethod
    def check_poisson(cls, v):
        if not (0 < v < 0.5):
            raise ValueError("Poisson's ratio must be in (0, 0.5)")
        return v


class PeriodicLoad(BaseModel):
    L0:                   float         = 0.0   # baseline load [Pa] — before t_start and during off-phase
    L1:                   float                 # active load level [Pa] — during on-phase
    t_start:              float         = 0.0   # time when cycling begins [s]
    period:               float                 # full cycle duration [s]
    duty_cycle:           float         = 0.5   # fraction of period spent at L1 (0–1)
    n_periods:            int           = -1    # number of cycles; -1 = infinite
    L_after:              Optional[float] = None  # load after cycling ends; default = (L1-L0)*duty_cycle
    transition_steps:     int           = 0     # steps in the L1→L0/L_after ramp; 0 = instant jump
    transition_step_dur:  float         = 100.0 # duration of each step [s]

    def _l_after(self) -> float:
        return self.L_after if self.L_after is not None else (self.L1 - self.L0) * self.duty_cycle

    def _transition(self, L_src: float, L_dst: float, t_since: float) -> float:
        """Stepped ramp from L_src to L_dst, t_since seconds after the start of the transition."""
        n = self.transition_steps
        if n <= 0 or L_src == L_dst:
            return L_dst
        total = n * self.transition_step_dur
        if t_since >= total:
            return L_dst
        k = int(t_since / self.transition_step_dur)   # 0-based step index
        return L_src + (k + 1) / n * (L_dst - L_src)

    def eval(self, t: float) -> float:
        """Return the load value at physical time t [s]."""
        if t < self.t_start:
            return self.L0
        t_rel = t - self.t_start
        l_after = self._l_after()

        if self.n_periods >= 0 and t_rel >= self.n_periods * self.period:
            # Level just before the n_periods boundary
            L_src = self.L1 if self.duty_cycle >= 1.0 else self.L0
            return self._transition(L_src, l_after, t_rel - self.n_periods * self.period)

        phase = (t_rel % self.period) / self.period
        if phase < self.duty_cycle:
            return self.L1
        # Off-phase: stepped transition L1 → L0
        t_since_fall = (t_rel % self.period) - self.duty_cycle * self.period
        return self._transition(self.L1, self.L0, t_since_fall)

    def switch_times(self, t_end: float) -> list:
        """Return sorted list of all load-change times in (0, t_end)."""
        times = set()
        t_on  = self.period * self.duty_cycle
        l_after = self._l_after()
        n_max = (self.n_periods if self.n_periods >= 0
                 else int((t_end - self.t_start) / self.period) + 2)

        def _add(t):
            if 1e-12 < t < t_end - 1e-12:
                times.add(t)

        def _add_trans(t0, L_src, L_dst):
            """Add switch times for a (possibly stepped) transition starting at t0."""
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
            # Rising edge: always an instant jump to L1
            if t_rise > 1e-12:
                _add(t_rise)
            # Falling edge within cycle (only when off-phase exists)
            if self.duty_cycle < 1.0:
                _add_trans(t_fall, self.L1, self.L0)

        # Final transition to L_after
        if self.n_periods >= 0:
            t_fin = self.t_start + self.n_periods * self.period
            L_src = self.L1 if self.duty_cycle >= 1.0 else self.L0
            _add_trans(t_fin, L_src, l_after)

        return sorted(times)


class BoundaryCondition(BaseModel):
    U_r:      Optional[float] = None
    U_z:      Optional[float] = None
    U_r_rigid: int = 0
    U_z_rigid: int = 0
    sig_rr:   Optional[float] = None
    sig_zz:   Optional[float] = None
    Pressure: Optional[float] = None
    periodic_load: Optional[PeriodicLoad] = None

    class Config:
        extra = 'forbid'


class BCCfg(BaseModel):
    bottom: Optional[BoundaryCondition] = BoundaryCondition()
    right:  Optional[BoundaryCondition] = BoundaryCondition()
    top:    Optional[BoundaryCondition] = BoundaryCondition()
    left:   Optional[BoundaryCondition] = BoundaryCondition()


class NumericalCfg(BaseModel):
    num_steps:     Optional[int]   = Field(default=None, description="Number of time steps (uniform fallback)")
    theta_cn:      float           = Field(gt=0, le=1, description="Crank-Nicholson theta (0:Explicit ; 0.5: C-N ; 1: Implicit)", default=0.5)
    end_time_tv:    float           = Field(gt=0, description="Dimensionless consolidation time T_v = c_v*t/(4H²); t_end[s] = 4H²/c_v * T_v", default=0.25)
    dt_min_s:      Optional[float] = Field(default=None, description="Minimum physical timestep [s]")
    dt_max_s:      Optional[float] = Field(default=None, description="Maximum physical timestep [s]")
    dt_factor:     float           = Field(default=1.5,  description="Geometric growth factor between timesteps")
    penalty_rigid: float           = 1e10

    _current_step: int  = 0
    _time_list:    list = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._current_step = 0
        self._time_list    = [0.0]

    def setup_timesteps(self, config):
        H   = config.mesh.H
        c_v = config.materials.c_v
        t_end = 4.0 * H * H / c_v * self.end_time_tv

        if self.dt_min_s is not None and self.dt_max_s is not None:
            self._build_geometric_timesteps(config, t_end)
        else:
            N  = self.num_steps if self.num_steps is not None else 1000
            dt = t_end / N
            self._expected_time_list = np.linspace(0.0, t_end, N + 1)
            self._expected_dt_list   = np.full(N + 1, dt)

    def _build_geometric_timesteps(self, config, t_end: float):
        """Build timestep array with geometric growth, resetting to dt_min after each load change.

        Around every load-switch time t_sw the pattern is:
          ... dt_max ... | t_sw - dt_min | t_sw | t_sw + dt_min | t_sw + dt_min*f | ...
        Both the pre-snap (t_sw - dt_min) and the switch itself (t_sw) reset dt to dt_min.
        """
        dt_min = self.dt_min_s

        # Collect all load-switch times (physical seconds)
        switch_times = set()
        for face in ('bottom', 'right', 'top', 'left'):
            bc = getattr(config.boundary_conditions, face, None)
            if bc is None or bc.periodic_load is None:
                continue
            switch_times.update(bc.periodic_load.switch_times(t_end))

        # Checkpoints = switch times + pre-snap points (t_sw - dt_min) + t_end
        # Every checkpoint resets dt = dt_min after being hit.
        checkpoints = set()
        for t_sw in switch_times:
            if 0.0 < t_sw < t_end - 1e-12:
                checkpoints.add(t_sw)
            t_pre = t_sw - dt_min
            if t_pre > 1e-12:
                checkpoints.add(t_pre)
        checkpoints.add(t_end)
        checkpoints = sorted(checkpoints)

        times  = [0.0]
        t      = 0.0
        dt     = dt_min
        cp_idx = 0

        while t < t_end - 1e-12:
            next_cp = checkpoints[cp_idx]
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
                dt = min(dt * self.dt_factor, self.dt_max_s)

        times = np.array(times)
        dts   = np.empty_like(times)
        dts[0]  = 0.0
        dts[1:] = np.diff(times)
        self._expected_time_list = times
        self._expected_dt_list   = dts
    
    def dt(self) -> float:
        return self._expected_dt_list[self._current_step]
    
    def current_time(self) -> float:
        return self._time_list[-1]
    
    def time_list(self) -> list:
        return self._time_list.copy()
    
    def expected_time_list(self) -> np.ndarray:
        return self._expected_time_list.copy()
    
    def end(self) -> bool:
        return self._current_step > len(self._expected_time_list) -2
    
    def next_ts(self):
        if self.end(): return None
        
        self._current_step += 1
        new_time = self.current_time() + self.dt()
        self._time_list.append(new_time)
        
        return self._current_step, self.dt(), new_time

    def __iter__(self):
        return self
    
    def __next__(self):
        result = self.next_ts()
        if result is None: raise StopIteration
        return result


class OutputCfg(BaseModel):
    results:             str = "outputs/results.bp"
    timeseries:          str = "outputs/fem_timeseries.nc"
    pressure_profile:    str = "outputs/pressure_profile.nc"
    invariants_nc:       str = "outputs/invariants.nc"
    n_profile_points:    int = 1000
    n_invariant_points:  int = 500


class Config(BaseModel):
    general:             GeneralCfg
    mesh:                MeshCfg
    materials:           MaterialCfg
    boundary_conditions: BCCfg
    numerical:           NumericalCfg
    output:              OutputCfg = Field(default_factory=OutputCfg)

    def __init__(self, config_file: str = "config.yaml", **data):
        if not data:
            with open(config_file) as f:
                data = yaml.safe_load(f)
        super().__init__(**data)
        self.numerical.setup_timesteps(self)
    
    def summary(self):
        lines = []
        lines.append("="*70)
        lines.append("CONFIGURATION SUMMARY")
        lines.append("="*70)
        lines.append(f"\nMesh:")
        lines.append(f"  Re (radius) = {self.mesh.Re} m")
        lines.append(f"  H (height)  = {self.mesh.H} m")
        lines.append(f"  N (elements per dimension) = {self.mesh.N}")
        lines.append(f"\nMaterial Properties:")
        lines.append(f"  E (Young's modulus) = {self.materials.E:.3e} Pa")
        lines.append(f"  nu (Poisson's ratio) = {self.materials.nu}")
        lines.append(f"  alpha (Biot coefficient) = {self.materials.alpha}")
        lines.append(f"  perm (permeability) = {self.materials.perm:.3e} m²")
        lines.append(f"  visc (viscosity) = {self.materials.visc:.3e} Pa·s")
        lines.append(f"  M (Biot modulus) = {self.materials.M:.3e} Pa")
        lines.append(f"  mu (computed) = {self.materials.mu:.3e} Pa")
        lines.append(f"  lambda (computed) = {self.materials.lmbda:.3e} Pa")
        lines.append(f"  K (computed) = {self.materials.K:.3e} Pa")
        lines.append(f"  K_u (computed) = {self.materials.K_u:.3e} Pa")
        lines.append(f"  nu_u (computed) = {self.materials.nu_u:.3e}")
        lines.append(f"  S (computed) = {self.materials.S:.3e}")
        lines.append(f"  c_v (computed) = {self.materials.c_v:.3e}")
        lines.append(f"  eta (computed) = {self.materials.eta:.3e}")
        n_steps = len(self.numerical._expected_time_list) - 1
        lines.append(f"\nNumerical Parameters:")
        lines.append(f"  End time* = {self.numerical.end_time_tv} (dimensionless)")
        lines.append(f"  Time steps = {n_steps}")
        lines.append(f"  dt_min = {self.numerical._expected_dt_list[1]:.4g} s"
                     f"  dt_max = {self.numerical._expected_dt_list[1:].max():.4g} s")
        lines.append(f"\nOutput:")
        lines.append(f"  Results file = {self.output.results}")
        lines.append(f"  Timeseries = {self.output.timeseries}")
        lines.append("="*70)
        return "\n".join(lines)


_instance = None


def load(config_file: str = "config.yaml"):
    global _instance
    _instance = Config(config_file)
    return _instance


def get() -> Config:
    if _instance is None:
        raise RuntimeError("Config not loaded. Call config.load() first.")
    return _instance
