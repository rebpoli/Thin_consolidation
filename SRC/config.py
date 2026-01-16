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


class BoundaryCondition(BaseModel):
    U_r:      Optional[float] = None
    U_z:      Optional[float] = None
    U_r_rigid: int = 0
    U_z_rigid: int = 0
    sig_rr:   Optional[float] = None
    sig_zz:   Optional[float] = None
    Pressure: Optional[float] = None
    
    class Config:
        extra = 'forbid'


class BCCfg(BaseModel):
    bottom: Optional[BoundaryCondition] = BoundaryCondition()
    right:  Optional[BoundaryCondition] = BoundaryCondition()
    top:    Optional[BoundaryCondition] = BoundaryCondition()
    left:   Optional[BoundaryCondition] = BoundaryCondition()


class NumericalCfg(BaseModel):
#     dt0:   float     = Field(gt=0, description="Initial timestep time")
#     dtmax: float     = Field(gt=0, description="Maximum timestep time")
#     dtk:   float     = Field(gt=1, description="Geometric factor to increase dt")
#     end_time:  float = Field(gt=0, description="End time [s]")
    num_steps: int    = Field(gt=1, description="Number of time steps")
    theta_cn: float   = Field(gt=0, le=1, description="Crank-Nicholson theta (0:Explicit ; 0.5: C-N ; 1: Implicit)", default=0.5)
    end_time_s: float = Field(gt=0, description="End time t* (dimensionless)", default=0.25)
    penalty_rigid: float = 1e10
    
    _current_step: int  = 0
    _time_list:    list = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        self._current_step = 0
        self._time_list    = [0.0]

#         self._expected_time_list = []
#         self._expected_dt_list = []
#         dt = self.dt0
#         time = 0.0
#         while time < self.end_time+dt:
#             dt = min(dt * self.dtk, self.dtmax)
#             time += dt
#             self._expected_time_list.append(time)
#             self._expected_dt_list.append(dt)
#         self._expected_time_list = np.array(self._expected_time_list)
#         self._expected_dt_list = np.array( self._expected_dt_list )

    def setup_timesteps( self, config ) :
        H = config.mesh.H
        c_v = config.materials.c_v
        ts1 = self.end_time_s
        N = self.num_steps
        dt_s = ts1/N
        dt = 4*H*H/c_v * dt_s
        t_end = 4*H*H/c_v * ts1
        self._expected_time_list = np.linspace(0, t_end, N + 1)
        self._expected_dt_list = np.full(N+1, dt)
        
#         self._expected_time_list = np.array(self._expected_time_list)
#         self._expected_dt_list = np.array( self._expected_dt_list )
    
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
    results:           str = "outputs/results.xdmf"
    timeseries:        str = "outputs/fem_timeseries.nc"


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
        lines.append(f"\nNumerical Parameters:")
        lines.append(f"  End time* = {self.numerical.end_time_s} (dimensionless)")
        lines.append(f"  Time steps = {self.numerical.num_steps}")
        lines.append(f"  dt = {self.numerical.dt()} s")
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
