from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
import yaml
import numpy as np

_instance = None


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
    sig0:  float = Field(              description="Initial stress [Pa]")
    
    mu:    float = None
    lmbda: float = None
    
    def __init__(self, **data):
        super().__init__(**data)
        self.mu    = self.E / (2 * (1 + self.nu))
        self.lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
    
    @field_validator('nu')
    @classmethod
    def check_poisson(cls, v):
        if not (0 < v < 0.5):
            raise ValueError("Poisson's ratio must be in (0, 0.5)")
        return v


class BoundaryCondition(BaseModel):
    U_r:      Optional[float] = None
    U_z:      Optional[float] = None
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
    end_time:  float = Field(gt=0, description="End time [s]")
    num_steps: int   = Field(gt=1, description="Number of time steps")
    
    _current_step: int  = 0
    _time_list:    list = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        self._current_step = 0
        self._time_list    = [0.0]
    
    def dt(self) -> float:
        return self.end_time / self.num_steps
    
    def current_time(self) -> float:
        return self._time_list[-1]
    
    def time_list(self) -> list:
        return self._time_list.copy()
    
    def expected_time_list(self) -> np.ndarray:
        return np.linspace(0, self.end_time, self.num_steps + 1)
    
    def end(self) -> bool:
        return self._current_step >= self.num_steps
    
    def next_ts(self):
        if self.end():
            return None
        
        self._current_step += 1
        new_time = self._current_step * self.dt()
        self._time_list.append(new_time)
        
        return self._current_step, self.dt(), new_time
    
    def __iter__(self):
        return self
    
    def __next__(self):
        result = self.next_ts()
        if result is None:
            raise StopIteration
        return result


class OutputCfg(BaseModel):
    vtk_file:                   str = "outputs/results"
    fem_timeseries_file:        str = "outputs/fem_timeseries.nc"
    analytical_timeseries_file: str = "outputs/analytical_timeseries.nc"
    output_dir:                 str = "outputs"


class Config(BaseModel):
    mesh:                MeshCfg
    materials:           MaterialCfg
    boundary_conditions: BCCfg
    numerical:           NumericalCfg
    output:              OutputCfg
    
    def __init__(self, config_file: str = "config.yaml", **data):
        if not data:
            with open(config_file) as f:
                data = yaml.safe_load(f)
        super().__init__(**data)
    
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
        lines.append(f"  sig0 (initial stress) = {self.materials.sig0:.3e} Pa")
        lines.append(f"  mu (computed) = {self.materials.mu:.3e} Pa")
        lines.append(f"  lambda (computed) = {self.materials.lmbda:.3e} Pa")
        lines.append(f"\nNumerical Parameters:")
        lines.append(f"  End time = {self.numerical.end_time} s")
        lines.append(f"  Time steps = {self.numerical.num_steps}")
        lines.append(f"  dt = {self.numerical.dt()} s")
        lines.append(f"\nOutput:")
        lines.append(f"  VTK file = {self.output.vtk_file}")
        lines.append(f"  FEM timeseries = {self.output.fem_timeseries_file}")
        lines.append(f"  Analytical timeseries = {self.output.analytical_timeseries_file}")
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
