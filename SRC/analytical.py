import numpy as np
import xarray as xr

import config
from mesh_generator import CylinderMesh

from dolfinx import mesh, fem, io
from mpi4py import MPI


class Analytical1DConsolidation:
    def __init__(self, n_spatial_points: int = 50, n_fourier_terms: int = 50):
        cfg = config.get()
        
        self.material_cfg  = cfg.materials
        self.mesh_cfg      = cfg.mesh
        self.numerical_cfg = cfg.numerical
        
        self.times = self.numerical_cfg.expected_time_list()
        
        self.n_z     = n_spatial_points
        self.n_terms = n_fourier_terms
        self.z_coords = np.linspace(0, self.mesh_cfg.H, self.n_z)
        
        self._compute_derived_properties()
        self._compute_history()
    
    def _compute_derived_properties(self):
        E     = self.material_cfg.E
        nu    = self.material_cfg.nu
        alpha = self.material_cfg.alpha
        perm  = self.material_cfg.perm
        visc  = self.material_cfg.visc
        M     = self.material_cfg.M
        sig0  = self.material_cfg.sig0
        mu    = self.material_cfg.mu
        lmbda = self.material_cfg.lmbda
        
        H  = self.mesh_cfg.H
        Re = self.mesh_cfg.Re
        
        self.eta = alpha * (1 - 2*nu) / (2 * (1 - nu))
        self.S   = 1/M + (alpha**2) * (1 - 2*nu) / (2 * mu * (1 - nu))
        self.c_v = perm / visc / self.S
        self.p0  = -sig0 * self.eta / mu / self.S
        
        self.K_u = lmbda + 2*mu
    
    def _compute_history(self):
        n_times = len(self.times)
        
        self._pressure_field = np.zeros((n_times, self.n_z))
        self._uz_field       = np.zeros((n_times, self.n_z))

        H  = self.mesh_cfg.H
        
        for i_t, t in enumerate(self.times):
            for i_z, z in enumerate(self.z_coords):
                self._pressure_field[i_t, i_z] = self._pressure(t, H-z)
                self._uz_field[i_t, i_z]       = self._uz(t, H-z)
        
        self._pressure_at_bottom = self._pressure_field[:, 0]
        self._uz_at_top          = self._uz_field[:, -1]
        self._volume_drained     = self._compute_volume_drained()
    
    def _pressure(self, t: float, z: float) -> float:
        if t == 0:
            return self.p0
        
        H   = self.mesh_cfg.H
        T_v = self.c_v * t / (H**2)
        
        pressure_sum = 0.0
        for n in range(self.n_terms):
            m = (2*n + 1) * np.pi / 2
            pressure_sum += (2.0 / m) * np.sin(m * z / H) * np.exp(-m**2 * T_v)
        
        return self.p0 * pressure_sum
    
    def _uz(self, t: float, z: float) -> float:
        H = self.mesh_cfg.H
        
        if t == 0:
            uz_elastic = -self.eta * self.p0 * z / self.K_u
            return uz_elastic
        
        T_v = self.c_v * t / (H**2)
        
        uz_final = -self.p0 * H / self.K_u * (self.eta + self.S * self.K_u * (1 - z/H))
        
        sum_term = 0.0
        for n in range(self.n_terms):
            m = (2*n + 1) * np.pi / 2
            sum_term += (2.0 / m**2) * np.sin(m * z / H) * (1 - np.exp(-m**2 * T_v))
        
        uz_time = self.p0 * H / self.K_u * sum_term
        
        return uz_final + uz_time
    
    def _compute_volume_drained(self) -> np.ndarray:
        H   = self.mesh_cfg.H
        Re  = self.mesh_cfg.Re
        
        volume_drained = np.zeros(len(self.times))
        
        for i_t, t in enumerate(self.times):
            if t == 0:
                volume_drained[i_t] = 0.0
            else:
                T_v = self.c_v * t / (H**2)
                
                sum_term = 0.0
                for n in range(self.n_terms):
                    m = (2*n + 1) * np.pi / 2
                    sum_term += (1.0 / m**2) * (1 - np.exp(-m**2 * T_v))
                
                Q_total = np.pi * Re**2 * self.p0 * H / self.K_u * sum_term
                volume_drained[i_t] = Q_total
        
        return volume_drained
    
    def pressure_at_bottom(self, t: float) -> float:
        idx = np.argmin(np.abs(self.times - t))
        return self._pressure_at_bottom[idx]
    
    def uz_at_top(self, t: float) -> float:
        idx = np.argmin(np.abs(self.times - t))
        return self._uz_at_top[idx]
    
    def volume_drained(self, t: float) -> float:
        idx = np.argmin(np.abs(self.times - t))
        return self._volume_drained[idx]
    
    def get_history(self) -> dict:
        return {
            'times':              self.times,
            'pressure_at_bottom': self._pressure_at_bottom,
            'uz_at_top':          self._uz_at_top,
            'volume_drained':     self._volume_drained,
        }
    
    def save_vtk(self, domain, facets):
        
        cfg = config.get()
        comm = MPI.COMM_WORLD
        
        V_scalar = fem.functionspace(domain, ("Lagrange", 1))
        
        pressure_func = fem.Function(V_scalar, name="Pressure")
        uz_func       = fem.Function(V_scalar, name="Displacement_z")
        
        vtk_base = cfg.output.vtk_file.replace(".pvd", "")
        vtk_file = vtk_base + ".analytical.bp"
        
        vtx_writer = io.VTXWriter(comm, vtk_file, [pressure_func, uz_func], engine="BP4")
        
        for i_t, t in enumerate(self.times):
            def pressure_expr(x):
                z_vals = x[1]
                n_points = len(z_vals)
                result = np.zeros(n_points)
                
                if t == 0:
                    result[:] = self.p0
                else:
                    H = self.mesh_cfg.H
                    T_v = self.c_v * t / (H**2)
                    
                    for n in range(self.n_terms):
                        m = (2*n + 1) * np.pi / 2
                        result += (2.0 / m) * np.sin(m * z_vals / H) * np.exp(-m**2 * T_v)
                    
                    result *= self.p0
                
                return result
            
            def uz_expr(x):
                z_vals = x[1]
                n_points = len(z_vals)
                H = self.mesh_cfg.H
                
                if t == 0:
                    return -self.eta * self.p0 * z_vals / self.K_u
                
                T_v = self.c_v * t / (H**2)
                
                uz_final = -self.p0 * H / self.K_u * (self.eta + self.S * self.K_u * (1 - z_vals/H))
                
                sum_term = np.zeros(n_points)
                for n in range(self.n_terms):
                    m = (2*n + 1) * np.pi / 2
                    sum_term += (2.0 / m**2) * np.sin(m * z_vals / H) * (1 - np.exp(-m**2 * T_v))
                
                uz_time = self.p0 * H / self.K_u * sum_term
                
                return uz_final + uz_time
            
            pressure_func.interpolate(pressure_expr)
            uz_func.interpolate(uz_expr)
            
            vtx_writer.write(t)
            
            if comm.rank == 0:
                print(f"  Step {i_t+1}/{len(self.times)}: t={t:.3f}s")
        
        vtx_writer.close()
        
        if comm.rank == 0:
            print(f"✓ Analytical VTK saved: {vtk_file}")
            print(f"  Open in ParaView: {vtk_file}/ directory")
    
    def save_timeseries(self):
        cfg = config.get()
        history = self.get_history()
        
        ds = xr.Dataset(
            {
                'pressure_at_bottom': (['time'], history['pressure_at_bottom']),
                'uz_at_top':          (['time'], history['uz_at_top']),
                'volume_drained':     (['time'], history['volume_drained']),
            },
            coords={'time': history['times']},
            attrs={
                'E':     cfg.materials.E,
                'nu':    cfg.materials.nu,
                'alpha': cfg.materials.alpha,
                'perm':  cfg.materials.perm,
                'visc':  cfg.materials.visc,
                'M':     cfg.materials.M,
                'sig0':  cfg.materials.sig0,
                'Re':    cfg.mesh.Re,
                'H':     cfg.mesh.H,
                'c_v':   self.c_v,
                'p0':    self.p0,
                'eta':   self.eta,
                'S':     self.S,
            }
        )
        
        ds.to_netcdf(cfg.output.analytical_timeseries_file)
        
        if MPI.COMM_WORLD.rank == 0:
            print(f"✓ Analytical timeseries saved: {cfg.output.analytical_timeseries_file}")
