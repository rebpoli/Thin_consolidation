import numpy as np
import xarray as xr

import config
from mesh_generator import CylinderMesh

from dolfinx import mesh, fem, io
from mpi4py import MPI


class Analytical1DConsolidation:
    def __init__(self, n_spatial_points: int = 50):
        cfg = config.get()
        
        self.sig0  = -1e5
        self.material_cfg  = cfg.materials
        self.mesh_cfg      = cfg.mesh
        self.numerical_cfg = cfg.numerical
        
        self.times = self.numerical_cfg.expected_time_list()
        
        self.n_z     = n_spatial_points
        self.z_coords = np.linspace(0, self.mesh_cfg.H, self.n_z)

        self._compute_derived_properties()
        
    def compute() :
        self._compute_history()
    
    def _compute_derived_properties(self):
        mu    = self.material_cfg.mu
        S     = self.material_cfg.S
        eta   = self.material_cfg.eta
        sig0 = self.sig0
        self.p0  = -sig0 * eta / mu / S
    
    def _compute_history(self):
        n_times = len(self.times)
        
        self._pressure_field = np.zeros((n_times, self.n_z))
        self._uz_field       = np.zeros((n_times, self.n_z))

        H  = self.mesh_cfg.H
        
        for i_t, t in enumerate(self.times):
            for i_z, z in enumerate(self.z_coords):
                self._pressure_field[i_t, i_z] = self._pressure(t, z)
                self._uz_field[i_t, i_z]       = self._uz(t, z)
        
        self._pressure_at_top = self._pressure_field[:, -1]
        self._uz_at_bottom          = self._uz_field[:, 0]
        self._volume_drained     = self._compute_volume_drained()
    
    #
    #
    def _pressure(self, t: float, z: float) -> float:
        if t == 0:
            return self.p0
        
        H    = self.mesh_cfg.H
        c_v  = self.material_cfg.c_v

        T_v = c_v * t / (H**2) / 4
        
        # Ref: Cheng,2016
        F1 = 0.0
        for n in range(50):
            m = (2*n + 1) * np.pi / 2
            F1 += (2.0 / m) * np.sin(m * z / H) * np.exp(-4*m**2 * T_v)
        
        return self.p0 * F1
    
    #
    #
    def _uz(self, t: float, z: float) -> float:
        H = self.mesh_cfg.H
        S     = self.material_cfg.S
        K_u   = self.material_cfg.K_u
        nu_u  = self.material_cfg.nu_u
        nu    = self.material_cfg.nu
        mu    = self.material_cfg.mu
        eta   = self.material_cfg.eta
        c_v   = self.material_cfg.c_v
        
        T_v = c_v * t / (H**2) / 4
#         print(f"t=>t*: {t} => {T_v}")
        
        # Ref: Cheng,2016
        F2 = 0.0
        for n in range(50):
            m = (2*n + 1) * np.pi / 2
            F2 += (2.0 / m**2) * np.cos(m * z / H) * (1 - np.exp(-4*m**2 * T_v))

#         print(f"t*:{T_v} z*:{z/H} => F2={F2} p0:{self.p0}")
        
        uz  = -self.sig0 * H / 2 / mu * (1-2*nu_u) / (1-nu_u) * ( 1 - z/H )
        uz += -self.sig0 * H * F2 * ( nu_u - nu ) / ( 2 * mu ) / ( 1 - nu_u ) / ( 1 - nu )
        
        return uz
    
    #
    #
    def _V_drained( self, time ) :
        H    = self.mesh_cfg.H
        Re   = self.mesh_cfg.Re
        c_v  = self.material_cfg.c_v
        perm  = self.material_cfg.perm
        visc  = self.material_cfg.visc
        p0 = self.p0

        t_s = c_v * time / (4 * H**2)  
        F = 0.0
        for n in range(500):
            m = (2*n + 1) * np.pi
            F += (1.0 / m**2) * (1.0 - np.exp(-m**2 * t_s))

        vd = np.pi * Re**2 * (perm/visc) * (2*p0/H) * (4*H**2/c_v) * F

        return vd


    #
    #
    def _compute_volume_drained(self) -> np.ndarray:
        
        volume_drained = np.zeros(len(self.times))
        
        # DeltaV = ∫Q dτ = 
        #   = π·Re² (k/μ) (2p₀/H) (4H²/c_v) Σ (1/m²) [1 - exp(-m²Tv)]
        for i_t, t in enumerate(self.times):
            if t == 0: continue

            volume_drained[i_t] = self._V_drained(t)

        return volume_drained
    
    #
    #
    def pressure_at_top(self, t: float) -> float: return self._pressure(t,self.mesh_cfg.H)
    def uz_at_bottom(self, t: float) -> float: return self._uz(t,0)
    def volume_drained(self, t: float) -> float: return self._V_drained(t)
    
    def get_history(self) -> dict:
        return {
            'times':              self.times,
            'pressure_at_bottom': self._pressure_at_top,
            'uz_at_bottom':          self._uz_at_bottom,
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
                    
                    for n in range(50):
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
                for n in range(50):
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
                'uz_at_bottom':          (['time'], history['uz_at_bottom']),
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
