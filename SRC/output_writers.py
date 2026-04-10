"""Output management for the FEM solver.

One public entry point — OutputManager — owns all output files and field
references.  Individual writer classes handle one file each; the manager
delegates to them and shields the solver from file-level details.

Writers:
  - TimeseriesWriter      — one scalar record per timestep (pressure, uz, errors, …)
  - PressureProfileWriter — vertical centerline pressure profile per timestep
  - InvariantsWriter      — stress invariants sampled on a regular spatial grid
  - VTXWriter (dolfinx)   — ParaView BP4 field output

Public lifecycle (all on OutputManager):
    mgr.configure_vtx(domain_comm, functions)   # before open — registers field refs
    mgr.open()                                   # create all files
    mgr.write(time, record, p_values)            # once per timestep
    mgr.close()                                  # flush and close all files

All NetCDF methods silently no-op on MPI ranks ≠ 0 (via @_rank0 decorator).
Material parameters and mesh dimensions are written as NetCDF global attributes.
"""

import os
import numpy as np
import netCDF4 as nc4
from dolfinx import io
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

from config import CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank0(method):
    """Method decorator: execute only on MPI rank 0, silently skip otherwise."""
    def wrapper(self, *args, **kwargs):
        if self._comm.rank == 0:
            return method(self, *args, **kwargs)
    return wrapper


#
#
def _ensure_dir(path: str):
    """Create parent directories for path if they do not already exist."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


#
#
def _write_material_attrs(ds):
    """Write material parameters and geometry as NetCDF global attributes."""
    m = CONFIG.materials
    for k in ('E', 'nu', 'alpha', 'perm', 'visc', 'M'):
        setattr(ds, k, getattr(m, k))
    ds.Re = CONFIG.mesh.Re
    ds.H  = CONFIG.mesh.H


# ---------------------------------------------------------------------------
# Timeseries writer
# ---------------------------------------------------------------------------

class TimeseriesWriter:
    """Writes one scalar record per timestep to a NetCDF time-series file.

    Variables are listed in _VARS.  Each is stored as a 1-D array indexed
    by time.  Material attributes and geometry are stored as global attributes
    so the file is self-contained for post-processing.
    """

    _VARS = [
        'sig_zz_applied',
        'pressure_at_base', 'pressure_mean', 'pressure_p10', 'pressure_p90',
        'uz_at_bottom', 'uz_at_top',
        'volume_drained',
        'analytical_pressure', 'analytical_uz', 'analytical_volume',
        'pressure_error_percent', 'uz_error_percent', 'volume_error_percent',
    ]

    def __init__(self, comm):
        self._path = CONFIG.output.timeseries
        self._comm = comm
        self._ds   = None
        self._idx  = 0

    @_rank0
    def open(self):
        _ensure_dir(self._path)
        ds = nc4.Dataset(self._path, 'w', format='NETCDF3_64BIT_OFFSET')
        ds.createDimension('time', None)   # unlimited — grows each timestep
        ds.createVariable('time', 'f8', ('time',))
        for name in self._VARS:
            ds.createVariable(name, 'f8', ('time',))
        _write_material_attrs(ds)
        self._ds  = ds
        self._idx = 0
        print(f"✓ Timeseries NC opened: {self._path}")

    @_rank0
    def append(self, time: float, record: dict):
        """Append one timestep record.  record must contain all keys in _VARS."""
        i = self._idx
        self._ds['time'][i] = time
        for name in self._VARS:
            self._ds[name][i] = record[name]
        self._ds.sync()
        self._idx += 1

    @_rank0
    def close(self):
        self._ds.close()
        print(f"✓ Timeseries saved: {self._path}")


# ---------------------------------------------------------------------------
# Pressure profile writer
# ---------------------------------------------------------------------------

class PressureProfileWriter:
    """Writes the vertical centerline pressure profile each timestep.

    The profile is sampled by VerticalLineSampler (passed at construction)
    which performs cubic-spline interpolation from the pressure DOFs onto
    a fixed set of z-coordinates stored once in the NetCDF file.
    """

    def __init__(self, comm, v_sampler):
        self._path    = CONFIG.output.pressure_profile
        self._comm    = comm
        self._sampler = v_sampler
        self._ds      = None
        self._idx     = 0

    @_rank0
    def open(self):
        _ensure_dir(self._path)
        n  = self._sampler.num_points
        ds = nc4.Dataset(self._path, 'w', format='NETCDF3_64BIT_OFFSET')
        ds.createDimension('z_point', n)
        ds.createDimension('time', None)
        z_var = ds.createVariable('z_coord', 'f8', ('z_point',))
        ds.createVariable('time',     'f8', ('time',))
        ds.createVariable('pressure', 'f8', ('time', 'z_point'))
        z_var[:]                 = self._sampler.get_z_coordinates()
        z_var.units              = 'm'
        z_var.long_name          = 'Height along centerline (r=0)'
        ds['time'].units         = 's'
        ds['pressure'].units     = 'Pa'
        ds['pressure'].long_name = 'Pore pressure along vertical centerline'
        _write_material_attrs(ds)
        self._ds  = ds
        self._idx = 0
        print(f"✓ Pressure profile NC opened: {self._path}")

    @_rank0
    def append(self, time: float, p_values):
        """Sample and store the pressure profile at the current timestep."""
        profile = self._sampler.sample(p_values)
        i = self._idx
        self._ds['time'][i]        = time
        self._ds['pressure'][i, :] = profile
        self._ds.sync()
        self._idx += 1

    @_rank0
    def close(self):
        self._ds.close()
        print(f"✓ Pressure profile saved: {self._path}")


# ---------------------------------------------------------------------------
# Stress invariants writer
# ---------------------------------------------------------------------------

class InvariantsWriter:
    """Samples stress invariants on a regular (r, z) grid each timestep.

    The grid is built once in __init__ using dolfinx bounding-box collision
    to find which mesh cell contains each grid point.  Invalid points (outside
    the mesh) are silently discarded.  At each timestep the projected stress
    and pressure fields are evaluated at the valid grid points and the
    following invariants are computed and stored:

      I1      = tr(σ_total)                  [Pa]
      J2      = 0.5 · s:s  (deviatoric)      [Pa²]
      p_pore  = pore pressure                [Pa]
      p_eff   = I1/3 + α·p_pore  (Biot)     [Pa]
      p_eff_t = I1/3 + p_pore    (Terzaghi) [Pa]
      q       = √(3·J2)                      [Pa]
      u_r     = radial displacement          [m]
    """

    def __init__(self, comm, domain):
        self._path   = CONFIG.output.invariants_nc
        self._comm   = comm
        self._domain = domain
        self._ds     = None
        self._idx    = 0
        self._build_grid()

    def _build_grid(self):
        """Build the regular sampling grid and locate containing cells.

        Grid aspect ratio matches the domain (Re / H_mesh) so points are
        approximately square.  Points outside the mesh are discarded.
        """
        n      = CONFIG.output.n_invariant_points
        Re     = CONFIG.mesh.Re
        H_mesh = CONFIG.mesh.H / 2
        ratio  = Re / H_mesh
        n_z    = max(2, round(np.sqrt(n / ratio)))
        n_r    = max(2, round(n / n_z))

        r_1d   = np.linspace(0, Re,     n_r)
        z_1d   = np.linspace(0, H_mesh, n_z)
        R, Z   = np.meshgrid(r_1d, z_1d)
        r_flat = R.ravel();  z_flat = Z.ravel()
        pts    = np.column_stack([r_flat, z_flat, np.zeros(len(r_flat))])

        tree       = bb_tree(self._domain, self._domain.topology.dim)
        candidates = compute_collisions_points(tree, pts)
        colliding  = compute_colliding_cells(self._domain, candidates, pts)

        valid_idx, valid_cells = [], []
        for i in range(len(pts)):
            lnk = colliding.links(i)
            if len(lnk) > 0:
                valid_idx.append(i)
                valid_cells.append(int(lnk[0]))

        valid_idx   = np.array(valid_idx)
        self._pts   = pts[valid_idx]
        self._cells = np.array(valid_cells)
        self._r     = r_flat[valid_idx]
        self._z     = z_flat[valid_idx]

        if self._comm.rank == 0:
            print(f"✓ Invariant sampler: {len(valid_idx)} valid points "
                  f"on {n_r}×{n_z} grid (requested {n})")

    @_rank0
    def open(self):
        _ensure_dir(self._path)
        n  = len(self._pts)
        ds = nc4.Dataset(self._path, 'w', format='NETCDF3_64BIT_OFFSET')
        ds.createDimension('point', n)
        ds.createDimension('time',  None)

        r_var = ds.createVariable('r',    'f8', ('point',))
        z_var = ds.createVariable('z',    'f8', ('point',))
        ds.createVariable('time',   'f8', ('time',))
        for name in ('I1', 'J2', 'p_pore', 'p_eff', 'p_eff_t', 'q', 'u_r'):
            ds.createVariable(name, 'f4', ('time', 'point'))

        r_var[:] = self._r;  r_var.units = 'm';  r_var.long_name = 'Radial coordinate'
        z_var[:] = self._z;  z_var.units = 'm';  z_var.long_name = 'Axial coordinate'

        ds['time'].units     = 's'
        ds['I1'].units       = 'Pa';    ds['I1'].long_name      = 'First stress invariant I1 = tr(σ_total)'
        ds['J2'].units       = 'Pa²';   ds['J2'].long_name      = 'Second deviatoric invariant J2 = 0.5 s:s'
        ds['p_pore'].units   = 'Pa';    ds['p_pore'].long_name  = 'Pore pressure'
        ds['p_eff'].units    = 'Pa';    ds['p_eff'].long_name   = 'Biot effective mean stress = I1/3 + α·p'
        ds['p_eff_t'].units  = 'Pa';    ds['p_eff_t'].long_name = 'Terzaghi effective mean stress (α=1)'
        ds['q'].units        = 'Pa';    ds['q'].long_name       = 'Deviatoric stress q = √(3·J2)'
        ds['u_r'].units      = 'm';     ds['u_r'].long_name     = 'Radial displacement'

        _write_material_attrs(ds)
        self._ds  = ds
        self._idx = 0
        print(f"✓ Invariants NC opened: {self._path}  ({n} sample points)")

    @_rank0
    def append(self, time: float, sigma_rr_f, sigma_tt_f, sigma_zz_f,
               sigma_rz_f, p_out_f, u_out_f, alpha: float):
        """Evaluate fields at grid points, compute invariants, and write one record."""
        pts   = self._pts;  cells = self._cells
        srr   = sigma_rr_f.eval(pts, cells)[:, 0]
        stt   = sigma_tt_f.eval(pts, cells)[:, 0]
        szz   = sigma_zz_f.eval(pts, cells)[:, 0]
        srz   = sigma_rz_f.eval(pts, cells)[:, 0]
        p_pore = p_out_f.eval(pts, cells)[:, 0]
        u_r    = u_out_f.eval(pts, cells)[:, 0]

        I1      = srr + stt + szz
        p_mean  = I1 / 3.0
        J2      = 0.5 * ((srr - p_mean)**2 + (stt - p_mean)**2 + (szz - p_mean)**2) + srz**2
        p_eff   = p_mean + alpha * p_pore
        p_eff_t = p_mean + p_pore             # Terzaghi: α = 1
        q       = np.sqrt(np.maximum(3.0 * J2, 0.0))

        i = self._idx
        self._ds['time'][i]       = time
        self._ds['I1'][i, :]      = I1
        self._ds['J2'][i, :]      = J2
        self._ds['p_pore'][i, :]  = p_pore
        self._ds['p_eff'][i, :]   = p_eff
        self._ds['p_eff_t'][i, :] = p_eff_t
        self._ds['q'][i, :]       = q
        self._ds['u_r'][i, :]     = u_r
        self._ds.sync()
        self._idx += 1

    @_rank0
    def close(self):
        self._ds.close()
        print(f"✓ Invariants saved: {self._path}")


# ---------------------------------------------------------------------------
# Output manager — single entry point for the solver
# ---------------------------------------------------------------------------

class OutputManager:
    """Owns and coordinates all output writers for one simulation run.

    The solver creates one OutputManager, registers its field functions via
    configure_vtx(), then calls open() / write() / close() throughout the
    time loop.  Field functions are updated in-place by the solver before
    each write() call; the manager reads them by reference.

    Usage::

        mgr = OutputManager(comm, domain, v_sampler)
        mgr.configure_vtx(domain.comm, [u_out, p_out, ...],
                          sigma_rr_out, sigma_tt_out, sigma_zz_out,
                          sigma_rz_out, p_out, u_out)
        mgr.open()
        for ...:
            # solver updates field functions in-place
            mgr.write(time, record, p_values)
        mgr.close()
    """

    def __init__(self, comm, domain, v_sampler):
        self._comm = comm
        self._ts   = TimeseriesWriter(comm)
        self._pp   = PressureProfileWriter(comm, v_sampler)
        self._inv  = InvariantsWriter(comm, domain)
        self._vtx  = None
        # Field references set by configure_vtx — updated in-place by the solver
        self._stress_fields = None   # (sigma_rr, sigma_tt, sigma_zz, sigma_rz, p_out, u_out)

    def configure_vtx(self, domain_comm, vtx_functions,
                      sigma_rr_f, sigma_tt_f, sigma_zz_f,
                      sigma_rz_f, p_out_f, u_out_f):
        """Set up the VTX (ParaView) writer and register stress field references.

        Must be called before open().  vtx_functions is the list of
        dolfinx.fem.Function objects to stream; the stress/pressure/displacement
        references are used by InvariantsWriter on each write() call.
        """
        path = CONFIG.output.results
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._vtx = io.VTXWriter(domain_comm, path, vtx_functions, engine="BP4")
        self._stress_fields = (sigma_rr_f, sigma_tt_f, sigma_zz_f,
                               sigma_rz_f, p_out_f, u_out_f)
        if self._comm.rank == 0:
            print(f"✓ VTX output configured: {path}")

    #
    #
    def open(self):
        """Open all output files."""
        self._ts.open()
        self._pp.open()
        self._inv.open()

    #
    #
    def write(self, time: float, record: dict, p_values):
        """Write one timestep to all output files.

        p_values  — pressure DOF array (from W.sub(1).collapse())
        record    — dict matching TimeseriesWriter._VARS
        Field functions registered in configure_vtx() are read by reference
        (the solver updates them in-place before calling this).
        """
        self._ts.append(time, record)
        self._pp.append(time, p_values)
        self._inv.append(time, *self._stress_fields, CONFIG.materials.alpha)
        self._vtx.write(time)

    #
    #
    def close(self):
        """Flush and close all output files."""
        self._vtx.close()
        self._ts.close()
        self._pp.close()
        self._inv.close()
