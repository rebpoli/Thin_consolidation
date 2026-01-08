#!/usr/bin/env -S python -u

import numpy as np
import matplotlib.pyplot as plt
import gmsh
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, io
from dolfinx.io import gmsh as gmshio
import dolfinx.fem.petsc
import os

comm = MPI.COMM_WORLD
rank = comm.rank

# ==============================================================================
# MESH GENERATION
# ==============================================================================
def create_mesh(Re, H, N=10):
    hsize = H/N
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gdim = 2
    model_rank = 0
    gmsh.model.add("Cylinder")

    geom = gmsh.model.geo

    # Create rectangle in (r,z) plane
    p1 = geom.add_point(0, 0, 0)
    p2 = geom.add_point(Re, 0, 0)
    p3 = geom.add_point(Re, H, 0)
    p4 = geom.add_point(0, H, 0)

    bottom = geom.add_line(p1, p2)
    right = geom.add_line(p2, p3)
    top = geom.add_line(p3, p4)
    left = geom.add_line(p4, p1)

    boundary = geom.add_curve_loop([bottom, right, top, left])
    surf = geom.add_plane_surface([boundary])

    geom.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hsize)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hsize)

    gmsh.model.addPhysicalGroup(gdim, [surf], 1)
    gmsh.model.addPhysicalGroup(gdim - 1, [bottom], 1, name="bottom")
    gmsh.model.addPhysicalGroup(gdim - 1, [right], 2, name="right")
    gmsh.model.addPhysicalGroup(gdim - 1, [top], 3, name="top")
    gmsh.model.addPhysicalGroup(gdim - 1, [left], 4, name="left")

    gmsh.model.mesh.generate(gdim)

    mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, model_rank, gdim=gdim)
    domain = mesh_data.mesh
    facets = mesh_data.facet_tags

    gmsh.finalize()

    print(f"✓ Mesh created: {domain.topology.index_map(gdim).size_global} cells")

    return domain, facets


#
#
#
class Materials :
    def __init__( self, domain ) :
        self._floats = {
                'E': 1.44e10,
                'nu': 0.2,
                'alpha': 0.78,
                'perm': 1.9e-13,
                'visc': 1e-3,
                'M': 1.35e10,
                'sig0' : -1e5
        }
        E = self.float("E")
        nu = self.float("nu")
        self._floats['mu'] = E / 2 / ( 1 + nu )
        self._floats['lmbda'] = E*nu / (1+nu) / (1-2*nu)
        # Create the fems
        self._fems = {k: fem.Constant(domain, v) for k, v in self._floats.items()}

    # Getters
    def float(self, pname) :
        return self._floats[pname];
    def fem(self, pname) :
        return self._fems[pname];



# ==============================================================================
# FUNCTION SPACES (USING UFL MIXEDELEMENT)
# ==============================================================================
def setup_function_spaces(domain):
    """
    Create mixed function space using UFL MixedElement approach
    """
    from basix.ufl import element, mixed_element

    gdim = domain.geometry.dim
    cell_name = domain.topology.cell_name()

    # Define individual elements using basix
    # Displacement: P2 vector
    V_elem = element("Lagrange", cell_name, 2, shape=(gdim,))

    # Pressure: P1 scalar
    Q_elem = element("Lagrange", cell_name, 1)

    # Create mixed element
    mixed_elem = mixed_element([V_elem, Q_elem])

    # Create mixed function space
    W = fem.functionspace(domain, mixed_elem)

    print(f"✓ Function space created: {W.dofmap.index_map.size_global} total DOFs")

    return W, gdim

# ==============================================================================
# STRAIN AND STRESS DEFINITIONS
# ==============================================================================
def define_strain_stress(domain):
    """
    Define strain and stress operators for axisymmetric formulation

    Parameters:
    -----------
    domain : dolfinx.mesh.Mesh
        The computational domain

    Returns:
    --------
    functions : dict
        Dictionary containing strain and stress functions
    """
    x = ufl.SpatialCoordinate(domain)

    def eps(w):
        """Strain tensor in axisymmetric coordinates"""
        e_rr = w[0].dx(0)
        e_tt = w[0] / x[0]
        e_zz = w[1].dx(1)
        e_rz = 0.5 * (w[0].dx(1) + w[1].dx(0))

        return ufl.sym(ufl.as_tensor([
            [e_rr, 0,    e_rz],
            [0,    e_tt, 0   ],
            [e_rz, 0,    e_zz]
        ]))

    def eps_v(w):
        """Volumetric strain"""
        return w[0].dx(0) + w[0] / x[0] + w[1].dx(1)

    def sigma_eff(w):
        """Effective stress tensor"""
        lmbda = PROPS.fem('lmbda')
        mu = PROPS.fem('mu')
        return lmbda * ufl.tr(eps(w)) * ufl.Identity(3) + 2.0 * mu * eps(w)

    return {'eps': eps, 'eps_v': eps_v, 'sigma_eff': sigma_eff, 'x': x}


# ==============================================================================
# WEAK FORM
# ==============================================================================
def setup_weak_form(W, domain, facets, strain_stress, _dt):
    """
    Assemble weak form for time-dependent poroelasticity

    Parameters:
    -----------
    W : dolfinx.fem.FunctionSpace
        Mixed function space
    domain : dolfinx.mesh.Mesh
        Computational domain
    facets : dolfinx.mesh.MeshTags
        Boundary facet tags
    strain_stress : dict
        Strain and stress functions
    _dt : float
        Time step size

    Returns:
    --------
    forms : dict
        Dictionary containing bilinear and linear forms
    """
    # Extract functions
    eps = strain_stress['eps']
    eps_v = strain_stress['eps_v']
    sigma_eff = strain_stress['sigma_eff']
    x = strain_stress['x']
    r = x[0]

    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Previous solution
    wh_old = fem.Function(W, name="Previous")
    u_old, p_old = ufl.split(wh_old)

    # Time step constant
    dt = fem.Constant(domain, _dt)

    # Measures
    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facets)

    # Get the properties
    alpha = PROPS.fem('alpha')
    perm = PROPS.fem('perm')
    visc = PROPS.fem('visc')
    sig0 = PROPS.fem('sig0')
    M = PROPS.fem('M')

    # BILINEAR FORM
    a_form = (
    # Mechanical equilibrium
              + ufl.inner(  sigma_eff(u),   eps(v)  ) * r * dx
              - alpha * p * eps_v(v) * r * dx

    # Flow equation with time derivative
              + alpha / dt * eps_v(u) * q * r * dx
              + perm/visc * ufl.inner(ufl.grad(p), ufl.grad(q)) * r * dx
              + (1/M) * p/dt * q * r * dx
      )

    traction = ufl.as_vector([0, sig0])

    # LINEAR FORM (RHS)
    L_form = (
    # Applied traction
      + ufl.inner(traction, v) * r * ds(3)  # Top boundary

        # Old volumetric strain
          + alpha / dt * eps_v(u_old) * q * r * dx
          + 1/M * p_old/dt * q * r * dx

        # Source term for pressure
          +  fem.Constant(domain, 0.0) * q * r * dx
      )


    print("✓ Weak forms assembled")

    return {
        'a_form': a_form,
        'L_form': L_form,
        'wh_old': wh_old,
        'dx': dx,
        'ds': ds
    }


# ==============================================================================
# BOUNDARY CONDITIONS
# ==============================================================================
def setup_boundary_conditions(W, domain, facets, gdim):
    """
    Define boundary conditions for coupled problem

    Returns two sets of boundary conditions:
    - bcs_t0: For t=0 (no pressure BC - undrained at top)
    - bcs: For t>0 (includes pressure BC - drainage active)

    Physical interpretation:
    - At t=0: Load just applied, no drainage yet, p=p0 everywhere
    - At t>0: Drainage BC active, p=0 at top surface
    """
    # First level collapse - get displacement and pressure spaces
    W0_collapsed, W0_map = W.sub(0).collapse()
    W1_collapsed, W1_map = W.sub(1).collapse()

    # Second level collapse - get individual displacement components
    W0_r_collapsed, W0_r_map = W0_collapsed.sub(0).collapse()
    W0_z_collapsed, W0_z_map = W0_collapsed.sub(1).collapse()

    # ==========================================
    # DISPLACEMENT BCs (same for all times)
    # ==========================================
    bcs_displacement = []

    # Bottom: u_z = 0
    bottom_facets = facets.find(1)
    bottom_dofs_z = fem.locate_dofs_topological(
        (W.sub(0).sub(1), W0_z_collapsed),
        gdim - 1,
        bottom_facets
    )
    u_bottom = fem.Function(W0_z_collapsed)
    u_bottom.x.array[:] = 0.0
    bcs_displacement.append(fem.dirichletbc(u_bottom, bottom_dofs_z, W.sub(0).sub(1)))

    # Axis (left): u_r = 0
    axis_facets = facets.find(4)
    axis_dofs_r = fem.locate_dofs_topological(
        (W.sub(0).sub(0), W0_r_collapsed),
        gdim - 1,
        axis_facets
    )
    u_axis = fem.Function(W0_r_collapsed)
    u_axis.x.array[:] = 0.0
    bcs_displacement.append(fem.dirichletbc(u_axis, axis_dofs_r, W.sub(0).sub(0)))

    # Right boundary: u_r = 0
    axis_facets = facets.find(2)
    axis_dofs_r = fem.locate_dofs_topological(
        (W.sub(0).sub(0), W0_r_collapsed),
        gdim - 1,
        axis_facets
    )
    u_axis = fem.Function(W0_r_collapsed)
    u_axis.x.array[:] = 0.0
    bcs_displacement.append(fem.dirichletbc(u_axis, axis_dofs_r, W.sub(0).sub(0)))

    # ==========================================
    # PRESSURE BC (only for t > 0)
    # ==========================================
    # Top: p = 0 (drainage boundary condition)
    top_facets = facets.find(3)
    top_dofs_p = fem.locate_dofs_topological(
        (W.sub(1), W1_collapsed),
        gdim - 1,
        top_facets
    )
    p_top = fem.Function(W1_collapsed)
    p_top.x.array[:] = 0.0
    bc_pressure_top = fem.dirichletbc(p_top, top_dofs_p, W.sub(1))

    # ==========================================
    # CREATE TWO BC SETS
    # ==========================================
    # For t=0: Only displacement BCs (no drainage yet)
    bcs_t0 = bcs_displacement.copy()

    # For t>0: Displacement BCs + pressure BC (drainage active)
    bcs = bcs_displacement.copy()
    bcs.append(bc_pressure_top)

    print(f"✓ Boundary conditions defined:")
    print(f"  - Displacement BCs (all times): {len(bcs_displacement)}")
    print(f"    • Bottom: u_z = 0")
    print(f"    • Axis/sides: u_r = 0")
    print(f"  - For t=0 (undrained): {len(bcs_t0)} BCs total")
    print(f"  - For t>0 (draining): {len(bcs)} BCs total (includes p=0 at top)")

    return {'t0': bcs_t0, 'consolidation': bcs}


# ==============================================================================
# INITIAL CONDITION
# ==============================================================================
def solve_initial_condition(W, forms, bcs):
    """
    Solve for initial condition (t=0, instantaneous loading)

    This computes the UNDRAINED response at the instant of loading.
    No drainage has occurred yet, so the drainage BC p=0 at top is NOT applied.

    Parameters:
    -----------
    W : dolfinx.fem.FunctionSpace
        Mixed function space
    forms : dict
        Dictionary containing weak forms
    bcs : list
        Boundary conditions for t=0 (without pressure BC at top)

    Returns:
    --------
    wh : dolfinx.fem.Function
        Initial solution (undrained state)
    problem : dolfinx.fem.petsc.LinearProblem
        Problem object for time stepping
    """
    print("\n=== Computing initial condition (t=0, undrained state) ===")
    print("    Note: Drainage BC NOT applied at t=0")

    wh = fem.Function(W, name="Current")
    forms['wh_old'].x.array[:] = 0.0  # Start from zero

    problem = fem.petsc.LinearProblem(
        forms['a_form'],
        forms['L_form'],
        u=wh,
        bcs=bcs,
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "lu",
            "ksp_rtol": 1e-8,
        },
        petsc_options_prefix="terzaghi_"
    )

    problem.solve()

    # Copy to old solution
    forms['wh_old'].x.array[:] = wh.x.array

    # Print initial values
    uh_init, ph_init = wh.split()
    print(f"t = 0.0 s:")
    print(f"  Max u_z: {uh_init.sub(1).collapse().x.array.max():.6e}")
    print(f"  Max p: {ph_init.collapse().x.array.max():.6e}")
    print(f"  Min p: {ph_init.collapse().x.array.min():.6e}")

    return wh, problem


# ==============================================================================
# ANALYTICAL SOLUTION (1D TERZAGHI CONSOLIDATION)
# ==============================================================================
def analytical_solution_terzaghi(z, t, H, n_terms=50):
    """
    Compute analytical solution for 1D Terzaghi consolidation

    Parameters:
    -----------
    z : float or array
        Vertical coordinate (0 at bottom, H at top)
    t : float
        Time
    H : float
        Height of consolidating layer
    n_terms : int
        Number of terms in Fourier series

    Returns:
    --------
    p : float or array
        Excess pore pressure at (z, t)
    """

    if t == 0:
        return p0 * np.ones_like(z) if hasattr(z, '__len__') else p0

    M = PROPS.float('M')
    perm = PROPS.float('perm')
    visc = PROPS.float('visc')
    alpha = PROPS.float('alpha')
    nu = PROPS.float('nu')
    mu = PROPS.float('mu')
    sig0 = PROPS.float('sig0')

    eta = alpha * ( 1 - 2*nu ) / ( 2 * ( 1 - nu ) )

    S = 1/M + (alpha**2)*(1-2*nu) / (2*mu*(1-nu))
    c_v = perm/visc/S

    # Initial pressure
    p0 = -sig0 * eta / mu / S

    # Dimensionless stuff
    t_s = c_v * t / (4*(H**2))  # Time factor
    z_s = np.asarray(z/H)

    f1 = np.zeros_like(z, dtype=float)
    for n in range(n_terms):
        m = ( 2*n + 1 ) * np.pi # odd nums
        f1 += (4 / m ) * np.sin(m * z_s/2) * np.exp(-m**2 * t_s)

    p = f1 * p0

    return p


# def degree_of_consolidation_analytical(t, H, c_v, n_terms=50):
#     """
#     Compute degree of consolidation from analytical solution

#     Parameters:
#     -----------
#     t : float or array
#         Time(s)
#     H : float
#         Height of consolidating layer
#     c_v : float
#         Consolidation coefficient
#     n_terms : int
#         Number of terms in series

#     Returns:
#     --------
#     U : float or array
#         Degree of consolidation (0 to 100%)
#     """
#     t = np.asarray(t)

#     # Handle t=0 case
#     if np.isscalar(t):
#         if t == 0:
#             return 0.0
#         T_v = c_v * t / (H**2)
#     else:
#         T_v = np.zeros_like(t)
#         mask = t > 0
#         T_v[mask] = c_v * t[mask] / (H**2)

#     U = np.ones_like(T_v, dtype=float)

#     for n in range(n_terms):
#         M = (2*n + 1) * np.pi / 2
#         U -= (2 / M**2) * np.exp(-M**2 * T_v)

#     return U * 100  # Convert to percentage


# def settlement_analytical(t, H, c_v, S_final, n_terms=50):
#     """
#     Compute settlement from analytical solution

#     Parameters:
#     -----------
#     t : float or array
#         Time(s)
#     H : float
#         Height
#     c_v : float
#         Consolidation coefficient
#     S_final : float
#         Final settlement
#     n_terms : int
#         Number of terms

#     Returns:
#     --------
#     S : float or array
#         Settlement at time t
#     """
#     U = degree_of_consolidation_analytical(t, H, c_v, n_terms) / 100
#     return S_final * U


# ==============================================================================
# TIME STEPPING WITH ANALYTICAL COMPARISON
# ==============================================================================
def time_stepping_loop(wh, problem, forms, W, bcs_consolidation, T_final, num_steps,
                       H, strain_stress, domain, output_dir="results"):
    """
    Perform time integration with analytical solution comparison

    Parameters:
    -----------
    wh : dolfinx.fem.Function
        Initial solution from t=0
    problem : dolfinx.fem.petsc.LinearProblem
        Problem object (will be recreated with consolidation BCs)
    forms : dict
        Dictionary containing weak forms
    W : dolfinx.fem.FunctionSpace
        Mixed function space
    bcs_consolidation : list
        Boundary conditions for t>0 (includes p=0 at top)
    T_final : float
        Final time
    num_steps : int
        Number of time steps
    H : float
        Height
    strain_stress : dict
        Dictionary containing strain and stress functions
    domain : dolfinx.mesh.Mesh
        Computational domain
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Recreating problem with consolidation BCs (p=0 at top) ===")

    # Recreate problem with consolidation boundary conditions (includes p=0 at top)
    problem = fem.petsc.LinearProblem(
        forms['a_form'],
        forms['L_form'],
        u=wh,
        bcs=bcs_consolidation,
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "lu",
            "ksp_rtol": 1e-8,
        },
        petsc_options_prefix="terzaghi_"
    )
    os.makedirs(output_dir, exist_ok=True)

    dt = T_final / num_steps
    domain = wh.function_space.mesh

    # Collapsed spaces for the mixed function
    W0_collapsed, _ = W.sub(0).collapse()
    W1_collapsed, _ = W.sub(1).collapse()

    # Create P1 spaces for output (XDMF requires same degree as mesh)
    gdim = domain.geometry.dim
    V_P1 = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    Q_P1 = fem.functionspace(domain, ("Lagrange", 1))

    u_file = io.XDMFFile(MPI.COMM_WORLD, f"{output_dir}/displacement.xdmf", "w")
    p_file = io.XDMFFile(MPI.COMM_WORLD, f"{output_dir}/pressure.xdmf", "w")
    u_file.write_mesh(domain)
    p_file.write_mesh(domain)

    # Extract and collapse initial solution
    uh_init, ph_init = wh.split()
    uh_init_collapsed = uh_init.collapse()
    ph_init_collapsed = ph_init.collapse()

    times = [0.0]

    # GATHER FROM ALL MPI PROCS
    max_p = ph_init_collapsed.x.array.max()
    max_p = comm.allreduce(max_p, op=MPI.MAX)
    max_pressures = [max_p]

    max_displacements = [uh_init_collapsed.sub(1).collapse().x.array.max()]

    # Analytical solution storage
    U_analytical = [0.0]
    max_pressures_analytical = [0.0]

    # Volume variation storage
    volume_history = {'times': [], 'delta_V': []}

    # Create P1 functions for saving
    u_save = fem.Function(V_P1, name="Displacement")
    p_save = fem.Function(Q_P1, name="Pressure")

    # Interpolate and save initial condition
    u_save.interpolate(uh_init_collapsed)
    p_save.interpolate(ph_init_collapsed)
    u_file.write_function(u_save, 0.0)
    p_file.write_function(p_save, 0.0)

    # Track volume variation at t=0
    if rank == 0:
        print("\n=== Computing Volume Variation ===")
    delta_V_init = track_volume_variation(wh, W, strain_stress, domain, 0.0, volume_history)

    print("\n=== Starting time integration ===")
    print(f"{'Time [s]':>10} {'Max p (FEM) [kPa]':>18} {'Max p (Analytical) [kPa]':>25} "
          f"{'U (FEM) [%]':>12} {'U (Analytical) [%]':>18} {'Error U [%]':>12}")
    print("-" * 115)

    for step in range(1, num_steps + 1):
        t = step * dt

        problem.solve()

        # Extract and collapse solution
        uh, ph = wh.split()
        uh_collapsed = uh.collapse()
        ph_collapsed = ph.collapse()

        max_p = ph_collapsed.x.array.max()
        max_p = comm.allreduce(max_p, op=MPI.MAX)
        max_uz = uh_collapsed.sub(1).collapse().x.array.max()

        # Compute analytical solution
#         U_anal = degree_of_consolidation_analytical(t, H, c_v)
        # Analytical pressure (at bottom, z=0)
        p_anal_bottom = analytical_solution_terzaghi(0, t, H)

        # Compute degree of consolidation from FEM
        if max_displacements[-1] != max_displacements[0]:
            U_fem = (max_uz - max_displacements[0]) / (max_displacements[-1] - max_displacements[0] + 1e-15) * 100
        else:
            U_fem = 0.0

        times.append(t)
        max_pressures.append(max_p)
        max_displacements.append(max_uz)
#         U_analytical.append(U_anal)
        max_pressures_analytical.append(p_anal_bottom)

        # Print comparison
        if step % 5 == 0 or step == num_steps:
#             error_U = abs(U_fem - U_anal)
            print(f"{t:10.2f} {max_p/1e3:18.3e} {p_anal_bottom/1e3:25.3e} "
                  f"{U_fem:12.2f} {0:18.2f} {0:12.2f}")
#                   f"{U_fem:12.2f} {U_anal:18.2f} {error_U:12.2f}")

        # Interpolate to P1 and save
        u_save.interpolate(uh_collapsed)
        p_save.interpolate(ph_collapsed)
        u_file.write_function(u_save, t)
        p_file.write_function(p_save, t)

        # Track volume variation
        track_volume_variation(wh, W, strain_stress, domain, t, volume_history)

        forms['wh_old'].x.array[:] = wh.x.array

    u_file.close()
    p_file.close()

    print(f"\n✓ Time integration completed")
    print(f"✓ Results saved to {output_dir}/")

    return {
        'times': np.array(times),
        'max_pressures': np.array(max_pressures),
        'max_displacements': np.array(max_displacements),
        'U_analytical': np.array(U_analytical),
        'max_pressures_analytical': np.array(max_pressures_analytical),
        'volume_history': volume_history
    }

# ==============================================================================
# VOLUME VARIATION POSTPROCESSING
# ==============================================================================
def compute_volume_variation(wh, W, strain_stress, domain):
    """
    Compute the volume variation for the whole axisymmetric domain.

    Volume variation is calculated as:
    ΔV = 2π ∫_Ω (α*ε_v + p/M) * r dΩ

    where:
    - α: Biot coefficient
    - ε_v: volumetric strain
    - p: pore pressure
    - M: Biot modulus
    - r: radial coordinate (for axisymmetric revolution)

    Parameters:
    -----------
    wh : dolfinx.fem.Function
        Current solution (displacement and pressure)
    W : dolfinx.fem.FunctionSpace
        Mixed function space
    strain_stress : dict
        Dictionary containing strain functions
    domain : dolfinx.mesh.Mesh
        Computational domain

    Returns:
    --------
    delta_V : float
        Volume variation for the whole domain
    """
    # Extract displacement and pressure from solution
    u, p = wh.split()

    # Get material properties
    alpha = PROPS.fem('alpha')
    M = PROPS.fem('M')

    # Get volumetric strain function
    eps_v = strain_stress['eps_v']

    # Get spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    r = x[0]  # Radial coordinate

    # Define the integrand: (α*ε_v + p/M) * r for axisymmetric case
    # Factor of 2π is included for full revolution
    integrand = (alpha * eps_v(u) + p / M) * r

    # Measure for integration
    dx = ufl.Measure("dx", domain=domain)

    # Assemble the integral
    form = fem.form(integrand * dx)
    local_integral = fem.assemble_scalar(form)

    # Reduce across all MPI processes
    global_integral = comm.allreduce(local_integral, op=MPI.SUM)

    # Multiply by 2π for the axisymmetric revolution
    delta_V = 2.0 * np.pi * global_integral

    return delta_V


def track_volume_variation(wh, W, strain_stress, domain, time, volume_history):
    """
    Track volume variation over time and store in history dictionary.

    Parameters:
    -----------
    wh : dolfinx.fem.Function
        Current solution
    W : dolfinx.fem.FunctionSpace
        Mixed function space
    strain_stress : dict
        Dictionary containing strain functions
    domain : dolfinx.mesh.Mesh
        Computational domain
    time : float
        Current time
    volume_history : dict
        Dictionary to store volume variation history
    """
    # Compute volume variation
    delta_V = compute_volume_variation(wh, W, strain_stress, domain)

    # Store in history
    volume_history['times'].append(time)
    volume_history['delta_V'].append(delta_V)

    if rank == 0:
        print(f"  Volume variation at t={time:.4f} s: ΔV = {delta_V:.6e} m³")

    return delta_V


def save_volume_variation_table(volume_history, filename="volume_variation.dat"):
    """
    Save volume variation history to a table file.

    Parameters:
    -----------
    volume_history : dict
        Dictionary containing times and volume variations
    filename : str
        Output filename
    """
    if rank == 0:
        times = np.array(volume_history['times'])
        delta_V = np.array(volume_history['delta_V'])

        # Create output directory if it doesn't exist
        os.makedirs('consolidation_results', exist_ok=True)
        filepath = os.path.join('consolidation_results', filename)

        # Write header and data
        with open(filepath, 'w') as f:
            f.write("# Volume Variation over Time\n")
            f.write("# Calculated as: ΔV = 2π ∫_Ω (α*ε_v + p/M) * r dΩ\n")
            f.write(f"# α (Biot coefficient) = {PROPS.float('alpha'):.4f}\n")
            f.write(f"# M (Biot modulus) = {PROPS.float('M'):.4e} Pa\n")
            f.write("#\n")
            f.write("# Time [s]         Volume Variation [m³]\n")
            f.write("#" + "-"*50 + "\n")

            for t, dV in zip(times, delta_V):
                f.write(f"{t:14.6e}  {dV:24.12e}\n")

        print(f"\n✓ Volume variation table saved to {filepath}")

        # Also create a visualization
        plt.figure(figsize=(10, 6))
        plt.plot(times, delta_V, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Volume Variation, ΔV [m³]', fontsize=12)
        plt.title('Domain Volume Variation over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_filepath = os.path.join('consolidation_results', 'volume_variation.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Volume variation plot saved to {plot_filepath}")


# ==============================================================================
# STRESS COMPUTATION
# ==============================================================================
def compute_stresses(wh, W, strain_stress):
    """
    Compute stress components from the solution

    Parameters:
    -----------
    wh : dolfinx.fem.Function
        Mixed solution function containing displacement and pressure
    W : dolfinx.fem.FunctionSpace
        Mixed function space
    strain_stress : dict
        Dictionary containing strain and stress functions

    Returns:
    --------
    stress_dict : dict
        Dictionary containing:
        - 'sigma_eff': Effective stress tensor function
        - 'sigma_total': Total stress tensor function
        - 'sigma_rr': Radial stress component
        - 'sigma_tt': Hoop/circumferential stress component
        - 'sigma_zz': Axial stress component
        - 'sigma_rz': Shear stress component
        - 'von_mises': Von Mises equivalent stress
    """
    # Extract strain and stress functions
    eps = strain_stress['eps']
    sigma_eff = strain_stress['sigma_eff']

    # Split solution into displacement and pressure
    uh, ph = ufl.split(wh)

    # Compute effective stress tensor
    sigma_eff_tensor = sigma_eff(uh)

    # Compute total stress (Terzaghi's principle: sigma_total = sigma_eff - alpha*p*I)
    alpha = PROPS.fem('alpha')
    sigma_total = sigma_eff_tensor - alpha * ph * ufl.Identity(3)

    # Extract stress components
    sigma_rr = sigma_total[0, 0]
    sigma_tt = sigma_total[1, 1]
    sigma_zz = sigma_total[2, 2]
    sigma_rz = sigma_total[0, 2]

    # Compute Von Mises stress
    # von Mises = sqrt(0.5*((s11-s22)^2 + (s22-s33)^2 + (s33-s11)^2 + 6*(s12^2 + s23^2 + s31^2)))
    s_diff_rr_tt = sigma_rr - sigma_tt
    s_diff_tt_zz = sigma_tt - sigma_zz
    s_diff_zz_rr = sigma_zz - sigma_rr

    von_mises = ufl.sqrt(0.5 * (s_diff_rr_tt**2 + s_diff_tt_zz**2 + s_diff_zz_rr**2 + 6*sigma_rz**2))

    # Create function spaces for stress output (P1 for each component)
    domain = W.mesh
    S_scalar = fem.functionspace(domain, ("Lagrange", 1))

    # Create functions for stress components
    sigma_rr_func = fem.Function(S_scalar, name="Sigma_rr")
    sigma_tt_func = fem.Function(S_scalar, name="Sigma_theta")
    sigma_zz_func = fem.Function(S_scalar, name="Sigma_zz")
    sigma_rz_func = fem.Function(S_scalar, name="Sigma_rz")
    von_mises_func = fem.Function(S_scalar, name="Von_Mises_Stress")

    # Create expressions for interpolation
    sigma_rr_expr = fem.Expression(sigma_rr, S_scalar.element.interpolation_points)
    sigma_tt_expr = fem.Expression(sigma_tt, S_scalar.element.interpolation_points)
    sigma_zz_expr = fem.Expression(sigma_zz, S_scalar.element.interpolation_points)
    sigma_rz_expr = fem.Expression(sigma_rz, S_scalar.element.interpolation_points)
    von_mises_expr = fem.Expression(von_mises, S_scalar.element.interpolation_points)

    # Interpolate stress components
    sigma_rr_func.interpolate(sigma_rr_expr)
    sigma_tt_func.interpolate(sigma_tt_expr)
    sigma_zz_func.interpolate(sigma_zz_expr)
    sigma_rz_func.interpolate(sigma_rz_expr)
    von_mises_func.interpolate(von_mises_expr)

    print("\n=== Stress Statistics ===")
    print(f"σ_rr:  min = {sigma_rr_func.x.array.min()/1e3:8.3f} kPa, max = {sigma_rr_func.x.array.max()/1e3:8.3f} kPa")
    print(f"σ_θθ:  min = {sigma_tt_func.x.array.min()/1e3:8.3f} kPa, max = {sigma_tt_func.x.array.max()/1e3:8.3f} kPa")
    print(f"σ_zz:  min = {sigma_zz_func.x.array.min()/1e3:8.3f} kPa, max = {sigma_zz_func.x.array.max()/1e3:8.3f} kPa")
    print(f"σ_rz:  min = {sigma_rz_func.x.array.min()/1e3:8.3f} kPa, max = {sigma_rz_func.x.array.max()/1e3:8.3f} kPa")
    print(f"σ_vm:  min = {von_mises_func.x.array.min()/1e3:8.3f} kPa, max = {von_mises_func.x.array.max()/1e3:8.3f} kPa")

    return {
        'sigma_eff': sigma_eff_tensor,
        'sigma_total': sigma_total,
        'sigma_rr': sigma_rr_func,
        'sigma_tt': sigma_tt_func,
        'sigma_zz': sigma_zz_func,
        'sigma_rz': sigma_rz_func,
        'von_mises': von_mises_func
    }


def save_stresses(stresses, domain, output_dir="consolidation_output"):
    """
    Save stress fields to XDMF files for visualization

    Parameters:
    -----------
    stresses : dict
        Dictionary containing stress functions from compute_stresses
    domain : dolfinx.mesh.Mesh
        Computational domain
    output_dir : str
        Directory to save output files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save each stress component
    stress_files = {}
    stress_names = ['sigma_rr', 'sigma_tt', 'sigma_zz', 'sigma_rz', 'von_mises']

    for name in stress_names:
        filename = f"{output_dir}/{name}.xdmf"
        stress_files[name] = io.XDMFFile(MPI.COMM_WORLD, filename, "w")
        stress_files[name].write_mesh(domain)
        stress_files[name].write_function(stresses[name], 0.0)
        stress_files[name].close()
        print(f"✓ Saved {name} to {filename}")

    print(f"\n✓ All stress fields saved to {output_dir}/")


# ==============================================================================
# VISUALIZATION
# ==============================================================================
def visualize_results(wh, W, history, T_final):
    """Create visualization plots"""
    import pyvista
    from dolfinx import plot

    pyvista.OFF_SCREEN = True

    W0_collapsed, _ = W.sub(0).collapse()
    W1_collapsed, _ = W.sub(1).collapse()

    # Extract and collapse final solution
    uh_final, ph_final = wh.split()
    uh_final_collapsed = uh_final.collapse()
    ph_final_collapsed = ph_final.collapse()

    u_final = fem.Function(W0_collapsed, name="Displacement")
    u_final.x.array[:] = uh_final_collapsed.x.array
    p_final = fem.Function(W1_collapsed, name="Pressure")
    p_final.x.array[:] = ph_final_collapsed.x.array

    # Plot displacement
    topology, cell_types, geometry = plot.vtk_mesh(W0_collapsed)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    displacement_2d = u_final.x.array.reshape((geometry.shape[0], 2))
    displacement_3d = np.zeros((displacement_2d.shape[0], 3))
    displacement_3d[:, :2] = displacement_2d

    grid["displacement"] = displacement_3d
    warped = grid.warp_by_vector("displacement", factor=100)

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, style="wireframe", color="black", opacity=0.3, line_width=1)
    plotter.add_mesh(warped, show_edges=True, color="lightblue")
    plotter.camera_position = 'xy'
    plotter.view_xy()
    plotter.screenshot('consolidation_final_displacement.png', window_size=[1920, 1080])
    print("✓ Saved consolidation_final_displacement.png")

    # Plot pressure
    topology_p, cell_types_p, geometry_p = plot.vtk_mesh(W1_collapsed)
    grid_p = pyvista.UnstructuredGrid(topology_p, cell_types_p, geometry_p)
    grid_p["pressure"] = p_final.x.array

    plotter2 = pyvista.Plotter()
    plotter2.add_mesh(grid_p, scalars="pressure", show_edges=True,
                      cmap="viridis", clim=[0, max(history['max_pressures'])],
                      scalar_bar_args={'title': 'Pore Pressure [Pa]'})
    plotter2.camera_position = 'xy'
    plotter2.view_xy()
    plotter2.screenshot('consolidation_final_pressure.png', window_size=[1920, 1080])
    print("✓ Saved consolidation_final_pressure.png")

    # Time history plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(history['times'], history['max_pressures'] / 1e3, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('Maximum Pressure [kPa]', fontsize=12)
    ax1.set_title('Pore Pressure Dissipation', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['times'], history['max_displacements'] * 1000, 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Maximum Vertical Displacement [mm]', fontsize=12)
    ax2.set_title('Settlement Evolution', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('consolidation_history.png', dpi=150, bbox_inches='tight')
    print("✓ Saved consolidation_history.png")

    # Degree of consolidation
    U = (history['max_displacements'] - history['max_displacements'][0]) / \
        (history['max_displacements'][-1] - history['max_displacements'][0]) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['times'], U, 'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Degree of Consolidation U [%]', fontsize=12)
    ax.set_title('Consolidation Progress', fontsize=14)
    ax.axhline(y=90, color='r', linestyle='--', label='90% consolidation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('degree_of_consolidation.png', dpi=150, bbox_inches='tight')
    print("✓ Saved degree_of_consolidation.png")

    # Print final statistics
    print(f"\nFinal state (t = {T_final} s):")
    print(f"  Max pressure: {history['max_pressures'][-1]:.3e} Pa " +
          f"({history['max_pressures'][-1]/history['max_pressures'][1]*100:.1f}% of initial)")
#     print(f"  Max settlement: {history['max_displacements'][-1]*1000:.3f} mm")
#     print(f"  Degree of consolidation: {U[-1]:.1f}%")


# ==============================================================================
# ANALYTICAL SOLUTION XDMF EXPORT
# ==============================================================================
def save_analytical_solution_xdmf(domain, H, T_final, num_steps, PROPS,
                                   output_dir="results_analytical", n_terms=50):
    """
    Compute and save the analytical Terzaghi solution for all timesteps to XDMF files

    Parameters:
    -----------
    domain : dolfinx.mesh.Mesh
        The computational mesh
    H : float
        Height of the consolidating layer (m)
    T_final : float
        Final simulation time (s)
    num_steps : int
        Number of time steps
    PROPS : Materials object
        Material properties object containing all material parameters
    output_dir : str, optional
        Directory to save output files (default: "results_analytical")
    n_terms : int, optional
        Number of terms in Fourier series for analytical solution (default: 50)

    Returns:
    --------
    dict containing times, max pressures and displacements from analytical solution
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    dt = T_final / num_steps

    # Get material properties
    M = PROPS.float('M')
    perm = PROPS.float('perm')
    visc = PROPS.float('visc')
    alpha = PROPS.float('alpha')
    nu = PROPS.float('nu')
    mu = PROPS.float('mu')
    sig0 = PROPS.float('sig0')

    # Calculate derived parameters
    eta = alpha * (1 - 2*nu) / (2 * (1 - nu))
    S = 1/M + (alpha**2)*(1-2*nu) / (2*mu*(1-nu))
    c_v = perm/visc/S

    # Initial pressure
    p0 = -sig0 * eta / mu / S

    # Calculate final settlement
    E_oed = (1-nu)*(2*mu*(1+nu)) / ((1+nu)*(1-2*nu))
    uz_final = -sig0 * H / E_oed

    # Calculate immediate (undrained) settlement
    lmbda = PROPS.float('lmbda')
    K_u = lmbda + 2*mu + (alpha**2)*M  # Undrained constrained modulus
    uz_immediate = -sig0 * H / K_u

    # Consolidation settlement (additional settlement due to drainage)
    uz_consolidation = uz_final - uz_immediate

    if rank == 0:
        print(f"\n=== Analytical Solution Parameters ===")
        print(f"Consolidation coefficient c_v: {c_v:.6e} m²/s")
        print(f"Initial pressure p0: {p0:.6e} Pa ({p0/1e3:.3f} kPa)")
        print(f"Undrained modulus K_u: {K_u:.6e} Pa")
        print(f"Drained modulus E_oed: {E_oed:.6e} Pa")
        print(f"Immediate settlement (t=0): {uz_immediate*1000:.6f} mm")
        print(f"Consolidation settlement: {uz_consolidation*1000:.6f} mm")
        print(f"Final settlement (t=∞): {uz_final*1000:.6f} mm")
        print(f"Ratio K_u/E_oed: {K_u/E_oed:.3f}")

    # Create function spaces (P1 for output)
    gdim = domain.geometry.dim
    V_P1 = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    Q_P1 = fem.functionspace(domain, ("Lagrange", 1))

    # Create functions for analytical solution
    p_analytical = fem.Function(Q_P1, name="Pressure_Analytical")
    u_analytical = fem.Function(V_P1, name="Displacement_Analytical")

    # Open XDMF files
    p_file = io.XDMFFile(comm, f"{output_dir}/pressure_analytical.xdmf", "w")
    u_file = io.XDMFFile(comm, f"{output_dir}/displacement_analytical.xdmf", "w")

    p_file.write_mesh(domain)
    u_file.write_mesh(domain)

    # Get mesh coordinates
    x = domain.geometry.x

    # Storage for history
    times = [0.0]
    max_pressures_analytical = []
    max_displacements_analytical = []
    volume_variation_analytical = []
    flux_top_analytical = []

    if rank == 0:
        print(f"\n=== Computing Analytical Solution for All Timesteps ===")

    # Loop over all timesteps
    for step in range(num_steps + 1):
        t = step * dt

        # Get z coordinates
        z_coords = x[:, 1]

        # Compute pressure and displacement
        if t == 0:
            # ========================================
            # t = 0: INSTANT OF LOADING (t = 0⁺)
            # ========================================
            # At this instant:
            # - Load σ₀ is suddenly applied at the top
            # - Pore pressure jumps uniformly to p₀ EVERYWHERE (including at top!)
            # - Drainage BC p=0 has NOT taken effect yet (no time has passed)
            # - Undrained elastic displacement occurs instantly
            #
            # This is the TRUE undrained state before any consolidation
            # ========================================

            # Pressure is uniform everywhere (no spatial variation)
            p_values = p0 * np.ones_like(z_coords)  # p = p₀ even at z=H (top)

            # Displacement profile (linear with depth, zero at bottom)
            # This is the immediate undrained elastic response
            uz_values = uz_immediate * (z_coords / H)

            # ========================================
            # ANALYTICAL VOLUME VARIATION AND FLUX (t=0)
            # ========================================
            # Volumetric strain: ε_v = ∂u_z/∂z (constant in z for linear displacement)
            eps_v_analytical = uz_immediate / H  # ∂(uz_immediate·z/H)/∂z = uz_immediate/H

            # Volume variation: ΔV = π·Re² · ∫₀^H (α·ε_v + p/M) dz
            # At t=0: ε_v is constant, p is constant
            integrand_volume = alpha * eps_v_analytical + p0 / M
            delta_V_analytical = np.pi * Re**2 * integrand_volume * H

            # Flux at top boundary: q_z = -(k/μ) ∂p/∂z|_{z=H}
            # At t=0: pressure is uniform, so ∂p/∂z = 0 everywhere
            dp_dz_top = 0.0
            q_z_top = -(perm / visc) * dp_dz_top

            # Total flux rate leaving top: Q = π·Re² · q_z
            Q_top_analytical = np.pi * Re**2 * q_z_top

        else:
            # ========================================
            # t > 0: CONSOLIDATION PHASE
            # ========================================
            # For all t > 0 (even the first timestep t=dt):
            # - Drainage BC p=0 at top is now active
            # - Pore pressure dissipates from top surface
            # - Additional consolidation settlement occurs
            # - Solution uses Fourier series which satisfies BC p(H,t)=0
            # ========================================
            # Dimensionless time
            t_s = c_v * t / (4 * H**2)
            z_s = z_coords / H

            # Pressure using Fourier series
            # NOTE: Using (1-z_s) instead of z_s to match boundary conditions:
            # Top (z=H): p=0 (drained), Bottom (z=0): impermeable
            p_values = np.zeros_like(z_coords)
            for n in range(n_terms):
                m = (2*n + 1) * np.pi
                p_values += (4/m) * np.sin(m * (1 - z_s) / 2) * np.exp(-m**2 * t_s)
            p_values *= p0

            # Degree of consolidation
            U = 1.0
            for n in range(n_terms):
                M_n = (2*n + 1) * np.pi / 2
                U -= (2 / M_n**2) * np.exp(-M_n**2 * t_s)

            # Vertical displacement
            # Total displacement = immediate (undrained) + consolidation
            # uz(z,t) = [uz_immediate + (uz_final - uz_immediate)*U(t)] * (z/H)
            # Bottom (z=0): u_z=0 (BC), Top (z=H): maximum displacement
            uz_values = (uz_immediate + (uz_final - uz_immediate) * U) * (z_coords / H)

            # ========================================
            # ANALYTICAL VOLUME VARIATION AND FLUX (t>0)
            # ========================================
            # Volumetric strain: ε_v = ∂u_z/∂z (constant in z for linear displacement)
            eps_v_analytical = (uz_immediate + (uz_final - uz_immediate) * U) / H

            # Volume variation: ΔV = π·Re² · ∫₀^H (α·ε_v + p/M) dz
            # Need to integrate pressure over height
            # Use numerical integration over the mesh points
            z_sorted_idx = np.argsort(z_coords)
            z_sorted = z_coords[z_sorted_idx]
            p_sorted = p_values[z_sorted_idx]

            # Integrate using trapezoidal rule
            integrand_p = p_sorted / M
            integral_p = np.trapz(integrand_p, z_sorted)

            # Add constant strain contribution
            integral_eps_v = alpha * eps_v_analytical * H

            # Total volume variation
            delta_V_analytical = np.pi * Re**2 * (integral_eps_v + integral_p)

            # Flux at top boundary: q_z = -(k/μ) ∂p/∂z|_{z=H}
            # Pressure gradient from Fourier series at z=H (z_s = 1)
            # p(z,t) = p0 · ∑ (4/m) · sin(m(1-z_s)/2) · exp(-m²·t_s)
            # ∂p/∂z = p0 · ∑ (4/m) · (-m/2H) · cos(m(1-z_s)/2) · exp(-m²·t_s)
            # At z=H (z_s=1): ∂p/∂z = p0 · ∑ (4/m) · (-m/2H) · cos(0) · exp(-m²·t_s)
            #                       = -p0/H · ∑ (2) · exp(-m²·t_s)
            dp_dz_top = 0.0
            for n in range(n_terms):
                m = (2*n + 1) * np.pi
                dp_dz_top += 2.0 * np.exp(-m**2 * t_s)
            dp_dz_top *= (-p0 / H)

            # Darcy flux (positive outward from domain)
            q_z_top = -(perm / visc) * dp_dz_top

            # Total flux rate leaving top: Q = π·Re² · q_z
            Q_top_analytical = np.pi * Re**2 * q_z_top

        # Create displacement vector
        u_values = np.zeros((len(x), gdim))
        u_values[:, 1] = uz_values

        # Assign to functions
        p_analytical.x.array[:] = p_values
        u_analytical.x.array[:] = u_values.flatten()

        # Write to files
        p_file.write_function(p_analytical, t)
        u_file.write_function(u_analytical, t)

        # Statistics
        max_p = np.max(np.abs(p_values))
        max_uz = np.max(np.abs(uz_values))

        max_p = comm.allreduce(max_p, op=MPI.MAX)
        max_uz = comm.allreduce(max_uz, op=MPI.MAX)

        max_pressures_analytical.append(max_p)
        max_displacements_analytical.append(max_uz)
        volume_variation_analytical.append(delta_V_analytical)
        flux_top_analytical.append(Q_top_analytical)

        if step > 0:
            times.append(t)

        if (step % 10 == 0 or step == num_steps) and rank == 0:
            if t == 0:
                U_percent = 0.0  # No consolidation yet at t=0
                # Verify t=0 state
                print(f"t = {t:8.4f} s | Max p = {max_p/1e3:10.3f} kPa | "
                      f"Max uz = {max_uz*1000:10.6f} mm | U = {U_percent:6.2f}%")
                print(f"             | p at top (z=H) = {p_values[np.argmax(z_coords)]/1e3:10.3f} kPa (should = {p0/1e3:.3f} kPa)")
                print(f"             | ΔV = {delta_V_analytical:.6e} m³ | Q_top = {Q_top_analytical:.6e} m³/s")
                print(f"             | Note: At t=0, drainage BC has NOT taken effect yet")
            else:
                # U = consolidation that has occurred (0 to 100%)
                # Current settlement = immediate + consolidation*U
                # U = (current - immediate) / (final - immediate)
                U_percent = (max_uz - abs(uz_immediate)) / (abs(uz_final) - abs(uz_immediate)) * 100
                print(f"t = {t:8.4f} s | Max p = {max_p/1e3:10.3f} kPa | "
                      f"Max uz = {max_uz*1000:10.6f} mm | U = {U_percent:6.2f}%")
                print(f"             | ΔV = {delta_V_analytical:.6e} m³ | Q_top = {Q_top_analytical:.6e} m³/s")

    p_file.close()
    u_file.close()

    if rank == 0:
        print(f"✓ Analytical solution saved to {output_dir}/")

    return {
        'times': np.array(times),
        'max_pressures_analytical': np.array(max_pressures_analytical),
        'max_displacements_analytical': np.array(max_displacements_analytical),
        'volume_variation_analytical': np.array(volume_variation_analytical),
        'flux_top_analytical': np.array(flux_top_analytical)
    }


def save_analytical_volume_and_flux_tables(analytical_history, PROPS, Re, H,
                                           output_dir="consolidation_results"):
    """
    Save analytical volume variation and flux data to table files

    Parameters:
    -----------
    analytical_history : dict
        Dictionary containing analytical solution history
    PROPS : Materials
        Material properties object
    Re : float
        External radius [m]
    H : float
        Height [m]
    output_dir : str
        Output directory
    """
    if rank == 0:
        times = analytical_history['times']
        delta_V = analytical_history['volume_variation_analytical']
        Q_top = analytical_history['flux_top_analytical']

        os.makedirs(output_dir, exist_ok=True)

        # Save volume variation table
        filepath_vol = os.path.join(output_dir, 'volume_variation_analytical.dat')
        with open(filepath_vol, 'w') as f:
            f.write("# Analytical Volume Variation over Time\n")
            f.write("# Calculated as: ΔV = π·Re² · ∫₀^H (α·ε_v + p/M) dz\n")
            f.write(f"# α (Biot coefficient) = {PROPS.float('alpha'):.4f}\n")
            f.write(f"# M (Biot modulus) = {PROPS.float('M'):.4e} Pa\n")
            f.write(f"# Re (radius) = {Re:.4f} m\n")
            f.write(f"# H (height) = {H:.4f} m\n")
            f.write("#\n")
            f.write("# Time [s]         Volume Variation [m³]\n")
            f.write("#" + "-"*50 + "\n")

            for t, dV in zip(times, delta_V):
                f.write(f"{t:14.6e}  {dV:24.12e}\n")

        print(f"\n✓ Analytical volume variation table saved to {filepath_vol}")

        # Save flux table
        filepath_flux = os.path.join(output_dir, 'flux_top_analytical.dat')
        with open(filepath_flux, 'w') as f:
            f.write("# Analytical Fluid Flux at Top Boundary over Time\n")
            f.write("# Calculated as: Q = π·Re² · q_z where q_z = -(k/μ) ∂p/∂z|_{z=H}\n")
            f.write(f"# k (permeability) = {PROPS.float('perm'):.4e} m²\n")
            f.write(f"# μ (viscosity) = {PROPS.float('visc'):.4e} Pa·s\n")
            f.write(f"# Re (radius) = {Re:.4f} m\n")
            f.write(f"# H (height) = {H:.4f} m\n")
            f.write("#\n")
            f.write("# Time [s]         Flux Rate Q [m³/s]      Note\n")
            f.write("#" + "-"*70 + "\n")

            for i, (t, Q) in enumerate(zip(times, Q_top)):
                note = ""
                if i == 0:
                    note = "  # t=0: no drainage yet"
                elif i == 1:
                    note = "  # t=dt: drainage begins"
                f.write(f"{t:14.6e}  {Q:24.12e}{note}\n")

        print(f"✓ Analytical flux table saved to {filepath_flux}")

        # Save integrated flux table
        filepath_int_flux = os.path.join(output_dir, 'integrated_flux_analytical.dat')
        with open(filepath_int_flux, 'w') as f:
            f.write("# Integrated Analytical Fluid Flux (Cumulative Volume Drained)\n")
            f.write("# Calculated as: ∫₀ᵗ Q(τ) dτ where Q = π·Re² · q_z\n")
            f.write(f"# k (permeability) = {PROPS.float('perm'):.4e} m²\n")
            f.write(f"# μ (viscosity) = {PROPS.float('visc'):.4e} Pa·s\n")
            f.write(f"# Re (radius) = {Re:.4f} m\n")
            f.write(f"# H (height) = {H:.4f} m\n")
            f.write("#\n")
            f.write("# Time [s]         Integrated Flux [m³]    Note\n")
            f.write("#" + "-"*80 + "\n")

            # Compute integrated flux again for saving
            integrated_flux_save = np.zeros_like(Q_top)
            for i in range(1, len(times)):
                integrated_flux_save[i] = integrated_flux_save[i-1] + 0.5 * (Q_top[i] + Q_top[i-1]) * (times[i] - times[i-1])

            for i, (t, int_Q) in enumerate(zip(times, integrated_flux_save)):
                note = ""
                if i == 0:
                    note = "  # t=0: no drainage yet"
                elif i == len(times) - 1:
                    note = f"  # Final cumulative volume"
                f.write(f"{t:14.6e}  {int_Q:24.12e}{note}\n")

        print(f"✓ Integrated flux table saved to {filepath_int_flux}")

        # Create plots
        import matplotlib.pyplot as plt

        # Volume variation plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, delta_V, 'r-o', linewidth=2, markersize=6, label='Analytical')
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Volume Variation, ΔV [m³]', fontsize=12)
        ax.set_title('Analytical Volume Variation over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        plt.tight_layout()

        plot_filepath = os.path.join(output_dir, 'volume_variation_analytical.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Analytical volume variation plot saved to {plot_filepath}")

        # Flux plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, Q_top, 'b-s', linewidth=2, markersize=6, label='Analytical')
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Flux Rate, Q [m³/s]', fontsize=12)
        ax.set_title('Analytical Fluid Flux at Top Boundary', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.tight_layout()

        plot_filepath = os.path.join(output_dir, 'flux_top_analytical.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Analytical flux plot saved to {plot_filepath}")

        # Integrated flux plot (cumulative volume drained)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Compute cumulative integral of flux using cumulative trapezoidal rule
        integrated_flux = np.zeros_like(Q_top)
        for i in range(1, len(times)):
            integrated_flux[i] = integrated_flux[i-1] + 0.5 * (Q_top[i] + Q_top[i-1]) * (times[i] - times[i-1])

        ax.plot(times, integrated_flux, 'g-d', linewidth=2, markersize=6, label='∫Q dt (Cumulative Volume Drained)')
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Cumulative Volume Drained [m³]', fontsize=12)
        ax.set_title('Integral of Flux: Total Volume Drained Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        # Add annotation for final value
        if len(integrated_flux) > 0:
            final_vol = integrated_flux[-1]
            ax.text(times[-1], final_vol, f'  Final: {final_vol:.3e} m³',
                   verticalalignment='center', fontsize=10, color='darkgreen')

        plt.tight_layout()

        plot_filepath = os.path.join(output_dir, 'integrated_flux_analytical.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Integrated flux plot saved to {plot_filepath}")

        # Combined plot: Volume variation, flux, and integrated flux
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))

        # Panel 1: Volume variation
        ax1.plot(times, delta_V, 'r-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Time [s]', fontsize=12)
        ax1.set_ylabel('Volume Variation, ΔV [m³]', fontsize=12)
        ax1.set_title('Volume Variation', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Panel 2: Flux
        ax2.plot(times, Q_top, 'b-s', linewidth=2, markersize=4)
        ax2.set_xlabel('Time [s]', fontsize=12)
        ax2.set_ylabel('Flux Rate, Q [m³/s]', fontsize=12)
        ax2.set_title('Fluid Flux Leaving Top Boundary', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

        # Panel 3: Integrated flux
        ax3.plot(times, integrated_flux, 'g-d', linewidth=2, markersize=4)
        ax3.set_xlabel('Time [s]', fontsize=12)
        ax3.set_ylabel('Cumulative Volume Drained [m³]', fontsize=12)
        ax3.set_title('Integrated Flux: ∫Q dt', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_filepath = os.path.join(output_dir, 'analytical_volume_flux_integrated.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Combined analytical plot (3-panel) saved to {plot_filepath}")

        # Conservation verification plot: Integrated flux vs Volume change
        fig, ax = plt.subplots(figsize=(10, 6))

        # Volume change from initial state
        volume_change = delta_V - delta_V[0]  # Change from initial

        ax.plot(times, -volume_change, 'r-o', linewidth=2, markersize=6,
                label='Volume Decrease: -(ΔV(t) - ΔV(0))', alpha=0.8)
        ax.plot(times, integrated_flux, 'b--s', linewidth=2, markersize=6,
                label='Cumulative Flux: ∫Q dt', alpha=0.8)
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Volume [m³]', fontsize=12)
        ax.set_title('Conservation Verification: Volume Drained vs Volume Change',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        # Add annotation showing the difference
        if len(times) > 1:
            final_diff = integrated_flux[-1] - (-volume_change[-1])
            rel_error = abs(final_diff / integrated_flux[-1]) * 100 if integrated_flux[-1] != 0 else 0
            ax.text(0.5, 0.95, f'Final difference: {final_diff:.3e} m³ ({rel_error:.2f}%)',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plot_filepath = os.path.join(output_dir, 'conservation_check.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Conservation verification plot saved to {plot_filepath}")

        # Direct comparison plot: Volume Variation vs Cumulative Drainage
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Left y-axis: Volume variation
        color1 = 'tab:red'
        ax1.set_xlabel('Time [s]', fontsize=13)
        ax1.set_ylabel('Volume Variation, ΔV [m³]', fontsize=13, color=color1)
        line1 = ax1.plot(times, delta_V, 'r-o', linewidth=2.5, markersize=6,
                        label='Volume Variation (ΔV)', alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3, linestyle=':')

        # Right y-axis: Cumulative drainage
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Cumulative Volume Drained [m³]', fontsize=13, color=color2)
        line2 = ax2.plot(times, integrated_flux, 'g-s', linewidth=2.5, markersize=6,
                        label='Cumulative Drained (∫Q dt)', alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color2)

        # Title and legend
        ax1.set_title('Volume Variation vs Cumulative Drainage\n(Conservation: ΔV(0) - ΔV(t) ≈ ∫Q dt)',
                     fontsize=14, fontweight='bold', pad=20)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.9)

        # Add annotation showing relationship
        mid_idx = len(times) // 2
        if len(times) > 1:
            ax1.annotate('', xy=(times[mid_idx], delta_V[mid_idx]),
                        xytext=(times[mid_idx], delta_V[0]),
                        arrowprops=dict(arrowstyle='<->', color='darkred', lw=2, alpha=0.5))
            ax1.text(times[mid_idx], (delta_V[mid_idx] + delta_V[0])/2,
                    '  Volume\n  decrease', fontsize=9, color='darkred',
                    verticalalignment='center')

            ax2.annotate('', xy=(times[mid_idx], integrated_flux[mid_idx]),
                        xytext=(times[mid_idx], 0),
                        arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2, alpha=0.5))
            ax2.text(times[mid_idx], integrated_flux[mid_idx]/2,
                    'Volume\ndrained  ', fontsize=9, color='darkgreen',
                    verticalalignment='center', horizontalalignment='right')

        plt.tight_layout()
        plot_filepath = os.path.join(output_dir, 'volume_variation_vs_cumulative_drainage.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Volume variation vs cumulative drainage plot saved to {plot_filepath}")

        # Single-axis comparison (normalized)
        fig, ax = plt.subplots(figsize=(12, 7))

        # Normalize both quantities to [0, 1] for comparison
        if len(times) > 1 and integrated_flux[-1] != 0:
            # For volume variation: map from initial to final
            delta_V_normalized = (delta_V - delta_V[0]) / (delta_V[-1] - delta_V[0])

            # For integrated flux: map from 0 to final
            integrated_flux_normalized = integrated_flux / integrated_flux[-1]

            ax.plot(times, delta_V_normalized, 'r-o', linewidth=2.5, markersize=6,
                   label='Normalized ΔV: (ΔV - ΔV₀)/(ΔV_f - ΔV₀)', alpha=0.8)
            ax.plot(times, integrated_flux_normalized, 'g-s', linewidth=2.5, markersize=6,
                   label='Normalized ∫Q dt: (∫Q dt) / (∫Q dt)_final', alpha=0.8)

            ax.set_xlabel('Time [s]', fontsize=13)
            ax.set_ylabel('Normalized Progress [0 to 1]', fontsize=13)
            ax.set_title('Normalized Comparison: Volume Change vs Drainage\n(Both curves should overlap if conservation holds)',
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=11, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.1, 1.1])

            # Add horizontal lines at key percentages
            for pct in [0.25, 0.5, 0.75, 1.0]:
                ax.axhline(y=pct, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
                ax.text(times[-1]*1.02, pct, f'{pct*100:.0f}%',
                       verticalalignment='center', fontsize=9, color='gray')

            # Calculate and display correlation (if scipy available)
            try:
                from scipy.stats import pearsonr
                if len(delta_V_normalized) > 2:
                    corr, _ = pearsonr(delta_V_normalized, integrated_flux_normalized)
                    ax.text(0.02, 0.98, f'Correlation: {corr:.6f}',
                           transform=ax.transAxes, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            except ImportError:
                # scipy not available, skip correlation
                pass

        plt.tight_layout()
        plot_filepath = os.path.join(output_dir, 'normalized_volume_drainage_comparison.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Normalized comparison plot saved to {plot_filepath}")

        # Print statistics
        print(f"\n=== Analytical Volume Variation Statistics ===")
        print(f"Initial (t=0): {delta_V[0]:.6e} m³")
        print(f"Final (t={times[-1]:.2f}s): {delta_V[-1]:.6e} m³")
        print(f"Total change: {delta_V[-1] - delta_V[0]:.6e} m³")

        print(f"\n=== Analytical Flux Statistics ===")
        print(f"Maximum flux: {np.max(Q_top):.6e} m³/s at t={times[np.argmax(Q_top)]:.4f} s")
        if len(times) > 1:
            total_volume_out = np.trapz(Q_top, times)
            print(f"Total volume drained (∫Q dt): {total_volume_out:.6e} m³")
            print(f"Final integrated flux: {integrated_flux[-1]:.6e} m³")

        print(f"\n=== Conservation Check ===")
        if len(times) > 1:
            volume_decrease = -(delta_V[-1] - delta_V[0])  # Positive = volume lost
            flux_integral = integrated_flux[-1]
            difference = flux_integral - volume_decrease
            rel_error = abs(difference / flux_integral) * 100 if flux_integral != 0 else 0

            print(f"Volume decrease from domain: {volume_decrease:.6e} m³")
            print(f"Volume drained via flux:     {flux_integral:.6e} m³")
            print(f"Difference:                  {difference:.6e} m³")
            print(f"Relative error:              {rel_error:.4f}%")

            if rel_error < 1.0:
                print("✓ Conservation satisfied (error < 1%)")
            elif rel_error < 5.0:
                print("⚠ Conservation acceptable (error < 5%)")
            else:
                print("✗ Conservation check failed (error > 5%)")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # --- PARAMETERS ---
    # Geometry
    Re = 1.0          # Outer radius (m)
    H = 1.0           # Height (m)
    N = 50       # Mesh size (m)

    # Time parameters
    T_final = 1   # Final time (s)
    num_steps = 1000    # Number of time steps
    dt = T_final / num_steps

    print("="*70)
    print("AXISYMMETRIC CONSOLIDATION SIMULATION")
    print("="*70)
    print(f"Geometry: R = {Re} m, H = {H} m")
    print(f"Time: T = {T_final} s, dt = {dt:.3f} s, steps = {num_steps}")
    print("="*70)

    # --- SETUP ---
    domain, facets = create_mesh(Re, H, N)
    PROPS = Materials(domain)

    W, gdim = setup_function_spaces(domain)
    strain_stress = define_strain_stress(domain)
    forms = setup_weak_form(W, domain, facets, strain_stress, dt)
    bcs_dict = setup_boundary_conditions(W, domain, facets, gdim)

    # --- SAVE ANALYTICAL SOLUTION ---
    if rank == 0:
        print("\n" + "="*70)
        print("SAVING ANALYTICAL SOLUTION TO XDMF")
        print("="*70)
    analytical_history = save_analytical_solution_xdmf(
        domain, H, T_final, num_steps, PROPS,
        output_dir="consolidation_analytical"
    )

    # --- SAVE ANALYTICAL VOLUME VARIATION AND FLUX ---
    if rank == 0:
        print("\n" + "="*70)
        print("SAVING ANALYTICAL VOLUME VARIATION AND FLUX TABLES")
        print("="*70)
    save_analytical_volume_and_flux_tables(analytical_history, PROPS, Re, H)

    # --- SOLVE ---
    # Solve t=0 with undrained BCs (no pressure BC at top)
    wh, problem = solve_initial_condition(W, forms, bcs_dict['t0'])
    # Time stepping with consolidation BCs (includes p=0 at top)
    history = time_stepping_loop(wh, problem, forms, W, bcs_dict['consolidation'],
                                 T_final, num_steps, H, strain_stress, domain)

    # --- SAVE VOLUME VARIATION TABLE ---
    if rank == 0:
        print("\n" + "="*70)
        print("SAVING VOLUME VARIATION TABLE")
        print("="*70)
    save_volume_variation_table(history['volume_history'])

    # --- STRESSES ---
    stresses = compute_stresses(wh, W, strain_stress)
    save_stresses(stresses, domain)

    # --- VISUALIZE ---
    visualize_results(wh, W, history, T_final)

    # --- COMPARE FEM vs ANALYTICAL ---
    if rank == 0:
        print("\n" + "="*70)
        print("CREATING FEM vs ANALYTICAL COMPARISON PLOTS")
        print("="*70)

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        times = history['times']

        # Plot 1: Pressure comparison
        ax = axes[0, 0]
        ax.plot(times, history['max_pressures']/1e3, 'b-o',
                label='FEM', linewidth=2, markersize=4)
        ax.plot(times, analytical_history['max_pressures_analytical']/1e3, 'r--s',
                label='Analytical', linewidth=2, markersize=4)
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Maximum Pressure [kPa]', fontsize=12)
        ax.set_title('Pressure Dissipation Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Plot 2: Displacement comparison
        ax = axes[0, 1]
        ax.plot(times, history['max_displacements']*1000, 'b-o',
                label='FEM', linewidth=2, markersize=4)
        ax.plot(times, analytical_history['max_displacements_analytical']*1000, 'r--s',
                label='Analytical', linewidth=2, markersize=4)
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Maximum Vertical Displacement [mm]', fontsize=12)
        ax.set_title('Settlement Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Plot 3: Pressure error
        ax = axes[1, 0]
        p_error = np.abs(history['max_pressures'] -
                        analytical_history['max_pressures_analytical'])
        p_error_percent = (p_error / analytical_history['max_pressures_analytical'][0]) * 100
        ax.plot(times, p_error_percent, 'g-o', linewidth=2, markersize=4)
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Relative Pressure Error [%]', fontsize=12)
        ax.set_title('Pressure Error (FEM vs Analytical)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 4: Displacement error
        ax = axes[1, 1]
        u_error = np.abs(history['max_displacements'] -
                        analytical_history['max_displacements_analytical'])
        u_error_mm = u_error * 1000
        ax.plot(times, u_error_mm, 'm-o', linewidth=2, markersize=4)
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Absolute Displacement Error [mm]', fontsize=12)
        ax.set_title('Displacement Error (FEM vs Analytical)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig('fem_vs_analytical_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved fem_vs_analytical_comparison.png")

        # Volume Variation Comparison Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Get FEM and analytical volume variation data
        times_fem = history['times']
        delta_V_fem = np.array(history['volume_history']['delta_V'])
        delta_V_analytical = analytical_history['volume_variation_analytical']

        ax.plot(times_fem, delta_V_fem, 'b-o', label='FEM', linewidth=2, markersize=6)
        ax.plot(times, delta_V_analytical, 'r--s', label='Analytical', linewidth=2, markersize=6)
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Volume Variation, ΔV [m³]', fontsize=12)
        ax.set_title('Volume Variation Comparison (FEM vs Analytical)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('volume_variation_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved volume_variation_comparison.png")

        # Volume variation error
        delta_V_error = np.abs(delta_V_fem - delta_V_analytical)
        delta_V_error_percent = (delta_V_error / np.abs(delta_V_analytical[0])) * 100

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(times_fem, delta_V_error_percent, 'g-o', linewidth=2, markersize=4)
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Relative Volume Variation Error [%]', fontsize=12)
        ax.set_title('Volume Variation Error (FEM vs Analytical)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig('volume_variation_error.png', dpi=300, bbox_inches='tight')
        print("✓ Saved volume_variation_error.png")

        # FEM Volume Variation vs Inverted Cumulative Drainage (same axis)
        fig, ax = plt.subplots(figsize=(12, 8))

        # Get FEM volume variation
        delta_V_fem = np.array(history['volume_history']['delta_V'])
        times_fem = np.array(history['volume_history']['times'])

        # Get analytical integrated flux - compute analytically
        integrated_flux = analytical_history['flux_top_analytical']
        times_analytical = analytical_history['times']

        # Analytical integration of flux
        # Q(t) = π·Re² · (k/μ) · (2p₀/H) · Σ exp(-m²·Tv)
        # ∫Q dt = π·Re² · (k/μ) · (2p₀/H) · (4H²/c_v) · Σ (1/m²) · (1 - exp(-m²·Tv))

        # Get material properties
        M = PROPS.float('M')
        perm = PROPS.float('perm')
        visc = PROPS.float('visc')
        alpha = PROPS.float('alpha')
        nu = PROPS.float('nu')
        mu = PROPS.float('mu')
        sig0 = PROPS.float('sig0')

        # Calculate derived parameters
        eta = alpha * (1 - 2*nu) / (2 * (1 - nu))
        S = 1/M + (alpha**2)*(1-2*nu) / (2*mu*(1-nu))
        c_v = perm/visc/S
        p0 = -sig0 * eta / mu / S

        # Number of Fourier terms
        n_terms = 50

        # Compute analytical integral for each time
        integrated_flux_cumulative = np.zeros_like(times_analytical)
        for i, t in enumerate(times_analytical):
            if t == 0:
                integrated_flux_cumulative[i] = 0.0
            else:
                t_s = c_v * t / (4 * H**2)  # Dimensionless time

                # Sum over Fourier series
                sum_integral = 0.0
                for n in range(n_terms):
                    m = (2*n + 1) * np.pi
                    sum_integral += (1.0 / m**2) * (1.0 - np.exp(-m**2 * t_s))

                # Complete integral: π·Re² · (k/μ) · (2p₀/H) · (4H²/c_v) · Σ(...)
                integrated_flux_cumulative[i] = np.pi * Re**2 * (perm/visc) * (2*p0/H) * (4*H**2/c_v) * sum_integral

        # Conservation: ΔV(t) = ΔV(0) - ∫Q dt
        # So: -∫Q dt should match ΔV(t) - ΔV(0)
        # To plot both starting from ΔV(0), add ΔV(0) to -∫Q dt
#         integrated_flux_adjusted = delta_V_fem[0] - integrated_flux_cumulative
        integrated_flux_adjusted = - integrated_flux_cumulative

        # Compute difference (conservation error)
        # Interpolate to FEM time points for comparison
        try:
            from scipy.interpolate import interp1d
            f_flux = interp1d(times_analytical, integrated_flux_adjusted, kind='linear',
                             bounds_error=False, fill_value='extrapolate')
            flux_interp = f_flux(times_fem)
            difference = delta_V_fem - flux_interp
            plot_difference = True
        except ImportError:
            plot_difference = False

        # Plot curves
        ax.plot(times_fem, delta_V_fem, 'b-o', linewidth=2.5, markersize=2,
               label='FEM Volume Variation (ΔV)', alpha=0.85)
        ax.plot(times_analytical, integrated_flux_adjusted, 'r--s', linewidth=2.5, markersize=2,
               label='ΔV₀ - ∫Q dt (should match ΔV)', alpha=0.85)

        if plot_difference:
            ax.plot(times_fem, difference, 'g-^', linewidth=2, markersize=2,
                   label='Difference (ΔV - [ΔV₀ - ∫Q dt])', alpha=0.85)

        ax.set_xlabel('Time [s]', fontsize=13, fontweight='bold')
        ax.set_ylabel('Volume [m³]', fontsize=13, fontweight='bold')
        ax.set_title('Conservation Check: FEM Volume Variation vs ΔV₀ - ∫Q dt',
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)

        plt.tight_layout()
        plt.savefig('volume_variation_vs_inverted_drainage.png', dpi=300, bbox_inches='tight')
        print("✓ Saved volume_variation_vs_inverted_drainage.png")

        # Print error statistics
        print(f"\n=== Error Statistics ===")
        print(f"Maximum pressure error: {np.max(p_error)/1e3:.6f} kPa ({np.max(p_error_percent):.4f}%)")
        print(f"Maximum displacement error: {np.max(u_error)*1000:.6f} mm")
        print(f"RMS pressure error: {np.sqrt(np.mean(p_error**2))/1e3:.6f} kPa")
        print(f"RMS displacement error: {np.sqrt(np.mean(u_error**2))*1000:.6f} mm")
        print(f"Maximum volume variation error: {np.max(delta_V_error):.6e} m³ ({np.max(delta_V_error_percent):.4f}%)")
        print(f"RMS volume variation error: {np.sqrt(np.mean(delta_V_error**2)):.6e} m³")

    print("\n" + "="*70)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("="*70)

