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
    """Define boundary conditions for coupled problem"""
    # First level collapse - get displacement and pressure spaces
    W0_collapsed, W0_map = W.sub(0).collapse()
    W1_collapsed, W1_map = W.sub(1).collapse()

    # Second level collapse - get individual displacement components
    W0_r_collapsed, W0_r_map = W0_collapsed.sub(0).collapse()
    W0_z_collapsed, W0_z_map = W0_collapsed.sub(1).collapse()

    bcs = []

    # Bottom: u_z = 0
    bottom_facets = facets.find(1)
    bottom_dofs_z = fem.locate_dofs_topological(
        (W.sub(0).sub(1), W0_z_collapsed),
        gdim - 1,
        bottom_facets
    )
    u_bottom = fem.Function(W0_z_collapsed)
    u_bottom.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_bottom, bottom_dofs_z, W.sub(0).sub(1)))

    # Axis: u_r = 0
    axis_facets = facets.find(4)
    axis_dofs_r = fem.locate_dofs_topological(
        (W.sub(0).sub(0), W0_r_collapsed),
        gdim - 1,
        axis_facets
    )
    u_axis = fem.Function(W0_r_collapsed)
    u_axis.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_axis, axis_dofs_r, W.sub(0).sub(0)))

    # Axis: u_r = 0
    axis_facets = facets.find(2)
    axis_dofs_r = fem.locate_dofs_topological(
        (W.sub(0).sub(0), W0_r_collapsed),
        gdim - 1,
        axis_facets
    )
    u_axis = fem.Function(W0_r_collapsed)
    u_axis.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_axis, axis_dofs_r, W.sub(0).sub(0)))

    # Top: p = 0
    top_facets = facets.find(3)
    top_dofs_p = fem.locate_dofs_topological(
        (W.sub(1), W1_collapsed),
        gdim - 1,
        top_facets
    )
    p_top = fem.Function(W1_collapsed)
    p_top.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(p_top, top_dofs_p, W.sub(1)))

    print(f"✓ Boundary conditions defined: {len(bcs)} BCs")
    print(f"  - Bottom: u_z = 0")
    print(f"  - Axis: u_r = 0")
    print(f"  - Top: p = 0")

    return bcs


# ==============================================================================
# INITIAL CONDITION
# ==============================================================================
def solve_initial_condition(W, forms, bcs):
    """
    Solve for initial condition (t=0, instantaneous loading)
    
    Parameters:
    -----------
    W : dolfinx.fem.FunctionSpace
        Mixed function space
    forms : dict
        Dictionary containing weak forms
    bcs : list
        Boundary conditions
        
    Returns:
    --------
    wh : dolfinx.fem.Function
        Initial solution
    """
    print("\n=== Computing initial condition (t=0) ===")
    
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
def time_stepping_loop(wh, problem, forms, W, T_final, num_steps,
                       H, output_dir="results"):
    """Perform time integration with analytical solution comparison"""
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

    # Create P1 functions for saving
    u_save = fem.Function(V_P1, name="Displacement")
    p_save = fem.Function(Q_P1, name="Pressure")

    # Interpolate and save initial condition
    u_save.interpolate(uh_init_collapsed)
    p_save.interpolate(ph_init_collapsed)
    u_file.write_function(u_save, 0.0)
    p_file.write_function(p_save, 0.0)

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
        'max_pressures_analytical': np.array(max_pressures_analytical)
    }

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
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # --- PARAMETERS ---
    # Geometry
    Re = 1.0          # Outer radius (m)
    H = 1.0           # Height (m)
    N = 10       # Mesh size (m)
    
    # Time parameters
    T_final = 1   # Final time (s)
    num_steps = 50    # Number of time steps
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
    bcs = setup_boundary_conditions(W, domain, facets, gdim)
    
    # --- SOLVE ---
    wh, problem = solve_initial_condition(W, forms, bcs)
    history = time_stepping_loop(wh, problem, forms, W, T_final, num_steps, H )
    
    # --- STRESSES ---
    stresses = compute_stresses(wh, W, strain_stress)
    save_stresses(stresses, domain)

    # --- VISUALIZE ---
    visualize_results(wh, W, history, T_final)
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("="*70)
