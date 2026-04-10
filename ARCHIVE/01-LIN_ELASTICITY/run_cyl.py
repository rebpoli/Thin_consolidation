#!/usr/bin/env -S python -i

import numpy as np
import matplotlib.pyplot as plt

import gmsh
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, io
from dolfinx.io import gmsh as gmshio
import dolfinx.fem.petsc

# Geometry parameters
Re = 1.0    # Outer radius (solid cylinder, no inner hole)
H = 2.0     # Height of cylinder
hsize = 0.1  # Mesh size

# Create mesh using Gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gdim = 2
model_rank = 0
gmsh.model.add("Cylinder")

geom = gmsh.model.geo

# Create rectangle in (r,z) plane for axisymmetric cylinder
# Points: (0,0), (Re,0), (Re,H), (0,H)
p1 = geom.add_point(0, 0, 0)     # bottom-left (axis)
p2 = geom.add_point(Re, 0, 0)    # bottom-right
p3 = geom.add_point(Re, H, 0)    # top-right
p4 = geom.add_point(0, H, 0)     # top-left (axis)

# Create lines
bottom = geom.add_line(p1, p2)   # z=0 (bottom)
right = geom.add_line(p2, p3)    # r=Re (outer surface)
top = geom.add_line(p3, p4)      # z=H (top)
left = geom.add_line(p4, p1)     # r=0 (axis)

# Create surface
boundary = geom.add_curve_loop([bottom, right, top, left])
surf = geom.add_plane_surface([boundary])

geom.synchronize()

# Set mesh size
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hsize)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hsize)

# Add physical groups
gmsh.model.addPhysicalGroup(gdim, [surf], 1)
gmsh.model.addPhysicalGroup(gdim - 1, [bottom], 1, name="bottom")
gmsh.model.addPhysicalGroup(gdim - 1, [right], 2, name="right")
gmsh.model.addPhysicalGroup(gdim - 1, [top], 3, name="top")
gmsh.model.addPhysicalGroup(gdim - 1, [left], 4, name="left")

gmsh.model.mesh.generate(gdim)

# Import mesh to DOLFINx
mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, model_rank, gdim=gdim)
domain = mesh_data.mesh
facets = mesh_data.facet_tags
cell_tags = mesh_data.cell_tags

gmsh.finalize()

# Spatial coordinate for axisymmetric formulation
x = ufl.SpatialCoordinate(domain)

# Strain definition for axisymmetric coordinates
def eps(v):
    """Strain tensor in axisymmetric coordinates (r,z)"""
    e_rr = v[0].dx(0)                           # ∂u_r/∂r
    e_tt = v[0] / x[0]                          # u_r/r (hoop strain)
    e_zz = v[1].dx(1)                           # ∂u_z/∂z
    e_rz = 0.5 * (v[0].dx(1) + v[1].dx(0))     # Shear strain

    return ufl.sym(
        ufl.as_tensor([
            [e_rr, 0,    e_rz],
            [0,    e_tt, 0   ],
            [e_rz, 0,    e_zz]
        ])
    )

# Material properties
E = fem.Constant(domain, 1e5)
nu = fem.Constant(domain, 0.3)
mu = E / 2 / (1 + nu)
lmbda = E * nu / (1 + nu) / (1 - 2 * nu)

def sigma(v):
    """Stress tensor"""
    return lmbda * ufl.tr(eps(v)) * ufl.Identity(3) + 2.0 * mu * eps(v)

# Function space
V = fem.functionspace(domain, ("P", 2, (gdim,)))
du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)

# Define measures with subdomain data
dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facets)

# Bilinear form (including r factor for axisymmetric)
a_form = ufl.inner(sigma(du), eps(u_)) * x[0] * dx

# Applied traction on top (sigma_zz = -1e5, compression)
applied_stress = fem.Constant(domain, -1e0)
# Traction vector: T = sigma·n, on top n = (0, 1)
# So T_z = sigma_zz = applied_stress
traction = ufl.as_vector([0, applied_stress])
L_form = ufl.inner(traction, u_) * x[0] * ds(3)  # Apply on top boundary

# Boundary conditions
# Bottom: u_z = 0 (only vertical displacement constrained)
# Axis (left): u_r = 0 (symmetry condition, radial displacement = 0)

Vx, _ = V.sub(0).collapse()  # r-component subspace
Vy, _ = V.sub(1).collapse()  # z-component subspace

# Bottom boundary: u_z = 0
bottom_dofs_z = fem.locate_dofs_topological((V.sub(1), Vy), gdim - 1, facets.find(1))
u_bottom = fem.Function(Vy)
u_bottom.x.array[:] = 0.0

# Axis boundary: u_r = 0 (symmetry)
axis_dofs_r = fem.locate_dofs_topological((V.sub(0), Vx), gdim - 1, facets.find(4))
u_axis = fem.Function(Vx)
u_axis.x.array[:] = 0.0

bcs = [
    fem.dirichletbc(u_bottom, bottom_dofs_z, V.sub(1)),  # u_z = 0 on bottom
    fem.dirichletbc(u_axis, axis_dofs_r, V.sub(0)),      # u_r = 0 on axis
]

# Solve
u = fem.Function(V, name="Displacement")
problem = fem.petsc.LinearProblem(
    a_form,
    L_form,
    u=u,
    bcs=bcs,
    petsc_options={
        "ksp_type": "bcgsl",         # Enhanced BiCGStab(L)
        "ksp_bcgsl_ell": 2,          # Search directions
        "pc_type": "gamg",           # Algebraic multigrid
        "pc_gamg_type": "agg",
        "ksp_rtol": 1e-8,
        "ksp_monitor": None,         # See convergence
        "ksp_converged_reason": None # See why it converged
        },
    petsc_options_prefix="elasticity_"
)
problem.solve()

print("✓ Solution completed")
print(f"  Max u_r: {u.sub(0).collapse().x.array.max():.6e}")
print(f"  Max u_z: {u.sub(1).collapse().x.array.max():.6e}")

# Visualization with pyvista
import pyvista
from dolfinx import plot

pyvista.OFF_SCREEN = True

# Create pyvista grid
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach 2D displacement and pad to 3D for pyvista
displacement_2d = u.x.array.reshape((geometry.shape[0], 2))
displacement_3d = np.zeros((displacement_2d.shape[0], 3))
displacement_3d[:, 0] = displacement_2d[:, 0]  # r component
displacement_3d[:, 1] = displacement_2d[:, 1]  # z component

grid["displacement"] = displacement_3d

# Create warped mesh (amplified deformation)
warped = grid.warp_by_vector("displacement", factor=10)

# Plot original and deformed
plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="black", opacity=0.5, line_width=2)
plotter.add_mesh(warped, show_edges=True, color="lightblue")
plotter.camera_position = 'xy'
plotter.view_xy()
plotter.screenshot('cylinder_deformation.png', window_size=[1920, 1080])
print("✓ Saved cylinder_deformation.png")

# Plot displacement magnitude
displacement_mag = np.linalg.norm(displacement_3d, axis=1)
warped.point_data["displacement_magnitude"] = displacement_mag

plotter2 = pyvista.Plotter()
plotter2.add_mesh(warped, scalars="displacement_magnitude", show_edges=True,
                  cmap="jet", scalar_bar_args={'title': 'Displacement Magnitude'})
plotter2.camera_position = 'xy'
plotter2.view_xy()
plotter2.screenshot('cylinder_displacement_magnitude.png', window_size=[1920, 1080])
print("✓ Saved cylinder_displacement_magnitude.png")

# Save results - interpolate P2 to P1 for XDMF
# V_P1 = fem.functionspace(domain, ("P", 1, (gdim,)))
# u_P1 = fem.Function(V_P1, name="Displacement")
# u_P1.interpolate(u)
# with io.XDMFFile(domain.comm, "cylinder_results.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_function(u_P1)
# print("✓ Saved cylinder_results.xdmf")


# Calculate and visualize sigma_zz (vertical stress)

# Define DG space for stress (piecewise constant per element)
DG = fem.functionspace(domain, ("DG", 0))

# Create expression for sigma_zz component
sigma_zz_expr = sigma(u)[2, 2]

# Project to DG space
stress_expr = fem.Expression(sigma_zz_expr, DG.element.interpolation_points)
sigma_zz_fem = fem.Function(DG, name="Sigma_zz")
sigma_zz_fem.interpolate(stress_expr)

print(f"  Max σ_zz: {sigma_zz_fem.x.array.max():.6e}")
print(f"  Min σ_zz: {sigma_zz_fem.x.array.min():.6e}")

# Visualize stress with pyvista
topology, cell_types, geometry = plot.vtk_mesh(domain)
grid_stress = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach stress values as CELL data
grid_stress.cell_data["sigma_zz"] = sigma_zz_fem.x.array

# Plot stress distribution
plotter3 = pyvista.Plotter()
plotter3.add_mesh(grid_stress, scalars="sigma_zz", show_edges=True,
                  cmap="coolwarm",
                  scalar_bar_args={'title': 'Vertical Stress σ_zz [Pa]'})
plotter3.camera_position = 'xy'
plotter3.view_xy()
plotter3.screenshot('cylinder_stress_zz.png', window_size=[1920, 1080])
print("✓ Saved cylinder_stress_zz.png")

# Save stress to XDMF
with io.XDMFFile(domain.comm, "cylinder_stress.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(sigma_zz_fem)
print("✓ Saved cylinder_stress.xdmf")
