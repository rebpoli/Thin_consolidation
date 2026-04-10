#!/usr/bin/env -S python

# linear_elasticity_minimal.py
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

# Material parameters
E = 1e5
nu = 0.3
mu = E / (2*(1 + nu))
lambda_ = E*nu / ((1+nu)*(1-2*nu))

# Create mesh
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([1.0, 0.1, 0.1])],
    [20, 5, 5],
    cell_type=mesh.CellType.hexahedron,
)

# Function space - THIS IS THE KEY!
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

# Strain and stress
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

# Boundary condition - fixed left end
def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
u_D = np.array([0, 0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0, 0, -1e3)))

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx

# Solve
problem = LinearProblem(
    a, L,
    bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="elasticity"
)
uh = problem.solve()

# Save results
with io.XDMFFile(domain.comm, "displacement.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "Displacement"
    xdmf.write_function(uh)

print(f"Max displacement: {uh.x.array.max():.6e}")
print("✓ Solution complete!")


# ============================================
# VISUALIZATION WITH PYVISTA
# ============================================
from dolfinx import mesh, fem, io, plot, default_scalar_type
import pyvista

# Create pyvista grid
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach displacement values
grid["Displacement"] = uh.x.array.reshape((geometry.shape[0], 3))

# Plot 1: Original and deformed mesh
p = pyvista.Plotter()
p.add_mesh(grid, style="wireframe", color="black", line_width=1, label="Original")
warped = grid.warp_by_vector("Displacement", factor=10)  # Scale for visibility
p.add_mesh(warped, show_edges=True, color="lightblue", label="Deformed")
p.add_legend()
p.show_axes()
p.camera_position = 'iso'
p.show()

# Compute von Mises stress
s = sigma(uh) - 1.0 / 3 * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
von_Mises = ufl.sqrt(3.0 / 2 * ufl.inner(s, s))

V_von_mises = fem.functionspace(domain, ("DG", 0))
stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
stresses = fem.Function(V_von_mises)
stresses.interpolate(stress_expr)

# Add stress to warped mesh
warped.cell_data["VonMises"] = stresses.x.array

# Plot 2: Stress visualization
p2 = pyvista.Plotter()
warped.set_active_scalars("VonMises")
p2.add_mesh(warped, show_edges=True, scalar_bar_args={'title': 'Von Mises Stress'})
p2.show_axes()
p2.camera_position = 'iso'
p2.show()

# Save screenshot
p3 = pyvista.Plotter(off_screen=True)
p3.add_mesh(warped, show_edges=True)
p3.camera_position = 'iso'
p3.screenshot("displacement.png")
print("✓ Saved displacement.png")
