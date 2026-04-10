from dolfinx import fem, mesh
from basix.ufl import element, mixed_element
from mpi4py import MPI

# Create mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

# Velocity element (vector, degree 2)
P2 = element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim,))

# Pressure element (scalar, degree 1)
P1 = element("Lagrange", domain.topology.cell_name(), 1)

# Mixed element
W_el = mixed_element([P2, P1])

# Function space
W = fem.functionspace(domain, W_el)

# Now you can collapse subspaces
u_space, _ = W.sub(0).collapse()
p_space, _ = W.sub(1).collapse()

print(f"Mixed space has {W.dofmap.index_map.size_local} DOFs")
print(f"Velocity space has {u_space.dofmap.index_map.size_local} DOFs")
print(f"Pressure space has {p_space.dofmap.index_map.size_local} DOFs")
