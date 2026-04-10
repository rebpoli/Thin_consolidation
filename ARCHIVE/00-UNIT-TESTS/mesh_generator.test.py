#!/usr/bin/env python

import sys
sys.path.append('../../SRC')

#!/usr/bin/env python

from mpi4py import MPI
import numpy as np
import config
from mesh_generator import CylinderMesh

print("="*70)
print("MESH GENERATOR - INTERACTIVE TEST")
print("="*70)

config.load("config.yaml")
cfg = config.get()

Re = cfg.mesh.Re
H = cfg.mesh.H
N = cfg.mesh.N


print("\n--- CREATING FENICSX DOMAIN ---")
mesh = CylinderMesh(comm=MPI.COMM_WORLD)
domain, facets = mesh.domain, mesh.facets

print("\n--- FENICSX DOMAIN INFORMATION ---")

tdim = domain.topology.dim
gdim = domain.geometry.dim

print(f"\nDimensions:")
print(f"  Topological dimension: {tdim}")
print(f"  Geometrical dimension: {gdim}")

print(f"\nTopology:")
num_cells_local = domain.topology.index_map(tdim).size_local
num_cells_global = domain.topology.index_map(tdim).size_global
print(f"  Cells (local):  {num_cells_local}")
print(f"  Cells (global): {num_cells_global}")

domain.topology.create_connectivity(tdim-1, tdim)
num_facets = domain.topology.index_map(tdim-1).size_local
print(f"  Facets (local): {num_facets}")

domain.topology.create_connectivity(0, tdim)
num_vertices = domain.topology.index_map(0).size_local
print(f"  Vertices (local): {num_vertices}")

print(f"\nGeometry:")
coords = domain.geometry.x
print(f"  Coordinate array shape: {coords.shape}")
print(f"  Min coordinates: r={coords[:, 0].min():.6f}, z={coords[:, 1].min():.6f}")
print(f"  Max coordinates: r={coords[:, 0].max():.6f}, z={coords[:, 1].max():.6f}")

print(f"\nCell type: {domain.topology.cell_name()}")

print(f"\n--- BOUNDARY FACETS ---")
unique_markers = np.unique(facets.values)
print(f"Boundary markers present: {sorted(unique_markers)}")

marker_names = {1: "bottom", 2: "right", 3: "top", 4: "left"}
total_boundary_facets = 0

for marker in sorted(unique_markers):
    facet_indices = facets.find(marker)
    num_facets_marker = len(facet_indices)
    total_boundary_facets += num_facets_marker
    name = marker_names.get(marker, "unknown")
    print(f"  Marker {marker} ({name:6s}): {num_facets_marker:4d} facets")

print(f"  Total boundary facets: {total_boundary_facets}")

print(f"\n--- MESH QUALITY CHECKS ---")

cell_volumes = []
for cell_idx in range(num_cells_local):
    cell = domain.geometry.x[domain.geometry.dofmap[cell_idx]]
    
    r_coords = cell[:, 0]
    z_coords = cell[:, 1]
    
    r_center = np.mean(r_coords)
    
    if r_center > 0:
        area_2d = 0.5 * abs(
            (r_coords[1] - r_coords[0]) * (z_coords[2] - z_coords[0]) -
            (r_coords[2] - r_coords[0]) * (z_coords[1] - z_coords[0])
        )
        volume_axisym = 2 * np.pi * r_center * area_2d
        cell_volumes.append(volume_axisym)

if cell_volumes:
    cell_volumes = np.array(cell_volumes)
    print(f"Cell volume statistics (axisymmetric):")
    print(f"  Min:  {cell_volumes.min():.6e} m³")
    print(f"  Max:  {cell_volumes.max():.6e} m³")
    print(f"  Mean: {cell_volumes.mean():.6e} m³")
    print(f"  Std:  {cell_volumes.std():.6e} m³")
    
    total_volume = cell_volumes.sum()
    expected_volume = np.pi * Re**2 * H
    print(f"\nTotal volume: {total_volume:.6e} m³")
    print(f"Expected volume (πR²H): {expected_volume:.6e} m³")
    print(f"Relative error: {abs(total_volume - expected_volume)/expected_volume * 100:.4f}%")

print("\n" + "="*70)
print("✓ MESH GENERATION TEST COMPLETED")
print("="*70)
print("\nMesh files can be visualized using:")
print("  - ParaView (if VTK files are exported)")
print("  - gmsh (rerun this script)")
print("\nThe mesh is ready for use in the FEM solver!")
print("="*70)
