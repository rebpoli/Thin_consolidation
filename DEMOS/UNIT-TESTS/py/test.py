#!/usr/bin/env python3
"""
Generate perfectly orthogonal quad mesh for FEniCSx.
All lines are perfectly vertical/horizontal with refinement near right boundary.
"""

import gmsh
import numpy as np

# ============================================================================
# MESH PARAMETERS
# ============================================================================

Re = 10.0           # Width
H = 1.0           # Height
N_X = 20           # Number of elements in X
N_Y = 100          # Number of elements in Y

REFINEMENT_FACTOR = 0.6  # >1 = finer near right

OUTPUT_MSH = "quad_mesh.msh"

# ============================================================================

def generate_x_coordinates(n_x, length, refinement_factor):
    """Generate X coordinates with geometric progression towards right"""
    if abs(refinement_factor - 1.0) < 1e-10:
        return np.linspace(0, length, n_x + 1)

    r = refinement_factor
    n = n_x

    x = np.zeros(n + 1)
    x[0] = 0.0

    # First element size
    dx_0 = length * (r - 1) / (r**n - 1)

    for i in range(n):
        dx_i = dx_0 * r**i
        x[i + 1] = x[i] + dx_i

    # Normalize
    x = x * (length / x[-1])

    return x

def create_mesh():
    """Create orthogonal quad mesh for FEniCSx"""

    gmsh.initialize()
    gmsh.model.add("OrthogonalQuadMesh")
    gmsh.option.setNumber("General.Terminal", 1)

    print("="*70)
    print("GENERATING ORTHOGONAL QUAD MESH FOR FENICSX")
    print("="*70)

    # Generate coordinates
    x_coords = generate_x_coordinates(N_X, Re, REFINEMENT_FACTOR)
    y_coords = np.linspace(0, H, N_Y + 1)

    print(f"X spacing: {np.diff(x_coords).min():.6f} to {np.diff(x_coords).max():.6f}")
    print(f"Y spacing: {np.diff(y_coords)[0]:.6f}")

    # Create all points in a structured grid
    points = {}
    for j in range(N_Y + 1):
        for i in range(N_X + 1):
            tag = j * (N_X + 1) + i + 1
            points[(i, j)] = gmsh.model.geo.addPoint(x_coords[i], y_coords[j], 0, tag=tag)

    # Create horizontal lines
    h_lines = {}
    line_tag = 1
    for j in range(N_Y + 1):
        for i in range(N_X):
            h_lines[(i, j)] = gmsh.model.geo.addLine(points[(i, j)], points[(i+1, j)], tag=line_tag)
            line_tag += 1

    # Create vertical lines
    v_lines = {}
    for i in range(N_X + 1):
        for j in range(N_Y):
            v_lines[(i, j)] = gmsh.model.geo.addLine(points[(i, j)], points[(i, j+1)], tag=line_tag)
            line_tag += 1

    # Create curve loops and surfaces
    surfaces = []
    surf_tag = 1
    for j in range(N_Y):
        for i in range(N_X):
            bottom = h_lines[(i, j)]
            right = v_lines[(i+1, j)]
            top = h_lines[(i, j+1)]
            left = v_lines[(i, j)]

            loop = gmsh.model.geo.addCurveLoop([bottom, right, -top, -left], tag=surf_tag)
            surf = gmsh.model.geo.addPlaneSurface([loop], tag=surf_tag)
            surfaces.append(surf)
            surf_tag += 1

    gmsh.model.geo.synchronize()

    # Mark each quad surface as transfinite (1x1 structured)
    for surf in surfaces:
        gmsh.model.mesh.setTransfiniteSurface(surf)
        gmsh.model.mesh.setRecombine(2, surf)

    # Set all curves as transfinite with 2 points (just endpoints)
    for line in list(h_lines.values()) + list(v_lines.values()):
        gmsh.model.mesh.setTransfiniteCurve(line, 2)

    # Physical groups - CRITICAL for FEniCSx
    # Bottom
    bottom_lines = [h_lines[(i, 0)] for i in range(N_X)]
    gmsh.model.addPhysicalGroup(1, bottom_lines, tag=1, name="bottom")

    # Right
    right_lines = [v_lines[(N_X, j)] for j in range(N_Y)]
    gmsh.model.addPhysicalGroup(1, right_lines, tag=2, name="right")

    # Top
    top_lines = [h_lines[(i, N_Y)] for i in range(N_X)]
    gmsh.model.addPhysicalGroup(1, top_lines, tag=3, name="top")

    # Left
    left_lines = [v_lines[(0, j)] for j in range(N_Y)]
    gmsh.model.addPhysicalGroup(1, left_lines, tag=4, name="left")

    # Domain - ALL surfaces in one physical group
    gmsh.model.addPhysicalGroup(2, surfaces, tag=1, name="domain")

    # Mesh options
    gmsh.option.setNumber("Mesh.RecombineAll", 1)

    # Generate
    gmsh.model.mesh.generate(2)

    # Verify
    print("\nVerification:")
    elem_types = gmsh.model.mesh.getElementTypes(dim=2)
    for elem_type in elem_types:
        props = gmsh.model.mesh.getElementProperties(elem_type)
        elems = gmsh.model.mesh.getElementsByType(elem_type)
        print(f"  {props[0]}: {len(elems[0])} elements")

    # Save
    gmsh.write(OUTPUT_MSH)
    print(f"\n✓ Saved: {OUTPUT_MSH}")

    gmsh.fltk.run()
    gmsh.finalize()

if __name__ == "__main__":
    create_mesh()
