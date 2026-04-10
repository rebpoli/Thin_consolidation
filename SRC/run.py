#!/usr/bin/env -S python -u
"""Entry point for the poroelastic consolidation solver.

Reads config.yaml from the current working directory, builds the mesh,
runs the time integration, and prints a results summary.  Symlinked into
each single-run demo directory.
"""

import config
from mesh_generator import CylinderMesh
from fem_solver import PoroelasticitySolver

#
#

cfg = config.load()

print(cfg.summary())

print("\n--- MESH GENERATION ---")
mesh = CylinderMesh()

print("\n--- FEM SOLVER SETUP ---")
solver = PoroelasticitySolver(mesh)

solver.solve()

print(solver.summary())
