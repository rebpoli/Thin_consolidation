#!/usr/bin/env -S python -u

import config
from mesh_generator import CylinderMesh
from fem_solver import PoroelasticitySolver

cfg = config.load()

print(cfg.summary())

print("\n--- MESH GENERATION ---")
mesh = CylinderMesh()

print("\n--- FEM SOLVER SETUP ---")
solver = PoroelasticitySolver(mesh)

solver.solve()

print(solver.summary())
