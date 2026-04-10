#!/usr/bin/env python

import sys
sys.path.append('../../SRC')

import numpy as np
import config
from mpi4py import MPI
from analytical import Analytical1DConsolidation
from mesh_generator import CylinderMesh

print("="*70)
print("TESTING ANALYTICAL SOLUTION")
print("="*70)

config.load("config.yaml")
cfg = config.get()

print("\n--- CONFIGURATION ---")
print(f"H = {cfg.mesh.H} m")
print(f"Re = {cfg.mesh.Re} m")
print(f"End time = {cfg.numerical.end_time} s")
print(f"Time steps = {cfg.numerical.num_steps}")

print("\n--- CREATING ANALYTICAL SOLUTION ---")
analytical = Analytical1DConsolidation(n_spatial_points=50, n_fourier_terms=50)

print("\n--- DERIVED PROPERTIES ---")
print(f"c_v (consolidation coefficient) = {analytical.c_v:.6e} m²/s")
print(f"p0 (initial pressure)           = {analytical.p0:.6e} Pa")
print(f"eta                             = {analytical.eta:.6f}")
print(f"S (storage coefficient)         = {analytical.S:.6e} Pa⁻¹")
print(f"K_u (undrained bulk modulus)    = {analytical.K_u:.6e} Pa")

print("\n--- TIME HISTORY ---")
history = analytical.get_history()
times = history['times']
pressures = history['pressure_at_bottom']
displacements = history['uz_at_top']
volumes = history['volume_drained']

print(f"Number of time points: {len(times)}")
print(f"Time range: {times[0]:.2f} to {times[-1]:.2f} s")

print("\n--- INITIAL CONDITIONS (t=0) ---")
print(f"Pressure at bottom: {pressures[0]:.6e} Pa")
print(f"Displacement at top: {displacements[0]:.6e} m")
print(f"Volume drained: {volumes[0]:.6e} m³")

print("\n--- FINAL STATE ---")
print(f"Pressure at bottom: {pressures[-1]:.6e} Pa")
print(f"Displacement at top: {displacements[-1]:.6e} m")
print(f"Volume drained: {volumes[-1]:.6e} m³")

print("\n--- PRESSURE DECAY ---")
decay_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
for idx in decay_indices:
    t = times[idx]
    p = pressures[idx]
    p_ratio = p / analytical.p0 if analytical.p0 != 0 else 0
    print(f"t={t:8.1f}s: p={p:12.3e} Pa ({p_ratio*100:6.2f}% of initial)")

print("\n--- DISPLACEMENT EVOLUTION ---")
for idx in decay_indices:
    t = times[idx]
    uz = displacements[idx]
    print(f"t={t:8.1f}s: uz={uz:12.6e} m ({uz*1000:8.4f} mm)")

print("\n--- VOLUME DRAINED ---")
for idx in decay_indices:
    t = times[idx]
    V = volumes[idx]
    print(f"t={t:8.1f}s: V={V:12.6e} m³")

print("\n--- SPATIAL DISTRIBUTION AT SELECTED TIMES ---")
z_test = [0.0, cfg.mesh.H/4, cfg.mesh.H/2, 3*cfg.mesh.H/4, cfg.mesh.H]
t_test = [0.0, cfg.numerical.end_time/4, cfg.numerical.end_time/2, cfg.numerical.end_time]

for t in t_test:
    print(f"\nTime t={t:.1f}s:")
    for z in z_test:
        p = analytical._pressure(t, z)
        uz = analytical._uz(t, z)
        print(f"  z={z:6.2f}m: p={p:12.3e} Pa, uz={uz:12.6e} m")

print("\n--- CHECKING BOUNDARY CONDITIONS ---")
t_mid = cfg.numerical.end_time / 2
p_top = analytical._pressure(t_mid, cfg.mesh.H)
p_bottom = analytical._pressure(t_mid, 0.0)
print(f"At t={t_mid:.1f}s:")
print(f"  Pressure at top (z=H):    {p_top:.6e} Pa (should → 0)")
print(f"  Pressure at bottom (z=0): {p_bottom:.6e} Pa")

print("\n--- MONOTONICITY CHECKS ---")
pressure_decreasing = all(pressures[i] >= pressures[i+1] for i in range(len(pressures)-1))
displacement_decreasing = all(displacements[i] <= displacements[i+1] for i in range(len(displacements)-1))
volume_increasing = all(volumes[i] <= volumes[i+1] for i in range(len(volumes)-1))

print(f"Pressure decreasing with time: {pressure_decreasing}")
print(f"Displacement magnitude increasing: {displacement_decreasing}")
print(f"Volume drained increasing: {volume_increasing}")

if pressure_decreasing and displacement_decreasing and volume_increasing:
    print("\n✓ MONOTONICITY CHECKS PASSED")
else:
    print("\n✗ MONOTONICITY CHECKS FAILED")

print("\n--- SAVING RESULTS ---")

analytical.save_timeseries()
print("✓ Timeseries saved successfully")

print("\n--- GENERATING MESH FOR XDMF OUTPUT ---")
mesh = CylinderMesh(MPI.COMM_WORLD)
domain, facets = mesh.generate()
analytical.save_vtk(domain, facets)
print("✓ XDMF saved successfully")

print("\n" + "="*70)
print("✓ ANALYTICAL SOLUTION TEST COMPLETED")
print("="*70)
