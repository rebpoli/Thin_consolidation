#!/usr/bin/env python

import sys
sys.path.append('../../SRC')

import config

print("="*70)
print("TESTING CONFIGURATION SYSTEM")
print("="*70)

config.load("config.yaml")
cfg = config.get()

print("\n--- MESH ---")
print(f"Re = {cfg.mesh.Re} m")
print(f"H  = {cfg.mesh.H} m")
print(f"N  = {cfg.mesh.N}")

print("\n--- MATERIALS ---")
print(f"E     = {cfg.materials.E:.3e} Pa")
print(f"nu    = {cfg.materials.nu}")
print(f"alpha = {cfg.materials.alpha}")
print(f"mu    = {cfg.materials.mu:.3e} Pa (computed)")
print(f"lmbda = {cfg.materials.lmbda:.3e} Pa (computed)")

print("\n--- BOUNDARY CONDITIONS ---")
print(f"Bottom: U_r={cfg.boundary_conditions.bottom.U_r}, U_z={cfg.boundary_conditions.bottom.U_z}")
print(f"Right:  U_r={cfg.boundary_conditions.right.U_r}, U_z={cfg.boundary_conditions.right.U_z}")
print(f"Top:    sig_zz={cfg.boundary_conditions.top.sig_zz}, Pressure={cfg.boundary_conditions.top.Pressure}")
print(f"Left:   U_r={cfg.boundary_conditions.left.U_r}, U_z={cfg.boundary_conditions.left.U_z}")

print("\n--- NUMERICAL ---")
print(f"End time  = {cfg.numerical.end_time} s")
print(f"Num steps = {cfg.numerical.num_steps}")
print(f"dt        = {cfg.numerical.dt()} s")

print("\n--- TESTING ITERATOR ---")
print("First 5 timesteps:")
count = 0
for i_ts, dt, time in cfg.numerical:
    print(f"  Step {i_ts}: t={time:.2f}s, dt={dt:.2f}s")
    count += 1
    if count >= 5:
        break

print("\n--- EXPECTED TIME LIST ---")
expected_times = cfg.numerical.expected_time_list()
print(f"Expected times shape: {expected_times.shape}")
print(f"First 5: {expected_times[:5]}")
print(f"Last 5:  {expected_times[-5:]}")

print("\n✓ Configuration system test PASSED")
print("="*70)

