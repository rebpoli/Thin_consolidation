# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a finite element analysis framework for modeling **poroelastic consolidation** (Biot theory) in geotechnical engineering. It simulates how excess pore pressure dissipates in thin cylindrical soil/rock specimens under applied stress, comparing FEM solutions against 1D analytical solutions.

## Running Simulations

```bash
# Run from a DEMO directory (config.yaml is read from the current working directory)
cd DEMOS/02-SINGLE-CONSOLIDATION
python ../../SRC/run.py

# Install dependencies
make install_deps
```

The solver runs inside Docker: base image is `dolfinx/dolfinx:stable` (see `CONTAINER/Dockerfile`).

## Architecture

### Data Flow

```
config.yaml → Config (Pydantic) → CylinderMesh (gmsh→dolfinx) → PoroelasticitySolver → XDMF + NetCDF outputs
```

### Source Files (`SRC/`)

- **`run.py`** — Entry point. Loads config, builds mesh, runs solver.
- **`config.py`** — Pydantic config classes. `Config.load()` reads `config.yaml` from the **current working directory**. Key computed properties (shear modulus μ, Lamé λ, consolidation coefficient c_v) are derived in `MaterialCfg`.
- **`mesh_generator.py`** — `CylinderMesh` generates structured quad meshes via gmsh for an axisymmetric r-z domain. Boundaries are tagged: bottom=1, right=2, top=3, left=4.
- **`formulation.py`** — `PoroelasticityFormulation` defines the UFL weak form. Uses P2 elements for displacement, P1 for pressure. Implements axisymmetric Biot poroelasticity with Crank-Nicolson time integration. All integrals include `r dΩ` weighting for axisymmetry.
- **`fem_solver.py`** — `PoroelasticitySolver` orchestrates time integration. Solves coupled system via PETSc/MUMPS at each step. Projects stresses to DG0 space. Computes errors vs analytical at each timestep.
- **`analytical.py`** — `Analytical1DConsolidation` computes the Terzaghi 1D solution via Fourier series. Used for error validation throughout the simulation.
- **`export_pvd.py`** — Organizes XDMF output into ParaView-friendly structure with a generated Python macro.

### Key Physics

- **Mixed FEM**: P2 (displacement) + P1 (pressure) — avoids locking
- **Effective stress**: σ_total = σ' − α·p·I (Biot coefficient α)
- **Time factor**: T_v = c_v·t / (4H²) — used to set adaptive time stepping in `NumericalCfg.setup_timesteps()`
- **Boundary conditions**: Dirichlet for displacement/pressure; Neumann for stress (σ_zz applied loads); penalty method for rigid body constraints

### Demos

| Directory | Purpose |
|-----------|---------|
| `00-UNIT-TESTS` | Mesh and solver unit tests |
| `01-LIN_ELASTICITY` | Linear elasticity validation |
| `02-SINGLE-CONSOLIDATION` | Jacketed test (drained right boundary) |
| `03-SINGLE-CONSOLIDATION-UNJACKETED` | Unjacketed test |
| `04-MULTIPLE-ASPECT-RATIO-10ND` | Aspect ratio parameter sweep |
| `05-06` | Batch runs with different geometric/BC configurations |

Each demo has a `config.yaml` and writes outputs to an `outputs/` subdirectory (XDMF fields + `fem_timeseries.nc` NetCDF time series).

## Configuration Structure

`config.yaml` maps to these Pydantic classes:
- `GeneralCfg`: run metadata (description, run_dir, run_id)
- `MeshCfg`: Re (radius), H (height), N (elements per dimension)
- `MaterialCfg`: E, nu, alpha, perm, visc, M — derived properties computed automatically
- `BCCfg`: per-surface BCs (bottom/top/left/right), each with optional `U_r`, `U_z`, `P`, `sig_zz`
- `NumericalCfg`: num_steps, theta_cn (Crank-Nicolson θ, 0.5=CN, 1.0=backward Euler)
- `OutputCfg`: paths for XDMF and NetCDF output

## Dependencies

Core: `dolfinx`, `petsc4py` (MUMPS solver), `gmsh`, `ufl`, `basix`, `mpi4py`
Data: `xarray`, `netCDF4`, `numpy`, `scipy`, `pandas`
Config: `pydantic`, `PyYAML`
Visualization: `pyvista`, `meshio`, `matplotlib`
