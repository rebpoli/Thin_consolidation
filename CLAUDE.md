# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a finite element analysis framework for modeling **poroelastic consolidation** (Biot theory) in geotechnical engineering. It simulates how excess pore pressure dissipates in thin cylindrical soil/rock specimens under applied stress, comparing FEM solutions against 1D analytical solutions.

## Running Simulations

```bash
# Single-run demo (config.yaml read from cwd)
cd DEMOS/VALIDATION/01-Traditional-Consolidation
make all           # run + postproc

# Sweep demo
cd DEMOS/PAPER-01/OAT-SEALED
make gen           # generate runs/*/config.yaml
make run           # run all cases in parallel
make postproc      # generate plots

# Run everything
cd DEMOS && make all
```

The solver runs inside Docker: base image is `dolfinx/dolfinx:stable` (see `CONTAINER/Dockerfile`).

## Repository Layout

```
SRC/        Library modules (FEM solver, config, mesh, analytical)
TOOLS/      Generic CLI tools (batch runner, postproc dispatcher, viewer, animator)
DEMOS/      All test cases and paper demos
ARCHIVE/    Legacy demos (not maintained)
CONTAINER/  Docker build files
```

## Architecture

### Data Flow

```
config.yaml → Config (Pydantic) → CylinderMesh (gmsh→dolfinx) → PoroelasticitySolver → XDMF + NetCDF outputs
```

### Source Files (`SRC/`)

- **`run.py`** — Entry point. Loads config, builds mesh, runs solver. Symlinked by demos.
- **`config.py`** — Pydantic config classes. `Config.load()` reads `config.yaml` from the **current working directory**. Key computed properties (shear modulus μ, Lamé λ, consolidation coefficient c_v) are derived in `MaterialCfg`.
- **`mesh_generator.py`** — `CylinderMesh` generates structured quad meshes via gmsh for an axisymmetric r-z domain. Boundaries are tagged: bottom=1, right=2, top=3, left=4.
- **`formulation.py`** — `PoroelasticityFormulation` defines the UFL weak form. Uses P2 elements for displacement, P1 for pressure. Implements axisymmetric Biot poroelasticity with Crank-Nicolson time integration. All integrals include `r dΩ` weighting for axisymmetry.
- **`fem_solver.py`** — `PoroelasticitySolver` orchestrates time integration. Solves coupled system via PETSc/MUMPS at each step. Projects stresses to DG0 space. Computes errors vs analytical at each timestep.
- **`analytical.py`** — `Analytical1DConsolidation` computes the Terzaghi 1D solution via Fourier series. Used for error validation throughout the simulation.
- **`export_pvd.py`** — Organizes XDMF output into ParaView-friendly structure with a generated Python macro.
- **`vertical_line_sampler.py`** — `VerticalLineSampler` extracts pressure profiles along the vertical centerline for NetCDF export.

### Tools (`TOOLS/`)

All tools are executable (`./tool.py`) and designed to be called from a demo directory:

- **`run_sweep.py`** — Generic parallel batch runner. Discovers `runs/*/config.yaml` and runs the solver in each, showing a live progress table. Options: `--n-jobs`, `--dry-run`, `--rerun-failed`.
- **`postproc.py`** — Dispatcher: calls `py/postproc.py` in the current demo directory, forwarding all CLI args. Use `--script py/other.py` to run a different script.
- **`animate.py`** — Animates pore pressure evolution along the vertical centerline from a NetCDF pressure profile file.
- **`view_fields.py`** — PyVista-based interactive viewer for XDMF field outputs.

### Demos (`DEMOS/`)

#### Standard demo contract

```
DEMOS/<GROUP>/<NAME>/
├── config.yaml        # single-run: FEM config; sweep: base template
├── sweep.yaml         # (sweep demos only) parameter sweep definition
├── run.py             # symlink → SRC/run.py  (single-run demos only)
├── runs/              # generated per-run configs + outputs  (gitignored)
├── outputs/           # single-run outputs  (gitignored)
├── png/               # generated plots  (gitignored)
└── py/                # demo-specific scripts
    ├── gen_configs.py  # (sweep demos) generates runs/ from parameters
    └── postproc.py     # plot/analysis entry point
```

All scripts in `py/` are executable (`./py/script.py`).

#### Makefile targets

| Target | Single-run demos | Sweep demos |
|--------|-----------------|-------------|
| `run` | run FEM solver | run all sweep cases via `TOOLS/run_sweep.py` |
| `gen` | — | generate `runs/*/config.yaml` via `py/gen_configs.py` |
| `postproc` | call `TOOLS/postproc.py` | call `TOOLS/postproc.py` |
| `clean` | remove `outputs/` | remove `runs/ log/ png/` |
| `all` | `run + postproc` | `gen + run + postproc` |

#### Demo inventory

| Path | Type | Purpose |
|------|------|---------|
| `UNIT-TESTS/` | unit tests | Config, mesh, and analytical solution unit tests |
| `VALIDATION/01-Traditional-Consolidation/` | single-run | Jacketed consolidation vs Terzaghi analytical |
| `VALIDATION/02-Thin-Disc-Consolidation/` | single-run | Unjacketed thin disc, drained side, converges to analytical |
| `PAPER-01/OAT-SEALED/` | sweep (19 cases) | OAT sensitivity, sealed lateral boundary |
| `PAPER-01/OAT-DRAINED/` | sweep (19 cases) | OAT sensitivity, drained lateral boundary |
| `PAPER-01/py/compare.py` | analysis | Cross-comparison sealed vs drained |

## Configuration Structure

`config.yaml` maps to these Pydantic classes:
- `GeneralCfg`: run metadata (description, run_dir, run_id)
- `MeshCfg`: Re (radius), H (height), N (elements per dimension)
- `MaterialCfg`: E, nu, alpha, perm, visc, M — derived properties computed automatically
- `BCCfg`: per-surface BCs (bottom/top/left/right), each with optional `U_r`, `U_z`, `P`, `sig_zz`, `periodic_load`
- `NumericalCfg`: theta_cn (Crank-Nicolson θ), end_time_tv, dt_min_s, dt_max_s, dt_factor
- `OutputCfg`: paths for field results and NetCDF time series

## Dependencies

Core: `dolfinx`, `petsc4py` (MUMPS solver), `gmsh`, `ufl`, `basix`, `mpi4py`
Data: `xarray`, `netCDF4`, `numpy`, `scipy`, `pandas`
Config: `pydantic`, `PyYAML`
Visualization: `pyvista`, `meshio`, `matplotlib`
