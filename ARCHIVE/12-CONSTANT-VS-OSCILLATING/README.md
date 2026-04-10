# DEMO 12: Constant vs Oscillating Load Comparison

## Purpose

Validates the theta-weighting scheme for stress boundary conditions by comparing:

1. **Constant Stress Case** (5 MPa): Monotonic consolidation with constant applied load
2. **Oscillating Load Case** (0-10 MPa): Dynamic consolidation with periodic load cycling

Both cases use identical material properties and mesh geometry to isolate the effect of load time-dependence on the consolidation response.

## Key Questions

- Is the asymmetric theta-weighting (no weighting on stress vs theta-weighted diffusion) causing errors?
- Does the oscillating load case show smooth or oscillatory behavior?
- How do constant and oscillating cases compare in terms of consolidation patterns?

## Configuration

### Case 1: Constant 5 MPa Stress
```yaml
sig_zz: -5000000.0  # Fixed 5 MPa
```
Expected: Smooth, monotonic consolidation

### Case 2: Oscillating Load 0-10 MPa
```yaml
periodic_load:
  L0: 0.0            # Baseline: 0 Pa
  L1: -10000000.0    # Active: -10 MPa
  t_start: 0.0       # Start immediately
  period: 100.0      # 100 s per cycle
  duty_cycle: 0.5    # 50% at L1, 50% at L0
  n_periods: 10      # 10 complete cycles = 1000 s total
```
Expected: Periodic pressure/displacement oscillations superimposed on consolidation trend

## Running the Demo

### Run constant stress case:
```bash
cd runs/constant_5mpa
python ../../SRC/run.py
```

### Run oscillating load case:
```bash
cd runs/oscillating_0_10mpa
python ../../SRC/run.py
```

### Plot comparison:
```bash
./plot_comparison.py
```

## Mesh & Material Properties

- **Geometry**: Re=0.1 m (10 cm radius), H=0.02 m (2 cm height)
- **Mesh**: 100 elements per dimension
- **Materials**:
  - E = 14.4 GPa
  - ν = 0.2
  - α (Biot) = 0.75
  - k = 1×10⁻¹⁷ m² (very low permeability)
  - M = 13.5 GPa
- **Time integration**: θ_CN = 0.75 (Crank-Nicolson variant)

## Expected Behavior

### Constant Case
- Initial pressure spike at t=0 due to instantaneous load application
- Smooth exponential decay of pressure over 1000+ seconds
- Corresponding smooth settlement at bottom boundary
- Should match 1D analytical solution well

### Oscillating Case
- Pressure oscillations with load cycling
- Additional component on top of consolidation trend
- Settlement should show "sawtooth" pattern with load cycling
- **May reveal theta-weighting issues** if oscillations appear damped or distorted

## Output Files

- `outputs/fem_timeseries.nc`: Time series of key quantities (pressure, displacement, stress)
- `outputs/pressure_profile.nc`: Vertical line pressure profile over time
- `outputs/results.bp`: Full spatial field data for visualization
- `png/comparison.png`: Side-by-side comparison plots

## Hypothesis

If theta-weighting is missing on stress BCs:
- Constant case should be **correct** (no discontinuities)
- Oscillating case **may show artifacts** at load transitions
- The asymmetry in time discretization could introduce phase errors during load cycling
