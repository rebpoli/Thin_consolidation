# DEMO 12: Theta-Weighting Validation Protocol

## Purpose

Determine whether the asymmetric theta-weighting in the weak form causes significant errors in consolidation results.

**The Issue**: 
- Flow equation (diffusion): theta-weighted time integration
- Stress BC (applied load): NO theta-weighting (only evaluated at t_{n+1})
- This asymmetry could introduce phase errors during load transitions

## Experimental Design

### Case A: Constant Stress (5 MPa)
- **Load**: Fixed constant 5 MPa throughout simulation
- **Expected behavior**: Smooth monotonic consolidation, no load discontinuities
- **Predicted outcome**: Should be unaffected by theta-weighting asymmetry
- **Baseline**: Reference for correct behavior with constant loading

### Case B: Oscillating Load (0-10 MPa, 10 cycles)
- **Load**: Piecewise constant switching between 0 and -10 MPa every 50 seconds
- **Duration**: 10 complete cycles = 1000 seconds total
- **Expected behavior**: Periodic oscillations superimposed on consolidation trend
- **Predicted outcome**: May reveal errors if theta-weighting is missing

## What to Look For

### 1. Consolidation Trends
Compare pressure dissipation and settlement between cases:
- Constant case should show monotonic decay
- Oscillating case should show same trend but with oscillations on top

**Question**: Do the consolidation rates match (same half-life)?

### 2. Oscillation Pattern (Oscillating Case)
- **Ideal**: Symmetric square-wave pressure/displacement response to load cycling
- **If theta-weighting wrong**: 
  - Asymmetric oscillations (more damping during load increase?)
  - Phase lag or leading
  - Amplitude decay

### 3. Pressure Peaks
- When load is ON (10 MPa): pressure should spike up
- When load is OFF (0 Pa): pressure should relax down
- Pattern should repeat cleanly every 100 seconds

**Quantitative check**:
```
ΔP_peak = P_max(load=ON) - P_min(load=OFF)
```
Should be consistent across all 10 cycles (no growth or decay)

### 4. Settlement Pattern
- Vertical displacement should increase during load-ON phases
- Should stabilize/reduce during load-OFF phases
- Net displacement should be monotonically increasing (consolidation)

## Analysis Steps

### Step 1: Run both cases
```bash
cd runs/constant_5mpa && python ../../SRC/run.py
cd runs/oscillating_0_10mpa && python ../../SRC/run.py
```

### Step 2: Generate comparison plot
```bash
./plot_comparison.py
```

### Step 3: Inspect outputs visually
- Do pressure curves match between constant and oscillating during uniform stress periods?
- Is oscillating case smooth or noisy?
- Are oscillations symmetric?

### Step 4: Quantitative metrics

For oscillating case at timesteps with load ON (t mod 100 in [0, 50)):
```python
import xarray as xr
ds = xr.open_dataset('runs/oscillating_0_10mpa/outputs/fem_timeseries.nc')

# Filter by load state
t = ds['time'].values
sig = ds['sig_zz_applied'].values
p_mean = ds['pressure_mean'].values

# Find indices where load is ON (sig = -10 MPa)
load_on = np.abs(sig - (-1e7)) < 1e5  # Within 1% of -10 MPa

p_peaks = p_mean[load_on]
print(f"Pressure when load ON:")
print(f"  Mean: {np.mean(p_peaks):.2e} Pa")
print(f"  Std:  {np.std(p_peaks):.2e} Pa (should be small if consolidating steadily)")
```

### Step 5: Compare to constant case

Extract constant case at same times:
```python
ds_const = xr.open_dataset('runs/constant_5mpa/outputs/fem_timeseries.nc')

# Interpolate to same times as oscillating case
p_const_interp = np.interp(t, ds_const['time'].values, ds_const['pressure_mean'].values)

# Compare at similar stress periods
# (constant = 5 MPa, oscillating = average 5 MPa over cycle)
error = np.mean(np.abs(p_const_interp[load_on] - p_mean[load_on]) / np.abs(p_mean[load_on]))
print(f"Relative difference: {error*100:.2f}%")
```

## Interpretation Guide

### If results are nearly identical:
✓ **Conclusion**: Theta-weighting asymmetry has **negligible effect**
- No correction needed
- Current implementation is sufficient
- Asymmetry is cancelled by other discretization choices

### If oscillating case shows artifacts:
✗ **Conclusion**: Theta-weighting asymmetry **matters**
- Phase lag or damping in oscillations
- Pressure peaks decay over cycles
- Settlement pattern non-monotonic
- **Recommendation**: Apply theta-weighting to stress BCs

### If oscillating case is clean but constant case is wrong:
? **Unexpected finding**
- Suggests issue is elsewhere (analytical solution validation?)
- Both cases should be internally consistent

## References

The theta-weighted weak form for stress BCs should be:

```python
# Current (potential issue):
sig_zz = bc_spec.periodic_load.eval(t_new)

# Proposed fix:
sig_zz_old = bc_spec.periodic_load.eval(t_old)
sig_zz_new = bc_spec.periodic_load.eval(t_new)
sig_zz = (1 - theta) * sig_zz_old + theta * sig_zz_new
```

This would be applied in `formulation.py:_apply_neumann_bcs()` (lines 122-125).

