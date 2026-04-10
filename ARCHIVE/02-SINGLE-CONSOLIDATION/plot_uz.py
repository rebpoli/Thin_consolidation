#!/usr/bin/env -S python
"""Plot uz at top and bottom vs time, with the applied sig_zz overlaid."""
import xarray as xr
import matplotlib.pyplot as plt

NC_FILE = "outputs/fem_timeseries.nc"

ds = xr.open_dataset(NC_FILE, engine="scipy")

missing = [v for v in ("uz_at_top", "sig_zz_applied") if v not in ds]
if missing:
    raise KeyError(
        f"Variables {missing} not found in {NC_FILE}. "
        "Re-run the solver to regenerate the file with the updated code."
    )

t          = ds["time"].values
uz_bottom  = ds["uz_at_bottom"].values * 1e6   # m → μm
uz_top     = ds["uz_at_top"].values    * 1e6   # m → μm
sig_zz     = ds["sig_zz_applied"].values / 1e3  # Pa → kPa

fig, ax1 = plt.subplots(figsize=(11, 4))

ax1.plot(t, uz_bottom, color="steelblue",  label="uz bottom (z=0)")
ax1.plot(t, uz_top,    color="darkorange", label="uz top    (z=H)")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Vertical displacement [μm]", color="black")
ax1.tick_params(axis="y")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.step(t, sig_zz, color="crimson", linewidth=1.2, where="post", label="σ_zz applied")
ax2.set_ylabel("Applied σ_zz [kPa]", color="crimson")
ax2.tick_params(axis="y", labelcolor="crimson")
ax2.legend(loc="upper right")

plt.title("Z-displacement and applied load vs time")
plt.tight_layout()
plt.savefig("uz_vs_time.png", dpi=150)
print("Saved uz_vs_time.png")
plt.show()
