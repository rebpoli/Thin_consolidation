#!/usr/bin/env -S python -u  -i

import pandas as pd
from pathlib import Path
from glob import glob

runs_dir = 'runs'
output_file = 'all_results.pkl'

dfs = []
for pkl_file in glob('runs/run.*/outputs/summary.pkl'):
    print(pkl_file)
    dfs.append(pd.read_pickle(pkl_file))

df = pd.concat(dfs, ignore_index=True)

print(df)
# combined.to_pickle(output_file)
# print(f"Combined {len(dfs)} runs into {output_file}")
# print(combined)


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(df['Re/H'], df['volume_error_percent'], s=100, alpha=0.6)
plt.xlabel('Aspect Ratio (Re/H)')
plt.ylabel('Volume Error (%)')
plt.title('Volume Error vs Aspect Ratio')
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.tight_layout()
plt.savefig('volume_error_vs_aspect_ratio.png', dpi=150)

# plt.figure(figsize=(10, 6))
# plt.scatter(df['Re/H'], df['pressure_error_percent'], s=100, alpha=0.6)
# plt.xlabel('Aspect Ratio (Re/H)')
# plt.ylabel('Pressure Error (%)')
# plt.title('Pressure Error vs Aspect Ratio')
# plt.grid(True, alpha=0.3)
# plt.xscale('log')
# plt.tight_layout()
# plt.savefig('pressure_error_vs_aspect_ratio.png', dpi=150)

# plt.figure(figsize=(10, 6))
# plt.scatter(df['Re/H'], df['uz_error_percent'], s=100, alpha=0.6)
# plt.xlabel('Aspect Ratio (Re/H)')
# plt.ylabel('Uz Error (%)')
# plt.title('Uz Error vs Aspect Ratio')
# plt.grid(True, alpha=0.3)
# plt.xscale('log')
# plt.tight_layout()
# plt.savefig('uz_error_vs_aspect_ratio.png', dpi=150)

plt.figure(figsize=(10, 6))
plt.scatter(df['Re/H'], df['final_time']/60/60, s=100, alpha=0.6)
plt.xlabel('Aspect Ratio (Re/H)')
plt.ylabel('Final time (h)')
plt.title('Final test time')
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('uz_error_vs_aspect_ratio.png', dpi=150)
plt.show()
