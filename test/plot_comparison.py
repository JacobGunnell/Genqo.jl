"""Create box and whisker plots comparing benchmark results between Python and Julia genqo implementations."""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Load benchmark results
with open(".benchmarks/py-bench.json") as f:
    py = json.load(f)
with open(".benchmarks/jl-bench.json") as f:
    jl = json.load(f)

py_times = {bm["name"].removeprefix("test_").replace("__", "."): np.array(bm["stats"]["data"]) for bm in py["benchmarks"]}
jl_times = {name: np.array(bm[1]["times"])/1e9 for (name, bm) in jl[1][0][1]["data"].items()}

# Get all functions and sort by decreasing median time
all_funcs = list(set(py_times.keys()) | set(jl_times.keys()))
all_funcs.sort(key=lambda f: max(np.median(py_times.get(f, 0)), np.median(jl_times.get(f, 0))), reverse=True)

# Create paired boxplots
fig, ax = plt.subplots(figsize=(14, 6))

positions = []
data = []
colors = []
labels = []
for i, func in enumerate(all_funcs):
    if func in py_times:
        positions.append(3*i + 1)
        data.append(py_times[func])
        colors.append('lightblue')
    
    if func in jl_times:
        positions.append(3*i + 2)
        data.append(jl_times[func])
        colors.append('lightcoral')

bp = ax.boxplot(
    data, 
    positions=positions, 
    medianprops=dict(color="black", lw=0.5), 
    boxprops=dict(lw=0.5), 
    whiskerprops=dict(lw=0.5), 
    capprops=dict(lw=0.5), 
    widths=1.0, 
    patch_artist=True, 
    showfliers=False
)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set_xticks([3*i + 1.5 for i in range(len(all_funcs))])
ax.set_xticklabels(all_funcs, rotation=45, ha='right')
ax.set_yscale('log')
ax.set_ylabel("Time (s)")
ax.set_title("Genqo Benchmark Comparison")
ax.grid(axis='y', alpha=0.3)

# Add legend
legend_elements = [
    Patch(facecolor='lightblue', label='Python'),
    Patch(facecolor='lightcoral', label='Julia')
]
ax.legend(handles=legend_elements, loc='upper right')

fig.tight_layout()
fig.savefig(".benchmarks/benchmark_comparison.svg")
