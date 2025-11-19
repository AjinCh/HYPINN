import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from pathlib import Path

# ---------------------------------------------------------------------
# reproducible Path
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
root = REPO_ROOT / 'data' / 'Dataset_A'
files = os.listdir(root)

# ---------------------------------------------------------------------
# visualization of Dataset_A capacity trajectories
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(4, 2), dpi=200)

colors = [
    "#4477AA",  # blue
    "#66CCEE",  # cyan
    "#228833",  # green
    "#CCBB44",  # yellow
    "#EE6677",  # red
    "#AA3377",  # purple
]

markers = ['o', 'v', 'D', 'p', 's', '^']
legends = ['batch 1', 'batch 2', 'batch 3', 'batch 4', 'batch 5', 'batch 6']
batches = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']

line_width = 1.2
markevery = 50
markeredgewidth = 0.8

for i in range(6):
    for f in files:
        if batches[i] in f:
            path = root / f
            data = pd.read_csv(path)
            capacity = data['capacity'].values
            plt.plot(
                capacity[1:],
                color=colors[i],
                alpha=0.9,
                linewidth=line_width,
                marker=markers[i],
                markersize=2.2,
                markerfacecolor="none",
                markeredgewidth=markeredgewidth,
                markevery=markevery,
            )

plt.xlabel('Cycle')
plt.ylabel('Capacity (Ah)')

custom_lines = [
    Line2D([0], [0], color=colors[i], linewidth=line_width,
           marker=markers[i], markersize=2.5)
    for i in range(6)
]

plt.legend(custom_lines, legends, loc='upper right',
            bbox_to_anchor=(1.0, 1), frameon=False,
            ncol=3, fontsize=6)

plt.ylim([1.55, 2.05])
plt.tight_layout()

# ---------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------
save_dir = REPO_ROOT / 'results for analysis' / 'figures'
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / 'DatasetA_capacity_trajectories.png'

plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f" Figure saved at: {save_path.resolve()}")
