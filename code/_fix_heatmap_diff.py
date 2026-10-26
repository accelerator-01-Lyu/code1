# -*- coding: utf-8 -*-
"""One-off fix: fig_heatmap_diff colorbar in pp (consistent with cell labels)."""
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CODE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(CODE, "results"); FIG = os.path.join(CODE, "figures", "final")
DATASETS = ["iris", "wine", "glass", "ecoli", "aggregation", "compound"]
METHODS = ["Random", "Entropy", "CoreSet", "BADGE", "OriginalP2",
           "P2_Full", "P2_NoRobust", "P2_NoRedundancy"]
frames = [pd.read_csv(f) for f in glob.glob(os.path.join(RES, "shards", "*.csv"))
          if "diagnostics" not in f]
long = pd.concat(frames, ignore_index=True)
M = np.zeros((len(DATASETS), len(METHODS)))
for i, d in enumerate(DATASETS):
    base = long[(long.dataset == d) & (long.method == "P2_NoExploration")
                & (long.budget == .10) & (long.seed_set.astype(str) == "30")].ACC.mean()
    for j, m in enumerate(METHODS):
        M[i, j] = (long[(long.dataset == d) & (long.method == m)
                        & (long.budget == .10) & (long.seed_set.astype(str) == "30")
                        ].ACC.mean() - base)
Mpp = M * 100.0
cmap = LinearSegmentedColormap.from_list(
    "frozen", ["#7FA4C8", "#F7F5F2", "#D08AB6", "#6A5A8F"])
fig, ax = plt.subplots(figsize=(7.1, 2.2))
im = ax.imshow(Mpp, cmap=cmap, aspect="auto", vmin=-10, vmax=10)
for i in range(len(DATASETS)):
    for j in range(Mpp.shape[1]):
        ax.text(j, i, f"{Mpp[i, j]:.1f}", ha="center", va="center",
                fontsize=5, color="#3F345B")
ax.set_xticks(range(len(METHODS)))
ax.set_xticklabels(METHODS, rotation=45, ha="right", fontsize=6)
ax.set_yticks(range(len(DATASETS))); ax.set_yticklabels(DATASETS)
ax.set_title("ACC difference vs NoExplo (pp), 30-seed mean at 10% budget",
             fontsize=8, color="#3F345B")
cbar = plt.colorbar(im, ax=ax, fraction=.025,
                    label="ACC difference vs NoExplo (pp)")
cbar.set_ticks([-10, -5, 0, 5, 10])
cbar.set_ticklabels(["\u221210", "\u22125", "0", "+5", "+10"])
cbar.ax.tick_params(labelsize=7)
fig.text(.01, .01,
         "OriginalP2 is the archived-bug baseline and is excluded from ranking.",
         fontsize=6, color="#3F345B")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_heatmap_diff.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG, "fig_heatmap_diff.png"), dpi=300, bbox_inches="tight")
print("heatmap_diff fixed (pp)")
