# -*- coding: utf-8 -*-
"""add2_cd_redraw.py — Clean CD diagram with non-overlapping labels."""
import os, sys
import matplotlib.pyplot as plt
sys.path.insert(0, r"D:\AL_FuzzyClustering_Paper\v2_unified\code")
import additional_common as ac

ac.setup_style()

# method, avg_rank, color
methods = [
    ("CoreSet",         1.83, "#4F9D9D"),
    ("P2_NoRedundancy", 3.50, "#B8B5D8"),
    ("P2_NoExploration",3.83, "#D08AB6"),
    ("P2_NoRobust",     4.83, "#8B86C2"),
    ("P2_Full",         5.33, "#6A5A8F"),
    ("Entropy",         5.33, "#6B96C7"),
    ("Random",          5.50, "#A9C4DE"),
    ("BADGE",           5.83, "#8A8F98"),
]
CD = 4.29

fig, ax = plt.subplots(figsize=(8.5, 3.2))
ax.axhline(y=0, color="#333333", lw=1.5)

# label vertical offsets to avoid overlap
offsets = [0.55, 0.35, 0.55, 0.55, 0.35, 0.55, 0.35, 0.55]
for (name, rank, color), off in zip(methods, offsets):
    ax.plot([rank, rank], [0, off], color=color, lw=1.2)
    ax.plot(rank, 0, "o", color=color, ms=8, zorder=5)
    ax.text(rank, off+0.05, name, ha="center", va="bottom",
            fontsize=9.5, color=color, fontweight="bold")

# CD bar
ax.plot([2.0, 2.0+CD], [-0.35, -0.35], color="#222222", lw=5, solid_capstyle="butt")
ax.annotate("", xy=(2.0, -0.55), xytext=(2.0+CD, -0.55),
            arrowprops=dict(arrowstyle="<->", color="#222222", lw=1.2))
ax.text(2.0+CD/2, -0.75, f"CD = {CD:.2f}", ha="center", va="top", fontsize=10)

ax.set_xlim(1, 7)
ax.set_ylim(-1.0, 1.1)
ax.set_yticks([])
ax.set_xlabel("Average rank (left = better)", fontsize=11)
ax.set_title("All pre-registered methods at 10% (appendix; CoreSet reported)", fontsize=11)
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(ac.FIG_ADD,"add2_cd_full.png"), dpi=600, bbox_inches="tight")
fig.savefig(os.path.join(ac.FIG_ADD,"add2_cd_full.pdf"), bbox_inches="tight")
print("saved add2_cd_full")
