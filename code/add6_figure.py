# -*- coding: utf-8 -*-
"""add6_figure.py v2 — clean rho x beta heatmaps, colorbar inside each panel."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, r"D:\AL_FuzzyClustering_Paper\v2_unified\code")
import additional_common as ac
from matplotlib.colors import LinearSegmentedColormap

ac.setup_style()
df = pd.read_csv(os.path.join(ac.RES_ADD, "add6_rhobeta_grid.csv"))

rhos = [0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 4.0]
betas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
cmap = LinearSegmentedColormap.from_list("cool",
    ["#EDEFF2","#C9D8E8","#A9C4DE","#6B96C7","#3E7AAB"])

rho_sets = {"ecoli": [0.0,0.05,0.1,0.15,0.25,0.5,1.0,2.0,4.0],
            "compound": [0.0,0.05,0.1,0.25,0.5,1.0]}
for ax, ds in zip(axes, ["ecoli","compound"]):
    rhos = rho_sets[ds]
    M = np.full((len(rhos), len(betas)), np.nan)
    for _,r in df[df.dataset==ds].iterrows():
        if r.rho in rhos and r.beta in betas:
            M[rhos.index(r.rho), betas.index(r.beta)] = r.acc_mean
    vmin = np.nanmin(M); vmax = np.nanmax(M)
    im = ax.imshow(M, aspect="auto", cmap=cmap, origin="lower",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(betas))); ax.set_xticklabels([str(b) for b in betas])
    ax.set_yticks(range(len(rhos))); ax.set_yticklabels([str(r) for r in rhos])
    ax.set_xlabel("beta (redundancy weight)", fontsize=10)
    ax.set_ylabel("rho (exploration weight)", fontsize=10)
    ax.set_title(ds.upper(), fontsize=11, fontweight="bold")
    for i in range(len(rhos)):
        for j in range(len(betas)):
            if not np.isnan(M[i,j]):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                        fontsize=7, color="#222222")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)

fig.tight_layout(pad=0.5)
out_png = os.path.join(ac.FIG_ADD, "add6_rhobeta_heatmap.png")
out_pdf = os.path.join(ac.FIG_ADD, "add6_rhobeta_heatmap.pdf")
fig.savefig(out_png, dpi=600, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print("saved", out_png)
