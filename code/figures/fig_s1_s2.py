"""fig_s1_heatmap.py + fig_s2_violin.py"""
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from palette import COLORS, INK
from scipy.cluster.hierarchy import linkage, leaves_list

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(FIG, "final")
DATASETS = ["iris","wine","glass","ecoli","aggregation","compound"]
METHODS = ["Random","Entropy","CoreSet","BADGE","OriginalP2",
           "P2_Full","P2_NoExploration","P2_NoRobust","P2_NoRedundancy"]
frames=[pd.read_csv(f) for f in glob.glob(os.path.join(CODE,"results","shards","*.csv")) if "diagnostics" not in f]
long=pd.concat(frames,ignore_index=True)
sub=long[(long.budget==.10)]

# z-score per dataset
M=np.zeros((len(DATASETS),len(METHODS)))
for i,ds in enumerate(DATASETS):
    row=sub[sub.dataset==ds]
    vals=[row[row.method==m].ACC.mean() for m in METHODS]
    M[i]=(np.array(vals)-np.nanmean(vals))/np.nanstd(vals)
# cluster columns
Z=linkage(M.T,method="average"); order=leaves_list(Z)
fig,ax=plt.subplots(figsize=(7.1,2.2))
im=ax.imshow(M[:,order],cmap="PuRd",aspect="auto")
ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([METHODS[j] for j in order],rotation=45,ha="right",fontsize=6)
ax.set_yticks(range(len(DATASETS))); ax.set_yticklabels(DATASETS)
plt.colorbar(im,ax=ax,fraction=.025); plt.tight_layout()
plt.savefig(os.path.join(FINAL,"fig_s1_heatmap_zscore.pdf")); plt.savefig(os.path.join(FINAL,"fig_s1_heatmap_zscore.png"),dpi=300)
print("s1 done")

# violin
fig,axes=plt.subplots(2,3,figsize=(7.1,4),sharey=True)
for ax,ds in zip(axes.flat,DATASETS):
    data=[sub[(sub.dataset==ds)&(sub.method==m)].ACC.values for m in METHODS]
    parts=ax.violinplot(data,showmeans=False,showmedians=True)
    for pc,m in zip(parts["bodies"],METHODS): pc.set_facecolor(COLORS[m])
    ax.set_title(ds,fontsize=8); ax.set_xticks([])
plt.tight_layout(); plt.savefig(os.path.join(FINAL,"fig_s2_violin.pdf")); plt.savefig(os.path.join(FINAL,"fig_s2_violin.png"),dpi=300)
print("s2 done")
