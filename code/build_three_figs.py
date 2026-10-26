"""build_three_figs.py"""
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figures.palette import COLORS, INK

CODE=os.path.dirname(os.path.abspath(__file__))
RES=os.path.join(CODE,"results"); FIG=os.path.join(CODE,"figures","final")
DATASETS=["iris","wine","glass","ecoli","aggregation","compound"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7})

# fig_mechanism
p2c=pd.read_csv(os.path.join(RES,"p2c_eccentricity.csv"))
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.1,2.6))
x=np.arange(len(DATASETS)); w=.35
f=[p2c[(p2c.dataset==d)&(p2c.method=="P2_Full")&(p2c.budget==.10)].rate_low_entropy_high_h.mean() for d in DATASETS]
ne=[p2c[(p2c.dataset==d)&(p2c.method=="P2_NoExploration")&(p2c.budget==.10)].rate_low_entropy_high_h.mean() for d in DATASETS]
ax1.bar(x-w/2,f,w,color=COLORS["P2_Full"],label="Full")
ax1.bar(x+w/2,ne,w,color=COLORS["P2_NoExploration"],label="NoExplo")
ax1.set_xticks(x); ax1.set_xticklabels(DATASETS,rotation=45,ha="right",fontsize=6)
ax1.set_title("(a) Eccentric-point selection rate @10%",fontsize=7); ax1.legend(fontsize=6)
g=pd.read_csv(os.path.join(RES,"gradient_inner_product.csv"))
classes=["eccentric","boundary","overlap"]
data=[g[g.point_class==c].inner_product.values for c in classes]
ax2.violinplot(data,showmedians=True)
ax2.set_xticks(range(1,len(classes)+1)); ax2.set_xticklabels(classes)
ax2.set_title("(b) s_i by point class (Full)",fontsize=7); ax2.axhline(0,color="k",lw=.5)
fig.text(.01,.01,"Post-hoc exploratory — not part of pre-registered P1-P4",fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_mechanism.pdf")); plt.savefig(os.path.join(FIG,"fig_mechanism.png"),dpi=300)
print("mechanism done")

# heatmap diff
METHODS=["Random","Entropy","CoreSet","BADGE","OriginalP2","P2_Full","P2_NoRobust","P2_NoRedundancy"]
frames=[pd.read_csv(f) for f in glob.glob(os.path.join(RES,"shards","*.csv")) if "diagnostics" not in f]
long=pd.concat(frames,ignore_index=True)
M=np.zeros((len(DATASETS),len(METHODS)))
for i,d in enumerate(DATASETS):
    base=long[(long.dataset==d)&(long.method=="P2_NoExploration")&(long.budget==.10)&(long.seed_set.astype(str)=="30")].ACC.mean()
    for j,m in enumerate(METHODS):
        M[i,j]=long[(long.dataset==d)&(long.method==m)&(long.budget==.10)&(long.seed_set.astype(str)=="30")].ACC.mean()-base
fig,ax=plt.subplots(figsize=(7.1,2.2))
im=ax.imshow(M,cmap="RdBu_r",aspect="auto",vmin=-.1,vmax=.1)
ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS,rotation=45,ha="right",fontsize=6)
ax.set_yticks(range(len(DATASETS))); ax.set_yticklabels(DATASETS)
plt.colorbar(im,ax=ax,fraction=.025)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_heatmap_diff.pdf")); plt.savefig(os.path.join(FIG,"fig_heatmap_diff.png"),dpi=300)
print("heatmap_diff done")
