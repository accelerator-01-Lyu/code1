"""rebuild_heatmap_mechanism_s6.py"""
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figures.palette import COLORS, INK

CODE=os.path.dirname(os.path.abspath(__file__))
RES=os.path.join(CODE,"results"); FIG=os.path.join(CODE,"figures","final")
DATASETS=["iris","wine","glass","ecoli","aggregation","compound"]
METHODS=["Random","Entropy","CoreSet","BADGE","OriginalP2","P2_Full","P2_NoRobust","P2_NoRedundancy"]
frames=[pd.read_csv(f) for f in glob.glob(os.path.join(RES,"shards","*.csv")) if "diagnostics" not in f]
long=pd.concat(frames,ignore_index=True)
M=np.zeros((len(DATASETS),len(METHODS)))
for i,d in enumerate(DATASETS):
    base=long[(long.dataset==d)&(long.method=="P2_NoExploration")&(long.budget==.10)&(long.seed_set.astype(str)=="30")].ACC.mean()
    for j,m in enumerate(METHODS):
        M[i,j]=long[(long.dataset==d)&(long.method==m)&(long.budget==.10)&(long.seed_set.astype(str)=="30")].ACC.mean()-base
cmap=LinearSegmentedColormap.from_list("frozen",["#7FA4C8","#F7F5F2","#D08AB6","#6A5A8F"])
fig,ax=plt.subplots(figsize=(7.1,2.2))
im=ax.imshow(M,cmap=cmap,aspect="auto",vmin=-.1,vmax=.1)
for i in range(len(DATASETS)):
    for j in range(len(METHODS)):
        ax.text(j,i,f"{M[i,j]*100:.1f}",ha="center",va="center",fontsize=5)
ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS,rotation=45,ha="right",fontsize=6)
ax.set_yticks(range(len(DATASETS))); ax.set_yticklabels(DATASETS)
plt.colorbar(im,ax=ax,fraction=.025,label="ACC diff (vs NoExplo)")
fig.text(.01,.01,"OriginalP2 is the archived-bug baseline and is excluded from ranking.",fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_heatmap_diff.pdf")); plt.savefig(os.path.join(FIG,"fig_heatmap_diff.png"),dpi=300)
print("heatmap_diff rebuilt")

# mechanism b: overlay Full + NoExplo
g1=pd.read_csv(os.path.join(RES,"gradient_inner_product.csv"))
g2=pd.read_csv(os.path.join(RES,"gradient_inner_product_nonexplo.csv"))
classes=["eccentric","boundary","overlap"]
fig,ax=plt.subplots(figsize=(3.5,2.4))
for k,(g,c,lab) in enumerate([(g1,COLORS["P2_Full"],"Full"),(g2,COLORS["P2_NoExploration"],"NoExplo")]):
    data=[g[g.point_class==cl].inner_product.values for cl in classes]
    pos=np.arange(len(classes))+(k-.5)*.2
    bp=ax.violinplot(data,positions=pos,widths=.2,showmedians=True)
    for pc in bp["bodies"]: pc.set_facecolor(c); pc.set_alpha(.5)
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
ax.axhline(0,color="k",lw=.5); ax.set_title("s_i by point class (post-hoc)",fontsize=7)
fig.text(.01,.01,"Post-hoc exploratory",fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_mechanism_b.pdf")); plt.savefig(os.path.join(FIG,"fig_mechanism_b.png"),dpi=300)
print("mechanism_b done")
