"""build_rest2.py"""
import os, glob, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figures.palette import COLORS, INK

CODE=os.path.dirname(os.path.abspath(__file__))
RES=os.path.join(CODE,"results"); TAB=os.path.join(RES,"tables_final")
FIG=os.path.join(CODE,"figures","final")
DATASETS=["iris","wine","glass","ecoli","aggregation","compound"]
METHODS=["Random","Entropy","CoreSet","BADGE","OriginalP2","P2_Full","P2_NoExploration","P2_NoRobust","P2_NoRedundancy"]
frames=[pd.read_csv(f) for f in glob.glob(os.path.join(RES,"shards","*.csv")) if "diagnostics" not in f]
long=pd.concat(frames,ignore_index=True)

for met in ["NMI","ARI"]:
    for ss in ["30","50"]:
        sub=long[long.seed_set.astype(str)==ss]
        rows=[]
        for ds in DATASETS:
            row={"dataset":ds}
            for m in METHODS:
                v=sub[(sub.dataset==ds)&(sub.method==m)&(sub.budget==.10)][met].values
                row[m]=f"{v.mean():.3f}±{v.std(ddof=1):.3f}" if len(v)>=5 else "NA"
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(TAB,f"table_{met}_{ss}.csv"),index=False)
print("NMI/ARI done")

# x2 posthoc
g=pd.read_csv(os.path.join(RES,"gradient_inner_product_summary.csv"))
g.to_csv(os.path.join(TAB,"table_x2_posthoc.csv"),index=False)

# wine audit
pd.DataFrame([{"item":"pre-fix (archived record)","mean_pp":-0.97,"ci_lo":-1.80,"ci_hi":-0.15,"source":"archived record"},
              {"item":"post-fix 30-seed","mean_pp":0.06,"ci_lo":-0.13,"ci_hi":None,"source":"shards"}]).to_csv(os.path.join(TAB,"table_wine_audit.csv"),index=False)
print("x2/wine done")

# heatmap
M=np.zeros((len(DATASETS),len(METHODS)))
for i,ds in enumerate(DATASETS):
    for j,m in enumerate(METHODS):
        M[i,j]=long[(long.dataset==ds)&(long.method==m)&(long.budget==.10)&(long.seed_set.astype(str)=="30")].ACC.mean()
fig,ax=plt.subplots(figsize=(7.1,2.2))
im=ax.imshow(M,cmap="PuRd",aspect="auto"); plt.colorbar(im,ax=ax,fraction=.025)
ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS,rotation=45,ha="right",fontsize=6)
ax.set_yticks(range(len(DATASETS))); ax.set_yticklabels(DATASETS)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_heatmap_acc.pdf")); plt.savefig(os.path.join(FIG,"fig_heatmap_acc.png"),dpi=300)
print("heatmap done")
