"""fig_forest.py — revised forest with sig stars."""
import os, glob, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from palette import COLORS, INK

CODE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG=os.path.dirname(os.path.abspath(__file__)); FINAL=os.path.join(FIG,"final")
DATASETS=["iris","wine","glass","ecoli","aggregation","compound"]
frames=[pd.read_csv(f) for f in glob.glob(os.path.join(CODE,"results","shards","*.csv")) if "diagnostics" not in f]
long=pd.concat(frames,ignore_index=True)

rows=[]
for ss in ["30","50"]:
    sub=long[long.seed_set.astype(str)==ss]
    for ds in DATASETS:
        f=sub[(sub.dataset==ds)&(sub.method=="P2_Full")&(sub.budget==.10)].sort_values("seed")
        ne=sub[(sub.dataset==ds)&(sub.method=="P2_NoExploration")&(sub.budget==.10)].sort_values("seed")
        m=f.merge(ne,on="seed",suffixes=("_f","_ne"))
        d=m.ACC_ne.values-m.ACC_f.values; n=len(d)
        se=d.std(ddof=1)/np.sqrt(n); tc=stats.t.ppf(.95,n-1)
        p=2*(1-stats.t.cdf(abs(d.mean()/se),n-1))
        rows.append({"ss":ss,"ds":ds,"m":d.mean(),"lo":d.mean()-tc*se,"p":p})
r=pd.DataFrame(rows)
fig,ax=plt.subplots(figsize=(4.5,3.2))
for i,ds in enumerate(DATASETS):
    for k,ss in enumerate(["30","50"]):
        row=r[(r.ds==ds)&(r.ss==ss)].iloc[0]; y=i+(k-.5)*.32
        sig = row.p<.05
        c=COLORS["P2_NoExploration"] if sig else "white"
        ec=COLORS["P2_NoExploration"] if sig else "gray"
        ax.plot([row.lo,row.m],[y,y],color=COLORS["P2_Full"],lw=1.2)
        ax.plot(row.m,y,"o",mfc=c,mec=ec,ms=5)
        star="*" if row.p<.05 else ("**" if row.p<.01 else "")
        if star: ax.text(row.m+.002,y,star,fontsize=8,color=INK)
ax.axvline(0,color="k",lw=.8)
ax.axvline(-.005,color="gray",ls="--",lw=.8)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker="o",color="w",mfc=COLORS["P2_NoExploration"],mec=COLORS["P2_NoExploration"],label="50-seed ext sig"),
                    Line2D([],[],marker="o",color="w",mfc="white",mec="gray",label="30-seed confirmatory")],
          fontsize=6,loc="lower right")
ax.set_yticks(range(len(DATASETS))); ax.set_yticklabels(DATASETS)
ax.set_xlabel("NoExplo - Full ACC @10%")
fig.text(.02,.02,"Paired seeds; n=30 confirmatory / n=50 extension.",fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(FINAL,"fig_forest.pdf")); plt.savefig(os.path.join(FINAL,"fig_forest.png"),dpi=300)
print("forest done")
