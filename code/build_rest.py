"""build_rest.py — remaining tables + figs."""
import os, glob, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figures.palette import COLORS, INK

CODE=os.path.dirname(os.path.abspath(__file__))
RES=os.path.join(CODE,"results"); TAB=os.path.join(RES,"tables_final")
FIG=os.path.join(CODE,"figures","final"); os.makedirs(TAB,exist_ok=True); os.makedirs(FIG,exist_ok=True)
DATASETS=["iris","wine","glass","ecoli","aggregation","compound"]
METHODS=["Random","Entropy","CoreSet","BADGE","OriginalP2",
         "P2_Full","P2_NoExploration","P2_NoRobust","P2_NoRedundancy"]
frames=[pd.read_csv(f) for f in glob.glob(os.path.join(RES,"shards","*.csv")) if "diagnostics" not in f]
long=pd.concat(frames,ignore_index=True)

def pair(ss,ds,m1="P2_NoExploration",m2="P2_Full",metric="ACC",bud=.10):
    sub=long[long.seed_set.astype(str)==ss]
    a=sub[(sub.dataset==ds)&(sub.method==m1)&(sub.budget==bud)].sort_values("seed")
    b=sub[(sub.dataset==ds)&(sub.method==m2)&(sub.budget==bud)].sort_values("seed")
    m=a.merge(b,on="seed",suffixes=("_a","_b"))
    d=m[metric+"_a"].values-m[metric+"_b"].values
    n=len(d); se=d.std(ddof=1)/np.sqrt(n); tc=stats.t.ppf(.975,n-1)
    return dict(n=n,mean=d.mean(),lo=d.mean()-tc*se,hi=d.mean()+tc*se,
                dz=d.mean()/d.std(ddof=1),p=2*(1-stats.t.cdf(abs(d.mean()/se),n-1)))

# forest_contrasts
rows=[]
for ss in ["30","50"]:
    for ds in DATASETS:
        for bud in [.05,.10,.15,.20]:
            r=pair(ss,ds,bud=bud); r.update(seed_set=ss,dataset=ds,budget=bud)
            rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(TAB,"table_forest_contrasts.csv"),index=False)

# ecoli focus
rows=[]
for ss in ["30","50"]:
    for bud in [.05,.10,.15,.20]:
        r=pair(ss,"ecoli",bud=bud); r.update(seed_set=ss,budget=bud); rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(TAB,"table_ecoli_focus_30_50.csv"),index=False)
print("contrasts done")

# p2c posthoc
p2c=pd.read_csv(os.path.join(RES,"p2c_summary.csv"))
p2c.to_csv(os.path.join(TAB,"table_p2c_posthoc.csv"),index=False)
print("p2c done")

# ablation bars
fig,ax=plt.subplots(figsize=(7.1,2.6))
x=np.arange(len(DATASETS)); w=.13
for k,m in enumerate(["P2_Full","P2_NoExploration","P2_NoRobust","P2_NoRedundancy","OriginalP2"]):
    means=[]; los=[]
    for ds in DATASETS:
        v=long[(long.dataset==ds)&(long.method==m)&(long.budget==.10)&(long.seed_set.astype(str)=="30")].ACC.values
        mm=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); tc=stats.t.ppf(.975,len(v)-1)
        means.append(mm); los.append(tc*se)
    ax.bar(x+(k-2)*w,means,w,yerr=los,color=COLORS[m],label=m,capsize=2)
ax.set_xticks(x); ax.set_xticklabels(DATASETS); ax.legend(fontsize=6,ncol=3)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_ablation_bars.pdf")); plt.savefig(os.path.join(FIG,"fig_ablation_bars.png"),dpi=300)
print("ablation done")
