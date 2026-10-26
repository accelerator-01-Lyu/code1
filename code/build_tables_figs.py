"""Build 30-seed tables and figures from shards. Post-hoc reporting only."""
import os, numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(CODE, "results")
TAB = os.path.join(RES, "tables_30"); os.makedirs(TAB, exist_ok=True)
ASSET = r"D:\AL_FuzzyClustering_Paper\v2_unified\assets"; os.makedirs(ASSET, exist_ok=True)
DATASETS = ["iris","wine","glass","ecoli","aggregation","compound"]
METHODS = ["Random","Entropy","CoreSet","BADGE","OriginalP2",
           "P2_Full","P2_NoExploration","P2_NoRobust","P2_NoRedundancy"]
df = pd.read_csv(os.path.join(RES, "shards", "done.txt"), skiprows=0) if False else None
# load long
import glob
frames = []
for f in glob.glob(os.path.join(RES, "shards", "*.csv")):
    if "diagnostics" in f: continue
    frames.append(pd.read_csv(f))
long = pd.concat(frames, ignore_index=True)
long = long[long.seed_set.astype(str)=="30"]
print("long", long.shape)

def cell(vals):
    n=len(vals); m=vals.mean(); sd=vals.std(ddof=1)
    se=sd/np.sqrt(n); tc=stats.t.ppf(.975,n-1)
    rng=np.random.RandomState(0); boots=[vals[rng.randint(0,n,n)].mean() for _ in range(1000)]
    lo_b,hi_b=np.percentile(boots,[2.5,97.5])
    return f"{m:.3f}±{sd:.3f} [{m-tc*se:.3f},{m+tc*se:.3f}; {lo_b:.3f},{hi_b:.3f}]"

# table_V3 ACC @10%
rows=[]
for ds in DATASETS:
    row={"dataset":ds}
    for mth in METHODS:
        v=long[(long.dataset==ds)&(long.method==mth)&(long.budget==.10)].ACC.values
        row[mth]=cell(v) if len(v)>=30 else "NA"
    rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(TAB,"table_V3_ACC.csv"),index=False)
print("V3 done")

# ecoli focus
rows=[]
for bud in [.05,.10,.15,.20]:
    f=long[(long.dataset=="ecoli")&(long.method=="P2_Full")&(long.budget==bud)].sort_values("seed")
    ne=long[(long.dataset=="ecoli")&(long.method=="P2_NoExploration")&(long.budget==bud)].sort_values("seed")
    m=f.merge(ne,on="seed",suffixes=("_f","_ne"))
    d=m.ACC_f.values-m.ACC_ne.values
    n=len(d); se=d.std(ddof=1)/np.sqrt(n); tc=stats.t.ppf(.975,n-1)
    rows.append({"budget":bud,"diff":d.mean(),"t_lo":d.mean()-tc*se,"t_hi":d.mean()+tc*se,
                  "dz":d.mean()/d.std(ddof=1)})
pd.DataFrame(rows).to_csv(os.path.join(TAB,"table_ecoli_focus.csv"),index=False)
print("ecoli focus done")

# learning curves
fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
for ax,ds in zip(axes.flat,DATASETS):
    for mth in ["Random","Entropy","CoreSet","P2_Full","P2_NoExploration"]:
        means=[]; los=[]; his=[]
        for bud in [.05,.10,.15,.20]:
            v=long[(long.dataset==ds)&(long.method==mth)&(long.budget==bud)].ACC.values
            m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); tc=stats.t.ppf(.975,len(v)-1)
            means.append(m); los.append(m-tc*se); his.append(m+tc*se)
        ax.plot([5,10,15,20],means,marker="o",label=mth)
        ax.fill_between([5,10,15,20],los,his,alpha=.15)
    ax.set_title(ds); ax.set_xlabel("budget %"); ax.set_ylabel("ACC")
axes.flat[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(ASSET,"fig_learning_curves.png"),dpi=300)
print("fig done")
