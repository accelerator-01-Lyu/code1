"""build_tables_final.py — one-shot table pack."""
import os, glob, numpy as np, pandas as pd
from scipy import stats

CODE=os.path.dirname(os.path.abspath(__file__))
RES=os.path.join(CODE,"results"); TAB=os.path.join(RES,"tables_final")
os.makedirs(TAB,exist_ok=True)
DATASETS=["iris","wine","glass","ecoli","aggregation","compound"]
METHODS=["Random","Entropy","CoreSet","BADGE","OriginalP2",
         "P2_Full","P2_NoExploration","P2_NoRobust","P2_NoRedundancy"]
frames=[pd.read_csv(f) for f in glob.glob(os.path.join(RES,"shards","*.csv")) if "diagnostics" not in f]
long=pd.concat(frames,ignore_index=True)

def cell(v):
    n=len(v); m=v.mean(); sd=v.std(ddof=1); se=sd/np.sqrt(n); tc=stats.t.ppf(.975,n-1)
    rng=np.random.RandomState(0); boot=[v[rng.randint(0,n,n)].mean() for _ in range(1000)]
    lo_b,hi_b=np.percentile(boot,[2.5,97.5])
    return f"{m:.3f}±{sd:.3f} [{m-tc*se:.3f},{m+tc*se:.3f};{lo_b:.3f},{hi_b:.3f}]"

for ss in ["30","50"]:
    sub=long[long.seed_set.astype(str)==ss]
    rows=[]
    for ds in DATASETS:
        row={"dataset":ds}
        for m in METHODS:
            v=sub[(sub.dataset==ds)&(sub.method==m)&(sub.budget==.10)].ACC.values
            row[m]=cell(v) if len(v)>=5 else "NA"
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(TAB,f"table_main_ACC_{ss}.csv"),index=False)
print("tables done")
