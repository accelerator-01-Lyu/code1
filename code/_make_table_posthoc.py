# POST-HOC MECHANISM VALIDATION — not part of pre-registered P1-P4 verdicts.
"""Final supplementary table: 3 geometries x 3 sigma x 10 seeds + swissroll
budget-tightness control.  Adds 'budget' column; original 9 rows = 0.10.
"""
import os, pickle
import numpy as np
import pandas as pd
from scipy import stats
HERE=os.path.dirname(os.path.abspath(__file__)); SYN=os.path.join(HERE,"results","synthetic_3d")
summ=pd.read_csv(os.path.join(SYN,"summary_all.csv"))
traj=pd.read_csv(os.path.join(SYN,"trajectory_all.csv"))
bud=pd.read_csv(os.path.join(SYN,"summary_swissroll_budget.csv"))
bud_t=pd.read_csv(os.path.join(SYN,"trajectory_swissroll_budget.csv"))

GEO={"spiral":[0.4,0.6,0.8],"swissroll":[0.4,0.6,0.8],"hourglass":[0.3,0.4,0.6]}
rows=[]
# main 3-geometry rows at budget=0.10
for geo,sigmas in GEO.items():
    for s in sigmas:
        g=summ[(summ.dataset==geo)&np.isclose(summ.sigma,s)]
        rec={"geometry":geo,"sigma":s,"budget":0.10}
        for meth in ["P2_Full","P2_NoExploration"]:
            mm=g[g.method==meth]
            for ind in ["acc","nmi","ari"]:
                rec[f"{meth}_{ind}_mean"]=mm[ind].mean()
                rec[f"{meth}_{ind}_sd"]=mm[ind].std(ddof=1)
        for ind in ["acc","nmi","ari"]:
            f=g[g.method=="P2_Full"].sort_values("seed"); n=g[g.method=="P2_NoExploration"].sort_values("seed")
            m=f.merge(n,on="seed",suffixes=("_f","_n")); d=m[f"{ind}_n"].values-m[f"{ind}_f"].values
            h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
            _,p=stats.ttest_rel(m[f"{ind}_n"].values,m[f"{ind}_f"].values)
            rec[f"gap_{ind}_mean"]=d.mean(); rec[f"gap_{ind}_ci_lo"]=d.mean()-h
            rec[f"gap_{ind}_ci_hi"]=d.mean()+h; rec[f"gap_{ind}_p"]=p
        pk=pickle.load(open(os.path.join(SYN,f"{geo}_sigma{s}.pkl"),"rb")); y=pk["y"]; nn=int((y>=0).sum())
        sub=traj[(traj.dataset==geo)&np.isclose(traj.sigma,s)]
        for meth in ["P2_Full","P2_NoExploration"]:
            sel=sub[sub.method==meth].selected_idx.values
            nd=int((sel>=nn).sum()); rec[f"{meth}_decoy_hits"]=nd
            rec[f"{meth}_decoy_rate"]=nd/len(sel)
        rec["note"]=("NEGATIVE control: cone-top rings symmetric about apex; "
                     "FCM m=2 shares membership, no local single-cluster target"
                     if geo=="hourglass" else "")
        rows.append(rec)

# swissroll budget rows 0.02 / 0.03
for s in [0.4,0.6,0.8]:
    for bg in [0.02,0.03]:
        g=bud[np.isclose(bud.sigma,s)&np.isclose(bud.budget,bg)]
        rec={"geometry":"swissroll","sigma":s,"budget":bg}
        for meth in ["P2_Full","P2_NoExploration"]:
            mm=g[g.method==meth]
            for ind in ["acc","nmi","ari"]:
                rec[f"{meth}_{ind}_mean"]=mm[ind].mean(); rec[f"{meth}_{ind}_sd"]=mm[ind].std(ddof=1)
        for ind in ["acc","nmi","ari"]:
            f=g[g.method=="P2_Full"].sort_values("seed"); n=g[g.method=="P2_NoExploration"].sort_values("seed")
            m=f.merge(n,on="seed",suffixes=("_f","_n")); d=m[f"{ind}_n"].values-m[f"{ind}_f"].values
            h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
            _,p=stats.ttest_rel(m[f"{ind}_n"].values,m[f"{ind}_f"].values)
            rec[f"gap_{ind}_mean"]=d.mean(); rec[f"gap_{ind}_ci_lo"]=d.mean()-h
            rec[f"gap_{ind}_ci_hi"]=d.mean()+h; rec[f"gap_{ind}_p"]=p
        rec["note"]=("L0-only, zero active selections; two methods identical by "
                     "construction" if abs(bg-0.02)<1e-9 else "")
        rows.append(rec)

df=pd.DataFrame(rows)
out=os.path.join(SYN,"table_synthetic3d_posthoc.csv")
df.to_csv(out,index=False)
print("wrote",out,"rows",len(df))
