# -*- coding: utf-8 -*-
"""add2e_layered_fisher.py — Per-dataset sign tests + Fisher's combined p.
Also reports actual seed counts for S4 letter/segment."""
import os, sys, glob
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, r"D:\AL_FuzzyClustering_Paper\v2_unified\code")
import additional_common as ac

SHARDS=os.path.join(ac.HERE,"results","shards")
frames=[]
for fp in glob.glob(os.path.join(SHARDS,"*.csv")):
    try: df=pd.read_csv(fp)
    except: continue
    if "budget" not in df.columns: continue
    frames.append(df)
big=pd.concat(frames,ignore_index=True)
big=big[(np.isclose(big.alpha0,50.0))&(np.isclose(big.lam_mult,1.0))]
sub=big[np.isclose(big.budget,0.10)]
piv=sub.pivot_table(index=["dataset","seed"],columns="method",values="ACC",aggfunc="mean").dropna(subset=["P2_Full","P2_NoExploration"])
piv["delta"]=piv.P2_NoExploration-piv.P2_Full

print("Per-dataset sign test (one-sided H1: NoExplo>Full):")
pvals=[]
rows=[]
for ds,g in piv.reset_index().groupby("dataset"):
    w=int((g.delta>0).sum()); l=int((g.delta<0).sum()); t=int((g.delta==0).sum())
    p=float(stats.binomtest(w,w+l,0.5,alternative="greater").pvalue) if (w+l)>0 else 1.0
    pvals.append(p)
    rows.append((ds,w,l,t,len(g),g.delta.mean(),p))
    print(f"  {ds:14s} wins {w:2d} loss {l:2d} tie {t:2d}  n={len(g):2d}  meanΔ={g.delta.mean():+.4f}  p={p:.4g}")

# Fisher's combined
chi=-2*sum(np.log(max(p,1e-12)) for p in pvals)
df=2*len(pvals)
p_fisher=float(stats.chi2.sf(chi,df))
print(f"\nFisher's combined: chi2={chi:.3f} df={df}  p={p_fisher:.4g}")
# also two-sided overall
w=sum(r[1] for r in rows); l=sum(r[2] for r in rows)
print(f"Pooled: {w} wins {l} losses, two-sided binomial p={float(stats.binomtest(w,w+l,0.5).pvalue):.4g}")

# S4 seed counts
print("\nS4 paired CSV seed counts:")
p4=pd.read_csv(os.path.join(ac.RES_ADD,"add4_large_paired.csv"))
print(p4.groupby("dataset").size() if "dataset" in p4.columns else p4.columns.tolist())
