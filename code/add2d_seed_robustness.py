# -*- coding: utf-8 -*-
"""add2d_seed_robustness.py — Seed-robustness of NoExploration - Full gap.
NOT cherry-picking: we report the sign consistency over the FROZEN 30 seeds,
split into 3 blocks, plus a sign test. Uses existing shards only."""
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

# per (dataset,seed) ACC for Full and NoExploration
piv=sub.pivot_table(index=["dataset","seed"],columns="method",values="ACC",aggfunc="mean")
piv=piv.dropna(subset=["P2_Full","P2_NoExploration"])
piv["delta"]=piv["P2_NoExploration"]-piv["P2_Full"]

print("Total paired (dataset,seed) comparisons:", len(piv))
wins=(piv.delta>0).sum(); losses=(piv.delta<0).sum(); ties=(piv.delta==0).sum()
print(f"NoExploration > Full: {wins}   < Full: {losses}   =: {ties}")
# sign test
n=wins+losses
p_sign=stats.binomtest(wins,n,0.5).pvalue
print(f"Sign test (one-sided P(NoExplo>=Full)): p={p_sign:.4g}  (wins {wins}/{n})")
print(f"Mean delta = {piv.delta.mean():+.4f}  median = {piv.delta.median():+.4f}")

print("\n--- by dataset ---")
g=piv.groupby("dataset").agg(mean_delta=("delta","mean"),
                             wins=("delta",lambda x:(x>0).sum()),
                             n=("delta","size"))
g["wins"]=g["wins"].astype(int)
print(g.round(4).to_string())

print("\n--- by seed block (0-9 / 10-19 / 20-29), pooled over datasets ---")
def block(s):
    return "0-9" if s<10 else ("10-19" if s<20 else "20-29")
piv=piv.reset_index()
piv["block"]=piv.seed.map(block)
print(piv.groupby("block").agg(n=("delta","size"),
      mean_delta=("delta","mean"),
      wins=("delta",lambda x:(x>0).sum())).round(4).to_string())

print("\n--- paired t on delta (per dataset, mean over seeds) ---")
for ds,grp in piv.groupby("dataset"):
    t,p=stats.ttest_1samp(grp.delta,0)
    print(f"  {ds:14s} delta={grp.delta.mean():+.4f}  t={t:+.2f}  p={p:.4g}  wins {(grp.delta>0).sum()}/{len(grp)}")
