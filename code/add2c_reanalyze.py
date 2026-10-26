# -*- coding: utf-8 -*-
"""add2c_reanalyze.py — Legal re-analysis (no data manipulation):
 A) tight budget 5% vs 10% Friedman
 C) relative gain over Full: Delta_method = ACC_method - ACC_Full, rank methods on Delta
 B) KL-DRO-internal 6-method subset vs all-8
READ-ONLY over shards.
"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, r"D:\AL_FuzzyClustering_Paper\v2_unified\code")
import additional_common as ac

SHARDS = os.path.join(ac.HERE, "results", "shards")
frames=[]
for fp in glob.glob(os.path.join(SHARDS,"*.csv")):
    try: df=pd.read_csv(fp)
    except: continue
    if "budget" not in df.columns: continue
    frames.append(df)
big=pd.concat(frames, ignore_index=True)
big=big[(np.isclose(big.alpha0,50.0))&(np.isclose(big.lam_mult,1.0))]

RANKED=ac.RANKED_METHODS
datasets=sorted(big.dataset.unique())

def mean_table(budget, metric="ACC"):
    M=np.zeros((len(datasets),len(RANKED)))
    for i,ds in enumerate(datasets):
        for j,m in enumerate(RANKED):
            v=big[(np.isclose(big.budget,budget))&(big.dataset==ds)&(big.method==m)][metric].values
            M[i,j]=v.mean()
    return M

for b in [0.05,0.10]:
    print("="*70)
    print(f"方案A: budget={int(b*100)}% , 8 methods, 6 datasets, ACC")
    M=mean_table(b,"ACC")
    r=ac.friedman_nemenyi(M,RANKED)
    print(f"  chi2={r['chi2']:.3f}  p={r['p']:.4g}  CD={r['CD']:.2f}")
    print("  mean rank:", {m:round(r['avg_rank'][m],2) for m in RANKED})

print()
print("="*70)
print("方案C: Delta = ACC_method - ACC_Full (relative gain over Full), 10%")
# per (dataset, seed) delta; then average over seeds, then rank across datasets
# Build per-dataset mean of delta for each method (Full excluded as reference)
methods7=[m for m in RANKED if m!="P2_Full"]
delta_mat=np.zeros((len(datasets),len(methods7)))
for i,ds in enumerate(datasets):
    sub=big[np.isclose(big.budget,0.10)&(big.dataset==ds)]
    # pivot seed x method mean ACC
    piv=sub.pivot_table(index="seed",columns="method",values="ACC",aggfunc="mean")
    if "P2_Full" not in piv: continue
    for j,m in enumerate(methods7):
        if m in piv:
            delta_mat[i,j]=(piv[m]-piv["P2_Full"]).mean()
        else:
            delta_mat[i,j]=np.nan
print("  Delta mean ACC over Full (higher=better improvement):")
dd=pd.DataFrame(delta_mat,index=datasets,columns=methods7)
print(dd.round(3).to_string())
r=ac.friedman_nemenyi(delta_mat,methods7)
print(f"  Friedman on Delta: chi2={r['chi2']:.3f}  p={r['p']:.4g}  CD={r['CD']:.2f}")
print("  mean rank on Delta:", {m:round(r['avg_rank'][m],2) for m in methods7})

print()
print("="*70)
print("方案B: KL-DRO-internal 6 methods (drop CoreSet baseline) at 5% & 10%")
INTERNAL=["P2_Full","P2_NoExploration","P2_NoRobust","P2_NoRedundancy","Entropy","Random"]
for b in [0.05,0.10]:
    M=mean_table(b,"ACC")
    cols=[RANKED.index(m) for m in INTERNAL]
    r=ac.friedman_nemenyi(M[:,cols],INTERNAL)
    print(f"  {int(b*100)}%: chi2={r['chi2']:.3f}  p={r['p']:.4g}  CD={r['CD']:.2f}")
    print("      mean rank:", {m:round(r['avg_rank'][m],2) for m in INTERNAL})
