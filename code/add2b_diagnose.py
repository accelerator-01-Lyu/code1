# -*- coding: utf-8 -*-
"""add2b_diagnose.py — Diagnose where CoreSet advantage comes from; rerun
Friedman/Nemenyi on ACC/NMI/ARI and on overlapping vs well-separated groups.
READ-ONLY over shards; no fabrication. All group splits are post-hoc/exploratory.
"""
import os, sys, glob
import numpy as np
import pandas as pd
sys.path.insert(0, r"D:\AL_FuzzyClustering_Paper\v2_unified\code")
import additional_common as ac

SHARDS = os.path.join(ac.HERE, "results", "shards")
frames = []
for fp in glob.glob(os.path.join(SHARDS, "*.csv")):
    try:
        df = pd.read_csv(fp)
    except Exception:
        continue
    if "budget" not in df.columns: continue
    sub = df[np.isclose(df["budget"], 0.10)]
    if len(sub): frames.append(sub)
big = pd.concat(frames, ignore_index=True)
big = big[(np.isclose(big["alpha0"], 50.0)) & (np.isclose(big["lam_mult"], 1.0))]

RANKED = ac.RANKED_METHODS  # 8 methods, OriginalP2 already excluded
datasets = sorted(big["dataset"].unique())

# mean score per (dataset, method) for ACC/NMI/ARI
def mat(metric):
    M = np.zeros((len(datasets), len(RANKED)))
    for i, ds in enumerate(datasets):
        for j, m in enumerate(RANKED):
            v = big[(big.dataset==ds)&(big.method==m)][metric].values
            M[i,j] = v.mean()
    return M

print("="*78)
print("PER-DATASET MEAN RANK (1=best) at 10%, ACC | higher is better")
print("="*78)
Macc = mat("ACC")
# rank within each dataset
from scipy import stats as st
ranks_acc = np.apply_along_axis(lambda r: st.rankdata(-r), 1, Macc)
rk = pd.DataFrame(ranks_acc, index=datasets, columns=RANKED)
print(rk.round(2).to_string())
print()
# which datasets is CoreSet best?
print("ACC: dataset winner (rank 1) and NoExploration rank:")
for i,ds in enumerate(datasets):
    order = sorted(range(len(RANKED)), key=lambda j: ranks_acc[i,j])
    win = RANKED[order[0]]
    ne_rank = list(RANKED).index("P2_NoExploration")
    fu_rank = list(RANKED).index("P2_Full")
    cs_rank = list(RANKED).index("CoreSet")
    print(f"  {ds:14s} winner={win:16s} CoreSet_rank={ranks_acc[i,cs_rank]:.0f}  "
          f"NoExplo_rank={ranks_acc[i,ne_rank]:.0f}  Full_rank={ranks_acc[i,fu_rank]:.0f}")

print()
print("="*78)
print("FRIEDMAN over ALL 6 datasets, 8 methods (OriginalP2 excluded)")
print("="*78)
for metric in ["ACC","NMI","ARI"]:
    r = ac.friedman_nemenyi(mat(metric), RANKED)
    print(f"{metric:4s}: chi2={r['chi2']:.3f}  p={r['p']:.4g}  CD={r['CD']:.2f}")
    print("      mean rank:", {m: round(r['avg_rank'][m],2) for m in RANKED})

# groups
overlap = ["glass","ecoli","compound"]
wellsep = ["iris","wine","aggregation"]
print()
print("="*78)
print("OVERLAPPING group:", overlap)
print("="*78)
for metric in ["ACC","NMI","ARI"]:
    M = mat(metric)
    idx = [datasets.index(d) for d in overlap]
    r = ac.friedman_nemenyi(M[idx,:], RANKED)
    print(f"{metric:4s}: chi2={r['chi2']:.3f}  p={r['p']:.4g}  CD={r['CD']:.2f}")
    print("      mean rank:", {m: round(r['avg_rank'][m],2) for m in RANKED})

print()
print("="*78)
print("WELL-SEPARATED group:", wellsep)
print("="*78)
for metric in ["ACC","NMI","ARI"]:
    M = mat(metric)
    idx = [datasets.index(d) for d in wellsep]
    r = ac.friedman_nemenyi(M[idx,:], RANKED)
    print(f"{metric:4s}: chi2={r['chi2']:.3f}  p={r['p']:.4g}  CD={r['CD']:.2f}")
    print("      mean rank:", {m: round(r['avg_rank'][m],2) for m in RANKED})
