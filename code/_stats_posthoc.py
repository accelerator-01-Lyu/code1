import pandas as pd, numpy as np
from scipy import stats
s = pd.read_csv(r"results\synthetic_3d\summary_all.csv")
print("rows", len(s), "NaN", int(s.acc.isna().sum()))
print()
print("=== paired gap ACC(NoExplo)-ACC(Full), n=10, 95%CI ===")
for (k, sg), g in s.groupby(["dataset", "sigma"]):
    f = g[g.method == "P2_Full"].sort_values("seed")
    n = g[g.method == "P2_NoExploration"].sort_values("seed")
    m = f.merge(n, on="seed", suffixes=("_f", "_n"))
    d = m.acc_n.values - m.acc_f.values
    h = stats.t.ppf(.975, len(d)-1) * d.std(ddof=1) / np.sqrt(len(d))
    print(f"{k:10s} s={sg:<4} gap={d.mean():+.3f}  "
          f"[{d.mean()-h:+.3f},{d.mean()+h:+.3f}]  "
          f"Full={f.acc.mean():.3f} NoEx={n.acc.mean():.3f}")

# decoy selection frequency: y==-1 indices are the LAST 10 points appended
# load a pkl to get count of normal points = N-10
import pickle
tr = pd.read_csv(r"results\synthetic_3d\trajectory_all.csv")
print()
print("=== decoy (y=-1) selection frequency ===")
for kind, sg in [("spiral",0.6),("swissroll",0.6),("hourglass",0.4)]:
    d = tr[(tr.dataset==kind)&(tr.sigma==sg)]
    with open(f"results/synthetic_3d/{kind}_sigma{sg}.pkl","rb") as f:
        pk = pickle.load(f); y = pk["y"] if isinstance(pk,dict) else pk[1]
    n_normal = int((y>=0).sum())
    for method in ["P2_Full","P2_NoExploration"]:
        dm = d[d.method==method]
        sel = dm.selected_idx.values
        ndec = int((sel>=n_normal).sum())
        print(f"{kind} s={sg} {method:18s} decoy selections={ndec}/{len(sel)} "
              f"({ndec/len(sel)*100:.1f}%)")
