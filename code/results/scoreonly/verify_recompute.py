"""Independent recomputation of P2_ScoreOnly statistics from RAW shard CSVs.

Reads:
  - new ScoreOnly shards:  code/results/scoreonly/{ds}_P2_ScoreOnly_30_{seed}_...csv
  - existing Table-3 shards: code/results/shards/{ds}_{P2_NoExploration,P2_Full}_30_{seed}_...csv
Computes ACC@0.10 mean±SD and paired contrasts (pp / t-CI / p / d_z) from scratch.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

SHARDS = r"D:\AL_FuzzyClustering_Paper\v2_unified\code\results\shards"
SO = r"D:\AL_FuzzyClustering_Paper\v2_unified\code\results\scoreonly"
SEEDS = list(range(30))


def load_acc(ds, method, base):
    vals = {}
    for s in SEEDS:
        if method == "P2_ScoreOnly":
            p = os.path.join(base, f"{ds}_P2_ScoreOnly_30_{s}_0_0.0_50.0_1.0.csv")
        else:
            p = os.path.join(base, f"{ds}_{method}_30_{s}_0_0.0_50.0_1.0.csv")
        if not os.path.exists(p):
            return None, f"missing {p}"
        df = pd.read_csv(p)
        row = df[df.budget == 0.10]
        if len(row) != 1:
            return None, f"bad budget rows in {p}"
        vals[s] = float(row.ACC.iloc[0])
    return vals, None


def paired(a, b):
    d = np.array([a[s] - b[s] for s in SEEDS])
    n = len(d)
    mean = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    t, p = stats.ttest_rel(d, np.zeros(n))
    return {"pp": mean * 100, "lo": (mean - tcrit * se) * 100,
            "hi": (mean + tcrit * se) * 100, "p": float(p),
            "dz": mean / sd, "n": n}


def cell(v):
    arr = np.array(list(v.values()))
    m, sd = arr.mean(), arr.std(ddof=1)
    se = sd / np.sqrt(len(arr))
    tc = stats.t.ppf(0.975, len(arr) - 1)
    rng = np.random.RandomState(0)
    boot = [arr[rng.randint(0, len(arr), len(arr))].mean() for _ in range(1000)]
    return f"{m:.3f}±{sd:.3f} [{m - tc*se:.3f},{m + tc*se:.3f};{np.percentile(boot,2.5):.3f},{np.percentile(boot,97.5):.3f}]"


for ds in ("ecoli", "glass"):
    so, e1 = load_acc(ds, "P2_ScoreOnly", SO)
    ne, e2 = load_acc(ds, "P2_NoExploration", SHARDS)
    fu, e3 = load_acc(ds, "P2_Full", SHARDS)
    assert not any([e1, e2, e3]), (e1, e2, e3)
    print(f"== {ds} ==")
    print("cell: ScoreOnly", cell(so))
    print("cell: NoExpl   ", cell(ne))
    print("cell: Full     ", cell(fu))
    for a, b, nm in ((so, ne, "ScoreOnly-NoExpl"), (so, fu, "ScoreOnly-Full")):
        r = paired(a, b)
        print(nm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
