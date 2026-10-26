import pandas as pd, numpy as np
from scipy import stats
s = pd.read_csv(r"results\synthetic_3d\summary_all.csv")
b = pd.read_csv(r"results\synthetic_3d\summary_swissroll_budget.csv")

# attach 10% from main summary (dataset=swissroll)
sw = s[s.dataset == "swissroll"].copy()
ten = sw[["sigma","method","seed","acc","nmi","ari"]].copy()
ten["budget"] = 0.10
b2 = pd.concat([b[["sigma","method","seed","acc","nmi","ari","budget"]], ten],
               ignore_index=True)

print("=== swissroll paired gap ACC(NoExplo-Full), by sigma x budget ===")
for sg in [0.4,0.6,0.8]:
    for bg in [0.02,0.03,0.10]:
        g = b2[(np.isclose(b2.sigma,sg))&(np.isclose(b2.budget,bg))]
        f = g[g.method=="P2_Full"].sort_values("seed")
        n = g[g.method=="P2_NoExploration"].sort_values("seed")
        m = f.merge(n,on="seed",suffixes=("_f","_n"))
        d = m.acc_n.values-m.acc_f.values
        h = stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
        print(f"sigma={sg} budget={bg:.2f}: gap={d.mean()*100:+.2f}pp "
              f"[{(d.mean()-h)*100:+.2f},{(d.mean()+h)*100:+.2f}] "
              f"n={len(d)}")
