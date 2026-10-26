"""snapshot_analysis.py - quick results for E4 baseline contrast and E2a factor overview.

E4: queried (c2_record/diagnostic.csv) vs random (random_baseline.csv) s_i<0 fraction
    and mean_s per overlap level + per-run paired contrast.
E2a: rho x beta marginals (ACC) from rho_beta.csv; check rho=0.1 default position.
"""
import os
import numpy as np
import pandas as pd

RES = r"D:\AL_FuzzyClustering_Paper\v2_unified\results"
THEO = os.path.join(RES, "theorem2")
FACT = os.path.join(RES, "factor")
OUT = r"D:\tmp\results_snapshot"

os.makedirs(OUT, exist_ok=True)

LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]

# ---------------- E4 baseline contrast ----------------
dg = pd.read_csv(os.path.join(THEO, "c2_record", "diagnostic.csv"), encoding="utf-8")
rb = pd.read_csv(os.path.join(THEO, "random_baseline.csv"), encoding="utf-8")
dg = dg.drop_duplicates(subset=["overlap_level", "seed", "method", "point_idx"])

q = dg.groupby(["overlap_level", "seed", "method"]).agg(
    n=("s_i", "size"), frac_q=("s_i", lambda s: float((s < 0).mean())),
    mean_q=("s_i", "mean")).reset_index()
r = rb.groupby(["overlap_level", "seed", "method"]).agg(
    n=("s_i", "size"), frac_r=("s_i", lambda s: float((s < 0).mean())),
    mean_r=("s_i", "mean")).reset_index()
m = q.merge(r, on=["overlap_level", "seed", "method"], suffixes=("_q", "_r"))
m["d_frac"] = m["frac_q"] - m["frac_r"]
m["d_mean"] = m["mean_q"] - m["mean_r"]

contrast = []
for lv in LEVELS:
    sub = m[m.overlap_level == lv]
    contrast.append({
        "overlap": lv,
        "queried_frac": float(sub["frac_q"].mean()),
        "random_frac": float(sub["frac_r"].mean()),
        "d_frac_pp": float(sub["d_frac"].mean() * 100),
        "queried_mean_s": float(sub["mean_q"].mean()),
        "random_mean_s": float(sub["mean_r"].mean()),
        "n_runs": int(len(sub)),
    })
cc = pd.DataFrame(contrast)
cc.to_csv(os.path.join(OUT, "e4_baseline_contrast.csv"), index=False, encoding="utf-8")

# pooled one-sample: is queried - random > 0 across runs?
d_all = m["d_frac"].values
t = d_all.mean() / (d_all.std(ddof=1) + 1e-12) / np.sqrt(len(d_all)) * np.sqrt(len(d_all))
pooled = {
    "pooled_queried_frac": float(m["frac_q"].mean()),
    "pooled_random_frac": float(m["frac_r"].mean()),
    "pooled_d_frac_pp": float(m["d_frac"].mean() * 100),
    "runs": int(len(m)),
}
pd.DataFrame([pooled]).to_csv(os.path.join(OUT, "e4_baseline_pooled.csv"),
                              index=False, encoding="utf-8")

# ---------------- E2a factor overview ----------------
rb2 = pd.read_csv(os.path.join(FACT, "rho_beta.csv"), encoding="utf-8")
rb2["ACC_pp"] = rb2["ACC"] * 100
rho_m = rb2.groupby("rho").agg(acc_pp=("ACC_pp", "mean"),
                               n=("ACC_pp", "size")).reset_index()
beta_m = rb2.groupby("beta_mult").agg(acc_pp=("ACC_pp", "mean"),
                                      n=("ACC_pp", "size")).reset_index()
rho_m.to_csv(os.path.join(OUT, "e2a_rho_marginal.csv"), index=False, encoding="utf-8")
beta_m.to_csv(os.path.join(OUT, "e2a_beta_marginal.csv"), index=False, encoding="utf-8")

# default cell (rho=0.1, beta_mult=1.0) position vs platform
cells = rb2.groupby(["rho", "beta_mult"]).agg(acc_pp=("ACC_pp", "mean"),
                                              n=("ACC_pp", "size")).reset_index()
cells.to_csv(os.path.join(OUT, "e2a_cells.csv"), index=False, encoding="utf-8")

# quick two-way ANOVA (balanced 12 instances x 5x5 x 10)
import scipy.stats as st
aov_rows = []
for f, levels in [("rho", sorted(rb2["rho"].unique())),
                  ("beta_mult", sorted(rb2["beta_mult"].unique()))]:
    groups = [rb2[rb2[f] == v]["ACC_pp"].values for v in levels]
    F, p = st.f_oneway(*groups)
    aov_rows.append({"factor": f, "F": float(F), "p": float(p)})
pd.DataFrame(aov_rows).to_csv(os.path.join(OUT, "e2a_quick_anova.csv"),
                              index=False, encoding="utf-8")

# overlap-stratified rho marginal (for the 0.5pp platform check)
rl = rb2.groupby(["overlap", "rho"]).agg(acc_pp=("ACC_pp", "mean")).reset_index()
rl_pivot = rl.pivot(index="overlap", columns="rho", values="acc_pp")
rl_pivot.to_csv(os.path.join(OUT, "e2a_rho_by_overlap.csv"), encoding="utf-8")

print("=== E4 baseline contrast (queried vs random) ===")
print(cc.round(4).to_string(index=False))
print("\n=== E4 pooled ===")
print(pooled)
print("\n=== E2a rho marginal (ACC %) ===")
print(rho_m.round(4).to_string(index=False))
print("\n=== E2a beta marginal (ACC %) ===")
print(beta_m.round(4).to_string(index=False))
print("\n=== E2a quick one-way ANOVA ===")
print(pd.DataFrame(aov_rows).round(4).to_string(index=False))
print("\n=== E2a rho x overlap (ACC %) ===")
print(rl_pivot.round(4).to_string())
