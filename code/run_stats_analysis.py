"""run_stats_analysis.py - E4 paired bootstrap + E2a REML mixed-effects.

1) E4: queried vs random mean_s, paired per (overlap, seed, method),
   stratified bootstrap 95% CI of the intensity gap per overlap level.
2) E2a: ACC_pp ~ C(rho)*C(beta_mult) + (1|instance_id), REML.
   LR tests via ML fits: rho main effect, beta main effect, interaction.
   Linear trend contrast for rho (as continuous 0/.05/.1/.25/.5).
Outputs to results/stats/e4_intensity_gap.csv and e2a_mixed_model.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats as st

RES = r"D:\AL_FuzzyClustering_Paper\v2_unified\results"
THEO = os.path.join(RES, "theorem2")
STAT = os.path.join(RES, "stats")
os.makedirs(STAT, exist_ok=True)
OUT = r"D:\tmp\stats_analysis"
os.makedirs(OUT, exist_ok=True)

LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
B = 10000
RNG = np.random.default_rng(20261023)

# ---------------- 1) E4 paired bootstrap ----------------
dg = pd.read_csv(os.path.join(THEO, "c2_record", "diagnostic.csv"), encoding="utf-8")
rb = pd.read_csv(os.path.join(THEO, "random_baseline.csv"), encoding="utf-8")
dg = dg.drop_duplicates(subset=["overlap_level", "seed", "method", "point_idx"])

q = dg.groupby(["overlap_level", "seed", "method"])["s_i"].mean().rename("mean_q")
r = rb.groupby(["overlap_level", "seed", "method"])["s_i"].mean().rename("mean_r")
m = pd.concat([q, r], axis=1).reset_index()
m["gap"] = m["mean_q"] - m["mean_r"]     # negative gap => queried more negative

rows = []
for lv in LEVELS:
    g = m[m.overlap_level == lv]["gap"].values
    n = len(g)
    obs = float(g.mean())
    idx = RNG.integers(0, n, size=(B, n))
    bd = g[idx].mean(1)
    rows.append({"overlap": lv, "n_runs": n,
                 "gap_mean": obs, "gap_lo": float(np.quantile(bd, 0.025)),
                 "gap_hi": float(np.quantile(bd, 0.975))})
    # paired t-test on gaps as a quick check
    t, p = st.ttest_1samp(g, 0.0)
    rows[-1]["ttest_p"] = float(p)
gap = pd.DataFrame(rows)
gap.to_csv(os.path.join(STAT, "e4_intensity_gap.csv"), index=False, encoding="utf-8")

# pooled gap across all runs
g_all = m["gap"].values
obs_all = float(g_all.mean())
idx = RNG.integers(0, len(g_all), size=(B, len(g_all)))
bd_all = g_all[idx].mean(1)
pool = {"overlap": "pooled", "n_runs": int(len(g_all)), "gap_mean": obs_all,
        "gap_lo": float(np.quantile(bd_all, 0.025)),
        "gap_hi": float(np.quantile(bd_all, 0.975)),
        "ttest_p": float(st.ttest_1samp(g_all, 0.0).pvalue)}
gap = pd.concat([gap, pd.DataFrame([pool])], ignore_index=True)
gap.to_csv(os.path.join(STAT, "e4_intensity_gap.csv"), index=False, encoding="utf-8")

print("=== E4 intensity gap (queried - random mean_s), 95% CI ===")
print(gap.round(4).to_string(index=False))

# ---------------- 2) E2a REML mixed-effects ----------------
rb2 = pd.read_csv(os.path.join(RES, "factor", "rho_beta.csv"), encoding="utf-8")
rb2["ACC_pp"] = rb2["ACC"] * 100
rb2["iid"] = rb2["instance_id"].astype(str)

import statsmodels.api as sm
import statsmodels.formula.api as smf

# full model (REML fit for reporting)
full_reml = smf.mixedlm("ACC_pp ~ C(rho)*C(beta_mult)", rb2,
                        groups=rb2["iid"]).fit(reml=True)

# LR tests need ML fits
full_ml = smf.mixedlm("ACC_pp ~ C(rho)*C(beta_mult)", rb2,
                      groups=rb2["iid"]).fit(reml=False)
no_int = smf.mixedlm("ACC_pp ~ C(rho)+C(beta_mult)", rb2,
                     groups=rb2["iid"]).fit(reml=False)
no_rho = smf.mixedlm("ACC_pp ~ C(beta_mult)", rb2,
                     groups=rb2["iid"]).fit(reml=False)
no_beta = smf.mixedlm("ACC_pp ~ C(rho)", rb2,
                      groups=rb2["iid"]).fit(reml=False)
null_m = smf.mixedlm("ACC_pp ~ 1", rb2, groups=rb2["iid"]).fit(reml=False)


def lr_test(ll_full, ll_red, df):
    stat = 2.0 * (ll_full - ll_red)
    p = float(st.chi2.sf(stat, df))
    return stat, p


rows = []
lr_rho = lr_test(full_ml.llf, no_rho.llf, df=4)
lr_beta = lr_test(full_ml.llf, no_beta.llf, df=4)
lr_int = lr_test(full_ml.llf, no_int.llf, df=16)
lr_full_vs_null = lr_test(full_ml.llf, null_m.llf, df=25)
rows.append({"term": "rho main", "LR": lr_rho[0], "df": 4, "p": lr_rho[1]})
rows.append({"term": "beta main", "LR": lr_beta[0], "df": 4, "p": lr_beta[1]})
rows.append({"term": "rho x beta", "LR": lr_int[0], "df": 16, "p": lr_int[1]})
rows.append({"term": "full vs null", "LR": lr_full_vs_null[0], "df": 25,
             "p": lr_full_vs_null[1]})

# rho linear trend: rho as continuous covariate
rb2["rho_c"] = rb2["rho"].astype(float)
trend = smf.mixedlm("ACC_pp ~ rho_c + C(beta_mult)", rb2,
                    groups=rb2["iid"]).fit(reml=True)
rows.append({"term": "rho linear trend",
             "LR": float(trend.params["rho_c"]),
             "df": 0, "p": float(trend.pvalues["rho_c"])})

me = pd.DataFrame(rows)
me.to_csv(os.path.join(STAT, "e2a_mixed_model.csv"), index=False, encoding="utf-8")

# partial eta^2 approximations from F (use wald_test_terms on REML fit)
try:
    wt = full_reml.wald_test_terms()
    w_rows = []
    for name in ["C(rho)", "C(beta_mult)", "C(rho):C(beta_mult)"]:
        w = wt.table.loc[name]
        Fv = float(w["F"])
        pv = float(w["P>|F|"])
        df1 = float(w["df denom"]) if "df denom" in w else np.nan
        w_rows.append({"term": name, "F": Fv, "p": pv, "df_denom": df1})
    wt_df = pd.DataFrame(w_rows)
    wt_df.to_csv(os.path.join(STAT, "e2a_wald_terms.csv"), index=False,
                 encoding="utf-8")
    print("\n=== E2a Wald terms (REML) ===")
    print(wt_df.round(4).to_string(index=False))
except Exception as e:
    print("[wald terms failed]", e)

print("\n=== E2a LR tests (ML) ===")
print(me.round(4).to_string(index=False))

# random-effect variance vs residual (ICC)
vc = full_reml.cov_re.iloc[0, 0]
vr = full_reml.scale
print("\n=== E2a variance components ===")
print(f"between-instance var={vc:.4f} residual var={vr:.4f} "
      f"ICC={vc/(vc+vr):.4f}")

# singular fit warning check
print("\n[singular] full_reml converged:", full_reml.converged,
      "message:", getattr(full_reml, "message", "n/a")[:120])
