"""analyze_e2c.py - rho x beta x kappa three-factor analysis (REML mixed effects)."""
import os
import numpy as np
import pandas as pd
from scipy import stats as st

RES = r"D:\AL_FuzzyClustering_Paper\v2_unified\results"
FACT = os.path.join(RES, "factor")
OUT = r"D:\tmp\e2c_out"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(FACT, "rho_beta_kappa.csv"), encoding="utf-8")
df["ACC_pp"] = df["ACC"] * 100
df["iid"] = df["instance_id"].astype(str)

# marginals
for f, fn in [("rho", "rho_marginal"), ("beta_mult", "beta_marginal"),
              ("kappa", "kappa_marginal")]:
    m = df.groupby(f).agg(acc_pp=("ACC_pp", "mean"), n=("ACC_pp", "size")).reset_index()
    m.to_csv(os.path.join(OUT, fn + ".csv"), index=False, encoding="utf-8")

# REML mixed effects: three two-way interactions (no three-way for stability)
import statsmodels.formula.api as smf

full = smf.mixedlm("ACC_pp ~ C(rho)*C(beta_mult)*C(kappa)", df,
                   groups=df["iid"]).fit(reml=True)
full_ml = smf.mixedlm("ACC_pp ~ C(rho)*C(beta_mult)*C(kappa)", df,
                      groups=df["iid"]).fit(reml=False)
no_rho = smf.mixedlm("ACC_pp ~ C(beta_mult)*C(kappa)", df,
                     groups=df["iid"]).fit(reml=False)
no_beta = smf.mixedlm("ACC_pp ~ C(rho)*C(kappa)", df,
                      groups=df["iid"]).fit(reml=False)
no_kap = smf.mixedlm("ACC_pp ~ C(rho)*C(beta_mult)", df,
                     groups=df["iid"]).fit(reml=False)
no_rb = smf.mixedlm("ACC_pp ~ C(rho)+C(beta_mult)+C(kappa)+C(rho):C(kappa)+C(beta_mult):C(kappa)",
                    df, groups=df["iid"]).fit(reml=False)
null = smf.mixedlm("ACC_pp ~ 1", df, groups=df["iid"]).fit(reml=False)

rows = []
for name, llr, llred, dfd in [
    ("rho main", full_ml.llf, no_rho.llf, 2),
    ("beta main", full_ml.llf, no_beta.llf, 2),
    ("kappa main", full_ml.llf, no_kap.llf, 2),
    ("rho x beta", full_ml.llf, no_rb.llf, 4),
    ("full vs null", full_ml.llf, null.llf, 26),
]:
    stat = 2.0 * (llr - llred)
    rows.append({"term": name, "LR": stat, "df": dfd,
                 "p": float(st.chi2.sf(stat, dfd))})
me = pd.DataFrame(rows)
me.to_csv(os.path.join(OUT, "mixed_model.csv"), index=False, encoding="utf-8")

vc = full.cov_re.iloc[0, 0]
vr = full.scale
print("=== rho marginal (ACC %) ===")
print(df.groupby("rho").agg(acc_pp=("ACC_pp", "mean")).round(4).to_string())
print("\n=== beta marginal (ACC %) ===")
print(df.groupby("beta_mult").agg(acc_pp=("ACC_pp", "mean")).round(4).to_string())
print("\n=== kappa marginal (ACC %) ===")
print(df.groupby("kappa").agg(acc_pp=("ACC_pp", "mean")).round(4).to_string())
print("\n=== mixed model LR ===")
print(me.round(4).to_string(index=False))
print("\nICC=%.4f (between %.2f, residual %.2f)" % (vc/(vc+vr), vc, vr))
