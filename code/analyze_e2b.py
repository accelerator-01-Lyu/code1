"""analyze_e2b.py - kappa x alpha factor analysis (REML mixed effects + LR)."""
import os
import numpy as np
import pandas as pd
from scipy import stats as st

RES = r"D:\AL_FuzzyClustering_Paper\v2_unified\results"
FACT = os.path.join(RES, "factor")
OUT = r"D:\tmp\e2b_out"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(FACT, "kappa_alpha.csv"), encoding="utf-8")
df["ACC_pp"] = df["ACC"] * 100
df["iid"] = df["instance_id"].astype(str)

# marginals
km = df.groupby("kappa").agg(acc_pp=("ACC_pp", "mean"), n=("ACC_pp", "size")).reset_index()
am = df.groupby("alpha").agg(acc_pp=("ACC_pp", "mean"), n=("ACC_pp", "size")).reset_index()
km.to_csv(os.path.join(OUT, "kappa_marginal.csv"), index=False, encoding="utf-8")
am.to_csv(os.path.join(OUT, "alpha_marginal.csv"), index=False, encoding="utf-8")

# cells + overlap-stratified kappa
cells = df.groupby(["kappa", "alpha"]).agg(acc_pp=("ACC_pp", "mean")).reset_index()
cells.to_csv(os.path.join(OUT, "cells.csv"), index=False, encoding="utf-8")
ko = df.groupby(["overlap", "kappa"]).agg(acc_pp=("ACC_pp", "mean")).reset_index()
ko.pivot(index="overlap", columns="kappa", values="acc_pp").to_csv(
    os.path.join(OUT, "kappa_by_overlap.csv"), encoding="utf-8")

# REML mixed effects
import statsmodels.formula.api as smf

full = smf.mixedlm("ACC_pp ~ C(kappa)*C(alpha)", df,
                   groups=df["iid"]).fit(reml=True)
full_ml = smf.mixedlm("ACC_pp ~ C(kappa)*C(alpha)", df,
                      groups=df["iid"]).fit(reml=False)
no_int = smf.mixedlm("ACC_pp ~ C(kappa)+C(alpha)", df,
                     groups=df["iid"]).fit(reml=False)
no_kap = smf.mixedlm("ACC_pp ~ C(alpha)", df,
                     groups=df["iid"]).fit(reml=False)
no_alp = smf.mixedlm("ACC_pp ~ C(kappa)", df,
                     groups=df["iid"]).fit(reml=False)
null = smf.mixedlm("ACC_pp ~ 1", df, groups=df["iid"]).fit(reml=False)

rows = []
for name, llr, llred, dfd in [
    ("kappa main", full_ml.llf, no_kap.llf, 3),
    ("alpha main", full_ml.llf, no_alp.llf, 3),
    ("kappa x alpha", full_ml.llf, no_int.llf, 9),
    ("full vs null", full_ml.llf, null.llf, 15),
]:
    stat = 2.0 * (llr - llred)
    rows.append({"term": name, "LR": stat, "df": dfd,
                 "p": float(st.chi2.sf(stat, dfd))})
me = pd.DataFrame(rows)
me.to_csv(os.path.join(OUT, "mixed_model.csv"), index=False, encoding="utf-8")

vc = full.cov_re.iloc[0, 0]
vr = full.scale
print("=== kappa marginal (ACC %) ===")
print(km.round(4).to_string(index=False))
print("\n=== alpha marginal (ACC %) ===")
print(am.round(4).to_string(index=False))
print("\n=== cells (kappa x alpha, ACC %) ===")
print(cells.pivot(index="kappa", columns="alpha", values="acc_pp").round(4).to_string())
print("\n=== kappa by overlap (ACC %) ===")
print(ko.pivot(index="overlap", columns="kappa", values="acc_pp").round(3).to_string())
print("\n=== mixed model LR ===")
print(me.round(4).to_string(index=False))
print("\nICC=%.4f (between %.2f, residual %.2f)" % (vc/(vc+vr), vc, vr))
