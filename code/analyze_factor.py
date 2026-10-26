"""analyze_factor.py — E2 多因子结果分析（数字全部来自真实跑批 CSV）。

读取 results/factor/{rho_beta,kappa_alpha,rho_beta_kappa}.csv。

统计模型（任务规定：主实现 statsmodels AnovaRM，MixedLM 复核）：
  E2a: ACC ~ rho * beta_mult  (instance 为 subject)
  E2b: ACC ~ kappa * alpha
  E2c: ACC ~ rho * beta_mult * kappa
设计平衡：12 instance × 因子全水平 × 10 seed 重复。先把 10 seed 聚合为
(instance × 因子格) 均值，再做重复测量 ANOVA。

偏 eta 平方（partial eta squared）：
    eta_p^2(effect) = SS_effect / (SS_effect + SS_error_of_effect)
其中 SS_error_of_effect 为该效应对应的 ×subject 残差项（两因素：rho×S、
beta×S、rho×beta×S；三因素：含各阶 ×S）。SS 由教科书边际均值分解得到，
并与 AnovaRM 的 F 值逐效应断言一致（|F/F_ref-1|<1e-6），保证分解正确。
MixedLM（instance 随机截距，10 次重复长格式）复核主效应 p 是否一致。

另：Tukey HSD（rho 5 水平两两）；rho 边际均值按 overlap 分层；
最优水平 + 对 instance bootstrap(1000) 95% CI。
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sstats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.normpath(os.path.join(HERE, "..", "results", "factor"))


# --------------------------------------------------------------------------- #
# 教科书边际均值分解（平衡设计，s=12 instance）
# --------------------------------------------------------------------------- #
def decompose_2way(cell):
    """cell: (s,a,b)。返回 {name: dict(SS,df,F,p,eta_p,SSe,dfe)}。"""
    s, a, b = cell.shape
    G = cell.mean()
    m_a = cell.mean(axis=(0, 2))
    m_b = cell.mean(axis=(0, 1))
    m_s = cell.mean(axis=(1, 2))
    m_sa = cell.mean(axis=2)
    m_sb = cell.mean(axis=1)
    m_ab = cell.mean(axis=0)

    eff = {"A": m_a - G, "B": m_b - G,
           "AB": m_ab - m_a[None, :].T - m_b[None, :] + G}
    err = {"A": m_sa - m_s[:, None] - m_a[None, :] + G,
           "B": m_sb - m_s[:, None] - m_b[None, :] + G,
           "AB": cell - m_sa[:, :, None] - m_sb[:, None, :] + m_s[:, None, None]}
    w = {"A": s * b, "B": s * a, "AB": s}
    we = {"A": b, "B": a, "AB": 1.0}
    d = {"A": a - 1, "B": b - 1, "AB": (a - 1) * (b - 1)}
    dfe = {"A": (a - 1) * (s - 1), "B": (b - 1) * (s - 1),
           "AB": (a - 1) * (b - 1) * (s - 1)}
    out = {}
    for k in ("A", "B", "AB"):
        ss = w[k] * np.sum(eff[k] ** 2)
        sse = we[k] * np.sum(err[k] ** 2)
        F = (ss / d[k]) / (sse / dfe[k])
        out[k] = {"SS": ss, "df": d[k], "SSe": sse, "dfe": dfe[k],
                  "F": float(F), "p": float(sstats.f.sf(F, d[k], dfe[k])),
                  "eta_p": float(ss / (ss + sse))}
    return out


def decompose_3way(cell):
    """cell: (s,a,b,c)。"""
    s, a, b, c = cell.shape
    G = cell.mean()
    m_a = cell.mean(axis=(0, 2, 3))
    m_b = cell.mean(axis=(0, 1, 3))
    m_c = cell.mean(axis=(0, 1, 2))
    m_s = cell.mean(axis=(1, 2, 3))
    m_sa = cell.mean(axis=(2, 3))
    m_sb = cell.mean(axis=(1, 3))
    m_sc = cell.mean(axis=(1, 2))
    m_ab = cell.mean(axis=(0, 3))
    m_ac = cell.mean(axis=(0, 2))
    m_bc = cell.mean(axis=(0, 1))
    m_sab = cell.mean(axis=3)
    m_sac = cell.mean(axis=2)
    m_sbc = cell.mean(axis=1)

    eff = {}
    eff["A"] = m_a - G
    eff["B"] = m_b - G
    eff["C"] = m_c - G
    eff["AB"] = m_ab - m_a[None, :].T - m_b[None, :] + G
    eff["AC"] = m_ac - m_a[:, None] - m_c[None, :] + G
    eff["BC"] = m_bc - m_b[:, None] - m_c[None, :] + G
    m_abc = cell.mean(axis=0)
    eff["ABC"] = (m_abc - eff["AB"][:, :, None] - eff["AC"][:, None, :]
                  - eff["BC"][None, :, :] - eff["A"][:, None, None]
                  - eff["B"][None, :, None] - eff["C"][None, None, :] - G)

    err = {}
    err["A"] = m_sa - m_s[:, None] - m_a[None, :] + G
    err["B"] = m_sb - m_s[:, None] - m_b[None, :] + G
    err["C"] = m_sc - m_s[:, None] - m_c[None, :] + G
    err["AB"] = m_sab - m_sa[:, :, None] - m_sb[:, None, :] \
        + m_s[:, None, None] - eff["AB"][None, :, :]
    err["AC"] = m_sac - m_sa[:, :, None] - m_sc[:, None, :] \
        + m_s[:, None, None] - eff["AC"][None, :, :]
    err["BC"] = m_sbc - m_sb[:, None, :] - m_sc[:, None, :] \
        + m_s[:, None, None] - eff["BC"][None, :, :]
    err["ABC"] = cell - m_sab[:, :, :, None] - m_sac[:, :, None, :] \
        - m_sbc[:, None, :, :] + m_sa[:, :, None, None] \
        + m_sb[:, None, :, None] + m_sc[:, None, None, :] \
        - m_s[:, None, None, None] - eff["ABC"][None, :, :, :]

    w = {"A": s * b * c, "B": s * a * c, "C": s * a * b,
         "AB": s * c, "AC": s * b, "BC": s * a, "ABC": s}
    we = {"A": b * c, "B": a * c, "C": a * b,
          "AB": c, "AC": b, "BC": a, "ABC": 1.0}
    d = {"A": a - 1, "B": b - 1, "C": c - 1,
         "AB": (a - 1) * (b - 1), "AC": (a - 1) * (c - 1),
         "BC": (b - 1) * (c - 1), "ABC": (a - 1) * (b - 1) * (c - 1)}
    dfe = {"A": (a - 1) * (s - 1), "B": (b - 1) * (s - 1),
           "C": (c - 1) * (s - 1),
           "AB": (a - 1) * (b - 1) * (s - 1),
           "AC": (a - 1) * (c - 1) * (s - 1),
           "BC": (b - 1) * (c - 1) * (s - 1),
           "ABC": (a - 1) * (b - 1) * (c - 1) * (s - 1)}
    out = {}
    for k in ("A", "B", "C", "AB", "AC", "BC", "ABC"):
        ss = w[k] * np.sum(eff[k] ** 2)
        sse = we[k] * np.sum(err[k] ** 2)
        F = (ss / d[k]) / (sse / dfe[k])
        out[k] = {"SS": ss, "df": d[k], "SSe": sse, "dfe": dfe[k],
                  "F": float(F), "p": float(sstats.f.sf(F, d[k], dfe[k])),
                  "eta_p": float(ss / (ss + sse))}
    return out


def build_grid(agg, cols):
    """agg: 已按 (instance_id, *cols) 平均。返回 (12, *levels_shape) array。"""
    iids = sorted(agg.instance_id.unique())
    levels = [sorted(agg[c].unique()) for c in cols]
    grid = np.zeros((len(iids), *[len(lv) for lv in levels]))
    for ii, iid in enumerate(iids):
        sub = agg[agg.instance_id == iid]
        for idx in np.ndindex(*[len(lv) for lv in levels]):
            mask = np.ones(len(sub), bool)
            for j, c in enumerate(cols):
                mask &= (sub[c] == levels[j][idx[j]]).values
            grid[(ii,) + idx] = sub.ACC.values[mask].mean()
    return grid, iids, levels


def boot_best_ci(agg, factor_col, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    inst = sorted(agg.instance_id.unique())
    means = agg.groupby(factor_col)["ACC"].mean()
    boots = {lv: [] for lv in means.index}
    for _ in range(n_boot):
        samp = rng.choice(inst, size=len(inst), replace=True)
        sub = agg[agg.instance_id.isin(samp)]
        for lv in means.index:
            boots[lv].append(sub[sub[factor_col] == lv]["ACC"].mean())
    res = {lv: (float(means[lv]),
                float(np.percentile(boots[lv], 2.5)),
                float(np.percentile(boots[lv], 97.5)))
           for lv in means.index}
    best = max(res, key=lambda lv: res[lv][0])
    return res, best


def dir_desc(mm):
    items = sorted(mm.items(), key=lambda kv: kv[0])
    best = max(items, key=lambda kv: kv[1])
    return f"best={best[0]:g} (ACC={best[1]:.4f})"


def main():
    rb = pd.read_csv(os.path.join(RES, "rho_beta.csv"))
    ka = pd.read_csv(os.path.join(RES, "kappa_alpha.csv"))
    rbk = pd.read_csv(os.path.join(RES, "rho_beta_kappa.csv"))
    print(f"loaded: rho_beta={len(rb)} kappa_alpha={len(ka)} "
          f"rho_beta_kappa={len(rbk)}", flush=True)

    agg_a = rb.groupby(["instance_id", "rho", "beta_mult"],
                       as_index=False)["ACC"].mean()
    agg_b = ka.groupby(["instance_id", "kappa", "alpha"],
                       as_index=False)["ACC"].mean()
    agg_c = rbk.groupby(["instance_id", "rho", "beta_mult", "kappa"],
                        as_index=False)["ACC"].mean()

    from statsmodels.stats.anova import AnovaRM

    # ---- E2a --------------------------------------------------------------- #
    grid_a, iids, (rho_lv, beta_lv) = build_grid(agg_a, ["rho", "beta_mult"])
    anova_a = decompose_2way(grid_a)
    ar_a = AnovaRM(agg_a, "ACC", "instance_id",
                   within=["rho", "beta_mult"]).fit()
    print("\n[AnovaRM E2a]", flush=True)
    print(ar_a.anova_table.to_string(), flush=True)
    # F 一致性校验
    ref = ar_a.anova_table
    for key, refname in (("A", "rho"), ("B", "beta_mult"),
                         ("AB", "rho:beta_mult")):
        f_man = anova_a[key]["F"]
        f_ref = float(ref.loc[refname, "F Value"])
        assert abs(f_man - f_ref) / max(f_ref, 1e-12) < 1e-6, \
            f"E2a {key}: manual F={f_man} vs AnovaRM F={f_ref}"
    print("[E2a] manual-SS F matches AnovaRM: OK", flush=True)

    # ---- E2b --------------------------------------------------------------- #
    grid_b, _, (kap_lv, alph_lv) = build_grid(agg_b, ["kappa", "alpha"])
    anova_b = decompose_2way(grid_b)
    ar_b = AnovaRM(agg_b, "ACC", "instance_id",
                   within=["kappa", "alpha"]).fit()
    print("\n[AnovaRM E2b]", flush=True)
    print(ar_b.anova_table.to_string(), flush=True)
    refb = ar_b.anova_table
    for key, refname in (("A", "kappa"), ("B", "alpha"),
                         ("AB", "kappa:alpha")):
        f_man = anova_b[key]["F"]
        f_ref = float(refb.loc[refname, "F Value"])
        assert abs(f_man - f_ref) / max(f_ref, 1e-12) < 1e-6
    print("[E2b] manual-SS F matches AnovaRM: OK", flush=True)

    # ---- E2c --------------------------------------------------------------- #
    grid_c, _, (rho_sub, beta_sub, kap_sub) = build_grid(
        agg_c, ["rho", "beta_mult", "kappa"])
    anova_c = decompose_3way(grid_c)
    ar_c = AnovaRM(agg_c, "ACC", "instance_id",
                   within=["rho", "beta_mult", "kappa"]).fit()
    print("\n[AnovaRM E2c]", flush=True)
    print(ar_c.anova_table.to_string(), flush=True)
    refc = ar_c.anova_table
    name_map_c = {"A": "rho", "B": "beta_mult", "C": "kappa",
                  "AB": "rho:beta_mult", "AC": "rho:kappa",
                  "BC": "beta_mult:kappa", "ABC": "rho:beta_mult:kappa"}
    for key, refname in name_map_c.items():
        f_man = anova_c[key]["F"]
        f_ref = float(refc.loc[refname, "F Value"])
        assert abs(f_man - f_ref) / max(f_ref, 1e-12) < 1e-6, \
            f"E2c {key}: manual F={f_man} vs AnovaRM F={f_ref}"
    print("[E2c] manual-SS F matches AnovaRM: OK", flush=True)

    # ---- MixedLM 复核主效应 p --------------------------------------------- #
    import statsmodels.formula.api as smf
    rb2 = rb.copy()
    rb2["rho"] = rb2.rho.astype("category")
    rb2["beta_mult"] = rb2.beta_mult.astype("category")
    md = smf.mixedlm("ACC ~ C(rho)*C(beta_mult)", rb2,
                     groups=rb2.instance_id).fit(reml=True, method="lbfgs")
    print("\n[MixedLM E2a] fixed effects (主效应 p 复核):", flush=True)
    fe = md.summary().tables[1]
    print(fe.as_text(), flush=True)

    # ---- 汇总表 ------------------------------------------------------------ #
    mm_rho = {lv: float(grid_a[:, i, :].mean())
              for i, lv in enumerate(rho_lv)}
    mm_beta = {lv: float(grid_a[:, :, i].mean())
               for i, lv in enumerate(beta_lv)}
    rows = [
        {"experiment": "E2a", "factor": "rho", "F": anova_a["A"]["F"],
         "p": anova_a["A"]["p"], "eta_sq": anova_a["A"]["eta_p"],
         "direction": dir_desc(mm_rho)},
        {"experiment": "E2a", "factor": "beta_mult",
         "F": anova_a["B"]["F"], "p": anova_a["B"]["p"],
         "eta_sq": anova_a["B"]["eta_p"], "direction": dir_desc(mm_beta)},
        {"experiment": "E2a", "factor": "rho:beta_mult",
         "F": anova_a["AB"]["F"], "p": anova_a["AB"]["p"],
         "eta_sq": anova_a["AB"]["eta_p"], "direction": "interaction"},
    ]
    mm_kap = {lv: float(grid_b[:, i, :].mean())
              for i, lv in enumerate(kap_lv)}
    mm_alp = {lv: float(grid_b[:, :, i].mean())
              for i, lv in enumerate(alph_lv)}
    rows += [
        {"experiment": "E2b", "factor": "kappa", "F": anova_b["A"]["F"],
         "p": anova_b["A"]["p"], "eta_sq": anova_b["A"]["eta_p"],
         "direction": dir_desc(mm_kap)},
        {"experiment": "E2b", "factor": "alpha", "F": anova_b["B"]["F"],
         "p": anova_b["B"]["p"], "eta_sq": anova_b["B"]["eta_p"],
         "direction": dir_desc(mm_alp)},
        {"experiment": "E2b", "factor": "kappa:alpha",
         "F": anova_b["AB"]["F"], "p": anova_b["AB"]["p"],
         "eta_sq": anova_b["AB"]["eta_p"], "direction": "interaction"},
    ]
    for key in ("A", "B", "C", "AB", "AC", "BC", "ABC"):
        rows.append({"experiment": "E2c", "factor": name_map_c[key],
                     "F": anova_c[key]["F"], "p": anova_c[key]["p"],
                     "eta_sq": anova_c[key]["eta_p"],
                     "direction": "main/interaction"})
    t10 = pd.DataFrame(rows)
    t10.to_csv(os.path.join(RES, "table10_main_effects.csv"), index=False)

    # ---- Tukey HSD: rho 5 水平 -------------------------------------------- #
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tk = pairwise_tukeyhsd(endog=rb["ACC"].values,
                           groups=rb["rho"].values, alpha=0.05)
    print("\n[Tukey HSD rho]", flush=True)
    print(tk.summary().as_text(), flush=True)
    sig_pairs = []
    for r in tk.summary().data[1:]:
        g1, g2, meandiff, p_adj, lo, hi, reject = r
        if bool(reject):
            sig_pairs.append({"rho_1": g1, "rho_2": g2,
                              "mean_diff": float(meandiff),
                              "p_adj": float(p_adj),
                              "ci_lo": float(lo), "ci_hi": float(hi)})
    pd.DataFrame(sig_pairs).to_csv(
        os.path.join(RES, "tukey_rho_pairs.csv"), index=False)

    # ---- rho 边际均值按 overlap 分层 -------------------------------------- #
    overlaps = sorted(rb.overlap.unique())
    mm_mat = pd.DataFrame(index=rho_lv)
    for ov in overlaps:
        mm_mat[f"overlap_{ov}"] = [
            float(rb[(rb.overlap == ov) & (rb.rho == lv)]["ACC"].mean())
            for lv in rho_lv]
    mm_mat.index.name = "rho"
    mm_mat.to_csv(os.path.join(RES, "table10_marginal_means.csv"))
    print("\n[rho marginal means by overlap]", flush=True)
    print(mm_mat.to_string(), flush=True)

    # ---- fig11_data.csv ---------------------------------------------------- #
    panel_a = (agg_a.groupby(["rho", "beta_mult"])["ACC"].mean()
               .reset_index())
    panel_a.insert(0, "panel", "a_rho_beta_grid")
    panel_a = panel_a.rename(columns={"beta_mult": "beta_mult"})
    panel_b_rows = []
    for lv in rho_lv:
        for ov in overlaps:
            panel_b_rows.append({"panel": "b_rho_by_overlap",
                                 "rho": lv, "overlap": ov,
                                 "ACC": mm_mat.loc[lv, f"overlap_{ov}"]})
    panel_b = pd.DataFrame(panel_b_rows)
    fig11 = pd.concat([panel_a, panel_b], ignore_index=True)
    fig11.to_csv(os.path.join(RES, "fig11_data.csv"), index=False)

    # ---- tableF2: kappa×alpha 主效应 + 边际均值 --------------------------- #
    f2_rows = [{"section": "main_effect", "factor": r["factor"],
                "F": r["F"], "p": r["p"], "eta_p2": r["eta_sq"],
                "note": r["direction"]}
               for r in rows if r["experiment"] == "E2b"]
    for lv, v in mm_kap.items():
        f2_rows.append({"section": "marginal_mean", "factor": "kappa",
                        "level": lv, "ACC": v, "F": np.nan, "p": np.nan,
                        "eta_p2": np.nan, "note": ""})
    for lv, v in mm_alp.items():
        f2_rows.append({"section": "marginal_mean", "factor": "alpha",
                        "level": lv, "ACC": v, "F": np.nan, "p": np.nan,
                        "eta_p2": np.nan, "note": ""})
    pd.DataFrame(f2_rows).to_csv(
        os.path.join(RES, "tableF2_kappa_alpha.csv"), index=False)

    # ---- 最优水平 + bootstrap CI ------------------------------------------ #
    boot_rows = []
    for fac in ("rho", "beta_mult"):
        res, best = boot_best_ci(agg_a, fac)
        for lv, (m, lo, hi) in res.items():
            boot_rows.append({"experiment": "E2a", "factor": fac,
                              "level": lv, "margin_mean": m, "ci_lo": lo,
                              "ci_hi": hi, "is_best": lv == best})
    for fac in ("kappa", "alpha"):
        res, best = boot_best_ci(agg_b, fac)
        for lv, (m, lo, hi) in res.items():
            boot_rows.append({"experiment": "E2b", "factor": fac,
                              "level": lv, "margin_mean": m, "ci_lo": lo,
                              "ci_hi": hi, "is_best": lv == best})
    pd.DataFrame(boot_rows).to_csv(
        os.path.join(RES, "optimal_levels_boot_ci.csv"), index=False)

    print("\n==== SUMMARY ====", flush=True)
    print("Tukey significant rho pairs:", flush=True)
    for r in sig_pairs:
        print(f"  {r['rho_1']} vs {r['rho_2']} diff={r['mean_diff']:.4f} "
              f"p={r['p_adj']:.3g}", flush=True)
    print("Best levels:", flush=True)
    for r in boot_rows:
        if r["is_best"]:
            print(f"  {r['experiment']} {r['factor']}={r['level']} "
                  f"mean={r['margin_mean']:.4f} "
                  f"CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]", flush=True)


if __name__ == "__main__":
    main()
