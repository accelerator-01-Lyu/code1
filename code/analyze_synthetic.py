"""analyze_synthetic.py — E1 synthetic sweep analysis.

Reads results/synthetic/summary.csv and produces:
  paired_contrasts.csv     per instance x budget x contrast (NoExploration-Full,
                           SlackUniform-Full): mean paired diff, bootstrap 95% CI, d_z
  table9_by_overlap.csv / by_c / by_d / by_block.csv
                           noninferiority proportion (budget=0.10) per stratum
                           + bootstrap 95% CI over instances
  trend_tests.csv          Spearman rho(effect, overlap) + bootstrap CI,
                           Jonckheere-Terpstra ordered trend test
  fig9_data.csv            overlap-stratified NI proportion + CI
  fig10_data.csv          per-instance 10%-budget effect size (pp) + CI
  results/stats/e1_bh.csv  BH-FDR over the 72 one-sided NI tests

Methods
-------
* Paired diff = ACC(NoExploration) - ACC(Full) on the same seed (20 seeds).
* Two-sided CIs in paired_contrasts.csv: bootstrap percentile 95% (10000 resamples)
  via run_experiments.paired_stats(n_boot=10000); d_z = mean(diff)/sd(diff).
* Noninferiority (primary, budget=0.10): one-sided 95% LOWER bound of the paired
  t-CI must exceed -DELTA_NI (=-0.005).  Raw p-value for BH = one-sided paired-t
  p for H0: mean_diff <= -0.005.
* BH-FDR: Benjamini-Hochberg (statsmodels multipletests, method='fdr_bh').
* JT test: scipy.stats.jonckheere_terpstra if available; otherwise the classic
  mean-rank approximation (documented in the code).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_experiments import paired_stats, DELTA_NI

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "synthetic")
STATS = os.path.join(ROOT, "results", "stats")
os.makedirs(STATS, exist_ok=True)

BUDGET_MAIN = 0.10
N_BOOT = 10000
rng_boot = np.random.RandomState(20261022)


def boot_prop_ci(mask, n_boot=N_BOOT):
    """Bootstrap 95% CI of the proportion True in a boolean vector (resample
    instances)."""
    mask = np.asarray(mask, bool)
    n = len(mask)
    ps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng_boot.randint(0, n, n)
        ps[b] = mask[idx].mean()
    return mask.mean(), float(np.percentile(ps, 2.5)), float(np.percentile(ps, 97.5))


def main():
    df = pd.read_csv(os.path.join(RES, "summary.csv"))
    # sanity: expected rows
    expect = 72 * 4 * 20 * 4
    print(f"[analyze] summary rows={len(df)} expected={expect}")
    assert len(df) == expect, f"row count mismatch: {len(df)} != {expect}"

    methods = sorted(df.method.unique())
    print("[analyze] methods:", methods)
    assert {"P2_Full", "P2_NoExploration", "SlackUniform", "Random"} <= set(methods)

    # ---- 1. paired contrasts ------------------------------------------------ #
    insts = sorted(df.instance_id.unique())
    budgets = sorted(df.budget.unique())
    contrasts = []
    for iid in insts:
        sub = df[df.instance_id == iid]
        for contrast, b_method in (("NoExploration-Full", "P2_NoExploration"),
                                    ("SlackUniform-Full", "SlackUniform")):
            for bud in budgets:
                a = sub[(sub.method == b_method) & (sub.budget == bud)].sort_values("seed")
                b = sub[(sub.method == "P2_Full") & (sub.budget == bud)].sort_values("seed")
                assert len(a) == 20 and len(b) == 20, (iid, contrast, bud, len(a), len(b))
                s = paired_stats(a.ACC.values, b.ACC.values, n_boot=N_BOOT,
                                 seed=int(1000 + iid), sides="two")
                contrasts.append({
                    "instance_id": iid, "contrast": contrast, "budget": bud,
                    "mean_diff": s["mean"], "ci_low": s["boot_lo"],
                    "ci_high": s["boot_hi"], "d_z": s["dz"],
                })
    cdf = pd.DataFrame(contrasts)
    cdf.to_csv(os.path.join(RES, "paired_contrasts.csv"), index=False, encoding="utf-8")
    print(f"[analyze] paired_contrasts.csv rows={len(cdf)}")

    # ---- 2. noninferiority at main budget (one-sided t lower bound) -------- #
    ni_rows = []
    for iid in insts:
        sub = df[(df.instance_id == iid) & (df.budget == BUDGET_MAIN)]
        ne = sub[sub.method == "P2_NoExploration"].sort_values("seed").ACC.values
        fu = sub[sub.method == "P2_Full"].sort_values("seed").ACC.values
        d = ne - fu
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        t_one = (d.mean() + DELTA_NI) / se if se > 0 else np.inf
        p_raw = float(stats.t.sf(t_one, n - 1))          # H0: mean <= -0.005
        lo_t = float(d.mean() - stats.t.ppf(0.95, n - 1) * se)   # one-sided 95% LB
        ni_raw = bool(lo_t > -DELTA_NI)
        meta = sub.iloc[0]
        ni_rows.append({
            "instance_id": iid, "overlap": meta["overlap"], "block": meta["block"],
            "c": int(meta["c"]), "d": int(meta["d"]), "N": int(meta["N"]),
            "mean_diff": float(d.mean()), "ci_low_onesided_t": lo_t,
            "p_raw": p_raw, "noninferior_raw": ni_raw,
        })
    ndf = pd.DataFrame(ni_rows)

    # ---- BH-FDR ------------------------------------------------------------- #
    try:
        from statsmodels.stats.multitest import multipletests
        q = multipletests(ndf.p_raw.values, method="fdr_bh")[1]
    except Exception:
        # hand-rolled Benjamini-Hochberg
        p = ndf.p_raw.values
        order = np.argsort(p)
        m = len(p); qv = np.empty(m)
        prev = 1.0
        for rank in range(m - 1, -1, -1):
            idx = order[rank]
            prev = min(prev, (m / (rank + 1)) * p[idx])
            qv[idx] = prev
        q = qv
    ndf["bh_q"] = q
    ndf["noninferior_bh"] = ndf["bh_q"] < 0.05
    ndf.to_csv(os.path.join(STATS, "e1_bh.csv"), index=False, encoding="utf-8")
    print(f"[analyze] e1_bh.csv rows={len(ndf)}; NI raw={ndf.noninferior_raw.sum()}/72 "
          f"NI bh={ndf.noninferior_bh.sum()}/72")

    # ---- 3. overall + stratified NI proportions ----------------------------- #
    def strat_table(col):
        rows = []
        for val, g in ndf.groupby(col):
            prop, lo, hi = boot_prop_ci(g.noninferior_raw.values)
            rows.append({"stratum": col, "value": val,
                         "n_instances": len(g),
                         "n_noninferior": int(g.noninferior_raw.sum()),
                         "prop_noninferior": prop,
                         "ci_low": lo, "ci_high": hi})
        return pd.DataFrame(rows)

    overall_prop, overall_lo, overall_hi = boot_prop_ci(ndf.noninferior_raw.values)
    print(f"[analyze] overall NI prop = {overall_prop:.3f} "
          f"95%CI=({overall_lo:.3f},{overall_hi:.3f})")

    tab_overlap = strat_table("overlap")
    tab_c = strat_table("c")
    tab_d = strat_table("d")
    tab_block = strat_table("block")
    tab_overlap.to_csv(os.path.join(RES, "table9_by_overlap.csv"), index=False, encoding="utf-8")
    tab_c.to_csv(os.path.join(RES, "table9_by_c.csv"), index=False, encoding="utf-8")
    tab_d.to_csv(os.path.join(RES, "table9_by_d.csv"), index=False, encoding="utf-8")
    tab_block.to_csv(os.path.join(RES, "table9_by_block.csv"), index=False, encoding="utf-8")
    for t in (tab_overlap, tab_c, tab_d, tab_block):
        print(t.to_string(index=False))

    # ---- 4. trend: Spearman rho(effect, overlap) + JT ---------------------- #
    x = ndf.overlap.values
    eff = ndf.mean_diff.values          # NoExploration - Full at 10% budget
    rho, p_rho = stats.spearmanr(x, eff)
    # bootstrap CI for rho
    rhos = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng_boot.randint(0, len(x), len(x))
        rhos[b] = stats.spearmanr(x[idx], eff[idx]).statistic
    rho_lo, rho_hi = np.percentile(rhos, [2.5, 97.5])
    print(f"[analyze] Spearman rho(effect, overlap)={rho:.3f} "
          f"95%CI=({rho_lo:.3f},{rho_hi:.3f}) p={p_rho:.4g}")

    groups = [eff[x == ov] for ov in sorted(np.unique(x))]
    try:
        jt = stats.jonckheere_terpstra(groups, alternative="decreasing")
        jt_stat, jt_p = float(jt.statistic), float(jt.pvalue)
        jt_impl = "scipy.stats.jonckheere_terpstra"
    except Exception:
        # classic Jonckheere-Terpstra via direct pairwise Mann-Whitney counts.
        # Groups are ordered by increasing overlap; U = sum_{i<j} U_ij with
        # U_ij = #{x in G_i, y in G_j : x > y} + 0.5 #{x=y}.  Large U means
        # earlier groups have larger values => decreasing trend.
        U = 0.0
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a = groups[i][:, None]
                b = groups[j][None, :]
                U += float((a > b).sum() + 0.5 * (a == b).sum())
        Ntot = sum(len(g) for g in groups)
        ni = np.array([len(g) for g in groups])
        e_u = (Ntot ** 2 - (ni ** 2).sum()) / 4.0
        v_u = (Ntot ** 2 * (2 * Ntot + 3) -
               (ni ** 2 * (2 * ni + 3)).sum()) / 72.0
        z = (U - e_u) / np.sqrt(v_u)
        jt_stat = float(U)
        jt_p = float(stats.norm.sf(z))   # one-sided: decreasing trend
        jt_impl = "hand-rolled pairwise Mann-Whitney JT (normal approx)"
    print(f"[analyze] JT (decreasing effect with overlap) stat={jt_stat:.2f} "
          f"p={jt_p:.4g} via {jt_impl}")

    pd.DataFrame([{
        "test": "spearman_rho_effect_vs_overlap", "statistic": float(rho),
        "ci_low": float(rho_lo), "ci_high": float(rho_hi), "p_value": float(p_rho),
        "alternative": "two-sided", "note": "effect = NoExploration-Full ACC @10% budget",
    }, {
        "test": "jonckheere_terpstra", "statistic": jt_stat, "ci_low": np.nan,
        "ci_high": np.nan, "p_value": jt_p, "alternative": "decreasing",
        "note": jt_impl,
    }]).to_csv(os.path.join(RES, "trend_tests.csv"), index=False, encoding="utf-8")

    # ---- 5. figure data ----------------------------------------------------- #
    tab_overlap.to_csv(os.path.join(RES, "fig9_data.csv"), index=False, encoding="utf-8")
    f10 = ndf[["instance_id", "overlap", "block", "c", "d", "N",
               "mean_diff", "ci_low_onesided_t"]].copy()
    f10["effect_pp"] = f10["mean_diff"] * 100.0
    f10["ci_low_pp"] = f10["ci_low_onesided_t"] * 100.0
    f10.to_csv(os.path.join(RES, "fig10_data.csv"), index=False, encoding="utf-8")

    # ---- 6. SlackUniform != NoExploration verification ---------------------- #
    su = df[(df.method == "SlackUniform") & (df.budget == BUDGET_MAIN)].groupby("instance_id").ACC.mean()
    ne2 = df[(df.method == "P2_NoExploration") & (df.budget == BUDGET_MAIN)].groupby("instance_id").ACC.mean()
    diff = (su - ne2).dropna()
    print(f"[analyze] SlackUniform vs NoExploration @10%: mean ACC diff "
          f"{diff.mean():+.4f}, |diff|>1e-6 in {int((diff.abs() > 1e-6).sum())}/"
          f"{len(diff)} instances; e.g. ids {list(diff.index[:8])}: "
          f"{[round(v,4) for v in diff.values[:8]]}")

    print("[analyze] DONE")


if __name__ == "__main__":
    main()
