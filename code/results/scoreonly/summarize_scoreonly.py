"""summarize_scoreonly.py — statistics for P2_ScoreOnly ablation.

Reads:
  - code/results/shards/  : P2_NoExploration, P2_Full existing 30-seed shards
  - code/results/scoreonly/: P2_ScoreOnly shards produced by run_scoreonly.py
Writes everything to code/results/scoreonly/summary/ as JSON + CSV.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

CODE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))          # code/
SHARDS = os.path.join(CODE, "results", "shards")
SO_DIR = os.path.join(CODE, "results", "scoreonly")
SUMDIR = os.path.join(SO_DIR, "summary")
os.makedirs(SUMDIR, exist_ok=True)

DATASETS = ["ecoli", "glass"]
SEEDS = list(range(30))

# Table 3 reference values (main draft, for hard alignment check)
TABLE3 = {
    ("ecoli", "P2_NoExploration"): (0.659, 0.047),
    ("ecoli", "P2_Full"):          (0.645, 0.064),
    ("glass", "P2_NoExploration"): (0.453, 0.023),
    ("glass", "P2_Full"):          (0.457, 0.030),
}


def acc_at10(shard_dir, ds, method, seed):
    p = os.path.join(shard_dir,
                     f"{ds}_{method}_30_{seed}_0_0.0_50.0_1.0.csv")
    df = pd.read_csv(p)
    row = df.loc[np.isclose(df["budget"], 0.10)]
    return float(row["ACC"].iloc[0])


def cell(v):
    """Identical format to build_tables_final.cell()."""
    n = len(v); m = v.mean(); sd = v.std(ddof=1); se = sd / np.sqrt(n)
    tc = stats.t.ppf(.975, n - 1)
    rng = np.random.RandomState(0)
    boot = [v[rng.randint(0, n, n)].mean() for _ in range(1000)]
    lo_b, hi_b = np.percentile(boot, [2.5, 97.5])
    return {"n": int(n), "mean": float(m), "sd": float(sd),
            "t_ci_lo": float(m - tc * se), "t_ci_hi": float(m + tc * se),
            "boot_ci_lo": float(lo_b), "boot_ci_hi": float(hi_b),
            "cell": f"{m:.3f}±{sd:.3f} [{m-tc*se:.3f},{m+tc*se:.3f};"
                    f"{lo_b:.3f},{hi_b:.3f}]"}


def paired(a, b):
    """a,b paired arrays; returns pp, 95% two-sided t-CI (pp), p, d_z."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    n = len(d); mean = d.mean(); sd = d.std(ddof=1)
    se = sd / np.sqrt(n); tc = stats.t.ppf(.975, n - 1)
    p = float(stats.ttest_rel(a, b).pvalue)
    return {"mean_diff_acc": float(mean),
            "pp": float(mean * 100),
            "t_ci_lo_pp": float((mean - tc * se) * 100),
            "t_ci_hi_pp": float((mean + tc * se) * 100),
            "p": p,
            "d_z": float(mean / sd),
            "n": int(n)}


report = {"rebuild_check": [], "scoreonly": {}, "contrasts": {}}
per_seed_rows = []

for ds in DATASETS:
    so, ne, fu = [], [], []
    for s in SEEDS:
        so_v = acc_at10(SO_DIR, ds, "P2_ScoreOnly", s)
        ne_v = acc_at10(SHARDS, ds, "P2_NoExploration", s)
        fu_v = acc_at10(SHARDS, ds, "P2_Full", s)
        so.append(so_v); ne.append(ne_v); fu.append(fu_v)
        per_seed_rows.append({"seed": s, "dataset": ds,
                              "ScoreOnly_ACC": so_v,
                              "NoExploration_ACC": ne_v,
                              "Full_ACC": fu_v,
                              "diff_vs_NoExpl": so_v - ne_v,
                              "diff_vs_Full": so_v - fu_v})
    so = np.array(so); ne = np.array(ne); fu = np.array(fu)

    # (a) rebuild NoExpl/Full and check against Table 3
    for method, arr in (("P2_NoExploration", ne), ("P2_Full", fu)):
        m_ref, sd_ref = TABLE3[(ds, method)]
        got_m, got_sd = float(arr.mean()), float(arr.std(ddof=1))
        report["rebuild_check"].append({
            "dataset": ds, "method": method,
            "rebuilt_mean": got_m, "rebuilt_sd": got_sd,
            "table3_mean": m_ref, "table3_sd": sd_ref,
            "mean_abs_diff": abs(got_m - m_ref),
            "sd_abs_diff": abs(got_sd - sd_ref),
            "within_tol_002": bool(abs(got_m - m_ref) <= 0.002
                                  and abs(got_sd - sd_ref) <= 0.002)})

    # (b) ScoreOnly cell
    report["scoreonly"][ds] = cell(so)

    # (c) paired contrasts
    report["contrasts"][ds] = {
        "ScoreOnly_minus_NoExploration": paired(so, ne),
        "ScoreOnly_minus_Full": paired(so, fu),
    }

# ---- write outputs ----
per_seed = pd.DataFrame(per_seed_rows)
per_seed.to_csv(os.path.join(SUMDIR, "paired_per_seed.csv"), index=False)

pd.DataFrame(report["rebuild_check"]).to_csv(
    os.path.join(SUMDIR, "rebuild_vs_table3.csv"), index=False)

sc_rows = [{"dataset": ds, **report["scoreonly"][ds]} for ds in DATASETS]
pd.DataFrame(sc_rows).to_csv(os.path.join(SUMDIR, "scoreonly_cell.csv"),
                             index=False)

out_json = os.path.join(SUMDIR, "summary.json")
json.dump(report, open(out_json, "w", encoding="utf-8"), indent=2)

# console digest
for ds in DATASETS:
    print(f"== {ds} ==")
    print(" ScoreOnly:", report["scoreonly"][ds]["cell"])
    c1 = report["contrasts"][ds]["ScoreOnly_minus_NoExploration"]
    c2 = report["contrasts"][ds]["ScoreOnly_minus_Full"]
    print(f" S-NoExpl: pp={c1['pp']:+.2f} 95%CI[{c1['t_ci_lo_pp']:+.2f},"
          f"{c1['t_ci_hi_pp']:+.2f}] p={c1['p']:.4g} d_z={c1['d_z']:+.3f}")
    print(f" S-Full  : pp={c2['pp']:+.2f} 95%CI[{c2['t_ci_lo_pp']:+.2f},"
          f"{c2['t_ci_hi_pp']:+.2f}] p={c2['p']:.4g}")
for r in report["rebuild_check"]:
    print(f" check {r['dataset']:8s} {r['method']:18s} "
          f"rebuilt {r['rebuilt_mean']:.3f}±{r['rebuilt_sd']:.3f} vs "
          f"table3 {r['table3_mean']:.3f}±{r['table3_sd']:.3f} "
          f"OK={r['within_tol_002']}")
print("summary dir:", SUMDIR)
