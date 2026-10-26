"""patch_e1_missing_row.py - rerun job (4, SlackUniform, 8), verify determinism
of the 3 existing budget rows, then append only the missing 0.05 row."""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_synthetic_sweep as sw

RES = r"D:\AL_FuzzyClustering_Paper\v2_unified\results\synthetic"
SUMMARY = os.path.join(RES, "summary.csv")
DONE = os.path.join(RES, "done.txt")

# 1. rerun the job through the same worker path
res = sw._worker((4, "SlackUniform", 8))
rows = res["rows"]
print(f"[patch] job rerun produced {len(rows)} budget rows, "
      f"elapsed={res['elapsed_s']:.1f}s rss={res['rss_peak_mb']:.0f}MB")

df = pd.DataFrame(rows)
df = df[sw.SUMMARY_COLS]
df["instance_id"] = df["instance_id"].astype(float)
new05 = df[df["budget"] == 0.05]
print("[patch] new 0.05 row:\n", new05.to_string(index=False))

# 2. load existing rows for this job and compare the three shared budgets
old = pd.read_csv(SUMMARY, encoding="utf-8")
old_job = old[(old["instance_id"] == 4) & (old["method"] == "SlackUniform") &
              (old["seed"] == 8)]
print(f"[patch] existing rows for this job: {len(old_job)}")
cols = ["overlap", "block", "c", "d", "N", "method", "seed", "budget",
        "ACC", "NMI", "ARI"]
merged = pd.merge(old_job[cols], df[cols], on="budget", suffixes=("_old", "_new"))
for _, r in merged.iterrows():
    acc_ok = abs(r["ACC_old"] - r["ACC_new"]) < 1e-12
    nmi_ok = abs(r["NMI_old"] - r["NMI_new"]) < 1e-12
    ari_ok = abs(r["ARI_old"] - r["ARI_new"]) < 1e-12
    flag = "OK" if (acc_ok and nmi_ok and ari_ok) else "MISMATCH"
    print(f"[patch] budget={r['budget']} ACC {r['ACC_old']:.4f} vs {r['ACC_new']:.4f} "
          f"NMI {r['NMI_old']:.6f} vs {r['NMI_new']:.6f} -> {flag}")
    if flag != "OK":
        print("[patch] ABORT: determinism violated, do not append")
        sys.exit(1)

# 3. append the missing 0.05 row (only if truly absent)
already = old[(old["instance_id"] == 4) & (old["method"] == "SlackUniform") &
              (old["seed"] == 8) & (old["budget"] == 0.05)]
if len(already) > 0:
    print("[patch] 0.05 row already present; nothing to do")
else:
    new05[sw.SUMMARY_COLS].to_csv(SUMMARY, mode="a", header=False, index=False,
                                  encoding="utf-8")
    print("[patch] appended 0.05 row")

# 4. verify final counts
chk = pd.read_csv(SUMMARY, encoding="utf-8")
g = chk.groupby(["instance_id", "method", "budget"]).size()
bad = g[g != 20]
print(f"[patch] final bad combos: {len(bad)}; total rows={len(chk)}")
