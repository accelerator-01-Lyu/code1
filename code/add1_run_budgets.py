# -*- coding: utf-8 -*-
"""add1_run_budgets.py — Task 1 (add1): run NEW budget points for the budget
curve extension on ecoli / compound / glass, P2_Full vs P2_NoExploration.

Existing points 5/10/15/20% already live in code/results/shards (seed labels
0..29, seed_set="30"). We DO NOT touch shards. We run the FROZEN AL loop from
run_experiments.run_traj (deterministic, seeded) for the missing points:

  job "low" : budgets=(0.01, 0.02, 0.03)   -> new points 1, 2, 3 %
  job "mid" : budgets=(0.05, 0.07)         -> new point 7 % (continues from the
                                              shard's L0->5% trajectory, so the
                                              reproduced 5% row is a determinism
                                              cross-check against the shards)
  job "high": budgets=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30) -> new points 25, 30 %
                                              (first four rows must reproduce the
                                              shard values bit-for-bit; we keep
                                              them only as a check, and the
                                              aggregate uses shard rows for
                                              5/10/15/20 per protocol)

Seeds mirror the frozen shards: range(30) with seed labels 0..29 (this is what
jobs_main() used; ac.SEEDS_30 numbers are NOT used because the shard seed column
is 0..29 and by-seed pairing requires identical seed labels).

Output: results/additional/add1_new_budget_raw.csv (long table, resumable).
Log   : results/additional/_add1_run.log
"""
import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\AL_FuzzyClustering_Paper\v2_unified\code")
import additional_common as ac
import run_experiments as re

DATASETS = ["ecoli", "compound", "glass"]
METHODS = ["P2_Full", "P2_NoExploration"]
SEEDS = list(range(30))                      # frozen shard seed labels
JOB_TYPES = {
    "low": (0.01, 0.02, 0.03),
    "mid": (0.05, 0.07),
    "high": (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
}
ALPHA0 = 50.0
LAM_MULT = 1.0

OUT = ac.res_path("add1_new_budget_raw.csv")
LOG = ac.res_path("_add1_run.log")


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done_keys():
    if not os.path.exists(OUT):
        return set()
    df = pd.read_csv(OUT)
    return set(zip(df.dataset, df.method, df.seed, df.job_type))


def main():
    t0 = time.time()
    cache = {}
    pending = []
    have = done_keys()
    for ds in DATASETS:
        for m in METHODS:
            for s in SEEDS:
                for jt, buds in JOB_TYPES.items():
                    if (ds, m, s, jt) in have:
                        continue
                    pending.append((ds, m, s, jt, buds))
    log("pending jobs: %d (already done: %d)" % (len(pending), len(have)))
    n_done = 0
    for ds, m, s, jt, buds in pending:
        if ds not in cache:
            cache[ds] = re.load_pkl(ds)
        X, y = cache[ds]
        c = re.NCLASSES[ds]
        rows = re.run_traj(X, y, c, ds, m, s, 0, 0.0, ALPHA0, LAM_MULT,
                           budgets=buds, run_x2=False)[0]
        out = pd.DataFrame(rows)
        out["dataset"] = ds
        out["method"] = m
        out["seed"] = s
        out["job_type"] = jt
        out["alpha0"] = ALPHA0
        out["lam_mult"] = LAM_MULT
        out["eta"] = 0.0
        out["l0"] = 0
        out["seed_set"] = "add1"
        hdr = not os.path.exists(OUT)
        out.to_csv(OUT, mode="a", header=hdr, index=False)
        n_done += 1
        if n_done % 10 == 0 or n_done == len(pending):
            el = time.time() - t0
            rate = n_done / max(el, 1e-9)
            eta = (len(pending) - n_done) / max(rate, 1e-9)
            log("  %d/%d done (%.2f jobs/s), ETA %.1f min | last %s/%s/seed%d/%s"
                % (n_done, len(pending), rate, eta / 60, ds, m, s, jt))
    log("ALL DONE in %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
