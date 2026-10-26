# POST-HOC MECHANISM VALIDATION — not part of pre-registered P1-P4 verdicts.
"""Budget-tightness control on swissroll: budgets 2% and 3% (10% reused).
2% = L0-only baseline (0 active picks, two methods identical by construction).
3% = 12 active picks.  Writes summary_swissroll_budget.csv /
trajectory_swissroll_budget.csv.
"""
import os, sys, csv, pickle
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_synthetic_3d import run_traj_thin, load_pkl, AL_SEEDS, METHODS, K

SYN = os.path.join(HERE, "results", "synthetic_3d")
SIGMAS = [0.4, 0.6, 0.8]
BUDGETS = [0.02, 0.03]   # 0.10 reused from summary_all.csv


def job(j):
    sigma, method, budget, seed = j
    X, y = load_pkl(os.path.join(SYN, f"swissroll_sigma{sigma}.pkl"))
    n_normal = int((y >= 0).sum())
    try:
        rows, order = run_traj_thin(X, y, K, method, seed,
                                    budgets=(budget,), n_budget=n_normal,
                                    record=True)
        m = rows[-1]
        return {"ok": True, "sigma": sigma, "method": method, "budget": budget,
                "seed": seed, "acc": m["ACC"], "nmi": m["NMI"],
                "ari": m["ARI"], "order": order}
    except Exception as e:
        return {"ok": False, "sigma": sigma, "method": method, "budget": budget,
                "seed": seed, "err": str(e)}


def main():
    out_s = os.path.join(SYN, "summary_swissroll_budget.csv")
    out_t = os.path.join(SYN, "trajectory_swissroll_budget.csv")
    jobs = [(sg, m, b, sd) for sg in SIGMAS for m in METHODS
            for b in BUDGETS for sd in AL_SEEDS]
    print(f"[budget] {len(jobs)} shards", flush=True)
    from multiprocessing import Pool
    sf = open(out_s, "w", newline="")
    w_s = csv.DictWriter(sf, fieldnames=["sigma","method","budget","seed",
                                         "acc","nmi","ari"])
    w_s.writeheader()
    tf = open(out_t, "w", newline="")
    w_t = csv.DictWriter(tf, fieldnames=["sigma","method","budget","seed","round",
                                         "selected_idx","acc","nmi","ari"])
    w_t.writeheader()
    with Pool(8) as p:
        for res in p.imap_unordered(job, jobs, chunksize=1):
            if res["ok"]:
                for step, sidx in enumerate(res["order"], start=1):
                    w_t.writerow({"sigma":res["sigma"],"method":res["method"],
                                  "budget":res["budget"],"seed":res["seed"],
                                  "round":step,"selected_idx":sidx,
                                  "acc":(res["acc"] if step==len(res["order"]) else ""),
                                  "nmi":(res["nmi"] if step==len(res["order"]) else ""),
                                  "ari":(res["ari"] if step==len(res["order"]) else "")})
                w_s.writerow({"sigma":res["sigma"],"method":res["method"],
                              "budget":res["budget"],"seed":res["seed"],
                              "acc":res["acc"],"nmi":res["nmi"],"ari":res["ari"]})
            else:
                w_s.writerow({"sigma":res["sigma"],"method":res["method"],
                              "budget":res["budget"],"seed":res["seed"],
                              "acc":"NaN","nmi":"NaN","ari":"NaN"})
                print("[fail]", res, flush=True)
            sf.flush(); tf.flush()
    sf.close(); tf.close()
    print("[wrote]", out_s, flush=True)


if __name__ == "__main__":
    main()
