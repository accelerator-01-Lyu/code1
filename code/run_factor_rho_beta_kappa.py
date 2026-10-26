"""run_factor_rho_beta_kappa.py — E2c: rho x beta_mult x kappa 三因子子集扫描。

rho_subset    = [0, 0.1, 0.25]
beta_subset   = [0.5, 1, 2]   (lam_mult)
kappa_subset  = [0.5, 1, 2]
固定 alpha0=50.0, budget=0.10, seeds 0..9。
规模: 12 实例 x 27 格 x 10 seed = 3,240 jobs。
输出: results/factor/rho_beta_kappa.csv
      columns: instance_id, overlap, c, d, rho, beta_mult, kappa, seed, ACC, NMI, ARI
"""
import os
import sys
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _factor_lib import (IDS, SEEDS, run_grid)  # noqa: E402

RES_DIR = os.path.normpath(os.path.join(HERE, "..", "results", "factor"))
os.makedirs(RES_DIR, exist_ok=True)

RHO_SUB = [0.0, 0.1, 0.25]
BETA_SUB = [0.5, 1.0, 2.0]
KAPPA_SUB = [0.5, 1.0, 2.0]
ALPHA0_FIX = 50.0

CSV_PATH = os.path.join(RES_DIR, "rho_beta_kappa.csv")
DONE_PATH = os.path.join(RES_DIR, "done_rho_beta_kappa.txt")
LOG_PATH = os.path.join(RES_DIR, "factor_run.log")


def key_of(job):
    iid, seed, rho, kappa, alpha0, lam_mult = job
    return f"{iid}/{rho}/{lam_mult}/{kappa}/{seed}"


def row_of(job, meta, m):
    iid, seed, rho, kappa, alpha0, lam_mult = job
    return {
        "instance_id": iid,
        "overlap": meta["overlap_target"],
        "c": meta["c"],
        "d": meta["d"],
        "rho": rho,
        "beta_mult": lam_mult,
        "kappa": kappa,
        "seed": seed,
        "ACC": m["ACC"],
        "NMI": m["NMI"],
        "ARI": m["ARI"],
    }


def main():
    # (iid, seed, rho, kappa, alpha0, lam_mult)
    jobs = []
    for iid, seed, rho, lam, kap in itertools.product(IDS, SEEDS, RHO_SUB,
                                                      BETA_SUB, KAPPA_SUB):
        jobs.append((iid, seed, rho, kap, ALPHA0_FIX, lam))
    run_grid(jobs, key_of, row_of, CSV_PATH, DONE_PATH, LOG_PATH,
             tag="E2c_rho_beta_kappa")


if __name__ == "__main__":
    main()
