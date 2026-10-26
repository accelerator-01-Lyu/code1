"""run_factor_rho_beta.py — E2a: rho x beta_mult 全因子扫描。

rho_levels   = [0, 0.05, 0.1, 0.25, 0.5]      (探索奖励权重 rho_expl)
beta_mult    = [0.25, 0.5, 1, 2, 4]           (lam_mult; beta = lam_mult * lambda2/n)
固定 kappa=1.0, alpha0=50.0, budget=0.10, seeds 0..9。
规模: 12 实例 x 25 格 x 10 seed = 3,000 jobs。
输出: results/factor/rho_beta.csv
      columns: instance_id, overlap, c, d, rho, beta_mult, seed, ACC, NMI, ARI
"""
import os
import sys
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _factor_lib import (IDS, SEEDS, run_grid)  # noqa: E402

RES_DIR = os.path.join(HERE, "..", "results", "factor")
RES_DIR = os.path.normpath(RES_DIR)
os.makedirs(RES_DIR, exist_ok=True)

RHO_LEVELS = [0.0, 0.05, 0.1, 0.25, 0.5]
BETA_LEVELS = [0.25, 0.5, 1.0, 2.0, 4.0]
KAPPA_FIX = 1.0
ALPHA0_FIX = 50.0

CSV_PATH = os.path.join(RES_DIR, "rho_beta.csv")
DONE_PATH = os.path.join(RES_DIR, "done_rho_beta.txt")
LOG_PATH = os.path.join(RES_DIR, "factor_run.log")


def key_of(job):
    iid, seed, rho, kappa, alpha0, lam_mult = job
    return f"{iid}/{rho}/{lam_mult}/{seed}"


def row_of(job, meta, m):
    iid, seed, rho, kappa, alpha0, lam_mult = job
    return {
        "instance_id": iid,
        "overlap": meta["overlap_target"],
        "c": meta["c"],
        "d": meta["d"],
        "rho": rho,
        "beta_mult": lam_mult,
        "seed": seed,
        "ACC": m["ACC"],
        "NMI": m["NMI"],
        "ARI": m["ARI"],
    }


def main():
    jobs = list(itertools.product(IDS, SEEDS, RHO_LEVELS, [KAPPA_FIX],
                                  [ALPHA0_FIX], BETA_LEVELS))
    # job 元组顺序必须与 _worker 解包一致: (iid, seed, rho, kappa, alpha0, lam_mult)
    assert jobs[0][2] in RHO_LEVELS and jobs[0][5] in BETA_LEVELS
    run_grid(jobs, key_of, row_of, CSV_PATH, DONE_PATH, LOG_PATH, tag="E2a_rho_beta")


if __name__ == "__main__":
    main()
