"""run_factor_kappa_alpha.py — E2b: kappa x alpha 全因子扫描。

kappa_levels = [0.25, 0.5, 1.0, 2.0]   (KL 模糊集半径, calculate_gain(kappa=...))
alpha_levels = [1, 10, 50, 100]       (reference-law Dirichlet 浓度 alpha0)
固定 rho=0.1, lam_mult=1.0, budget=0.10, seeds 0..9。
规模: 12 实例 x 16 格 x 10 seed = 1,920 jobs。
输出: results/factor/kappa_alpha.csv
      columns: instance_id, overlap, c, d, kappa, alpha, seed, ACC, NMI, ARI
"""
import os
import sys
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _factor_lib import (IDS, SEEDS, run_grid)  # noqa: E402

RES_DIR = os.path.normpath(os.path.join(HERE, "..", "results", "factor"))
os.makedirs(RES_DIR, exist_ok=True)

KAPPA_LEVELS = [0.25, 0.5, 1.0, 2.0]
ALPHA_LEVELS = [1.0, 10.0, 50.0, 100.0]
RHO_FIX = 0.1
LAM_FIX = 1.0

CSV_PATH = os.path.join(RES_DIR, "kappa_alpha.csv")
DONE_PATH = os.path.join(RES_DIR, "done_kappa_alpha.txt")
LOG_PATH = os.path.join(RES_DIR, "factor_run.log")


def key_of(job):
    iid, seed, rho, kappa, alpha0, lam_mult = job
    return f"{iid}/{kappa}/{alpha0}/{seed}"


def row_of(job, meta, m):
    iid, seed, rho, kappa, alpha0, lam_mult = job
    return {
        "instance_id": iid,
        "overlap": meta["overlap_target"],
        "c": meta["c"],
        "d": meta["d"],
        "kappa": kappa,
        "alpha": alpha0,
        "seed": seed,
        "ACC": m["ACC"],
        "NMI": m["NMI"],
        "ARI": m["ARI"],
    }


def main():
    jobs = list(itertools.product(IDS, SEEDS, [RHO_FIX], KAPPA_LEVELS,
                                  ALPHA_LEVELS, [LAM_FIX]))
    assert jobs[0][3] in KAPPA_LEVELS and jobs[0][4] in ALPHA_LEVELS
    run_grid(jobs, key_of, row_of, CSV_PATH, DONE_PATH, LOG_PATH,
             tag="E2b_kappa_alpha")


if __name__ == "__main__":
    main()
