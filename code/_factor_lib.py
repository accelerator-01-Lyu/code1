"""_factor_lib.py — E2 多因子扫描共享机制（rho/beta x kappa/alpha x 三因子）。

自写轨迹 run_traj_factor 与 run_experiments.run_traj 的 P2_Full 路径逐行一致，
唯一区别：
  1. rho / kappa / alpha0 / lam_mult(beta_mult) 全部参数化（run_traj 把 RHO 固定为
     模块常量 0.1，且 select_batch 不显式传 kappa -> 内部默认 1.0）；
  2. budgets=(0.10,) 单检查点（E2 设计：预算 10%，单批次 target-labeled.sum()，
     一次 FCM refit）；
  3. l0=0, eta=0, run_x2=False；纯观测性的 outlier/eccentric 诊断与 x2 诊断
     不反馈进 U/V/labeled（已逐行核对 run_traj：om/ecc 只计数，不参与 select），
     故省略——数值轨迹完全相同。

初始 FCM 缓存：L0 上的 plcfcm_fit 只依赖 (instance_id, seed)，与因子无关。
优先只读复用 E1 的 results/synthetic/init_cache（bit 级一致，因为 plcfcm_fit 确定
性）；缺失时自行计算并写入 results/factor/init_cache。results/synthetic 保持只读。
"""
import os
# 线程数固定为 1：20 核机器上 E1 已占 12 worker，BLAS 默认每进程 20 线程会导致
# 300 线程互相踩踏（CPU 仅 35% 却很慢）。必须在 import numpy 之前设置。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import time
import pickle
import threading
import multiprocessing as mp

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_experiments import metrics, _lambda2_over_n, INIT_FRAC, M_MC  # noqa: E402
from fcm_update import plcfcm_fit as _real_plcfcm_fit                  # noqa: E402
from acquisition import select_batch                                    # noqa: E402

ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "synthetic")
RES_DIR = os.path.join(ROOT, "results", "factor")
E1_CACHE = os.path.join(ROOT, "results", "synthetic", "init_cache")
MY_CACHE = os.path.join(RES_DIR, "init_cache")
os.makedirs(MY_CACHE, exist_ok=True)

IDS = [1, 3, 5, 19, 21, 23, 37, 39, 41, 55, 57, 59]
SEEDS = list(range(10))
BUDGET = 0.10
PROGRESS_EVERY = 300

_JOB_CTX = {"iid": None, "seed": None, "init_done": False}


def _cached_plcfcm_fit(X, c, labeled_idx=None, labels=None, seed=0, **kw):
    """L0 首次拟合按 (iid, seed) 读缓存；之后的 refit 直通真实实现。"""
    if _JOB_CTX["iid"] is not None and not _JOB_CTX["init_done"]:
        iid = _JOB_CTX["iid"]
        for base in (MY_CACHE, E1_CACHE):
            p = os.path.join(base, f"init_{iid:03d}_seed{seed}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    U0, V0 = pickle.load(f)
                _JOB_CTX["init_done"] = True
                return U0.copy(), V0.copy(), 0
        U, V, it = _real_plcfcm_fit(X, c=c, labeled_idx=labeled_idx,
                                    labels=labels, seed=seed, **kw)
        outp = os.path.join(MY_CACHE, f"init_{iid:03d}_seed{seed}.pkl")
        tmp = outp + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump((U.copy(), V.copy()), f)
        os.replace(tmp, outp)
        _JOB_CTX["init_done"] = True
        return U, V, it
    return _real_plcfcm_fit(X, c=c, labeled_idx=labeled_idx, labels=labels,
                            seed=seed, **kw)


plcfcm_fit = _cached_plcfcm_fit


def run_traj_factor(X, y, c, seed, rho, kappa, alpha0, lam_mult, budget=BUDGET):
    """参数化轨迹（与 run_traj 的 P2_Full 路径逐行一致，除上述三点）。"""
    rng = np.random.RandomState((seed * 1000 + 0) & 0x7FFFFFFF)
    n = X.shape[0]
    yt = y.copy()
    n_init = max(c, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, bool)
    labeled[rng.choice(n, size=n_init, replace=False)] = True

    U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                         labels=yt[labeled], seed=seed)

    target = int(round(budget * n))
    if labeled.sum() >= target:
        return metrics(yt, U)
    b = target - int(labeled.sum())
    unlab = np.where(~labeled)[0]
    beta = lam_mult * _lambda2_over_n(U)
    sel = select_batch(U[unlab], X[unlab], V, "P2_Full", b,
                        alpha_0=alpha0, M_MC=M_MC, kappa=kappa,
                        rho_expl=rho, lam_red=beta, seed=seed, u_all=U)
    sel = unlab[sel]
    labeled[sel] = True
    U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                         labels=yt[labeled], seed=seed)
    return metrics(yt, U)


def load_instance(iid):
    dpath = os.path.join(DATA_DIR, f"instance_{iid:03d}")
    X = np.load(os.path.join(dpath, "X.npy"))
    y = np.load(os.path.join(dpath, "y_true.npy"))
    meta = json.load(open(os.path.join(dpath, "meta.json"), encoding="utf-8"))
    return X, y, meta


# --------------------------------------------------------------------------- #
# 内存纪律：动态 worker
# --------------------------------------------------------------------------- #
def free_mb():
    try:
        return psutil.virtual_memory().available / (1024 ** 2)
    except Exception:
        return 9999.0


def e1_done_count():
    p = os.path.join(ROOT, "results", "synthetic", "done.txt")
    if not os.path.exists(p):
        return 0
    try:
        with open(p, encoding="utf-8-sig", errors="ignore") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def pick_workers():
    """0=暂停；1..6 个 worker（组织者许可上限 6）。"""
    f = free_mb()
    if f < 800:
        return 0
    if f < 1000:
        return 1
    if f < 1500:
        return 2
    if f < 2500:
        return 3
    if f < 3500:
        return 4
    return 6


def log_line(msg, log_path):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_grid(jobs, key_of, row_of, csv_path, done_path, log_path, tag,
             chunk=150):
    """幂等跑批。jobs: tuple 列表；row_of(job, meta, metrics) -> 行 dict。"""
    import csv as _csv
    done = set()
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8-sig", errors="ignore") as f:
            done = set(f.read().split())
    pending = [j for j in jobs if key_of(j) not in done]
    total = len(pending)
    log_line(f"[{tag}] grid={len(jobs)} done={len(done)} pending={total} "
             f"free_mem={free_mb():.0f}MB e1_done={e1_done_count()}", log_path)
    if total == 0:
        return

    cols = list(row_of(pending[0],
                       {"overlap_target": 0.0, "c": 1, "d": 1, "N": 1},
                       {"ACC": 0.0, "NMI": 0.0, "ARI": 0.0}).keys())
    write_header = not os.path.exists(csv_path)
    f_csv = open(csv_path, "a", encoding="utf-8", newline="")
    f_done = open(done_path, "a", encoding="utf-8")
    writer = _csv.DictWriter(f_csv, fieldnames=cols)
    if write_header:
        writer.writeheader()
        f_csv.flush()

    lock = threading.Lock()
    state = {"n": 0}
    t0 = time.time()

    def handle(job, meta, m):
        row = row_of(job, meta, m)
        with lock:
            writer.writerow(row)
            f_csv.flush()
            f_done.write(key_of(job) + "\n")
            f_done.flush()
            state["n"] += 1
            n = state["n"]
            if n % PROGRESS_EVERY == 0 or n == total:
                el = time.time() - t0
                rate = n / max(el, 1.0)
                eta = (total - n) / max(rate, 1e-9)
                log_line(f"[{tag}] {n}/{total}  {rate:.3f} jobs/s  "
                         f"elapsed={el/60:.1f}min ETA={eta/60:.1f}min "
                         f"free_mem={free_mb():.0f}MB", log_path)

    i = 0
    while i < len(pending):
        w = pick_workers()
        if w == 0:
            log_line(f"[{tag}] free_mem={free_mb():.0f}MB < 800MB, pause 60s",
                     log_path)
            time.sleep(60)
            continue
        group = pending[i:i + chunk]
        pool = mp.Pool(w)
        try:
            for res in pool.imap_unordered(_worker, group):
                handle(*res)
        finally:
            pool.close()
            pool.join()
        i += len(group)

    f_csv.close()
    f_done.close()
    log_line(f"[{tag}] ALL DONE: {total} jobs in "
             f"{(time.time()-t0)/60:.1f} min", log_path)


def _worker(job):
    """顶层可 pickle worker。job 结构由各 runner 约定，返回 (job, meta, metrics)。"""
    iid, seed, rho, kappa, alpha0, lam_mult = job
    _JOB_CTX["iid"] = iid
    _JOB_CTX["seed"] = seed
    _JOB_CTX["init_done"] = False
    X, y, meta = load_instance(iid)
    m = run_traj_factor(X, y, meta["c"], seed, rho, kappa, alpha0, lam_mult)
    import gc
    del X, y
    gc.collect()
    return job, meta, m


if __name__ == "__main__":
    # 冒烟：与冻结 run_traj 默认参数逐行一致性校验
    from run_experiments import run_traj
    iid, seed = 1, 0
    X, y, meta = load_instance(iid)
    c = meta["c"]
    mine = run_traj_factor(X, y, c, seed, rho=0.1, kappa=1.0, alpha0=50.0,
                           lam_mult=1.0)
    ref_rows, *_ = run_traj(X, y, c, f"syn{iid:03d}", "P2_Full", seed, l0=0,
                            eta=0.0, alpha0=50.0, lam_mult=1.0,
                            budgets=(BUDGET,), run_x2=False)
    ref = ref_rows[0]
    print("mine      :", mine)
    print("run_traj  :", ref)
    for k in ("ACC", "NMI", "ARI"):
        assert abs(mine[k] - ref[k]) < 1e-9, f"MISMATCH {k}: {mine[k]} vs {ref[k]}"
    print("[factor_lib smoke] IDENTITY OK")
