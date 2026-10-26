"""run_scoreonly.py — P2_ScoreOnly ablation driver (rho=0, beta=0).

Runs ONLY ds in {ecoli, glass} x seeds x 10% budget checkpoint.
Writes one shard CSV per (ds, seed) to code/results/scoreonly/, column layout
identical to the main shards. Crash-safe: existing target CSV is skipped.

Usage:
  python run_scoreonly.py --seeds 0-4  --datasets ecoli          # direction phase
  python run_scoreonly.py --seeds 0-29 --datasets ecoli,glass     # full phase
"""
import os, sys, time, argparse
# Pin BLAS to single thread per worker: 14 worker procs x default OpenBLAS
# thread pools oversubscribe the box (memory-allocation failures). Shards are
# tiny (n<=788), so per-worker BLAS threading buys nothing.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import multiprocessing as mp
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))  # code/  (this file lives in code/results/scoreonly/)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from run_experiments import run_traj, load_pkl, NCLASSES  # noqa: E402

OUTDIR = HERE
os.makedirs(OUTDIR, exist_ok=True)

COLS = ["budget", "NMI", "ARI", "ACC", "dataset", "method", "seed_set", "seed",
        "l0", "eta", "alpha0", "lam_mult", "outlier_rate", "outlier_rate_pool",
        "eccentric_rate"]


def _worker(job):
    """Top-level picklable worker. job = (ds, seed)."""
    ds, seed = job
    X, y = load_pkl(ds)
    c = NCLASSES[ds]
    rows, of, ot, ofp, otp, diag, ecc, ecct, x2_rec = run_traj(
        X, y, c, ds, "P2_ScoreOnly", seed, 0, 0.0, 50.0, 1.0,
        budgets=(0.10,), run_x2=False)
    import gc
    del X, y
    gc.collect()
    return {"ds": ds, "seed": seed, "rows": rows, "of": of, "ot": ot,
            "ofp": ofp, "otp": otp, "ecc": ecc, "ecct": ecct}


def _csv_path(ds, seed):
    return os.path.join(OUTDIR,
                        f"{ds}_P2_ScoreOnly_30_{seed}_0_0.0_50.0_1.0.csv")


def _handle(res):
    ds, seed = res["ds"], res["seed"]
    df = pd.DataFrame(res["rows"])
    df["dataset"] = ds
    df["method"] = "P2_ScoreOnly"
    df["seed_set"] = "30"
    df["seed"] = seed
    df["l0"] = 0
    df["eta"] = 0.0
    df["alpha0"] = 50.0
    df["lam_mult"] = 1.0
    df["outlier_rate"] = (res["of"] / res["ot"]) if res["ot"] else np.nan
    df["outlier_rate_pool"] = (res["ofp"] / res["otp"]) if res["otp"] else np.nan
    df["eccentric_rate"] = (res["ecc"] / res["ecct"]) if res["ecct"] else np.nan
    df = df[COLS]
    p = _csv_path(ds, seed)
    tmp = p + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="e.g. 0-4 or 0-29")
    ap.add_argument("--datasets", default="ecoli,glass")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]

    jobs = []
    skipped = 0
    for ds in datasets:
        for seed in seeds:
            if os.path.exists(_csv_path(ds, seed)):
                skipped += 1
                continue
            jobs.append((ds, seed))

    n_workers = max(1, int(0.7 * os.cpu_count()))
    print(f"[scoreonly] datasets={datasets} seeds={lo}..{hi} "
          f"pending={len(jobs)} skipped_existing={skipped} workers={n_workers} "
          f"(cpu_count={os.cpu_count()})", flush=True)
    if not jobs:
        print("[scoreonly] nothing to do", flush=True)
        return

    t0 = time.time()
    done = 0
    fails = []
    if n_workers <= 1:
        for job in jobs:
            try:
                _handle(_worker(job))
            except Exception as e:  # noqa: BLE001
                fails.append((job, repr(e)))
            done += 1
            if done % 5 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  [{el:7.0f}s] {done}/{len(jobs)} "
                      f"({done/max(el,1):.2f} sh/s)", flush=True)
    else:
        with mp.Pool(n_workers) as pool:
            for res in pool.imap_unordered(_worker, jobs):
                try:
                    _handle(res)
                except Exception as e:  # noqa: BLE001
                    fails.append(((res["ds"], res["seed"]), repr(e)))
                done += 1
                if done % 5 == 0 or done == len(jobs):
                    el = time.time() - t0
                    print(f"  [{el:7.0f}s] {done}/{len(jobs)} "
                          f"({done/max(el,1):.2f} sh/s)", flush=True)
    el = time.time() - t0
    print(f"[scoreonly] DONE in {el:.0f}s ({el/60:.1f} min). fails={fails}",
          flush=True)


if __name__ == "__main__":
    main()
