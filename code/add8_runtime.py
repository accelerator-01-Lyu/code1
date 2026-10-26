# -*- coding: utf-8 -*-
"""add8_runtime.py — Task 8: wall-clock scaling of P2_NoExploration.

Config A: fixed d=20, vary N in {500,1000,2000,5000} (2 blobs, reproducible).
Config B: fixed N=2000, vary d in {5,10,20,50}.

For each (config, d, N) and seed in 0..4, measure:
  * single_round : one greedy select_batch call over the unlabeled pool
                   (batch = 8% N, i.e. from 2% to 10% checkpoint).
  * full_cycle   : the whole AL segment 2% -> 10% (FCM init on 2% labels,
                   selection, FCM refit) wall-clock.

Outputs:
  results/additional/add8_runtime.csv
  figures/additional/add8_runtime_vs_N.png/.pdf
  figures/additional/add8_runtime_vs_d.png/.pdf
  results/additional/add8_unit.json
"""
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\AL_FuzzyClustering_Paper\v2_unified\code")
import additional_common as ac
from fcm_update import plcfcm_fit
from acquisition import select_batch
from run_experiments import _lambda2_over_n

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

ac.setup_style()

ALPHA0 = 50.0
M_MC = ac.M_MC            # 50, frozen
INIT_FRAC = ac.INIT_FRAC  # 0.02
CHECKPOINT = ac.CHECKPOINT  # 0.10
C = 2
COLOR_FULL = "#3E7AAB"    # deep fog blue (locked)
COLOR_SINGLE = "#6B96C7"  # mid blue

SEEDS = list(range(5))


def make_data(N, d, seed):
    X, y = make_blobs(n_samples=N, n_features=d, centers=2,
                      cluster_std=1.0, random_state=seed)
    X = StandardScaler().fit_transform(X)
    return X, y


def measure_once(N, d, seed):
    X, y = make_data(N, d, seed)
    n = X.shape[0]
    rng = np.random.RandomState((seed * 1000 + 0) & 0x7FFFFFFF)
    n_init = max(C, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, bool)
    labeled[rng.choice(n, size=n_init, replace=False)] = True
    target = int(round(CHECKPOINT * n))
    b = target - int(labeled.sum())

    # --- setup FCM (not part of either timed measurement) ---
    U, V, _ = plcfcm_fit(X, c=C, labeled_idx=np.where(labeled)[0],
                         labels=y[labeled], seed=seed)
    unlab = np.where(~labeled)[0]
    beta = _lambda2_over_n(U[unlab])

    # --- single selection round ---
    t0 = time.perf_counter()
    sel = select_batch(U[unlab], X[unlab], V, "P2_NoExploration", b,
                       alpha_0=ALPHA0, M_MC=M_MC, rho_expl=0.1,
                       lam_red=beta, seed=seed, u_all=U)
    t_single = time.perf_counter() - t0

    # --- full AL cycle: FCM init + selection + FCM refit (from scratch) ---
    labeled2 = np.zeros(n, bool)
    rng2 = np.random.RandomState((seed * 1000 + 0) & 0x7FFFFFFF)
    labeled2[rng2.choice(n, size=n_init, replace=False)] = True
    t0 = time.perf_counter()
    U2, V2, _ = plcfcm_fit(X, c=C, labeled_idx=np.where(labeled2)[0],
                           labels=y[labeled2], seed=seed)
    unlab2 = np.where(~labeled2)[0]
    beta2 = _lambda2_over_n(U2[unlab2])
    sel2 = select_batch(U2[unlab2], X[unlab2], V2, "P2_NoExploration", b,
                        alpha_0=ALPHA0, M_MC=M_MC, rho_expl=0.1,
                        lam_red=beta2, seed=seed, u_all=U2)
    labeled2[unlab2[np.array(sel2, dtype=int)]] = True
    U3, V3, _ = plcfcm_fit(X, c=C, labeled_idx=np.where(labeled2)[0],
                           labels=y[labeled2], seed=seed)
    t_full = time.perf_counter() - t0
    return t_single, t_full


def run_config(config, d, N):
    singles, fulls = [], []
    for sd in SEEDS:
        ts, tf = measure_once(N, d, sd)
        singles.append(ts); fulls.append(tf)
        print(f"  [{config}] N={N} d={d} seed={sd}  single={ts:.3f}s  full={tf:.3f}s",
              flush=True)
    return (float(np.mean(singles)), float(np.std(singles, ddof=1)),
            float(np.mean(fulls)), float(np.std(fulls, ddof=1)))


def main():
    rows = []
    # Config A: vary N at d=20
    dA = 20
    for N in [500, 1000, 2000, 5000]:
        sm, ss, fm, fs = run_config("A", dA, N)
        rows.append({"config": "A", "d": dA, "N": N,
                     "single_round_mean": sm, "single_round_sd": ss,
                     "full_cycle_mean": fm, "full_cycle_sd": fs})
    # Config B: vary d at N=2000
    NB = 2000
    for d in [5, 10, 20, 50]:
        sm, ss, fm, fs = run_config("B", d, NB)
        rows.append({"config": "B", "d": d, "N": NB,
                     "single_round_mean": sm, "single_round_sd": ss,
                     "full_cycle_mean": fm, "full_cycle_sd": fs})

    df = pd.DataFrame(rows)
    csv_path = ac.save_csv(df, "add8_runtime.csv")
    print("WROTE:", csv_path, flush=True)

    # ---- figure: runtime vs N (config A) ---- #
    subA = df[df.config == "A"].sort_values("N")
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.errorbar(subA["N"], subA["single_round_mean"], yerr=subA["single_round_sd"],
                color=COLOR_SINGLE, marker="o", ms=5, lw=1.6, capsize=3,
                label="Single selection round")
    ax.errorbar(subA["N"], subA["full_cycle_mean"], yerr=subA["full_cycle_sd"],
                color=COLOR_FULL, marker="s", ms=5, lw=1.8, capsize=3,
                label="Full AL cycle (2%->10%)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (pool size, d=20)")
    ax.set_ylabel("Wall-clock seconds (log scale)")
    ax.set_title("P2_NoExploration scaling vs N")
    ax.legend()
    ac.style_ax(ax)
    pngN = ac.savefig(fig, "add8_runtime_vs_N")
    print("WROTE:", pngN, flush=True)

    # ---- figure: runtime vs d (config B) ---- #
    subB = df[df.config == "B"].sort_values("d")
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.errorbar(subB["d"], subB["single_round_mean"], yerr=subB["single_round_sd"],
                color=COLOR_SINGLE, marker="o", ms=5, lw=1.6, capsize=3,
                label="Single selection round")
    ax.errorbar(subB["d"], subB["full_cycle_mean"], yerr=subB["full_cycle_sd"],
                color=COLOR_FULL, marker="s", ms=5, lw=1.8, capsize=3,
                label="Full AL cycle (2%->10%)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("d (feature dim, N=2000)")
    ax.set_ylabel("Wall-clock seconds (log scale)")
    ax.set_title("P2_NoExploration scaling vs d")
    ax.legend()
    ac.style_ax(ax)
    pngD = ac.savefig(fig, "add8_runtime_vs_d")
    print("WROTE:", pngD, flush=True)

    # ---- headline numbers ---- #
    rN5k = df[(df.config == "A") & (df.N == 5000)].iloc[0]
    rd50 = df[(df.config == "B") & (df.d == 50)].iloc[0]
    print(f"[N=5000] single={rN5k.single_round_mean:.3f}+/-{rN5k.single_round_sd:.3f}s  "
          f"full={rN5k.full_cycle_mean:.3f}+/-{rN5k.full_cycle_sd:.3f}s", flush=True)
    print(f"[d=50]   single={rd50.single_round_mean:.3f}+/-{rd50.single_round_sd:.3f}s  "
          f"full={rd50.full_cycle_mean:.3f}+/-{rd50.full_cycle_sd:.3f}s", flush=True)

    # ---- unit.json ---- #
    u = ac.new_unit("add8",
        "Runtime scaling vs N and d (P2_NoExploration)",
        "supportive", "done")
    u["protocol"] = {"datasets": ["synthetic make_blobs, c=2, StandardScaler"],
                     "methods": ["P2_NoExploration"],
                     "n_seeds": 5, "budgets": [0.02, 0.10],
                     "factors": {"config_A_N": [500, 1000, 2000, 5000], "d_A": 20,
                                 "config_B_d": [5, 10, 20, 50], "N_B": 2000},
                     "checkpoint": 0.10}
    u["repro_script"] = "add8_runtime.py"
    u["csvs"] = ["add8_runtime.csv"]
    uj = ac.save_unit(u)
    print("WROTE:", uj, flush=True)


if __name__ == "__main__":
    main()
