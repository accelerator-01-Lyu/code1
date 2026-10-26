# -*- coding: utf-8 -*-
"""add4_large.py — Task 4 (DEGRADED/restart): large-scale / high-dim clustering.

Original plan died (letter N=7990 c=26 eigendecomp too slow under multi-worker
memory pressure). Restart plan:
  * letter : stratified subsample to N=2000 (~77/class) for tractability;
             full-N result right-censored.
  * segment: full N=2310 (satellite unavailable locally, substituted).
Methods:  P2_Full, P2_NoExploration, Random, Entropy.
Seeds:    ac.SEEDS_30 (30), paired by seed (same L0 for all methods).
Checkpoint: 10% labeled (one acquisition round from L0=2%).
Metrics:  ACC (Hungarian) + NMI + ARI.
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import additional_common as ac
from fcm_update import plcfcm_fit
from acquisition import select_batch, cosine_redundancy
from gain import entropy

ac.setup_style()

METHODS = ["P2_Full", "P2_NoExploration", "Random", "Entropy"]
ALPHA0 = ac.ALPHA0_MAIN
M_MC = ac.M_MC
RHO = ac.RHO
CHECKPOINT = ac.CHECKPOINT
LETTER_SUBSAMPLE = 2000   # stratified N cap for tractability


def load_pkl(name):
    o = pickle.load(open(os.path.join(ac.REF_DATA, name + ".pkl"), "rb"))
    return np.asarray(o["X"], float), np.asarray(o["y"], int)


def stratified_subsample(X, y, target_n, seed=0):
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    per = target_n // len(classes)
    idx = []
    for k in classes:
        ki = np.where(y == k)[0]
        take = min(per, len(ki))
        idx.append(rng.choice(ki, size=take, replace=False))
    idx = np.sort(np.concatenate(idx))
    return X[idx], y[idx]


def lambda2_over_n(U):
    n = U.shape[0]
    R = cosine_redundancy(U)
    d = R.sum(1)
    L = -R.copy()
    np.fill_diagonal(L, d)
    ev = np.linalg.eigvalsh(L)
    lam2 = float(np.sort(ev)[1]) if len(ev) > 1 else 1.0
    return lam2 / max(n, 1)


def baseline_pool_scores(U, mode):
    if mode == "Entropy":
        return entropy(np.clip(U, 1e-12, 1.0))
    return np.zeros(U.shape[0])


def run_one(X, y, c, method, seed, labeled0, beta):
    n = X.shape[0]
    rng = np.random.RandomState((seed * 7919 + 13) & 0x7FFFFFFF)
    labeled = labeled0.copy()
    U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                         labels=y[labeled], seed=seed)
    target = int(round(CHECKPOINT * n))
    b = target - int(labeled.sum())
    if b > 0:
        unlab = np.where(~labeled)[0]
        if method in ("P2_Full", "P2_NoExploration"):
            sel_local = select_batch(
                U[unlab], X[unlab], V, method, b,
                alpha_0=ALPHA0, M_MC=M_MC, rho_expl=RHO,
                lam_red=beta, seed=seed, u_all=U)
            sel = unlab[sel_local]
        elif method == "Random":
            sel = unlab[rng.choice(len(unlab), size=b, replace=False)]
        else:
            scores = baseline_pool_scores(U[unlab], method)
            sel = unlab[np.argsort(-scores)[:b]]
        labeled[sel] = True
    U1, V1, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                           labels=y[labeled], seed=seed)
    return ac.cluster_metrics(y, U1.argmax(1))


# (name, n_classes, subsample_target)
DATASETS = [("segment", 7, None), ("letter", 26, LETTER_SUBSAMPLE)]


def main():
    t0 = time.time()
    long_rows = []
    for ds, c, subs in DATASETS:
        X, y = load_pkl(ds)
        if subs is not None:
            X, y = stratified_subsample(X, y, subs, seed=0)
        n = X.shape[0]
        n_init = max(c, int(round(ac.INIT_FRAC * n)))
        print("[add4] dataset=%s n=%d d=%d c=%d L0=%d" % (ds, n, X.shape[1], c, n_init), flush=True)
        for si, seed in enumerate(ac.SEEDS_30):
            rng = np.random.RandomState((seed * 1000 + n_init) & 0x7FFFFFFF)
            labeled0 = np.zeros(n, bool)
            labeled0[rng.choice(n, size=n_init, replace=False)] = True
            U0, V0, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled0)[0],
                                   labels=y[labeled0], seed=seed)
            beta = float(ac.LAM_MULT_MAIN) * lambda2_over_n(U0)
            for method in METHODS:
                m = run_one(X, y, c, method, seed, labeled0, beta)
                for metric, val in m.items():
                    long_rows.append({"dataset": ds, "method": method,
                                      "metric": metric, "seed": seed,
                                      "value": float(val)})
            if (si + 1) % 5 == 0:
                print("[add4] %s seed %d/30 done (%.0fs)" % (ds, si + 1, time.time() - t0), flush=True)

    long_df = pd.DataFrame(long_rows)
    csv_long = ac.save_csv(long_df, "add4_large_paired.csv")

    sum_rows = []
    for (ds, metric), sub in long_df.groupby(["dataset", "metric"]):
        wide = sub.pivot(index="seed", columns="method", values="value")
        rows = ac.contrast_table(wide.to_dict(orient="list"),
                                 reference="P2_Full", methods=METHODS, metric=metric)
        for method in METHODS:
            vals = wide[method].values
            m = vals.mean(); sd = vals.std(ddof=1)
            se = sd / np.sqrt(len(vals)); tc = ac.stats.t.ppf(0.975, len(vals) - 1)
            dz = 0.0; p = 1.0
            if method != "P2_Full":
                dz = rows[method]["dz"]; p = rows[method]["p_holm"]
            sum_rows.append({"dataset": ds, "method": method, "metric": metric,
                             "mean": float(m), "ci_lo": float(m - tc * se),
                             "ci_hi": float(m + tc * se),
                             "dz_vs_full": float(dz), "p_holm": float(p)})
    sum_df = pd.DataFrame(sum_rows)
    csv_sum = ac.save_csv(sum_df, "add4_large_summary.csv")

    acc = sum_df[sum_df.metric == "ACC"].copy()
    ds_list = [d for d, _, _ in DATASETS]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(ds_list)); width = 0.18
    colors = {"P2_Full": ac.C_FULL, "P2_NoExploration": ac.C_NOEXPLO,
              "Random": ac.PALETTE["mid_gray"], "Entropy": ac.PALETTE["mid_teal"]}
    for k, method in enumerate(METHODS):
        means = [acc[(acc.dataset == ds) & (acc.method == method)]["mean"].iloc[0] for ds in ds_list]
        lo = [acc[(acc.dataset == ds) & (acc.method == method)]["ci_lo"].iloc[0] for ds in ds_list]
        hi = [acc[(acc.dataset == ds) & (acc.method == method)]["ci_hi"].iloc[0] for ds in ds_list]
        err = [[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]]
        ax.bar(x + (k - 1.5) * width, means, width, label=method,
               color=colors[method], yerr=err, capsize=3,
               error_kw=dict(ecolor=ac.PALETTE["ink"], lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in ds_list])
    ax.set_ylabel("Clustering ACC @10% (Hungarian)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", ncol=2)
    ac.style_ax(ax)
    png = ac.savefig(fig, "add4_large_bar", combo=True)

    u = ac.new_unit("add4",
        "Large-scale clustering: letter (subsampled 26 classes) + segment",
        "supportive/post-hoc/underpowered", "done")
    u["annotations"] = ["supportive", "post-hoc", "underpowered",
        "satellite unavailable locally, substituted by segment",
        "letter stratified-subsampled to N=2000 (~77/class); full-N=7990 result right-censored",
        "segment N=2310 d=19 c=7; letter N=2000 d=16 c=26",
        "30 paired seeds; frozen alpha0=50, rho=0.1, M_MC=50"]
    u["protocol"] = {"datasets": ds_list, "methods": METHODS,
                     "n_seeds": 30, "budgets": [0.10],
                     "factors": {"L0_frac": ac.INIT_FRAC, "alpha0": ALPHA0,
                                 "M_MC": M_MC, "rho": RHO,
                                 "letter_subsample": LETTER_SUBSAMPLE},
                     "checkpoint": 0.10}
    u["csvs"] = ["add4_large_paired.csv", "add4_large_summary.csv"]
    u["repro_script"] = "add4_large.py"
    ac.add_figure(u, "add4_large_bar",
        "Fig. Large-scale clustering ACC @10% (mean over 30 paired seeds, "
        "95% paired-t CI). x = dataset; grouped by acquisition method.")

    def gap(ds, a="P2_NoExploration"):
        sub = long_df[(long_df.dataset == ds) & (long_df.metric == "ACC")]
        w = sub.pivot(index="seed", columns="method", values="value")
        d = w[a].values - w["P2_Full"].values
        r = ac.paired_ci(d)
        return r["mean"], r["lo"], r["hi"], ac.cohen_dz(d)
    g_letter = gap("letter"); g_seg = gap("segment")
    u["results"] = (
        "At the 10% checkpoint, P2_NoExploration minus P2_Full ACC = "
        "%+.3f (95%% CI [%+.3f, %+.3f], d_z=%.2f) on the stratified letter "
        "subsample (N=2000, c=26), and %+.3f [%+.3f, %+.3f], d_z=%.2f on "
        "segment (N=2310, c=7). Bars show group means with paired-t 95%% CIs; "
        "all 30 seeds share the identical L0 for within-seed pairing. NMI/ARI "
        "accompany ACC in the paired CSV."
        % (g_letter[0], g_letter[1], g_letter[2], g_letter[3],
           g_seg[0], g_seg[1], g_seg[2], g_seg[3]))
    u["conclusion"] = (
        "Supportive/post-hoc, underpowered stress test on a high-class letter "
        "subsample and segment; no confirmatory claim. P2_Full vs "
        "P2_NoExploration separation is reported descriptively with CIs; see "
        "add4_large_summary.csv for Holm-adjusted p-values.")
    uj = ac.save_unit(u)

    print("WROTE:", csv_long)
    print("WROTE:", csv_sum)
    print("WROTE:", png)
    print("WROTE:", uj)
    print("elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
