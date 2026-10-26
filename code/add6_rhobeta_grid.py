# -*- coding: utf-8 -*-
"""add6_rhobeta_grid.py — Task 6: rho x beta 2D sensitivity grid (RESUMABLE).

Saves each (dataset,rho,beta) cell to CSV immediately; on relaunch, skips cells
already present. Robust against external kills on this shared box.
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\AL_FuzzyClustering_Paper\v2_unified\code")
import additional_common as ac
import run_experiments as re

ac.setup_style()

RHO_GRID = [0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0]
BETA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
DATASETS = ["ecoli", "compound"]
ALPHA0 = 50.0
L0 = 0
ETA = 0.0
CSV_NAME = "add6_rhobeta_grid.csv"

SMOKE = "--smoke" in sys.argv
SEEDS = ac.SEEDS_30
if SMOKE:
    SEEDS = ac.SEEDS_30[:3]

def run_one(ds, c, X, y, rho_mult, beta_mult, seed):
    re.RHO = rho_mult * 0.1
    out = re.run_traj(X, y, c, ds, "P2_Full", seed, L0, ETA, ALPHA0,
                      beta_mult, budgets=(0.10,), run_x2=False)
    return float(out[0][0]["ACC"])

def main():
    t0 = time.time()
    csv_path = ac.res_path(CSV_NAME)
    done = set()
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        for _, r in old.iterrows():
            done.add((r["dataset"], float(r["rho"]), float(r["beta"])))
        print("resume: %d cells already done" % len(done), flush=True)

    for ds in DATASETS:
        c = re.NCLASSES[ds]
        X, y = re.load_pkl(ds)
        print(f"[{ds}] n={X.shape[0]} d={X.shape[1]} c={c}", flush=True)
        for rm in RHO_GRID:
            for bm in BETA_GRID:
                if (ds, float(rm), float(bm)) in done:
                    continue
                accs = []
                for sd in SEEDS:
                    accs.append(run_one(ds, c, X, y, rm, bm, sd))
                accs = np.array(accs)
                rec = {"dataset": ds, "rho": rm, "beta": bm,
                       "acc_mean": float(accs.mean()),
                       "acc_sd": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0}
                df = pd.DataFrame([rec])
                hdr = not os.path.exists(csv_path)
                df.to_csv(csv_path, mode="a", header=hdr, index=False)
                print(f"  rho={rm:5.2f} beta={bm:5.2f}  ACC={rec['acc_mean']:.4f} "
                      f"+/- {rec['acc_sd']:.4f}   ({time.time()-t0:.0f}s)", flush=True)
        del X, y

    df = pd.read_csv(csv_path)
    print("total cells:", len(df), flush=True)

    cmap = ac.gap_colormap()
    rho_present = sorted(df["rho"].unique())
    beta_present = sorted(df["beta"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    vmin = float(df["acc_mean"].min()); vmax = float(df["acc_mean"].max())
    im = None
    for ax, ds in zip(axes, DATASETS):
        sub = df[df.dataset == ds]
        M = np.zeros((len(rho_present), len(beta_present)))
        for i, rm in enumerate(rho_present):
            for j, bm in enumerate(beta_present):
                row = sub[(sub.rho == rm) & (sub.beta == bm)]["acc_mean"]
                M[i, j] = float(row.iloc[0]) if len(row) else np.nan
        im = ax.imshow(M, cmap=cmap, origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(beta_present)))
        ax.set_xticklabels([("%.2g" % b) for b in beta_present])
        ax.set_yticks(range(len(rho_present)))
        ax.set_yticklabels([("%.2g" % r) for r in rho_present])
        ax.set_xlabel(r"$\beta$ multiplier  ($\\times \\lambda_2(L_R)/n$)")
        ax.set_ylabel(r"$\rho$ multiplier  ($\\times 0.1$)")
        ax.set_title(ds, fontsize=10)
        for i in range(len(rho_present)):
            for j in range(len(beta_present)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center",
                            fontsize=7.5, color=ac.PALETTE["ink"])
        if 1.0 in rho_present and 1.0 in beta_present:
            bi = rho_present.index(1.0); bj = beta_present.index(1.0)
            ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                      edgecolor=ac.PALETTE["ink"], lw=1.6))
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="ACC @10%")
    fig.tight_layout()
    png = ac.savefig(fig, "add6_rhobeta_heatmap", combo=True)
    print("WROTE:", png, flush=True)

    for ds in DATASETS:
        sub = df[df.dataset == ds].reset_index(drop=True)
        if len(sub) == 0: continue
        ib = sub.acc_mean.idxmax(); iw = sub.acc_mean.idxmin()
        print(f"[{ds}] BEST  rho={sub.loc[ib,'rho']} beta={sub.loc[ib,'beta']} "
              f"ACC={sub.loc[ib,'acc_mean']:.4f}", flush=True)
        print(f"[{ds}] WORST rho={sub.loc[iw,'rho']} beta={sub.loc[iw,'beta']} "
              f"ACC={sub.loc[iw,'acc_mean']:.4f}", flush=True)

    u = ac.new_unit("add6", "Rho x Beta sensitivity grid on ecoli / compound",
                    "post-hoc", "done")
    u["support"] = "mixed"
    u["protocol"] = {"datasets": DATASETS, "methods": ["P2_Full (coefficient sweep)"],
                     "n_seeds": len(SEEDS), "budgets": [0.10],
                     "factors": {"rho_grid": RHO_GRID, "beta_grid": BETA_GRID,
                                 "rho_nominal": 0.1, "beta_scale": "lambda2(L_R)/n",
                                 "baseline": "(rho=1,beta=1)=frozen P2_Full"},
                     "checkpoint": 0.10}
    u["repro_script"] = "add6_rhobeta_grid.py"
    u["csvs"] = [CSV_NAME]
    ac.save_unit(u)
    print("DONE in %.0fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
