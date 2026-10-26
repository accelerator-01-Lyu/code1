# POST-HOC MECHANISM VALIDATION — not part of pre-registered P1-P4 verdicts.
"""make_main_posthoc.py — main-text figures: swiss-roll only.
Outputs to figures/posthoc/main/ as 300dpi PNG + PDF.
spiral/hourglass selection maps withdrawn to supplement (files kept).
"""
import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from fcm_update import plcfcm_fit

HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(HERE, "results", "synthetic_3d")
MAIN = os.path.join(HERE, "figures", "posthoc", "main")
os.makedirs(MAIN, exist_ok=True)

INK = "#3F345B"; GREY = "#B8B8B8"
FULL_C = "#D08AB6"; NOEXP_C = "#7FA4C8"
NOTE = "Post-hoc Mechanism Validation — not part of pre-registered P1-P4"
CAPTION = ("Warm-pink x: P2_Full; blue o: P2_NoExploration; "
           "black *: eccentric decoys. Post-hoc colors differ from main-text "
           "method identity colors. Spiral (near-inert) and hourglass "
           "(negative-control geometry) full results are reported in "
           "Supplementary Table X; all 180 shards and nine datasets are "
           "released in the repository.")


def load(kind, sigma):
    o = pickle.load(open(os.path.join(SYN, f"{kind}_sigma{sigma}.pkl"), "rb"))
    return o["X"].astype(float), o["y"].astype(int)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, colors=INK)


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(MAIN, f"{name}.{ext}"), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[main]", name)


# --- main selection figure: swissroll sigma=.6 top+side ------------------- #
def main_selection():
    kind = "swissroll"; sigma = 0.6
    X, y = load(kind, sigma)
    traj = pd.read_csv(os.path.join(SYN, "trajectory_all.csv"))
    sub = traj[(traj.dataset == kind) & (np.isclose(traj.sigma, sigma))]
    sel_f = sub[sub.method == "P2_Full"].selected_idx.values
    sel_ne = sub[sub.method == "P2_NoExploration"].selected_idx.values
    ecc = y == -1
    fig = plt.figure(figsize=(180/25.4, 80/25.4))
    for j, (elev, azim, ttl) in enumerate([(18, -60, "top view"),
                                            (8, 30, "side view")]):
        ax = fig.add_subplot(1, 2, j+1, projection="3d")
        ax.scatter(X[~ecc, 0], X[~ecc, 1], X[~ecc, 2], s=3, c=GREY,
                   alpha=0.3, edgecolors="none")
        ax.scatter(X[sel_f, 0], X[sel_f, 1], X[sel_f, 2], marker="x",
                   c=FULL_C, s=14, linewidths=0.8)
        ax.scatter(X[sel_ne, 0], X[sel_ne, 1], X[sel_ne, 2], marker="o",
                   facecolors="none", edgecolors=NOEXP_C, s=8)
        ax.scatter(X[ecc, 0], X[ecc, 1], X[ecc, 2], marker="*", c="black", s=40)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(ttl, fontsize=8, color=INK, pad=-12)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        leg = [Line2D([0],[0],marker="x",color=FULL_C,ls="None",ms=7,label="P2_Full"),
               Line2D([0],[0],marker="o",color="none",markerfacecolor="none",
                      markeredgecolor=NOEXP_C,ms=6,label="P2_NoExploration"),
               Line2D([0],[0],marker="*",color="black",ls="None",ms=9,
                      label="eccentric decoy")]
        ax.legend(handles=leg, fontsize=6, loc="upper left", frameon=False,
                  labelcolor=INK)
    fig.suptitle("Swiss-roll (sigma=0.6) — selected points, 10 seeds",
                 fontsize=11, color=INK, weight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.13, 1, 0.93])
    fig.text(0.5, 0.075, CAPTION, ha="center", fontsize=6.5, color=INK)
    fig.text(0.01, 0.015, NOTE, fontsize=6, color=INK, alpha=0.8, ha="left")
    _save(fig, "main_3d_swissroll_selection")


# --- main overlap: swissroll single line, 3 sigmas ------------------------ #
def main_overlap():
    summ = pd.read_csv(os.path.join(SYN, "summary_all.csv"))
    sigmas = [0.4, 0.6, 0.8]
    xs, mu, lo, hi = [], [], [], []
    for s in sigmas:
        f = summ[(summ.dataset=="swissroll") & np.isclose(summ.sigma,s) &
                 (summ.method=="P2_Full")].sort_values("seed")
        ne = summ[(summ.dataset=="swissroll") & np.isclose(summ.sigma,s) &
                  (summ.method=="P2_NoExploration")].sort_values("seed")
        m = f.merge(ne, on="seed", suffixes=("_f","_ne"))
        d = m.acc_ne.values - m.acc_f.values
        h = stats.t.ppf(0.975, len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
        xs.append(s); mu.append(d.mean()); lo.append(d.mean()-h); hi.append(d.mean()+h)
    fig, ax = plt.subplots(figsize=(88/25.4, 60/25.4))
    ax.plot(xs, mu, color=INK, lw=1.6, marker="o", ms=5)
    ax.fill_between(xs, lo, hi, color=FULL_C, alpha=0.18, lw=0)
    ax.axhline(0, color=INK, lw=0.8, ls="--")
    for x, v in zip(xs, mu):
        ax.annotate(f"{v*100:+.2f}pp", (x, v), textcoords="offset points",
                    xytext=(0, 8), fontsize=6, color=INK, ha="center")
    ax.set_xlabel("sigma", fontsize=8, color=INK)
    ax.set_ylabel("ACC(NoExplo) - ACC(Full)", fontsize=8, color=INK)
    ax.set_xticks(sigmas)
    _style(ax)
    fig.suptitle("Swiss-roll paired ACC gap, 95% CI, n=10",
                 fontsize=9.5, color=INK, weight="bold")
    fig.tight_layout(rect=[0,0.08,1,0.93])
    fig.text(0.5, 0.03, CAPTION, ha="center", fontsize=5.8, color=INK)
    fig.text(0.01, 0.01, NOTE, fontsize=6, color=INK, alpha=0.8, ha="left")
    _save(fig, "main_3d_swissroll_overlap")


# --- main entropy-eccentricity: swissroll panel only ---------------------- #
def main_ee():
    kind = "swissroll"; sigma = 0.6
    X, y = load(kind, sigma)
    U, _, _ = plcfcm_fit(X, c=6, labeled_idx=None, labels=None, seed=20261010)
    p = np.clip(U, 1e-12, 1.0)
    H = -(p*np.log(p)).sum(1); ub = U.mean(0)
    h = np.abs(U-ub).sum(1)
    traj = pd.read_csv(os.path.join(SYN, "trajectory_all.csv"))
    sub = traj[(traj.dataset==kind) & np.isclose(traj.sigma,sigma)]
    sf = set(sub[sub.method=="P2_Full"].selected_idx)
    sn = set(sub[sub.method=="P2_NoExploration"].selected_idx)
    ecc = y == -1
    other = ~(ecc | np.isin(np.arange(len(y)), list(sf)) |
              np.isin(np.arange(len(y)), list(sn)))
    fig, ax = plt.subplots(figsize=(88/25.4, 70/25.4))
    ax.scatter(H[other], h[other], s=5, c=GREY, alpha=0.4, edgecolors="none")
    ax.scatter(H[list(sf)], h[list(sf)], s=16, marker="x", c=FULL_C,
               label="P2_Full selected", linewidths=0.8)
    ax.scatter(H[list(sn)], h[list(sn)], s=14, marker="o", facecolors="none",
               edgecolors=NOEXP_C, label="NoExplo selected")
    ax.scatter(H[ecc], h[ecc], marker="*", s=55, c="black", label="eccentric")
    ax.set_xlabel("H(u_i)", fontsize=8, color=INK)
    ax.set_ylabel(r"eccentricity $h_i=\|u_i-\bar u\|_1$", fontsize=8, color=INK)
    ax.legend(fontsize=6, frameon=False, labelcolor=INK, loc="best")
    _style(ax)
    fig.suptitle("Swiss-roll selection map on (entropy, eccentricity)",
                 fontsize=9, color=INK, weight="bold")
    fig.tight_layout(rect=[0,0.12,1,0.92])
    fig.text(0.5, 0.05, CAPTION, ha="center", fontsize=5.8, color=INK)
    fig.text(0.01, 0.01, NOTE, fontsize=6, color=INK, alpha=0.8, ha="left")
    _save(fig, "main_3d_swissroll_entropy_eccentricity")


# --- main cosine heatmap: swissroll single column ------------------------ #
def main_cosine():
    kind = "swissroll"; sigma = 0.6
    X, y = load(kind, sigma)
    U, _, _ = plcfcm_fit(X, c=6, labeled_idx=None, labels=None, seed=20261010)
    order = np.argsort(y)
    U = U[order]
    R = U @ U.T
    R = R / (np.linalg.norm(U, axis=1, keepdims=True) @
             np.linalg.norm(U, axis=1, keepdims=True).T)
    fig, ax = plt.subplots(figsize=(88/25.4, 80/25.4))
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("p", ["#7FA4C8","#6A5A8F","#D08AB6"])
    im = ax.imshow(R, cmap=cm, vmin=0, vmax=1)
    ax.set_title("Swiss-roll cosine R=UU$^\\top$ (label-sorted)",
                 fontsize=9, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6, colors=INK)
    fig.text(0.01, 0.01, NOTE, fontsize=6, color=INK, alpha=0.8, ha="left")
    _save(fig, "main_3d_swissroll_cosine")


if __name__ == "__main__":
    main_selection(); main_overlap(); main_ee(); main_cosine()
