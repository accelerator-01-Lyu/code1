# POST-HOC MECHANISM VALIDATION — not part of pre-registered P1-P4 verdicts.
"""make_figures_posthoc.py — Figures A-D for the isolated 3-D post-hoc check.

Color mapping (task-specified for THIS post-hoc set; both colors in-palette):
    P2_Full          = #D08AB6 (warm pink, big x)
    P2_NoExploration = #7FA4C8 (blue, open circle)
    eccentric decoy  = black star
    other points     = grey
White background; top/right spines removed; ink #3F345B; 300 dpi PNG;
small "Post-hoc Mechanism Validation" note on every panel.
"""
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from fcm_update import plcfcm_fit

HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(HERE, "results", "synthetic_3d")
FIG = os.path.join(HERE, "figures", "posthoc")
os.makedirs(FIG, exist_ok=True)

INK = "#3F345B"
GREY = "#B8B8B8"
FULL_C = "#D08AB6"      # warm pink x
NOEXP_C = "#7FA4C8"     # blue circle
ECC_C = "black"
CMAP = LinearSegmentedColormap.from_list("posthoc",
                                        ["#7FA4C8", "#6A5A8F", "#D08AB6"])
NOTE = "Post-hoc Mechanism Validation — not part of pre-registered P1-P4"
MIDDLE = {"spiral": 0.6, "swissroll": 0.6, "hourglass": 0.4}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, colors=INK)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color(INK)


def _note(fig):
    fig.text(0.01, 0.01, NOTE, fontsize=6, color=INK, alpha=0.8, ha="left")


def load(kind, sigma):
    o = pickle.load(open(os.path.join(SYN, f"{kind}_sigma{sigma}.pkl"), "rb"))
    return o["X"].astype(float), o["y"].astype(int)


# --------------------------------------------------------------------------- #
# Fig C: cosine R = U U^T heatmap, points sorted by true cluster label.
# --------------------------------------------------------------------------- #
def fig_c():
    fig, axes = plt.subplots(1, 3, figsize=(180/25.4, 60/25.4))
    for ax, kind in zip(axes, ("spiral", "swissroll", "hourglass")):
        X, y = load(kind, MIDDLE[kind])
        U, _, _ = plcfcm_fit(X, c=6, labeled_idx=None, labels=None,
                             seed=20261010)
        order = np.argsort(y, kind="stable")
        Us = U[order]
        norms = np.linalg.norm(Us, axis=1, keepdims=True) + 1e-12
        R = (Us / norms) @ (Us / norms).T
        ax.imshow(R, cmap=CMAP, vmin=-0.1, vmax=1.0, aspect="equal")
        ax.set_title(kind, fontsize=9, color=INK, weight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        _style(ax)
    fig.suptitle("Cosine membership overlap $\\hat U\\hat U^\\top$, points sorted by true label",
                 fontsize=9.5, color=INK, weight="bold")
    _note(fig)
    out = os.path.join(FIG, "synthetic_3d_cosine_heatmap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[figC]", out)


# --------------------------------------------------------------------------- #
# Fig A: 3-D selection-frequency map (middle sigma), top + side views.
# --------------------------------------------------------------------------- #
def fig_a():
    import pandas as pd
    traj = pd.read_csv(os.path.join(SYN, "trajectory_all.csv"))
    for kind in ("spiral", "swissroll", "hourglass"):
        sigma = MIDDLE[kind]
        X, y = load(kind, sigma)
        ecc = y == -1
        sub = traj[(traj.dataset == kind) & (np.isclose(traj.sigma, sigma))]
        fig = plt.figure(figsize=(180/25.4, 70/25.4))
        for j, (elev, azim, ttl) in enumerate([(12, -60, "top view"),
                                               (12, 30, "side view")]):
            ax = fig.add_subplot(1, 2, j + 1, projection="3d")
            ax.scatter(X[~ecc, 0], X[~ecc, 1], X[~ecc, 2], s=3, c=GREY,
                       alpha=0.4, edgecolors="none")
            ax.scatter(X[ecc, 0], X[ecc, 1], X[ecc, 2], marker="*",
                       s=60, c=ECC_C, label="eccentric")
            for meth, c, mk, lab in [("P2_Full", FULL_C, "x", "P2_Full"),
                                     ("P2_NoExploration", NOEXP_C, "o",
                                      "P2_NoExplo")]:
                m = sub[sub.method == meth]
                cnt = m.selected_idx.value_counts()
                top = cnt.head(20)
                for idx, fr in top.items():
                    ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], marker=mk,
                               s=20 + 6*fr, c=c, edgecolors="none",
                               linewidths=0.6)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(ttl, fontsize=8, color=INK, pad=-14)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            # legend (proxy artists; colors differ from main-text identity colors)
            from matplotlib.lines import Line2D
            leg = [Line2D([0], [0], marker="x", color=FULL_C, linestyle="None",
                          markersize=7, label="P2_Full"),
                   Line2D([0], [0], marker="o", color="none",
                          markerfacecolor="none", markeredgecolor=NOEXP_C,
                          markersize=6, label="P2_NoExploration"),
                   Line2D([0], [0], marker="*", color="black", linestyle="None",
                          markersize=9, label="eccentric decoy")]
            ax.legend(handles=leg, fontsize=6, loc="upper left",
                      frameon=False, labelcolor=INK)
        tag = " (SUPPLEMENT: negative geometry control)" if kind == "hourglass" else ""
        fig.suptitle(f"{kind} (sigma={sigma}){tag} — selected points, 10 seeds",
                     fontsize=11, color=INK, weight="bold", y=0.98)
        fig.text(0.5, 0.06,
                 "Warm-pink x: P2_Full; blue o: P2_NoExploration; "
                 "black *: eccentric decoys. Post-hoc colors differ from "
                 "main-text method identity colors.",
                 ha="center", fontsize=7.5, color=INK)
        _note(fig)
        out = os.path.join(FIG, f"synthetic_3d_{kind}_selection.png")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("[figA]", out)


# --------------------------------------------------------------------------- #
# Fig B: overlap-curve, x=sigma, y=ACC(NoExplo)-ACC(Full), 95% CI band.
# --------------------------------------------------------------------------- #
def fig_b():
    import pandas as pd
    summ = pd.read_csv(os.path.join(SYN, "summary_all.csv"))
    summ = summ[summ.acc != "NaN"].copy()
    summ["acc"] = summ.acc.astype(float)
    sigmas = {"spiral": [0.4, 0.6, 0.8], "swissroll": [0.4, 0.6, 0.8],
              "hourglass": [0.3, 0.4, 0.6]}
    fig, axes = plt.subplots(1, 3, figsize=(180/25.4, 60/25.4), sharey=False)
    for ax, kind in zip(axes, ("spiral", "swissroll", "hourglass")):
        xs, lo, hi, mu = [], [], [], []
        for s in sigmas[kind]:
            f = summ[(summ.dataset == kind) & (np.isclose(summ.sigma, s)) &
                     (summ.method == "P2_Full")].sort_values("seed")
            ne = summ[(summ.dataset == kind) & (np.isclose(summ.sigma, s)) &
                      (summ.method == "P2_NoExploration")].sort_values("seed")
            m = f.merge(ne, on="seed", suffixes=("_f", "_ne"))
            d = m.acc_ne.values - m.acc_f.values
            if len(d) < 2:
                continue
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            xs.append(s); mu.append(d.mean()); lo.append(d.mean()-h); hi.append(d.mean()+h)
        ax.plot(xs, mu, color=INK, lw=1.6, marker="o", ms=4)
        ax.fill_between(xs, lo, hi, color=FULL_C, alpha=0.15, lw=0)
        ax.axhline(0, color=INK, lw=0.8, ls="--")
        ax.set_title(kind, fontsize=9, color=INK, weight="bold")
        ax.set_xlabel("sigma", fontsize=8, color=INK)
        _style(ax)
    axes[0].set_ylabel("ACC(NoExplo) - ACC(Full)", fontsize=8, color=INK)
    fig.suptitle("Overlap curve: paired ACC gap (NoExplo - Full), 95% CI, n=10 seeds",
                 fontsize=9.5, color=INK, weight="bold")
    _note(fig)
    out = os.path.join(FIG, "synthetic_3d_overlap_curve.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[figB]", out)


# --------------------------------------------------------------------------- #
# Fig D: eccentricity h_i vs entropy H(u) scatter.
# --------------------------------------------------------------------------- #
def fig_d():
    import pandas as pd
    traj = pd.read_csv(os.path.join(SYN, "trajectory_all.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(180/25.4, 60/25.4))
    for ax, kind in zip(axes, ("spiral", "swissroll", "hourglass")):
        sigma = MIDDLE[kind]
        X, y = load(kind, sigma)
        U, _, _ = plcfcm_fit(X, c=6, labeled_idx=None, labels=None,
                             seed=20261010)
        p = np.clip(U, 1e-12, 1.0)
        H = -(p * np.log(p)).sum(1)
        ub = U.mean(0)
        h = np.abs(U - ub).sum(1)
        sub = traj[(traj.dataset == kind) & (np.isclose(traj.sigma, sigma))]
        sel_f = set(sub[sub.method == "P2_Full"].selected_idx)
        sel_ne = set(sub[sub.method == "P2_NoExploration"].selected_idx)
        ecc = y == -1
        other = ~(ecc | np.isin(np.arange(len(y)), list(sel_f)) |
                  np.isin(np.arange(len(y)), list(sel_ne)))
        ax.scatter(H[other], h[other], s=4, c=GREY, alpha=0.4, edgecolors="none")
        ax.scatter(H[list(sel_f)], h[list(sel_f)], s=14, marker="x",
                   c=FULL_C, label="P2_Full selected", linewidths=0.7)
        ax.scatter(H[list(sel_ne)], h[list(sel_ne)], s=14, marker="o",
                   facecolors="none", edgecolors=NOEXP_C, label="NoExplo selected")
        ax.scatter(H[ecc], h[ecc], marker="*", s=50, c="black",
                   label="eccentric")
        ax.set_title(kind, fontsize=9, color=INK, weight="bold")
        ax.set_xlabel("H(u_i)", fontsize=8, color=INK)
        _style(ax)
    axes[0].set_ylabel("eccentricity $h_i=\\|u_i-\\bar u\\|_1$", fontsize=8, color=INK)
    axes[0].legend(fontsize=6, frameon=False, labelcolor=INK)
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])
    fig.suptitle("Selection map on the (entropy, eccentricity) plane",
                 fontsize=11, color=INK, weight="bold", y=0.97)
    fig.text(0.5, 0.055,
             "Warm-pink x: P2_Full selected; blue o: P2_NoExploration "
             "selected; black *: eccentric decoys. Post-hoc colors differ "
             "from main-text identity colors.",
             ha="center", fontsize=7.5, color=INK)
    _note(fig)
    out = os.path.join(FIG, "synthetic_3d_eccentricity_vs_entropy.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[figD]", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all")
    args = ap.parse_args()
    if args.only in ("C", "all"):
        fig_c()
    if args.only in ("A", "all"):
        fig_a()
    if args.only in ("B", "all"):
        fig_b()
    if args.only in ("D", "all"):
        fig_d()
