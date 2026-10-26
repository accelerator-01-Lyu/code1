"""make_fig_s6.py - Fig S6: membership-cosine graph-Laplacian spectra.

The lambda2 used by the experiments is NOT a feature-space kNN Laplacian; it
is lambda2 of the combinatorial Laplacian L_R of the membership-derived
cosine-redundancy graph R = cosine(normalized U rows) (see
run_experiments._lambda2_over_n). This figure reproduces that exact object at
the round-0 semi-supervised PLCFCM fit (seed=0, l0=0, frozen L0 protocol) and
stems the first 40 ordered eigenvalues, highlighting the lambda2 (Fiedler)
gap and the lambda2/n scale that Theorem 3 identifies as the first-principles
redundancy coefficient.

Deterministic; no AL batches, no randomness beyond the frozen seed=0 fit.
Frozen figure spec (notes/figure_style_spec.md, FROZEN 2026-10-10).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_experiments import load_pkl, NCLASSES, INIT_FRAC, _lambda2_over_n
from fcm_update import plcfcm_fit
from acquisition import cosine_redundancy

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "figures", "final")
os.makedirs(OUTDIR, exist_ok=True)

INK = "#3F345B"
BLUE = "#7FA4C8"
PINK = "#D08AB6"
GRID = "#E9E9EE"

DATASETS = ["iris", "wine", "glass", "ecoli", "aggregation", "compound"]
N_EIG = 40
SEED = 0


def round0_spectrum(name):
    X, y = load_pkl(name)
    n = X.shape[0]
    c = NCLASSES[name]
    rng = np.random.RandomState(SEED * 1000)
    n_init = max(c, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, bool)
    labeled[rng.choice(n, size=n_init, replace=False)] = True
    U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                         labels=y[labeled], seed=SEED)
    R = cosine_redundancy(U)
    d = R.sum(1)
    L = -R.copy()
    np.fill_diagonal(L, d)
    m = min(N_EIG, n)
    eigs = np.linalg.eigvalsh(L)
    order = np.sort(eigs)
    beta = _lambda2_over_n(U)
    return order[:m], order[1], beta, n


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.09, 4.2))
    for ax, name in zip(axes.ravel(), DATASETS):
        eigs, lam2, beta, n = round0_spectrum(name)
        x = np.arange(1, len(eigs) + 1)
        ax.plot([x.min(), x.max()], [0, 0], color=INK, lw=0.6)
        for xi, ei in zip(x, eigs):
            ax.plot([xi, xi], [0, ei],
                    color=PINK if xi == 2 else BLUE,
                    lw=1.8 if xi == 2 else 0.9,
                    solid_capstyle="butt")
        ax.plot(x[1:], eigs[1:], "o", ms=2.6, mfc=BLUE, mec=BLUE, zorder=3)
        ax.plot(x[0], eigs[0], "o", ms=2.6, mfc=BLUE, mec=BLUE, zorder=3)
        ax.plot(2, eigs[1], "o", ms=5.5, mfc=PINK, mec=INK, mew=0.6, zorder=4)
        ax.set_title(name, color=INK, fontsize=9)
        ax.annotate(f"$\\lambda_2$={lam2:.3f}\n$\\lambda_2/n$={beta:.4f}",
                    xy=(2, eigs[1]),
                    xytext=(8, max(eigs[1] * 1.05, eigs[10] * 0.6)),
                    fontsize=7, color=INK, va="bottom",
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_xlim(0.3, len(eigs) + 0.7)
        ax.set_ylim(0, None)
        ax.set_xlabel("eigenvalue index", fontsize=8)
        ax.set_ylabel(r"$\lambda_i(L_R)$", fontsize=8)
    fig.suptitle(r"Membership-cosine graph Laplacian spectra: $\lambda_2$ gap "
                 r"and the $\lambda_2/n$ scale (round-0 fit, seed=0)",
                 fontsize=10, color=INK)
    fig.text(0.99, 0.005,
             "Deterministic; R=cosine(normalized U), L_R=diag(R1)-R; "
             "lambda2/n is recomputed each AL round in the experiments.",
             ha="right", va="bottom", fontsize=5.5, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    pdf = os.path.join(OUTDIR, "fig_s6_laplacian_spectrum.pdf")
    png = os.path.join(OUTDIR, "fig_s6_laplacian_spectrum.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print("wrote", pdf)
    print("wrote", png)
    for name in DATASETS:
        _, lam2, beta, n = round0_spectrum(name)
        print(f"{name:12s} n={n:4d} lambda2={lam2:.4f} lambda2/n={beta:.5f}")


if __name__ == "__main__":
    main()
