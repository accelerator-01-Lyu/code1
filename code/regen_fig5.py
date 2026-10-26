# -*- coding: utf-8 -*-
"""Rebuild Fig.5 (E. coli 10% selection overlap) with KBS-grade styling.

First prints the overlap count for a given seed, then renders the figure.
Usage:  python regen_fig5.py [--seed 0] [--no-plot]
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_experiments import load_pkl, NCLASSES, INIT_FRAC, M_MC, RHO, _lambda2_over_n
from fcm_update import plcfcm_fit
from acquisition import select_batch

OUT_PNG = os.path.join(HERE, '..', 'latex', 'media', 'fig7_tsne.png')

LAM_MULT = 0.5
ALPHA0 = 50
BUDGET = 0.10


def query_set(X, y, c, method, seed=0, l0=0):
    rng = np.random.RandomState((seed * 1000 + l0) & 0x7FFFFFFF)
    n = X.shape[0]
    n_init = max(c, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, bool)
    labeled[rng.choice(n, size=n_init, replace=False)] = True
    U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                         labels=y[labeled], seed=seed)
    target = int(round(BUDGET * n))
    b = target - int(labeled.sum())
    unlab = np.where(~labeled)[0]
    beta = LAM_MULT * _lambda2_over_n(U)
    sel = select_batch(U[unlab], X[unlab], V, method, b,
                       alpha_0=ALPHA0, M_MC=M_MC, rho_expl=RHO,
                       lam_red=beta, seed=seed, u_all=U)
    sel = unlab[sel]
    queried = np.where(labeled)[0].tolist() + sel.tolist()
    return sorted(queried)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    X, y = load_pkl('ecoli')
    c = NCLASSES['ecoli']
    n = X.shape[0]
    print(f'[data] ecoli n={n} d={X.shape[1]} c={c}')

    qf = query_set(X, y, c, 'P2_Full', seed=args.seed)
    qn = query_set(X, y, c, 'P2_NoExploration', seed=args.seed)
    both = sorted(set(qf) & set(qn))
    only_f = sorted(set(qf) - set(qn))
    only_n = sorted(set(qn) - set(qf))
    print(f'[seed {args.seed}] Full queried={len(qf)} NoExpl queried={len(qn)}')
    print(f'[overlap] Both={len(both)}  FullOnly={len(only_f)}  NoExplOnly={len(only_n)}')
    print(f'[overlap] coincidence fraction = {100*len(both)/len(qf):.1f}%')
    print(f'[idx] both={both}')
    print(f'[idx] full_only={only_f}')
    print(f'[idx] noexpl_only={only_n}')

    if args.no_plot:
        return

    # ---- 2D embedding (t-SNE, random state fixed) ----
    tsne = TSNE(n_components=2, init='pca', random_state=42,
                perplexity=30, max_iter=1000, learning_rate='auto')
    Z = tsne.fit_transform(X)

    # ---- styling ----
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.linewidth': 0.8,
    })
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    all_idx = set(range(n))
    not_q = sorted(all_idx - set(qf) - set(qn))
    # background: not queried, very light gray, weak, cluster context only
    if not_q:
        ax.scatter(Z[not_q, 0], Z[not_q, 1], c='#E0E0E0', s=15, alpha=0.6,
                   edgecolors='none', zorder=1, label='Not queried')
    # core: coincident points, dark blue, LARGE, black edge (most salient)
    ax.scatter(Z[both, 0], Z[both, 1], c='#1f77b4', s=100,
               edgecolors='black', linewidths=1.2, zorder=2,
               label='Both (coincide)')
    # Full only: red, large, black edge
    ax.scatter(Z[only_f, 0], Z[only_f, 1], marker='o', s=90, c='#d62728',
               edgecolors='black', linewidths=1.0, zorder=3,
               label='Full only')
    # NoExploration only: cyan triangles, large, black edge
    ax.scatter(Z[only_n, 0], Z[only_n, 1], marker='^', s=90, c='#17becf',
               edgecolors='black', linewidths=1.0, zorder=3,
               label='NoExploration only')

    ax.legend(frameon=False, loc='upper right', fontsize=10,
              labelspacing=1.2, handletextpad=0.3, borderaxespad=0.4)

    fig.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('[fig] saved', OUT_PNG, os.path.getsize(OUT_PNG), 'bytes')


if __name__ == '__main__':
    main()
