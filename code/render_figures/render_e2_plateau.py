# -*- coding: utf-8 -*-
"""fig_e2_plateau: parameter robustness (E2).

(a) plateau width per factor (pp) -- heights taken directly from the
    authoritative E2_final_draft.md Table 10: rho 0.05, beta 0.02, kappa 0.01,
    alpha0 0.09, overall 0.17; 0.50 pp non-inferiority margin as dashed line.
(b) marginal ACC minus sub-design grand mean (pp), computed from the factor
    CSVs (results/factor/*.csv): all four profiles hug zero.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (BLUE, ORANGE, GREEN, VERMILLION, GREY, WIDTH_85MM,
                    despine, panel_label, save_both)

F = r'D:\AL_FuzzyClustering_Paper\v2_unified\results\factor'

# authoritative plateau widths (E2_final_draft.md Sec.3 Table 10), pp
WIDTHS = [0.05, 0.02, 0.01, 0.09, 0.17]
LABELS = [r'$\rho$', r'$\beta$', r'$\kappa$', r'$\alpha_0$', 'overall']
MARGIN = 0.50


def marginal_profiles():
    rb = pd.read_csv(os.path.join(F, 'rho_beta.csv'))
    ka = pd.read_csv(os.path.join(F, 'kappa_alpha.csv'))
    profs = {}
    profs[r'$\rho$'] = (sorted(rb.rho.unique()),
                        rb.groupby('rho').ACC.mean().reindex(sorted(rb.rho.unique())).values * 100
                        - rb.ACC.mean() * 100, BLUE)
    profs[r'$\beta$'] = (sorted(rb.beta_mult.unique()),
                         rb.groupby('beta_mult').ACC.mean().reindex(sorted(rb.beta_mult.unique())).values * 100
                         - rb.ACC.mean() * 100, ORANGE)
    profs[r'$\kappa$'] = (sorted(ka.kappa.unique()),
                         ka.groupby('kappa').ACC.mean().reindex(sorted(ka.kappa.unique())).values * 100
                         - ka.ACC.mean() * 100, GREEN)
    profs[r'$\alpha_0$'] = (sorted(ka.alpha.unique()),
                           ka.groupby('alpha').ACC.mean().reindex(sorted(ka.alpha.unique())).values * 100
                           - ka.ACC.mean() * 100, VERMILLION)
    for k, (xs, ys, c) in profs.items():
        print(f'marginal {k}: x={xs} y(pp)={np.round(ys, 4)}')
    return profs


def build():
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(WIDTH_85MM, 3.5),
        gridspec_kw=dict(height_ratios=[1.0, 1.15], hspace=0.12))

    # ---------------- Panel a: plateau-width bars ----------------
    x = np.arange(len(LABELS))
    colors = [BLUE, BLUE, BLUE, BLUE, ORANGE]
    ax1.bar(x, WIDTHS, width=0.62, color=colors, edgecolor='white',
            linewidth=0.5, zorder=3)
    ax1.axhline(MARGIN, color=GREY, ls='--', lw=1.1, zorder=2)
    ax1.text(len(LABELS) - 0.45, MARGIN + 0.015,
             '0.50 pp non-inferiority margin', fontsize=6.8, color=GREY,
             ha='right', va='bottom')
    for xi, w in zip(x, WIDTHS):
        ax1.text(xi, w + 0.012, f'{w:.2f}', ha='center', va='bottom',
                 fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS)
    ax1.set_ylabel('plateau width (pp)')
    ax1.set_ylim(0, 0.62)
    despine(ax1)
    panel_label(ax1, '(a)')

    # ---------------- Panel b: marginal profiles ----------------
    profs = marginal_profiles()
    # lay out four factor groups side by side on one categorical axis
    pos, i = {}, 0
    xticks, xticklabels, groups = [], [], []
    for name, (xs, ys, c) in profs.items():
        start = i
        for j, xv in enumerate(xs):
            pos[(name, j)] = i
            xticks.append(i)
            xticklabels.append(f'{xv:g}')
            i += 1
        groups.append((name, start, i - 1, c))
        i += 1  # gap between groups
    for name, (xs, ys, c) in profs.items():
        xx = [pos[(name, j)] for j in range(len(xs))]
        ax2.plot(xx, ys, '-o', color=c, lw=1.2, ms=3.5, zorder=4)
    ax2.axhline(0, color=GREY, lw=0.9, zorder=1)
    for gname, g0, g1, c in groups:
        ax2.text((g0 + g1) / 2, 0.24, gname, ha='center', va='bottom',
                 fontsize=7.5, color=c)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels, fontsize=6.5, rotation=35, ha='right')
    ax2.set_ylabel('ACC $-$ grand mean (pp)')
    ax2.set_xlabel('factor level')
    ax2.set_ylim(-0.20, 0.30)
    despine(ax2)
    panel_label(ax2, '(b)', x=-0.14)

    save_both(fig, 'fig_e2_plateau')


if __name__ == '__main__':
    build()
