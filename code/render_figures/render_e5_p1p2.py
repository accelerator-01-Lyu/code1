# -*- coding: utf-8 -*-
"""fig_e5_p1p2: statistical enhancement (E5).

(a) raw vs Holm-corrected non-inferiority p over six UCI datasets
    (results/stats/p1_holm.csv), 0.05 threshold.
(b) dz forest plot over all 60 NoExploration-vs-Full contrasts
    (results/stats/all_dz.csv): 12 synthetic + 42 cross-task + 6 UCI.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (BLUE, ORANGE, GREEN, GREY, VERMILLION, WIDTH_85MM,
                    despine, panel_label, save_both)

S = r'D:\AL_FuzzyClustering_Paper\v2_unified\results\stats'


def build():
    holm = pd.read_csv(os.path.join(S, 'p1_holm.csv'))
    dz = pd.read_csv(os.path.join(S, 'all_dz.csv'))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(WIDTH_85MM, 7.0),
        gridspec_kw=dict(height_ratios=[0.9, 3.6], hspace=0.15))

    # ---------------- Panel a: raw vs Holm p ----------------
    x = np.arange(len(holm))
    w = 0.36
    ax1.bar(x - w/2, holm.raw_p, width=w, color=BLUE, label='raw p',
            edgecolor='white', linewidth=0.4)
    ax1.bar(x + w/2, holm.holm_p, width=w, color=ORANGE, label='Holm p',
            edgecolor='white', linewidth=0.4)
    ax1.axhline(0.05, color=VERMILLION, ls='--', lw=1.0)
    ax1.text(-0.45, 0.062, r'$\alpha=0.05$', color=VERMILLION,
             fontsize=6.8, ha='left', va='bottom')
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(holm.dataset, rotation=25, ha='right')
    ax1.set_ylabel('non-inferiority p (log scale)')
    ax1.set_ylim(1e-7, 0.5)
    ax1.legend(frameon=False, loc='upper right', fontsize=6.8,
               bbox_to_anchor=(0.99, 0.85))
    despine(ax1)
    panel_label(ax1, '(a)', x=-0.12, y=1.02)

    # ---------------- Panel b: d_z forest ----------------
    # build row blocks in display order (top -> bottom)
    blocks = []  # (title, color, sub_df, row_labels)
    e1 = dz[dz.source == 'e1_synthetic'].copy()
    e1['num'] = e1.comparison.str.extract(r'instance_(\d+)').astype(int)
    e1 = e1.sort_values('num')
    blocks.append(('E1 synthetic (10% budget)', GREEN, e1,
                   ['i' + n for n in e1.comparison.str.extract(r'_(\d+):')[0]]))

    e3 = dz[dz.source == 'e3_cross_task'].copy()
    e3['taskname'] = e3.comparison.str.split(':').str[0]
    for task, sub in e3.groupby('taskname', sort=True):
        sub = sub.sort_values('budget')
        blocks.append((task, ORANGE, sub,
                       [f'{int(round(b*100))}%' for b in sub.budget]))

    uci = dz[dz.source == 'uci_50seed'].copy()
    uci['ds'] = uci.comparison.str.split(':').str[0]
    uci = uci.sort_values('ds')
    blocks.append(('UCI (50 seeds, 10% budget)', BLUE, uci,
                   list(uci.ds)))

    # flatten rows
    flat, labels, colors = [], [], []
    block_ranges = []  # (title, color, y_top, y_bot)
    y = 0
    for title, color, sub, lab in blocks:
        y0 = y
        for (_, r), lb in zip(sub.iterrows(), lab):
            flat.append(r.d_z)
            labels.append(lb)
            colors.append(color)
            y += 1
        block_ranges.append((title, color, y - 1, y0))
        y += 1  # gap between blocks
    n = len(flat)
    yy = np.arange(n)[::-1]  # top row first

    for yi, val, c in zip(yy, flat, colors):
        ax2.plot(val, yi, 'o', ms=3.0, color=c, zorder=4)
    ax2.axvline(0, color=GREY, lw=0.9, zorder=1)

    # block headers + separators (row-index coords -> plot coords: y_plot = n-1-r)
    ax2.set_xlim(-0.95, 2.4)
    for title, color, ytop, ybot in block_ranges:
        # block's first row = row index ybot; plot it at the TOP of the block
        head_plot = n - 1 - ybot + 0.45
        ax2.text(-0.92, head_plot, title, fontsize=5.6, color=color,
                 fontweight='bold', ha='left', va='bottom')
        if ytop < n - 1:
            ax2.axhline(n - 1 - (ytop + 0.5), color='#BBBBBB', lw=0.6, ls=':')

    ax2.set_yticks(yy)
    ax2.set_yticklabels(labels, fontsize=5.4)
    ax2.set_xlabel(r"standardized effect $d_z$ (NoExploration $-$ Full)")
    ax2.set_ylim(-1.2, n + 0.8)
    despine(ax2)
    panel_label(ax2, '(b)', x=-0.12, y=1.005)

    save_both(fig, 'fig_e5_p1p2')


if __name__ == '__main__':
    build()
