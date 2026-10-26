# -*- coding: utf-8 -*-
"""fig_e1_effects: per-instance effect forest plot (E1, 72 synthetic).

Each row = one synthetic instance; point = NoExploration - Full ACC at the
10% budget (pp), whisker = one-sided 95% lower bound. Points are grouped by
overlap stratum (colour) and balanced/imbalanced block (marker).
Data: results/synthetic/fig10_data.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (BLUE, ORANGE, GREEN, VERMILLION, GREY, WIDTH_85MM,
                    despine, save_both)

D = r'D:\AL_FuzzyClustering_Paper\v2_unified\results\synthetic\fig10_data.csv'

OVERLAP_COLORS = {0.0: BLUE, 0.3: ORANGE, 0.6: GREEN, 0.9: VERMILLION}


def build():
    d = pd.read_csv(D).sort_values(['overlap', 'instance_id']).reset_index(drop=True)
    n = len(d)
    yy = np.arange(n)[::-1]  # top row first

    fig, ax = plt.subplots(figsize=(WIDTH_85MM, 7.6))

    for yi, (_, r) in zip(yy, d.iterrows()):
        c = OVERLAP_COLORS[r.overlap]
        # one-sided whisker: lower bound -> point
        ax.plot([r.ci_low_pp, r.effect_pp], [yi, yi], color=c, lw=0.9,
                alpha=0.8, zorder=3)
        mk = 'o' if r.block == 'balanced' else '^'
        ax.plot(r.effect_pp, yi, mk, color=c, ms=3.2, zorder=5)

    ax.axvline(0, color=GREY, lw=0.9, zorder=1)

    # overlap group headers + separators
    heads = []
    for ov, sub in d.groupby('overlap', sort=True):
        rows = sub.index.values
        head_plot = n - 1 - rows.min() + 0.5
        heads.append((ov, head_plot, rows.max()))
    for ov, hp, rmax in heads:
        ax.text(-0.92, hp, f'overlap={ov:.1f}', fontsize=6.5,
                color=OVERLAP_COLORS[ov], fontweight='bold',
                ha='left', va='center')
        if rmax < n - 1:
            ax.axhline(n - 1 - (rmax + 0.5), color='#BBBBBB', lw=0.6, ls=':')

    ax.set_yticks(yy)
    ax.set_yticklabels([f'{int(i)}' for i in d.instance_id], fontsize=5.2)
    ax.set_xlabel(r'NoExploration $-$ Full ACC (pp) @10%')
    ax.set_ylim(-1.5, n + 1.0)

    # dynamic xlim
    xmin = min(d.ci_low_pp.min(), 0)
    xmax = d.effect_pp.max()
    ax.set_xlim(xmin - 0.4, xmax + 0.5)

    # marker legend: balanced vs imbalanced
    ax.plot([], [], 'o', color=GREY, ms=3.5, label='balanced block')
    ax.plot([], [], '^', color=GREY, ms=3.5, label='imbalanced block')
    ax.legend(loc='lower left', frameon=False, fontsize=6.5, ncol=1,
              bbox_to_anchor=(0.0, 0.02))

    despine(ax)
    save_both(fig, 'fig_e1_effects')


if __name__ == '__main__':
    build()
