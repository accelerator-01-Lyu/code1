# -*- coding: utf-8 -*-
"""fig_e4_diag (Fig.18, redrawn): Theorem 2 gradient-intensity diagnostic.

(a) mean s_i vs overlap level: queried (solid, both methods pooled; values
    from results/theorem2/table12.csv, pooled rows) vs random-query baseline
    (dashed; mean s_i per overlap level computed from
    results/theorem2/random_baseline.csv, both methods pooled).
(b) Intensity gap = queried - random mean s_i per overlap level with 95%
    CIs (results/stats/e4_intensity_gap.csv); the dashed horizontal line
    marks the pooled gap (-0.543); vertical reference at zero.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (BLUE, GREY, WIDTH_85MM, despine, panel_label, save_both)

TH = r'D:\AL_FuzzyClustering_Paper\v2_unified\results\theorem2'
ST = r'D:\AL_FuzzyClustering_Paper\v2_unified\results\stats'


def build():
    t12 = pd.read_csv(os.path.join(TH, 'table12.csv'))
    q = t12[t12['class'] == 'pooled'].sort_values('overlap_level')
    levels = q.overlap_level.values
    queried = q.mean_s.values

    rb = pd.read_csv(os.path.join(TH, 'random_baseline.csv'))
    rand = rb.groupby('overlap_level').s_i.mean().reindex(levels).values

    gap = pd.read_csv(os.path.join(ST, 'e4_intensity_gap.csv'))
    g = gap[gap.overlap != 'pooled'].sort_values('overlap')
    pooled = gap[gap.overlap == 'pooled'].iloc[0]
    gv, glo, ghi = g.gap_mean.values, g.gap_lo.values, g.gap_hi.values

    print('levels  :', levels)
    print('queried :', np.round(queried, 4))
    print('random  :', np.round(rand, 4))
    print('gap     :', np.round(gv, 4), 'CI', np.round(glo, 4), np.round(ghi, 4))
    print('pooled  :', pooled.gap_mean)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(WIDTH_85MM, 3.3),
        gridspec_kw=dict(height_ratios=[1.15, 1.0], hspace=0.12))
    x = np.arange(len(levels))

    # ---------------- Panel a: mean s_i, queried vs random ----------------
    ax1.plot(x, queried, '-o', color=BLUE, lw=1.5, ms=4.5,
             label='queried (pooled)', zorder=5)
    ax1.plot(x, rand, '--o', color=GREY, lw=1.3, ms=4.5, mfc='white',
             mec=GREY, label='random query (baseline)', zorder=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{v:.1f}' for v in levels])
    ax1.set_xlabel('cluster overlap level')
    ax1.set_ylabel(r'mean $s_i$')
    ax1.legend(loc='upper left', frameon=False, fontsize=7)
    despine(ax1)
    panel_label(ax1, '(a)')

    # ---------------- Panel b: intensity gap ----------------
    ax2.axhline(0, color=GREY, lw=0.9, zorder=1)
    ax2.axhline(pooled.gap_mean, color=GREY, ls=':', lw=1.0, zorder=2)
    ax2.text(-0.45, pooled.gap_mean - 0.04,
             f'pooled = {pooled.gap_mean:.2f}', fontsize=6.8, color=GREY,
             ha='left', va='top')
    ax2.bar(x, gv, width=0.55, color=BLUE, edgecolor='white', linewidth=0.5,
            zorder=3)
    ax2.errorbar(x, gv, yerr=[gv - glo, ghi - gv], fmt='none', ecolor=GREY,
                 elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{v:.1f}' for v in levels])
    ax2.set_xlabel('cluster overlap level')
    ax2.set_ylabel(r'intensity gap')
    despine(ax2)
    panel_label(ax2, '(b)')

    save_both(fig, 'fig_e4_diag')


if __name__ == '__main__':
    build()
