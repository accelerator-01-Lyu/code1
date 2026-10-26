# -*- coding: utf-8 -*-
"""fig_e3_meta: cross-task meta-analysis (E3).

Forest plot of per-task NoExploration-Full ACC effect at 10% and 20% budgets
with 95% CIs, plus pooled random-effects estimates. Data:
results/cross_task/meta_analysis.csv (fig12_data.csv companion; pooled rows,
I^2 and tau^2 live here).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (BLUE, ORANGE, GREY, WIDTH_85MM, despine, panel_label,
                    save_both)

M = r'D:\AL_FuzzyClustering_Paper\v2_unified\results\cross_task\meta_analysis.csv'


def build():
    d = pd.read_csv(M)
    tasks = d[d.row_type == 'task'].copy().sort_values(
        ['n_classes', 'task']).reset_index(drop=True)
    p10 = d[d.row_type == 'pooled_10pct'].iloc[0]
    p20 = d[d.row_type == 'pooled_20pct'].iloc[0]

    fig, ax = plt.subplots(figsize=(WIDTH_85MM, 3.5))
    n = len(tasks)
    yt = np.arange(n)[::-1]  # top row first
    y_pool10 = n + 0.9
    y_pool20 = n - 0.1

    for y, (_, r) in zip(yt, tasks.iterrows()):
        # 10% budget
        e10 = r.effect_10pct * 100
        ax.plot([r.ci_low_10pct * 100, r.ci_high_10pct * 100],
                [y + 0.13, y + 0.13], color=BLUE, lw=1.0, solid_capstyle='butt')
        ax.plot([r.ci_low_10pct * 100, r.ci_low_10pct * 100],
                [y + 0.08, y + 0.18], color=BLUE, lw=0.8)
        ax.plot([r.ci_high_10pct * 100, r.ci_high_10pct * 100],
                [y + 0.08, y + 0.18], color=BLUE, lw=0.8)
        ax.plot(e10, y + 0.13, 'o', color=BLUE, ms=3.5, zorder=5)
        # 20% budget
        e20 = r.effect_20pct * 100
        ax.plot([r.ci_low_20pct * 100, r.ci_high_20pct * 100],
                [y - 0.13, y - 0.13], color=ORANGE, lw=1.0,
                solid_capstyle='butt')
        ax.plot([r.ci_low_20pct * 100, r.ci_low_20pct * 100],
                [y - 0.18, y - 0.08], color=ORANGE, lw=0.8)
        ax.plot([r.ci_high_20pct * 100, r.ci_high_20pct * 100],
                [y - 0.18, y - 0.08], color=ORANGE, lw=0.8)
        ax.plot(e20, y - 0.13, 's', color=ORANGE, ms=3.2, zorder=5)

    # pooled diamonds
    def diamond(y, eff, lo, hi, color, label):
        ax.fill([lo * 100, eff * 100, hi * 100, eff * 100],
                [y, y + 0.16, y, y - 0.16], color=color, zorder=6)
        ax.text(-1.55, y, label, ha='right', va='center', fontsize=7)

    diamond(y_pool10, p10.pooled_effect, p10.pooled_ci_low, p10.pooled_ci_high,
            BLUE, 'pooled 10%')
    diamond(y_pool20, p20.pooled_effect, p20.pooled_ci_low, p20.pooled_ci_high,
            ORANGE, 'pooled 20%')

    # I^2 / tau^2 annotations
    ax.text(1.35, y_pool10,
            rf'$I^2$={p10.I2:.0f}%, $\tau^2$={p10.tau2:.1e}',
            fontsize=6.5, color=BLUE, va='center')
    ax.text(1.35, y_pool20,
            rf'$I^2$={p20.I2:.0f}%, $\tau^2$={p20.tau2:.1e}',
            fontsize=6.5, color=ORANGE, va='center')

    ax.axvline(0, color=GREY, lw=0.9, zorder=1)
    ax.set_yticks(list(yt))
    ylabels = [f'{r.task} ({int(r.n_classes)}-class)'
                for _, r in tasks.iterrows()]
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel(r'NoExploration $-$ Full ACC (pp)')
    ax.set_ylim(-0.9, y_pool10 + 0.9)
    ax.set_xlim(-1.9, 2.9)
    # compact legend: two markers, placed in empty lower-left corner
    ax.plot([], [], 'o', color=BLUE, ms=3.5, label='10% budget')
    ax.plot([], [], 's', color=ORANGE, ms=3.2, label='20% budget')
    ax.legend(loc='lower left', frameon=False, fontsize=6.8, ncol=1,
              borderaxespad=0.1)
    despine(ax)

    save_both(fig, 'fig_e3_meta')


if __name__ == '__main__':
    build()
