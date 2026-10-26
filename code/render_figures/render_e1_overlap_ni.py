# -*- coding: utf-8 -*-
"""fig_e1_overlap_ni: E1 non-inferiority rate by overlap stratum.

Bars: fraction of the 18 synthetic instances per overlap stratum on which
NoExploration is non-inferior to Full at the 10% budget; whiskers are 95%
CIs. Data: results/synthetic/fig9_data.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import BLUE, GREY, WIDTH_85MM, despine, panel_label, save_both

D = r'D:\AL_FuzzyClustering_Paper\v2_unified\results\synthetic\fig9_data.csv'


def build():
    d = pd.read_csv(D)
    x = np.arange(len(d))
    prop = d.prop_noninferior.values * 100
    lo = (d.prop_noninferior - d.ci_low).values * 100
    hi = (d.ci_high - d.prop_noninferior).values * 100

    fig, ax = plt.subplots(figsize=(WIDTH_85MM, 2.6))
    ax.bar(x, prop, width=0.58, color=BLUE, edgecolor='white',
           linewidth=0.5, zorder=3)
    ax.errorbar(x, prop, yerr=[lo, hi], fmt='none', ecolor=GREY,
                elinewidth=1.0, capsize=3.0, capthick=1.0, zorder=4)
    for xi, (_, r) in zip(x, d.iterrows()):
        ax.text(xi, r.prop_noninferior * 100 + 3.5,
                f'{int(r.n_noninferior)}/{int(r.n_instances)}',
                ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v:.1f}' for v in d.value])
    ax.set_xlabel('cluster overlap level')
    ax.set_ylabel('non-inferior instances (%)')
    ax.set_ylim(0, 108)
    despine(ax)
    save_both(fig, 'fig_e1_overlap_ni')


if __name__ == '__main__':
    build()
