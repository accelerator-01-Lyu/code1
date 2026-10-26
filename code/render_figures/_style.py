# -*- coding: utf-8 -*-
"""Shared style for new experiment figures (Fig 13+).

Conventions matched to the existing journal figures (Fig 1-12):
- single-column width 85 mm (3.35 in), 300 dpi, PDF + PNG
- Okabe-Ito colour-blind-friendly palette (no red-green pair)
- sans-serif labels, despined axes, compact legends
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Okabe-Ito palette
BLUE = '#0072B2'
ORANGE = '#E69F00'
GREEN = '#009E73'
VERMILLION = '#D55E00'
SKY = '#56B4E9'
YELLOW = '#F0E442'
PURPLE = '#CC79A7'
GREY = '#666666'

FIG_DIR = r'D:\AL_FuzzyClustering_Paper\v2_unified\figures'
MM = 1.0 / 25.4
WIDTH_85MM = 85 * MM  # 3.346 in

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.constrained_layout.use': True,
})


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def panel_label(ax, text, x=-0.12, y=1.03):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8.5,
            fontweight='bold', va='bottom', ha='left')


def save_both(fig, name):
    pdf = os.path.join(FIG_DIR, name + '.pdf')
    png = os.path.join(FIG_DIR, name + '.png')
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print('saved:', pdf)
    print('saved:', png)
    return pdf, png
