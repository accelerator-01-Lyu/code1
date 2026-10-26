"""make_fig_mechanism_b.py - corrected dual-violin mechanism figure.

s_i = <grad h_i, grad w^rob> on SELECTED points, pooled over ecoli+glass at
budget 10%, by point class, Full vs NoExplo. Frozen palette only.
Post-hoc exploratory; not part of pre-registered P1-P4.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figures", "final")

INK = "#3F345B"
FULL = "#6A5A8F"   # deep purple (method identity color, frozen)
NOEX = "#D08AB6"   # highlight pink (NoExplo identity color, frozen)
GRID = "#E9E9EE"

CLASSES = ["eccentric", "boundary", "overlap"]
BUDGET = 0.1


def load(path):
    df = pd.read_csv(path)
    df = df[df["budget"] == BUDGET]
    return df


def main():
    full = pd.concat([load(os.path.join(RES, "gradient_inner_product.csv"))])
    noex = pd.concat([load(os.path.join(RES,
                            "gradient_inner_product_nonexplo.csv"))])
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    positions, fdata, ndata = [], [], []
    for k, cls in enumerate(CLASSES):
        f = full.loc[full["point_class"] == cls, "inner_product"].to_numpy()
        q = noex.loc[noex["point_class"] == cls, "inner_product"].to_numpy()
        fdata.append(f); ndata.append(q)
        positions.append((k - 0.18, k + 0.18))
    parts_f = ax.violinplot(fdata, positions=[p[0] for p in positions],
                            widths=0.32, showmedians=True, showextrema=True)
    parts_n = ax.violinplot(ndata, positions=[p[1] for p in positions],
                            widths=0.32, showmedians=True, showextrema=True)
    for pc in parts_f["bodies"]:
        pc.set_facecolor(FULL); pc.set_edgecolor(INK); pc.set_alpha(.75); pc.set_lw(.6)
    for pc in parts_n["bodies"]:
        pc.set_facecolor(NOEX); pc.set_edgecolor(INK); pc.set_alpha(.75); pc.set_lw(.6)
    for key, col in (("cmedians", INK), ("cmins", INK), ("cmaxes", INK),
                     ("cbars", INK)):
        parts_f[key].set_color(col); parts_n[key].set_color(col)
        parts_f[key].set_lw(1.2); parts_n[key].set_lw(1.2)
    ax.axhline(0, color=INK, lw=0.8)
    for k, cls in enumerate(CLASSES):
        nf, nn = len(fdata[k]), len(ndata[k])
        ax.text(k - 0.18, ax.get_ylim()[0] * 0.96, f"n={nf}", ha="center",
                va="top", fontsize=6.5, color=INK)
        ax.text(k + 0.18, ax.get_ylim()[0] * 0.96, f"n={nn}", ha="center",
                va="top", fontsize=6.5, color=INK)
        if cls == "overlap":
            ax.text(k, -4.7, "n<10 pooled,\nunderpowered",
                    ha="center", va="top", fontsize=6.5, style="italic",
                    color=INK)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel(r"$s_i=\langle\nabla h_i,\nabla w^{\mathrm{rob}}\rangle$")
    ax.set_title(r"$s_i$ on selected points by class (ecoli+glass, @10%)",
                 color=INK, fontsize=9.5)
    ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
    ax.scatter([], [], color=FULL, s=36, label="Full")
    ax.scatter([], [], color=NOEX, s=36, label="NoExplo")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(0.01, 0.005,
             "Post-hoc exploratory — not part of pre-registered P1–P4.",
             ha="left", va="bottom", fontsize=6.5, color=INK, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    pdf = os.path.join(OUT, "fig_mechanism_b.pdf")
    png = os.path.join(OUT, "fig_mechanism_b.png")
    fig.savefig(pdf); fig.savefig(png, dpi=300)
    print("wrote", pdf, os.path.getsize(pdf))
    print("wrote", png, os.path.getsize(png))
    for k, cls in enumerate(CLASSES):
        print(f"{cls:10s} Full n={len(fdata[k]):4d} med={np.median(fdata[k]):.3f}"
              f" | NoExplo n={len(ndata[k]):4d} med={np.median(ndata[k]):.3f}")


if __name__ == "__main__":
    main()
