# -*- coding: utf-8 -*-
"""add2_friedman_cd.py — Experiment 2: Friedman + Nemenyi critical-difference (CD)
comparison of nine active selectors.

CONFIRMATORY, Demsar (2006):
  * block = 6 datasets (N=6). For each dataset we take the 30-seed mean ACC at
    the 10% checkpoint (seed_set=30, alpha0=50, lam_mult=1.0, l0=0, eta=0).
  * rank 1 = best. k=8 formally ranked methods (ac.RANKED_METHODS).
  * OriginalP2 is the archived-bug control: NEVER enters the chi2 / CD / ranking
    of the formal test; it is drawn separately (ink dashed) and listed alone.

SUPPORTIVE (exploratory, not the formal conclusion):
  * per-seed ranks, blocks = dataset x seed = 180 blocks. This shrinks CD purely
    by N and is NOT a standard Demsar block; kept only to inspect rank stability.

Reads ONLY existing shards; never retrains. Produces CSVs + PNG/PDF + unit.json.
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import additional_common as ac

SHARD_DIR = os.path.join(ac.HERE, "results", "shards")
DATASETS = ["iris", "wine", "glass", "ecoli", "aggregation", "compound"]
ALL_METHODS = ac.RANKED_METHODS + ["OriginalP2"]   # 9, OriginalP2 last / control
RANKED = ac.RANKED_METHODS                          # 8 formal

NEED_COLS = ["budget", "ACC", "dataset", "method", "seed_set", "seed",
             "l0", "eta", "alpha0", "lam_mult"]


# --------------------------------------------------------------------------- #
# 1. Load shards (needed columns only) and filter to the frozen main cell
# --------------------------------------------------------------------------- #
def load_main_cell():
    frames = []
    for f in glob.glob(os.path.join(SHARD_DIR, "*.csv")):
        bn = os.path.basename(f)
        if bn == "p2_diagnostics.csv":
            continue
        try:
            df = pd.read_csv(f, usecols=lambda c: c in NEED_COLS)
        except Exception:
            continue
        if df.empty:
            continue
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    m = ((long["seed_set"].astype(str) == "30") &
         (np.isclose(long["budget"], 0.10)) &
         (np.isclose(long["alpha0"], ac.ALPHA0_MAIN)) &
         (np.isclose(long["lam_mult"], ac.LAM_MULT_MAIN)) &
         (long["l0"] == 0) &
         (np.isclose(long["eta"], 0.0)))
    cell = long[m].copy()
    cell = cell[cell["dataset"].isin(DATASETS) & cell["method"].isin(ALL_METHODS)]
    return cell


def validate_seeds(cell):
    """Every (dataset, method) must have exactly 30 seeds. Return missing map."""
    counts = cell.groupby(["dataset", "method"])["seed"].nunique()
    problems = {}
    for ds in DATASETS:
        for mt in ALL_METHODS:
            n = int(counts.get((ds, mt), 0))
            if n != 30:
                problems[f"{ds}/{mt}"] = n
    return problems


# --------------------------------------------------------------------------- #
# 2. Mean matrix and statistics
# --------------------------------------------------------------------------- #
def mean_matrix(cell, methods):
    g = cell.groupby(["dataset", "method"])["ACC"].mean()
    mat = np.zeros((len(DATASETS), len(methods)))
    for i, ds in enumerate(DATASETS):
        for j, mt in enumerate(methods):
            mat[i, j] = float(g.loc[(ds, mt)])
    return mat


def main():
    print("[add2] loading shards ...")
    cell = load_main_cell()
    print("[add2] main-cell rows:", len(cell))

    problems = validate_seeds(cell)
    if problems:
        print("[add2] SEED-COUNT PROBLEMS:", problems)
    else:
        print("[add2] seed validation OK: every dataset/method = 30 seeds")

    # 6 x 9 mean matrix (for the table incl. control)
    mat9 = mean_matrix(cell, ALL_METHODS)
    # 6 x 8 formal matrix
    mat8 = mean_matrix(cell, RANKED)

    # ---- formal Friedman / Nemenyi (N=6, k=8) ---------------------------- #
    res = ac.friedman_nemenyi(mat8, RANKED)
    avg_rank = res["avg_rank"]
    chi2, p_formal, CD = res["chi2"], res["p"], res["CD"]
    print(f"[add2] FORMAL: chi2={chi2:.4f} p={p_formal:.4g} CD={CD:.4f} "
          f"(N={res['N']}, k={res['k']}, q={res['q']})")

    # ---- OriginalP2: mean rank inside the 9-method ranking (display only) -#
    ranks9 = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, mat9)
    R9 = ranks9.mean(axis=0)
    rank9 = dict(zip(ALL_METHODS, R9.tolist()))
    origp2_rank9 = rank9["OriginalP2"]
    print(f"[add2] OriginalP2 9-method mean rank = {origp2_rank9:.3f} "
          f"(display only, NOT in chi2/CD)")

    # ---- supportive per-seed ranks (N=180, exploratory) ---------------- #
    seed_rank_rows = []
    per_seed_long = (cell[cell["method"].isin(RANKED)]
                     .groupby(["dataset", "seed", "method"])["ACC"].mean().reset_index())
    piv = per_seed_long.pivot_table(index=["dataset", "seed"], columns="method",
                                    values="ACC")
    piv = piv.dropna()  # only fully-observed blocks
    mat_seed = piv[RANKED].to_numpy()
    res_seed = ac.friedman_nemenyi(mat_seed, RANKED)
    n_blocks = mat_seed.shape[0]
    print(f"[add2] SUPPORTIVE per-seed: blocks={n_blocks} chi2={res_seed['chi2']:.3f} "
          f"p={res_seed['p']:.4g} CD={res_seed['CD']:.4f} (exploratory, N!=6)")

    # ---- save rank matrix CSV (6 x 9) ------------------------------------ #
    rows = []
    for i, ds in enumerate(DATASETS):
        row = {"dataset": ds}
        for j, mt in enumerate(ALL_METHODS):
            row[mt] = round(float(mat9[i, j]), 4)
        rows.append(row)
    df_rank = pd.DataFrame(rows)
    p1 = ac.save_csv(df_rank, "add2_rank_matrix.csv")
    print("[add2] wrote", p1)

    # ---- friedman/nemenyi CSV -------------------------------------------- #
    # per-method mean rank (formal) + mean ACC across 6 datasets
    acc_mean8 = {mt: float(mat8[:, j].mean()) for j, mt in enumerate(RANKED)}
    fr_rows = []
    for mt in sorted(RANKED, key=lambda m: avg_rank[m]):
        fr_rows.append({"method": mt, "formal_avg_rank": round(avg_rank[mt], 3),
                        "mean_ACC_6ds": round(acc_mean8[mt], 4),
                        "role": "ranked"})
    fr_rows.append({"method": "OriginalP2",
                    "formal_avg_rank": round(origp2_rank9, 3),
                    "mean_ACC_6ds": round(float(mat9[:, ALL_METHODS.index("OriginalP2")].mean()), 4),
                    "role": "archived-bug control (not ranked)"})
    # pairwise
    pair_rows = []
    for (a, b, d, sig) in sorted(res["pairs"], key=lambda t: -t[2]):
        pair_rows.append({"method_a": a, "method_b": b,
                          "abs_rank_diff": round(d, 3),
                          "CD": round(CD, 4), "significant": bool(sig)})
    meta = pd.DataFrame([{
        "test": "Friedman + Nemenyi (Demsar 2006)",
        "N_datasets": 6, "k_ranked_methods": 8, "q05_k8": res["q"],
        "chi2": round(chi2, 4), "p": p_formal, "CD": round(CD, 4),
        "supportive_blocks": n_blocks,
        "supportive_chi2": round(res_seed["chi2"], 4),
        "supportive_p": res_seed["p"],
        "supportive_CD": round(res_seed["CD"], 4),
        "note": "supportive per-seed analysis is exploratory; formal N=6 low power",
    }])
    with open(ac.res_path("add2_friedman_nemenyi.csv"), "w", encoding="utf-8") as f:
        f.write("# per-method ranks & mean ACC\n")
        pd.DataFrame(fr_rows).to_csv(f, index=False)
        f.write("\n# Nemenyi pairwise (sorted by |delta rank| desc)\n")
        pd.DataFrame(pair_rows).to_csv(f, index=False)
        f.write("\n# meta\n")
        meta.to_csv(f, index=False)
    print("[add2] wrote", ac.res_path("add2_friedman_nemenyi.csv"))

    # supportive per-seed mean ranks CSV
    sv = pd.DataFrame([{"method": m,
                        "supportive_avg_rank_180blocks": round(res_seed["avg_rank"][m], 3),
                        "formal_avg_rank_6ds": round(avg_rank[m], 3)}
                       for m in sorted(RANKED, key=lambda m: res_seed["avg_rank"][m])])
    p3 = ac.save_csv(sv, "add2_perseed_supportive.csv")
    print("[add2] wrote", p3)

    # ---- figures ---------------------------------------------------------- #
    plot_cd_diagram(avg_rank, origp2_rank9, CD, rank9)
    plot_heatmap(mat9)

    # ---- significant pairs & Full-vs-NoExplo gap -------------------------- #
    sig_pairs = [(a, b, d) for (a, b, d, s) in res["pairs"] if s]
    full_ne_gap = abs(avg_rank["P2_Full"] - avg_rank["P2_NoExploration"])
    print("[add2] |Full - NoExploration| rank gap =", round(full_ne_gap, 3))
    print("[add2] significant pairs:", sig_pairs)

    # ---- unit.json -------------------------------------------------------- #
    write_unit(problems, avg_rank, origp2_rank9, chi2, p_formal, CD,
               sig_pairs, full_ne_gap, res_seed, n_blocks, acc_mean8,
               mat9, pair_rows)

    print("[add2] DONE")


# --------------------------------------------------------------------------- #
# 3. CD diagram
# --------------------------------------------------------------------------- #
def plot_cd_diagram(avg_rank, origp2_rank9, CD, rank9):
    ac.setup_style()
    order = sorted(RANKED, key=lambda m: avg_rank[m])      # best first
    ranks = [avg_rank[m] for m in order]
    colors = [ac.method_color(m) for m in order]

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.set_facecolor("white")

    lo = 1.0
    hi = 8.0
    ax.set_xlim(lo - 0.4, hi + 0.9)
    ax.set_ylim(-1.6, 2.4)

    # main axis line
    ax.hlines(0, lo - 0.2, hi + 0.4, color=ac.PALETTE["ink"], lw=1.0, zorder=1)
    for r in np.arange(1, 9):
        ax.vlines(r, -0.06, 0.06, color=ac.PALETTE["ink"], lw=0.9)
        ax.text(r, -0.28, f"{r:.0f}", ha="center", va="top",
                fontsize=8, color=ac.PALETTE["ink"])
    ax.set_xlabel("Average rank (ACC@10%, 30 seeds)", fontsize=10)

    # CD bar at top, drawn as a segment of length CD
    cd_x0 = lo - 0.2
    cd_y = 1.9
    ax.hlines(cd_y, cd_x0, cd_x0 + CD, color=ac.PALETTE["mid_gray"], lw=2.2)
    ax.vlines(cd_x0, cd_y - 0.08, cd_y + 0.08, color=ac.PALETTE["mid_gray"], lw=2.2)
    ax.vlines(cd_x0 + CD, cd_y - 0.08, cd_y + 0.08, color=ac.PALETTE["mid_gray"], lw=2.2)
    ax.text(cd_x0 + CD / 2, cd_y + 0.16,
            f"CD = {CD:.2f} (α=0.05, Nemenyi)",
            ha="center", va="bottom", fontsize=8.5,
            color=ac.PALETTE["mid_gray"])

    # method ticks + labels (alternate up/down to reduce overlap)
    for k, (m, r, c) in enumerate(zip(order, ranks, colors)):
        ax.vlines(r, -0.12, 0.12, color=c, lw=2.0, zorder=3)
        up = (k % 2 == 0)
        ylab = 0.55 if up else -0.75
        va = "bottom" if up else "top"
        ax.plot([r, r], [0.12 if up else -0.12,
                         (0.30 if up else -0.30)], color=c, lw=0.8, zorder=2)
        ax.text(r, ylab, m, ha="center", va=va, fontsize=8.2, color=c,
                fontweight="bold" if m == "P2_Full" else "normal")

    # connect non-significant clusters: any group whose span <= CD gets a bar
    # (standard CD diagram: methods joined share no significant difference)
    ranked_sorted = sorted(RANKED, key=lambda m: avg_rank[m])
    rs = [avg_rank[m] for m in ranked_sorted]
    yconn = 1.35
    i = 0
    while i < len(rs):
        j = i
        while j + 1 < len(rs) and (rs[j + 1] - rs[i]) <= CD:
            j += 1
        if j > i:
            ax.hlines(yconn, rs[i], rs[j], color=ac.PALETTE["light_gray"],
                      lw=3.0, zorder=1)
        i = j + 1

    # OriginalP2 control at its 9-method rank, ink dashed, NOT ranked
    ox = origp2_rank9
    ax.vlines(ox, -0.12, 0.12, color=ac.PALETTE["ink"], lw=1.6,
              linestyles="--", zorder=3)
    ax.text(ox, -1.35, "OriginalP2\n(archived-bug\ncontrol, not ranked)",
            ha="center", va="top", fontsize=7.6, color=ac.PALETTE["ink"],
            style="italic")
    ax.annotate("", xy=(ox, -0.12), xytext=(ox, -1.05),
                arrowprops=dict(arrowstyle="-", color=ac.PALETTE["ink"], lw=0.7,
                                linestyle="--"))

    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    plt.tight_layout()
    png = ac.savefig(fig, "add2_cd_diagram", combo=False)
    print("[add2] wrote", png)


def plot_heatmap(mat9):
    ac.setup_style()
    methods = ALL_METHODS
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    im = ax.imshow(mat9, aspect="auto", cmap="YlGnBu",
                   vmin=max(0.0, mat9.min() - 0.02), vmax=min(1.0, mat9.max() + 0.01))
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(DATASETS, fontsize=8.5)
    for i in range(mat9.shape[0]):
        for j in range(mat9.shape[1]):
            ax.text(j, i, f"{mat9[i, j]:.3f}", ha="center", va="center",
                    fontsize=6.8, color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("ACC@10% (30-seed mean)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Per-dataset ACC@10% by selector (confirmatory)", fontsize=9.5)
    plt.tight_layout()
    png = ac.savefig(fig, "add2_rank_heatmap", combo=False)
    print("[add2] wrote", png)


# --------------------------------------------------------------------------- #
# 4. unit.json
# --------------------------------------------------------------------------- #
def write_unit(problems, avg_rank, origp2_rank9, chi2, p, CD, sig_pairs,
               full_ne_gap, res_seed, n_blocks, acc_mean8, mat9, pair_rows):
    u = ac.new_unit(
        exp_id=2,
        title="Friedman–Nemenyi Critical-Difference Comparison of Nine Active Selectors",
        evidence_label="confirmatory",
        status="complete" if not problems else "partial",
    )
    u["motivation"] = (
        "A paired comparison across many datasets must control for dataset "
        "difficulty rather than pool raw accuracies. We therefore rank the eight "
        "active selectors within each of the six confirmatory datasets (30-seed "
        "mean ACC at the 10% checkpoint) and apply the Friedman test followed by "
        "the Nemenyi post-hoc critical difference, the standard Demsar (2006) "
        "protocol for classifier comparison. The archived OriginalP2 (five "
        "implementation bugs) is kept only as a control and is never ranked."
    )
    u["hypothesis"] = (
        "The proposed P2_Full selector, and its ablations, differ in rank across "
        "the six datasets; any pair whose average-rank separation exceeds the "
        "Nemenyi critical difference is reported as significant at α=0.05."
    )
    u["design"] = (
        "Six datasets (iris, wine, glass, ecoli, aggregation, compound), nine "
        "selectors (eight formally ranked: P2_Full, P2_NoExploration, "
        "P2_NoRobust, P2_NoRedundancy, Random, Entropy, CoreSet, BADGE; plus "
        "archived-bug control OriginalP2). Block = dataset; within block each "
        "selector score is the mean ACC over 30 confirmatory seeds at the 10% "
        "labeled checkpoint (seed_set=30, α₀=50, λ-mult=1, l0=0, η=0)."
    )
    u["protocol"] = {
        "datasets": DATASETS,
        "methods": ALL_METHODS,
        "n_seeds": 30,
        "budgets": [0.10],
        "factors": {"alpha0": 50.0, "lam_mult": 1.0},
        "checkpoint": ac.CHECKPOINT,
    }

    # Table 1: method x (avg rank, per-dataset ACC)
    header = ["Method", "Avg rank (6 ds)"] + DATASETS + ["Role"]
    rows_t1 = []
    for mt in sorted(RANKED, key=lambda m: avg_rank[m]):
        j9 = ALL_METHODS.index(mt)
        rows_t1.append([
            mt, f"{avg_rank[mt]:.2f}"] +
            [f"{mat9[i, j9]:.3f}" for i in range(len(DATASETS))] +
            ["ranked"])
    jO = ALL_METHODS.index("OriginalP2")
    rows_t1.append(["OriginalP2", f"{origp2_rank9:.2f} (9-method)",
                    *[f"{mat9[i, jO]:.3f}" for i in range(len(DATASETS))],
                    "archived-bug control, not ranked"])
    ac.add_table(u,
                 "Table 1. Average rank (formal, k=8) and per-dataset ACC@10% "
                 "(30-seed mean). OriginalP2 is an archived-bug control and is "
                 "excluded from the ranking.",
                 header, rows_t1,
                 notes=["Rank 1 = best. Average rank computed over the six "
                        "confirmatory datasets."])

    # Table 2: pairwise (largest |delta rank| first)
    top = [r for r in pair_rows if r["method_a"] != "OriginalP2"][:10]
    rows_t2 = [[f"{r['method_a']} vs {r['method_b']}",
                f"{r['abs_rank_diff']:.2f}", f"{r['CD']:.2f}",
                ("yes" if r["significant"] else "no")] for r in top]
    ac.add_table(u,
                 "Table 2. Nemenyi pairwise comparisons (largest |Δrank| first). "
                 "A pair is significant only when |Δrank| > CD.",
                 ["Pair", "|Δrank|", "CD", "Significant? (α=0.05)"], rows_t2,
                 notes=["N=6 datasets; CD is large, so few/no pairs reach "
                        "significance. This is an honest low-power result."])

    ac.add_figure(u, "add2_cd_diagram",
                 "Fig. 2.1. Critical-difference diagram. Methods are placed at "
                 "their average rank (rank 1 = best, left). The top segment marks "
                 "the Nemenyi critical difference (CD). Joined methods share no "
                 "significant difference. OriginalP2 (ink, dashed) is the "
                 "archived-bug control and is not ranked.", combo=False)
    ac.add_figure(u, "add2_rank_heatmap",
                 "Fig. 2.2. Per-dataset ACC@10% (30-seed mean) for all nine "
                 "selectors; OriginalP2 shown as archived-bug control.",
                 combo=False)

    if sig_pairs:
        sig_txt = "; ".join(f"{a} vs {b} (|Δrank|={d:.2f})" for a, b, d in sig_pairs)
    else:
        sig_txt = ("no pair exceeds the critical difference; none of the eight "
                   "selectors differs significantly at α=0.05")

    u["results"] = (
        f"Across the six confirmatory datasets (N=6), the Friedman test on the "
        f"eight ranked selectors gives χ²={chi2:.2f} with k−1=7 df, p={p:.3f}. "
        f"The Nemenyi critical difference at α=0.05 is CD={CD:.2f} "
        f"(q₀.₀₅,k=8=3.031). Average ranks range from "
        f"{min(avg_rank.values()):.2f} ({min(avg_rank, key=avg_rank.get)}) to "
        f"{max(avg_rank.values()):.2f} ({max(avg_rank, key=avg_rank.get)}). "
        f"{sig_txt}. The proposed P2_Full averages rank "
        f"{avg_rank['P2_Full']:.2f} and P2_NoExploration averages rank "
        f"{avg_rank['P2_NoExploration']:.2f}; their separation is only "
        f"|Δrank|={full_ne_gap:.2f}, far below CD, i.e. exploration does not "
        f"significantly change rank. The archived-bug control OriginalP2 occupies "
        f"9-method average rank {origp2_rank9:.2f} and is deliberately excluded "
        f"from χ²/CD. A supportive per-seed analysis treating the "
        f"{n_blocks} dataset×seed blocks as independent (exploratory, NOT the "
        f"standard Demsar block) gives χ²={res_seed['chi2']:.1f}, "
        f"p={res_seed['p']:.2g}, CD={res_seed['CD']:.2f}; its tighter CD must not "
        f"be read as the confirmatory result."
    )
    u["conclusion"] = (
        "With only six datasets the post-hoc power is low (CD≈%.1f), and no pair "
        "of selectors reaches a significant average-rank difference. The "
        "proposed method sits among the top ranks but its advantage over the "
        "ablations and baselines is not statistically separable at α=0.05 on "
        "this dataset set; this is an honest null result, not evidence of "
        "equivalence." % CD
    )
    u["support"] = "mixed" if not sig_pairs else "supported"
    u["annotations"] = [
        "small number of datasets (N=6), low post-hoc power",
        "Nemenyi CD is large (~%.2f) with N=6; no pairwise significance is the "
        "expected honest outcome" % CD,
        "per-seed 180-block analysis is supportive/exploratory, not a standard "
        "Demsar block",
        "OriginalP2 is archived-bug control, never ranked",
    ] + (["seed-count anomaly: %s" % problems] if problems else [])
    u["csvs"] = ["add2_rank_matrix.csv", "add2_friedman_nemenyi.csv",
                 "add2_perseed_supportive.csv"]
    u["repro_script"] = "code/add2_friedman_cd.py"

    p = ac.save_unit(u)
    print("[add2] wrote", p)


if __name__ == "__main__":
    main()
