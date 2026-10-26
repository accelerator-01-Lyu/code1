"""additional_common.py — SINGLE SOURCE OF TRUTH for the 8 additional-experiment units.

ALL FOUR WORKERS (A/B/C/D) MUST `import additional_common as ac` and reuse the
constants / statistics / plotting / unit-writers below. DO NOT re-implement
seeds, paired stats, Holm, TOST, d_z, Nemenyi CD, or the palette in worker code
— re-implementation causes protocol drift and is treated as a defect.

Hard scope (do not violate):
  * This effort writes ONLY under code/results/additional/, code/figures/additional/,
    code/data/, and new code/add_*.py scripts. Do NOT touch the main PR build
    (notes/build_PR.py), the main manuscript (draft/AL_DRAL_unified_PR.*),
    supplementary_material.*, code/results/shards/done.txt, or any frozen
    code/results/*.csv used by the main paper.
  * Workers produce CSV + figures (PNG+PDF) + one `add{N}_unit.json` per
    experiment. Workers DO NOT build the .docx — the organizer assembles one
    document from the eight unit.json files after all workers finish.

Frozen protocol (identical to the main paper; no chasing results):
  * Seeds: 30-seed confirmatory = 20261010..20261039 ; 20-seed = 20261010..20261029 ;
    small n=5 shards = 0..4. Every comparison is paired by-seed.
  * Clustering metrics: ACC after Hungarian (linear_sum_assignment) label matching,
    NMI (sklearn normalized_mutual_info_score), ARI (adjusted_rand_score).
  * Report: paired gap (method - reference), 95% paired-t CI, paired t p-value,
    Holm step-down across a family, TOST for equivalence with margin
    DELTA_NI = 0.005 (0.5 pp), standardized paired effect d_z.
  * Primary checkpoint = 10% labeled (CHECKPOINT=0.10); |L0| = max(c, floor(0.02 N)).
  * 30-seed results are confirmatory. Anything with extra datasets / 50 seeds /
    20 seeds / exploratory regression must be explicitly labelled
    "supportive" / "post-hoc" / "exploratory" / "underpowered" / "right-censored".
  * Never copy a user "expected" result; report what the data gives (supported /
    not supported / refuted / mixed). Do not fabricate numbers.

Theory red lines (do not regress):
  * Prop. 3 (formerly "Theorem 3") is a DATA-DEPENDENT first-order scale
    beta = lambda_2(L_R)/n = r_c ; it is NOT a universally optimal scale. Refer
    to it as "Proposition 3" only.
  * Prop. 2 score zero-mean holds exactly, with NO additive constant vector.
  * The double-counting pathology is characterized by the exact gradient
    condition (inner product of the bonus gradient with the worst-case robust
    gradient < 0).
  * OriginalP2 is the ARCHIVED artifact carrying five implementation bugs; it is
    an archived-bug control only and is NEVER ranked in the formal comparison.
  * The 50-seed E. coli / Compound results are supportive and do not survive Holm.

Math in unit.json text: use LaTeX inside single $...$ (e.g. $h_i=\\|u_i-\\bar u\\|_1$",
"$\\beta=\\lambda_2(L_R)/n$"). The final builder converts these to native OMML.
Do NOT paste raw LaTeX outside $...$ and do NOT embed images of equations.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DATA = os.path.join(HERE, "reference_repo", "data")
RES_ADD = os.path.join(HERE, "results", "additional")
FIG_ADD = os.path.join(HERE, "figures", "additional")
DATA_ADD = os.path.join(HERE, "data")
for _d in (RES_ADD, FIG_ADD, DATA_ADD):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Seeds and frozen protocol constants
# --------------------------------------------------------------------------- #
SEEDS_30 = list(range(20261010, 20261040))   # 30 confirmatory seeds
SEEDS_20 = list(range(20261010, 20261030))   # 20 seeds
SEEDS_5 = list(range(0, 5))                  # legacy small-n shards
INIT_FRAC = 0.02
M_MC = 50
KAPPA = 1.0
RHO = 0.1
DELTA_NI = 0.005
CHECKPOINT = 0.10
ALPHA0_MAIN = 50.0      # frozen main-shard value
LAM_MULT_MAIN = 1.0

# --------------------------------------------------------------------------- #
# Palette — identity colors (LOCKED) + low-saturation cool palette
# --------------------------------------------------------------------------- #
# Identity colors for the double-counting contrast: never change.
C_FULL = "#6A5A8F"            # P2_Full deep purple
C_NOEXPLO = "#D08AB6"         # P2_NoExploration pink (main paired-comparison figures)
C_NOEXPLO_MUTED = "#C9A0C4"   # desaturated pink ONLY for small multi-method legends

# Cool, low-saturation swatches (blue / purple / grey / teal). High-saturation
# red / orange / bright yellow are FORBIDDEN as main colors.
PALETTE = {
    "deep_purple": "#6A5A8F", "fog_purple": "#8B86C2", "light_purple": "#B8B5D8",
    "deep_blue": "#3E7AAB", "mid_blue": "#6B96C7", "light_blue": "#A9C4DE",
    "deep_teal": "#4F9D9D", "mid_teal": "#7FB8B4", "light_teal": "#B8D8D4",
    "ink": "#3D4048", "mid_gray": "#8A8F98", "light_gray": "#C9CDD3",
    "grid_gray": "#EDEFF2", "bg": "#FAFBFC",
}
# Auxiliary methods rotate fog_purple -> mid_blue -> mid_teal -> mid_gray ->
# light_purple -> light_blue (user-specified order); all from the cool swatches.
COLORS = {
    "P2_Full": "#6A5A8F",
    "P2_NoExploration": "#D08AB6",
    "Random": "#8B86C2",
    "Entropy": "#6B96C7",
    "CoreSet": "#7FB8B4",
    "BADGE": "#8A8F98",
    "P2_NoRobust": "#B8B5D8",
    "P2_NoRedundancy": "#A9C4DE",
    "OriginalP2": "#3D4048",       # ink, archived control (draw dashed, never ranked)
    "GMM_A": "#3E7AAB",
    "GMM_B": "#6B96C7",
    "teal": "#4F9D9D",
    "mid_teal": "#7FB8B4",
}
# coral / warm orange (#E8967A) is RETIRED; replace its former uses with
# mid_teal (#7FB8B4) or mid_gray (#8A8F98).

# Diverging heat-map colors (Exp 3 gap, Exp 6 rho-beta grid, any gap heatmap):
# positive = mid_teal, zero = grid_gray, negative = fog_purple. Never red-blue.
DIVERG_POS = "#7FB8B4"
DIVERG_ZERO = "#EDEFF2"
DIVERG_NEG = "#8B86C2"


def gap_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "gap_cool", [DIVERG_NEG, DIVERG_ZERO, DIVERG_POS])


def method_color(method):
    return COLORS.get(method, PALETTE["mid_gray"])


METHOD_ZORDER = ["P2_Full", "P2_NoExploration", "P2_NoRobust", "P2_NoRedundancy",
                 "Random", "Entropy", "CoreSet", "BADGE", "OriginalP2"]
# methods allowed in the FORMAL ranking (OriginalP2 excluded)
RANKED_METHODS = ["P2_Full", "P2_NoExploration", "P2_NoRobust", "P2_NoRedundancy",
                  "Random", "Entropy", "CoreSet", "BADGE"]


# --------------------------------------------------------------------------- #
# Clustering metrics
# --------------------------------------------------------------------------- #
def acc_hungarian(yt, yp):
    yt = np.asarray(yt, int); yp = np.asarray(yp, int)
    D = max(yt.max(), yp.max()) + 1
    C = np.zeros((D, D), int)
    for t, p in zip(yt, yp):
        C[t, p] += 1
    ri, ci = linear_sum_assignment(-C)
    return C[ri, ci].sum() / len(yt)


def cluster_metrics(yt, yp):
    from sklearn.metrics import normalized_mutual_info_score as nmi
    from sklearn.metrics import adjusted_rand_score as ari
    return {"ACC": acc_hungarian(yt, yp), "NMI": nmi(yt, yp), "ARI": ari(yt, yp)}


# --------------------------------------------------------------------------- #
# Paired statistics (gap = method - reference, paired by seed)
# --------------------------------------------------------------------------- #
def paired_ci(diff, alpha=0.05):
    """diff = paired differences (method - reference), length n. Returns mean
    gap, SE, two-sided t, raw p, 95% t-CI."""
    d = np.asarray(diff, float)
    n = len(d)
    m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
    tc = stats.t.ppf(1 - alpha / 2, n - 1)
    t = m / se if se > 0 else 0.0
    return {"n": n, "mean": float(m), "sd": float(sd), "se": float(se),
            "lo": float(m - tc * se), "hi": float(m + tc * se),
            "t": float(t), "p": float(2 * stats.t.sf(abs(t), n - 1)),
            "tcrit": float(tc)}


def cohen_dz(diff):
    d = np.asarray(diff, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def holm_adjust(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def tost(diff, margin=DELTA_NI):
    """Two one-sided tests of equivalence within [-margin, +margin].
    Returns p_tost (=max of the two one-sided p), and the two component p."""
    r = paired_ci(diff)
    n, m, se = r["n"], r["mean"], r["se"]
    t_lo = (m + margin) / se if se > 0 else np.inf      # H0: m <= -margin
    t_hi = (margin - m) / se if se > 0 else np.inf      # H0: m >= +margin
    p_lo = stats.t.sf(t_lo, n - 1)
    p_hi = stats.t.sf(t_hi, n - 1)
    return {"p_tost": float(max(p_lo, p_hi)),
            "p_low": float(p_lo), "p_high": float(p_hi),
            "equivalent": bool(max(p_lo, p_hi) < 0.05),
            "margin": margin}


def contrast_table(paired, reference, methods, metric="ACC"):
    """paired: dict method -> per-seed score array (all aligned by seed),
    reference: the reference method key. Returns rows of the standard contrast."""
    ref = np.asarray(paired[reference], float)
    pvals = {}
    rows = {}
    for mth in methods:
        if mth == reference:
            continue
        d = np.asarray(paired[mth], float) - ref
        r = paired_ci(d)
        rows[mth] = {"gap": r["mean"], "lo": r["lo"], "hi": r["hi"],
                     "p": r["p"], "dz": cohen_dz(d), "n": r["n"], "diff": d}
        pvals[mth] = r["p"]
    holm = holm_adjust([pvals[m] for m in pvals])
    for i, mth in enumerate(pvals):
        rows[mth]["p_holm"] = float(holm[i])
        rows[mth]["metric"] = metric
    return rows


# --------------------------------------------------------------------------- #
# Friedman + Nemenyi critical difference (Demsar 2006, alpha=0.05)
# --------------------------------------------------------------------------- #
# Studentized-range-based critical values used in CD = q*sqrt(k(k+1)/(6N))
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def friedman_nemenyi(score_matrix, method_names):
    """score_matrix: (N_datasets, k_methods) or (N, k) array where each row is a
    paired block (dataset x seed) with the ACC score for every method; higher is
    better. Returns average ranks (rank 1 = best), Friedman chi2/p, and CD."""
    W = np.asarray(score_matrix, float)
    N, k = W.shape
    ranks = np.apply_along_axis(lambda row: stats.rankdata(-row), 1, W)
    R = ranks.mean(axis=0)
    chi2 = 12.0 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4.0)
    p = float(stats.chi2.sf(chi2, k - 1))
    q = NEMENYI_Q05[k]
    CD = float(q * np.sqrt(k * (k + 1) / (6.0 * N)))
    pair = []
    for i in range(k):
        for j in range(i + 1, k):
            d = abs(R[i] - R[j])
            pair.append((method_names[i], method_names[j], float(d), bool(d > CD)))
    return {"avg_rank": dict(zip(method_names, R.tolist())),
            "chi2": float(chi2), "p": p, "CD": CD, "q": q, "N": N, "k": k,
            "pairs": pair}


# --------------------------------------------------------------------------- #
# Figure style (sans-serif, color-blind-safe, captions live in Word not image)
# --------------------------------------------------------------------------- #
def setup_style():
    ink = PALETTE["ink"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "text.color": ink,
        "axes.labelcolor": ink,
        "xtick.color": ink,
        "ytick.color": ink,
        "axes.edgecolor": ink,
        "figure.facecolor": "white",
        "axes.facecolor": PALETTE["bg"],
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid_gray"],
        "grid.alpha": 1.0,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    })


def style_ax(ax):
    """Horizontal-only light-gray grid on a near-white panel; ink-colored text."""
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(True, axis="y", color=PALETTE["grid_gray"], linewidth=0.8)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(PALETTE["ink"])
    ax.tick_params(colors=PALETTE["ink"])
    ax.xaxis.label.set_color(PALETTE["ink"])
    ax.yaxis.label.set_color(PALETTE["ink"])
    return ax


def ci_band(ax, x, lo, hi, color, alpha=0.15):
    """Soft paired-CI band: main color at alpha 0.12-0.18, no dark edge."""
    return ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)


def savefig(fig, stem, combo=False):
    """Line/single-panel art at 600 dpi; multi-panel combos at 500 dpi.
    Saves both PNG and PDF into code/figures/additional/. Do NOT put the figure
    caption (Fig. N. ...) inside the image; captions are supplied in unit.json."""
    dpi = 500 if combo else 600
    png = os.path.join(FIG_ADD, stem + ".png")
    pdf = os.path.join(FIG_ADD, stem + ".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


# --------------------------------------------------------------------------- #
# CSV / unit.json writers — the ONLY hand-off contract to the docx builder
# --------------------------------------------------------------------------- #
def res_path(name):
    return os.path.join(RES_ADD, name)


def save_csv(df, name):
    p = res_path(name)
    df.to_csv(p, index=False)
    return p


def new_unit(exp_id, title, evidence_label, status):
    """evidence_label in {confirmatory, supportive, post-hoc, exploratory};
    status in {complete, partial, blocked}. Tables are plain 2-D lists so the
    builder can render native three-line tables; no images of tables."""
    return {
        "exp_id": exp_id,
        "title": title,
        "evidence_label": evidence_label,
        "status": status,
        "motivation": "",          # English paragraph(s); math in $...$
        "hypothesis": "",          # the pre-registered hypothesis being tested
        "design": "",              # English: datasets, factors, methods, seeds, budgets
        "protocol": {              # machine-readable summary
            "datasets": [], "methods": [], "n_seeds": None,
            "budgets": [], "factors": {}, "checkpoint": CHECKPOINT,
        },
        "tables": [   # {"caption": str, "header": [..], "rows": [[..]], "notes": [..]}
        ],
        "figures": [  # {"file": "addN_x.png", "caption": "Fig. N. ... (English)"}
        ],
        "results": "",             # English narrative with the real numbers
        "conclusion": "",          # English, explicit
        "support": "mixed",        # supported | not supported | refuted | mixed
        "annotations": [],         # e.g. "post-hoc", "underpowered", "right-censored"
        "csvs": [],                # csv filenames under results/additional/
        "repro_script": "",        # code/add_*.py that reproduces this unit
    }


def save_unit(u):
    p = os.path.join(RES_ADD, "add%d_unit.json" % u["exp_id"])
    with open(p, "w", encoding="utf-8") as f:
        json.dump(u, f, ensure_ascii=False, indent=2)
    return p


def add_table(u, caption, header, rows, notes=None):
    u["tables"].append({"caption": caption, "header": header,
                        "rows": rows, "notes": notes or []})


def add_figure(u, stem, caption, combo=False):
    fn = stem + ".png"
    u["figures"].append({"file": fn, "caption": caption, "combo": combo})
    if fn not in u["csvs"]:
        u.setdefault("fig_files", []).append(fn)
    return os.path.join(FIG_ADD, fn)
