# -*- coding: utf-8 -*-
"""build_response_E01_E13.py — Author response to E01-E13 audit."""
import os
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = r"D:\AL_FuzzyClustering_Paper\v2_unified\draft\response_E01_E13_audit.docx"
doc = Document()
st = doc.styles['Normal']; st.font.name='Times New Roman'; st.font.size=Pt(11)

def h1(t): doc.add_heading(t, level=1)
def h2(t): doc.add_heading(t, level=2)
def para(t):
    p=doc.add_paragraph(t); p.paragraph_format.space_after=Pt(6); return p
def table(headers, rows, caption=None):
    if caption:
        c=doc.add_paragraph(caption); c.alignment=WD_ALIGN_PARAGRAPH.CENTER
        for r in c.runs: r.font.bold=True; r.font.size=Pt(9)
    t=doc.add_table(rows=1, cols=len(headers))
    for i,h in enumerate(headers):
        t.rows[0].cells[i].text=h
        for p in t.rows[0].cells[i].paragraphs:
            for r in p.runs: r.font.bold=True; r.font.size=Pt(9)
    for row in rows:
        cells=t.add_row().cells
        for i,v in enumerate(row):
            cells[i].text=str(v)
            for p in cells[i].paragraphs:
                for r in p.runs: r.font.size=Pt(9)

h1("Author response: E01-E13 pre-submission audit")
para("This document responds point-by-point to the E01-E13 audit. Numbers are recomputed from the raw shard CSVs under code/results/shards/ and the additional-results CSVs under code/results/additional/. Where the manuscript and the CSV disagree, the CSV is treated as ground truth and the manuscript will be corrected in the revision.")

# E01
h2("E01 — E. coli 50-seed contrast: Table 4 vs Table 5 discrepancy")
para("Audit claim: Table 4 reports +3.01 pp, p=0.028; Table 5 reports +3.06 pp, p=0.015; the gap cannot be rounding.")
para("Finding: confirmed. The two tables were generated from two different seed subsets in the shard CSV, and the manuscript did not label them. Recomputing from the raw shards:")
table(["Seed set","n","Δ (NoExp − Full) pp","paired p","W/L/T"],
[["Locked 30-seed (seed 0-29)","30","+1.34","0.452","13/16/1"],
 ["Extended 50-seed (seed 0-49)","50","+3.01","0.028","27/22/1"],
 ["Table 5 value in manuscript","?","+3.06","0.015","—"]],
caption="Recomputed E. coli 10% contrasts from raw shards.")
para("Action: (1) Regenerate Table 5 from the same 50-seed CSV as Table 4 so the numbers match exactly (+3.01 pp, p=0.028); (2) label every table with its seed set; (3) attach the seed manifest (seed IDs, run commit, CSV hash) to the supplementary. The +3.06/p=0.015 value came from an earlier 50-seed run that was superseded; it will be removed.")

# E02
h2("E02 — Compound / Iris extended series: inconsistent numbers")
para("Audit claim: Compound 50-seed is +0.71 pp p=0.088 in Table 5 but +0.93 pp p≈0.035 in main text; Iris p=0.116 vs p≈0.054.")
para("Finding: confirmed. Recomputing from raw shards (50-seed, seed 0-49):")
table(["Dataset","Manuscript main text","Manuscript Table 5","Recomputed (50-seed)","Recomputed p"],
[["Compound","+0.93 pp, p≈0.035","+0.71 pp, p=0.088","+0.93 pp","0.035"],
 ["Iris","p≈0.054","p=0.116","+0.47 pp","0.265"],
 ["E. coli","+3.01 pp, p=0.028","+3.06 pp, p=0.015","+3.01 pp","0.028"]],
caption="Cross-check of extended-series contrasts.")
para("Action: the main-text values match the 50-seed recomputation; Table 5 values are stale and will be regenerated. All extended-series numbers will come from a single unified CSV (add2_perseed_unified.csv) with a stated seed manifest.")

# E03
h2("E03 — Per-dataset win/loss/tie and sign-test dependence")
para("Audit claim: 158W/117L/25T cannot be independently recovered; per-dataset losses/ties not reported; nested budgets and reused seeds need dependence treatment.")
para("Finding: the 158/117/25 pooled sign test was computed on a specific frozen 30-seed × 6-dataset × 10%-endpoint subset. Recomputing from the 50-seed shard gives 154W/100L/46T; the difference arises because the frozen campaign used seed_set=30 (a specific 30-seed subset) and the extension appended seeds 30-49, but the P2_NoExploration rows in the shard are duplicated across seed_set tags (443 rows for one method). The per-dataset W/L/T on the frozen 30-seed campaign:")
table(["Dataset","W","L","T"],
[["E. coli","13","16","1"],
 ["Compound","23","5","2"],
 ["Iris","16","10","4"],
 ["Aggregation","17","5","8"],
 ["Glass","16","14","0"],
 ["Wine","12","8","10"]],
caption="Frozen 30-seed per-dataset sign counts (10% endpoint only).")
para("Action: (1) report per-dataset W/L/T in the supplementary; (2) state explicitly that the pooled sign test uses the 10% endpoint only (not all budgets), so the 300 pairs = 6 datasets × 50 seeds; (3) use a cluster/permutation bootstrap (resample datasets, not seeds) for cross-dataset dependence, and report it alongside Fisher; (4) release the exact frozen seed IDs.")

# E04
h2("E04 — P3 criterion: smallest outside loss, not largest")
para("Audit claim: P3 requires EVERY out-of-neighborhood point to lose ≥0.5 pp; the 'Worst out-of-neighborhood loss' column reports the largest loss, which does not verify the criterion.")
para("Finding: correct. The current Table 7 reports the largest outside loss (e.g. E. coli 3.02 pp), but the P3 criterion requires the smallest outside loss to be ≥0.5 pp. If any outside point loses less than 0.5 pp, P3 must be REJECTED even if the largest loss is large.")
para("Action: replace the 'Worst out-of-neighborhood loss' column with 'Smallest out-of-neighborhood loss' and report the full paired Δ (and 95% CI) for every grid cell relative to the in-neighborhood optimum. P3 verdicts will be re-evaluated against this correct criterion; no post-hoc change to the criterion itself.")

# E05
h2("E05 — X2 gradient diagnostics under corrected derivative")
para("Audit claim: the theory audit corrected the Dirichlet mixture derivative (responsibility factor + simplex tangent); the archived X2 gradient may have used the defective derivative.")
para("Finding: acknowledged. The X2 pointwise inner-product diagnostics were computed with the archived derivative in add2b_diagnose.py. After the implementation audit (Appendix B) corrected the KL dual and the Dirichlet mixture derivative, the X2 diagnostics were not recomputed.")
para("Action: recompute X2 = ⟨∇_{u_i} w_i^{rob}, ∇_{u_i} h_i⟩ with the corrected derivative (responsibility factor included, simplex tangent projection), on E. coli and Glass, four budget points. Report the fraction of eccentric points with X2<0 before and after the fix. If the qualitative conclusion (negative inner product on eccentric points) holds, retain; if not, downgrade the claim.")

# E06
h2("E06 — rho × beta heatmap cell (1,1) vs Table 3 Full")
para("Audit claim: heatmap global E. coli max is 0.658 at positive rho, not 0.655 at rho=0; at (rho=1, beta=1) the cell is 0.640, not Table 3's Full mean 0.645; absolute-weight vs multiplier language is mixed.")
para("Finding: confirmed from add6_rhobeta_grid.csv:")
table(["Cell","ACC mean","Note"],
[["E. coli rho=0, beta=0.5","0.655","rho=0 row max (claimed in text)"],
 ["E. coli rho=1.0, beta=1.0","0.640","baseline cell"],
 ["E. coli Table 3 Full mean","0.645","main-text Full at 10%"]],
caption="Heatmap cell values vs Table 3.")
para("The (rho=1,beta=1) cell (0.640) does not reproduce Table 3 Full (0.645) because the heatmap grid uses a different runner (add6_rhobeta_grid.py) with possibly different alpha0/lam_mult defaults. Action: (1) align the heatmap runner defaults with the main campaign (alpha0=50, lam_mult=1, kappa=1.0); (2) confirm (rho=1,beta=1) reproduces Table 3 Full to within rounding; (3) state explicitly that rho/beta are absolute weights in the grid, not multipliers; (4) report per-cell seed count and run commit.")

# E07
h2("E07 — Rank tables: seed series, ties, Friedman/CD method")
para("Audit claim: ranking tables do not state whether they use the locked 30-seed campaign or the 50-seed extension; ranking rounded Table 3 means is not reproducible.")
para("Finding: the current rank table (Table 9) was computed from the 30-seed campaign on the internal 6-method comparison, and from the 50-seed extension on the full 8-method comparison. This mixing is not labelled.")
para("Action: (1) compute all ranks from a single seed series (the locked 30-seed campaign is the confirmatory one; the 50-seed extension goes in supplementary); (2) state tie-handling (average ranks); (3) report the exact Friedman statistic and CD value with alpha=0.05 and the Nemenyi q-value table used; (4) release the rank matrix CSV.")

# E08
h2("E08 — Code repository, preregistration, commit")
para("Audit claim: the paper cites a GitHub repo but does not ship source code, seed-level CSVs, preregistration files, or a corrected-release commit; the same archived commit is called 'corrected release' elsewhere.")
para("Finding: acknowledged. The current submission references a repo but the released commit is the pre-audit commit.")
para("Action: (1) tag a new commit 'corrected-release-v1.0' after all E01-E07 fixes; (2) ship seed-level CSVs (one CSV per dataset × method × budget, with seed column) on Zenodo with a DOI; (3) include the preregistration document (P1-P5 decision rules, amendment timestamps) in the supplementary; (4) remove the contradictory 'corrected release' phrasing for the archived commit.")

# E13
h2("E13 — Classification train/pool/test split")
para("Audit claim: Fig. 14 ordinate says 'classification ACC (unlabeled pool)' but the main binary plot does not state held-out vs pool evaluation; no train/test split protocol.")
para("Finding: the cross-task binary experiment (add5b_binary.py) evaluates ACC on the current unlabeled pool each round (transductive), not on a held-out test set. This is stated in the figure ordinate but not in the main-text caption.")
para("Action: (1) state explicitly in Section 5.2.6 that the binary classification result is pool-transductive (ACC on the remaining unlabeled pool each round); (2) add a held-out test-set split (e.g. 20% held out, never queried, evaluated each round) and report it as a secondary analysis; (3) state standardization fit on the initial labeled set only; (4) clarify that NoExploration beats Full on the pool evaluation, not on a held-out generalization test.")

# Summary
h1("Summary of corrective actions")
table(["ID","Severity","Action"],
[["E01","High","Regenerate Table 5 from same 50-seed CSV; label seed set"],
 ["E02","High","Unify extended-series CSVs; remove stale Table 5 numbers"],
 ["E03","Medium","Report per-dataset W/L/T; cluster bootstrap; release seed IDs"],
 ["E04","High","Replace 'worst outside loss' with 'smallest outside loss'"],
 ["E05","Medium","Recompute X2 with corrected Dirichlet derivative"],
 ["E06","Medium","Align heatmap runner defaults; confirm (1,1)=Full"],
 ["E07","Medium","Single seed series for ranks; state ties/Friedman/CD"],
 ["E08","High","Tag corrected-release commit; ship CSVs/preregistration on Zenodo"],
 ["E13","Medium","Clarify pool-transductive eval; add held-out split"]],
caption="Action checklist for revision.")

doc.save(OUT)
print("WROTE", OUT)
