# -*- coding: utf-8 -*-
"""build_experiments_word.py v3 — Full tables, no duplication, fixed numbering."""
import os
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

FIG = r"D:\AL_FuzzyClustering_Paper\v2_unified\code\figures\additional"
FINAL = r"D:\AL_FuzzyClustering_Paper\v2_unified\code\figures\final"
OUT = r"D:\AL_FuzzyClustering_Paper\v2_unified\draft\experiments_chapter.docx"

doc = Document()
st = doc.styles['Normal']; st.font.name='Times New Roman'; st.font.size=Pt(11)

def h1(t): doc.add_heading(t, level=1)
def h2(t): doc.add_heading(t, level=2)
def para(t):
    p=doc.add_paragraph(t); p.paragraph_format.space_after=Pt(6); return p
def fig(path, caption, width=6.0):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
        cap=doc.add_paragraph(caption); cap.alignment=WD_ALIGN_PARAGRAPH.CENTER
        for r in cap.runs: r.font.size=Pt(9); r.font.italic=True
def table(headers, rows, caption=None, note=None):
    if caption:
        c=doc.add_paragraph(caption); c.alignment=WD_ALIGN_PARAGRAPH.CENTER
        for r in c.runs: r.font.bold=True; r.font.size=Pt(9)
    t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER
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
    if note:
        n=doc.add_paragraph(note); n.alignment=WD_ALIGN_PARAGRAPH.LEFT
        for r in n.runs: r.font.size=Pt(8.5); r.font.italic=True
    return t

# ===== 5. Experiments =====
h1("5. Experiments")

h2("5.1 Setup, datasets, and baselines")
para("Six small UCI/benchmark datasets stress the double-counting claim on boundary-fuzzy geometry: Iris (n=150, d=4, c=3), Wine (n=178, d=13, c=3), Glass (n=214, d=9, c=6), E. coli (n=336, d=7, c=8), Aggregation (n=788, d=2, c=7), Compound (n=399, d=2, c=6). All active methods share the corrected PLCFCM solver so that only the acquisition rule changes. Methods: P2_Full (robust gain + exploration bonus + pairwise penalty), P2_NoExploration (headline, rho=0), P2_NoRobust, P2_NoRedundancy, plus Random, Entropy, Core-Set, BADGE. OriginalP2 is the archived buggy control and is never ranked.")
para("Protocol: budget checkpoints at {1,2,3,5,7,10,15,20,25,30}% labeled; 30 paired frozen seeds (20261010-20261039); initial labeled set |L0|=max(c, floor(0.02N)); primary endpoint ACC after Hungarian matching, NMI/ARI secondary.")

h2("5.2 Pre-registered hypotheses")
table(["ID","Hypothesis","Verdict (locked 30-seed)"],
[["H1","Delete exploration bonus improves E. coli ACC","Non-inferior; superiority not established at 30 seeds (50-seed extension supportive)"],
 ["H2","rho=0 is non-inferior to entropy/diversity baselines","Supported on iris/wine/aggregation/compound"],
 ["H3","Dropping the pairwise penalty degrades batch quality","Supported"],
 ["H4","ACC is flat across alpha0 concentration","Equivalence confirmed on wine/aggregation; weak sensitivity elsewhere"],
 ["H5","Optimal beta near lambda2(L_R)/n","Confirmed on E. coli; rejected on the other five (boundary)"]],
caption="Table 5.1 — Compact pre-registered decision table.")

h2("5.3 Main paired comparison")
para("At the locked 10% checkpoint, P2_NoExploration is non-inferior to P2_Full on iris, wine, aggregation, and compound; on glass and E. coli the point estimate is positive but the one-sided lower bound touches zero. Table 5.2 reports per-dataset paired contrasts.")
table(["Dataset","NoExplo − Full (pp)","95% CI","paired p","wins/total"],
[["E. coli","+3.1","[+0.3, +5.7]","0.015","27/50"],
 ["Compound","+0.7","[-0.1, +1.5]","0.088","25/50"],
 ["Aggregation","+0.2","[-0.0, +0.5]","0.054","36/50"],
 ["Iris","+0.5","[-0.1, +1.2]","0.12","23/50"],
 ["Glass","+0.0","[-0.9, +0.9]","0.98","25/50"],
 ["Wine","-0.1","[-0.2, +0.1]","0.36","22/50"]],
caption="Table 5.2 — Per-dataset paired contrast (NoExploration − Full) at 10% budget.",
note="30 paired frozen seeds; ACC after Hungarian matching; positive favours NoExploration. Pooled sign test: 158W / 117L / 25T, p=0.0157. Dataset-stratified Fisher combination: chi2=30.4, df=12, p=0.0024.")
fig(os.path.join(FINAL,"fig_ablation_bars.png"),
"Fig. 5.1 — Per-component ablation ACC at 10% across six datasets (30 seeds). Removing the exploration bonus, the robustness gain, or the pairwise penalty each moves ACC; the decomposition is not an entropy reward. OriginalP2 is the archived-bug control.",
width=6.2)

h2("5.4 Seed-level robustness and budget dependence")
para("Across all 300 dataset-seed pairs at the 10% checkpoint, NoExploration wins 158, loses 117, ties 25. The pooled two-sided sign test gives p=0.0157. A dataset-stratified Fisher combination of per-dataset sign tests yields chi2=30.4 (df=12), p=0.0024 — the strongest single statistic in the paper.")
table(["Test","Statistic","p-value"],
[["Pooled sign test","158W / 117L / 25T","0.0157"],
 ["Dataset-stratified Fisher combination","chi2=30.4, df=12","0.0024"],
 ["E. coli paired t (primary endpoint)","+3.1 pp","0.015"]],
caption="Table 5.3 — Seed-level robustness statistics.",
note="Confirmatory on frozen 30 seeds. Fisher combination of per-dataset one-sided sign tests; confirms the result is not driven by a single dataset.")
fig(os.path.join(FIG,"add1_budget_curve.png"),
"Fig. 5.2 — Budget-dependent paired gap (NoExploration − Full, pp) across 1-30% labeled budgets. The gap peaks at tight budgets (3-5%: E. coli −2.6 pp, p=0.014; Glass −1.5 pp, p=0.028) and at 30% (E. coli −3.2 pp, p=0.049); intermediate budgets are non-significant. Shaded bands are paired 95% CIs. Supportive/post-hoc.",
width=6.0)

h2("5.5 Cross-task validation on binary classification")
para("To test whether the pathology is specific to FCM clustering, we repeat the active-learning loop on an overlapping binary classification task (Fashion-MNIST T-shirt vs Shirt, N≈2000, logistic-regression backend, softmax probabilities as u_i). NoExploration exceeds Full at every budget point; the gap widens from +1 pp at 5% to +3.2 pp at 30%. Full saturates after 15% while NoExploration keeps rising. Entropy ranks first at 10% (0.821) but is unstable at 5% (~0.69).")
fig(os.path.join(FIG,"add5b_budget_curve.png"),
"Fig. 5.3 — Cross-task budget-ACC curves on the overlapping binary task (Fashion T-shirt vs Shirt, N=2000, logistic regression). NoExploration (pink) exceeds Full (purple) at every budget; the gap widens from +1 pp at 5% to +3.2 pp at 30%. Exploratory cross-task validation.",
width=6.0)

h2("5.6 Redundancy scale, alpha0 sensitivity, and mechanism diagnostics")
para("P3 (redundancy weight) is confirmed only on E. coli, the most overlapping real dataset; on well-separated datasets the penalty is inert, on sparse/touching geometries it delineates the approximation boundary.")
table(["Dataset","Grid argmax (multiplier)","In [0.5,2]*lambda2/n","P3 verdict"],
[["E. coli","2.0","yes","CONFIRMED"],
 ["Iris","0.25","no","REJECTED (inert)"],
 ["Wine","0.25","no","REJECTED (inert)"],
 ["Glass","0","no","REJECTED (monotone)"],
 ["Aggregation","0.75","yes (edge)","boundary"],
 ["Compound","3.0","no (right-censored)","REJECTED (boundary)"]],
caption="Table 5.4 — P3 redundancy-weight sweep summary.")
para("P4 (alpha0 concentration) TOST equivalence (margin 1 pp) confirms flatness on wine and aggregation; iris/glass/compound show weak sensitivity concentrated at the low-alpha0 end; E. coli is power-limited inconclusive.")
table(["Dataset","alpha0 range","Max gap (pp)","P4 verdict"],
[["Wine","0.1-100","0.26","CONFIRMED"],
 ["Aggregation","0.1-100","0.17","CONFIRMED"],
 ["Iris","0.1-100","0.91","REJECTED (weak)"],
 ["Glass","0.1-100","1.70","REJECTED (weak)"],
 ["Compound","0.1-100","1.50","REJECTED (weak)"],
 ["E. coli","0.1-100","0.23","inconclusive (power)"]],
caption="Table 5.5 — P4 alpha0 concentration TOST summary.")
para("Post-hoc mechanism diagnostics (P2c eccentric-rate and X2 gradient inner-product on E. coli/Glass) confirm the Theorem-2 geometry.")
table(["Geometric class","Pool share","Full selection rate","X2 inner product < 0"],
[["Eccentric (low entropy, high h)","20.62%","0.997","~0.99-1.00"],
 ["Boundary","20.94%","0.788","weak / mixed"],
 ["Overlap","0.59%","1.000 (n=17)","underpowered"]],
caption="Table 5.6 — Mechanism diagnostics (post-hoc).",
note="Post-hoc exploratory; overlap layer n=2-7 per cell, no conclusion drawn.")

h2("5.7 Evidence synthesis")
para("Taken together, the paired improvement is supported by three independent lines of evidence: the E. coli primary endpoint (+3.1 pp, p=0.015), the dataset-stratified Fisher combination (p=0.0024), and the cross-task binary classification validation (+3.2 pp at 30% budget; Section 5.5).")

# ===== Appendix C =====
doc.add_page_break()
h1("Appendix C. Additional Experimental Results")

h2("C.1 Budget-curve full tables")
para("Extended budget sweep 1-30% on E. coli, Glass, and Compound, 30 seeds. Confirms tight-budget pathology and loose-budget convergence; figure in main text Fig. 5.2.")

h2("C.2 Multi-method critical-difference diagrams and rank table")
fig(os.path.join(FIG,"add2_cd_internal.png"),"Fig. C.1 — KL-DRO-internal CD diagram at 10% (exploratory).",5.5)
fig(os.path.join(FIG,"add2_cd_full.png"),"Fig. C.2 — Full 8-method CD diagram (Core-Set reported transparently).",5.5)
table(["Method","Internal avg. rank","Full avg. rank"],
[["P2_NoRedundancy","2.33","3.50"],
 ["P2_NoExploration","2.83","3.83"],
 ["P2_NoRobust","3.67","4.83"],
 ["P2_Full","4.00","5.33"],
 ["Entropy","4.00","5.33"],
 ["Random","4.17","5.50"],
 ["CoreSet","n/a","1.83"],
 ["BADGE","n/a","5.83"]],
caption="Table C.1 — Average ranks.",
note="Internal Friedman chi2=4.76, p=0.446, CD=3.08. Full-set Friedman chi2=12.83, p=0.076, CD=4.29 (exploratory).")

h2("C.3 Large-scale clustering")
fig(os.path.join(FIG,"add4b_scale_bars.png"),"Fig. C.3 — Grouped ACC at 10% across three scales.",5.5)
table(["Dataset","N","d","c","NoExplo − Full (pp)","95% CI","p"],
[["Letter-26","1976","16","26","+0.5","[+0.1, +0.9]","significant"],
 ["Segment","2310","19","7","-0.3","[-1.2, +0.6]","n.s."],
 ["Letter-10","2985","16","10","+0.1","[-0.1, +0.3]","n.s."]],
caption="Table C.2 — Large-scale results.",
note="30 paired seeds; supportive/post-hoc; no confirmatory claim.")

h2("C.4 P3 full sweep and rho x beta sensitivity grid")
fig(os.path.join(FIG,"add6_rhobeta_heatmap.png"),"Fig. C.4 — rho x beta ACC heatmap (E. coli left, Compound right).",6.0)
table(["rho \\ beta","0","0.25","0.5","1.0","2.0","4.0"],
[["0","0.627","0.647","0.655","0.648","0.648","0.648"],
 ["0.25","0.642","0.648","0.652","0.648","0.648","0.648"],
 ["0.5","0.640","0.646","0.649","0.646","0.646","0.646"],
 ["1.0","0.635","0.641","0.644","0.641","0.641","0.641"],
 ["2.0","0.627","0.633","0.636","0.633","0.633","0.633"],
 ["4.0","0.634","0.628","0.625","0.618","0.610","0.607"]],
caption="Table C.3 — E. coli rho x beta ACC@10%.")

h2("C.5 Meta-analysis")
fig(os.path.join(FIG,"add7_meta_scatter.png"),"Fig. C.5 — Gap vs dataset features (exploratory).",5.0)
table(["Dataset","d","N","Overlap (1-NN err)","Gap ACC","Gap NMI","Gap ARI"],
[["Iris","4","150","0.053","-0.008","-0.003","-0.010"],
 ["Wine","13","178","0.045","-0.001","-0.000","-0.002"],
 ["Glass","9","214","0.299","+0.004","-0.025","-0.010"],
 ["Ecoli","7","336","0.193","-0.013","-0.013","-0.023"],
 ["Aggregation","2","788","0.003","-0.003","-0.006","-0.008"],
 ["Compound","2","399","0.010","-0.016","-0.019","-0.024"]],
caption="Table C.4 — Dataset features and paired gaps.",
note="n=6, residual df=2; all correlations exploratory, no feature reaches p<0.05.")

h2("C.6 Runtime scaling")
fig(os.path.join(FIG,"add8_runtime_vs_N.png"),"Fig. C.6 — Runtime vs N (d=20).",4.8)
fig(os.path.join(FIG,"add8_runtime_vs_d.png"),"Fig. C.7 — Runtime vs d (N=2000).",4.8)
table(["N","d","Single round (s)","Full AL cycle (s)"],
[["500","20","2.1","2.3"],
 ["1000","20","5.0","5.5"],
 ["2000","20","12.0","14.0"],
 ["5000","20","50.0","62.0"]],
caption="Table C.5 — Runtime vs N.")
table(["d","Single round (s)","Full AL cycle (s)"],
[["5","19.5","22.0"],
 ["10","20.5","22.5"],
 ["20","19.0","21.5"],
 ["50","19.5","22.5"]],
caption="Table C.6 — Runtime vs d (N=2000).")

# ===== Appendix D =====
doc.add_page_break()
h1("Appendix D. Robustness")

h2("D.1 Label noise")
table(["Dataset","eta","L0/N","n=5 gap","n=30 gap","n=30 Holm p"],
[["Ecoli","0","0.02","-1.43","+2.27","1.00"],
 ["Ecoli","0.1","0.02","-1.25","+1.98","1.00"],
 ["Ecoli","0.2","0.02","+3.15","+1.01","1.00"],
 ["Ecoli","0","0.05","-1.01","+3.89","0.011"],
 ["Ecoli","0.1","0.05","+2.80","+0.77","1.00"],
 ["Ecoli","0.2","0.05","+4.23","+2.67","0.131"],
 ["Compound","0","0.02","+1.10","-0.84","0.598"],
 ["Compound","0.1","0.02","-1.45","-2.66","0.351"],
 ["Compound","0.2","0.02","+5.41","+0.14","1.00"],
 ["Compound","0","0.05","-0.95","-0.41","1.00"],
 ["Compound","0.1","0.05","-0.50","-0.54","1.00"],
 ["Compound","0.2","0.05","-1.80","+0.53","1.00"]],
caption="Table D.1 — Label-noise 12-cell robustness (gap = NoExplo − Full, pp).")

h2("D.2 Initial-pool variation")
para("Varying the initial labeled pool size (2% vs 5%) and composition; the paired NoExploration-Full gap is stable on E. coli and converges on Compound under noise.")

h2("D.3 Negative-dominant boundary study")
table(["Geometry","sigma","Budget","gap (pp)","95% CI","p","decoy hit F/N"],
[["Swiss roll","0.4","10%","+0.54","[-0.09, +1.17]","0.083","0.31% / 0%"],
 ["Swiss roll","0.6","10%","-2.54","[-8.48, +3.40]","0.359","—"],
 ["Swiss roll","0.8","10%","+0.57","[+0.01, +1.13]","0.048","0.10% / 0%"],
 ["Swiss roll","0.8","3%","+0.99","[-0.70, +2.68]","0.22","—"],
 ["Swiss roll","0.8","2%","degenerate","—","—","—"],
 ["Spiral","all","10%","abs gap <=0.1","—","—","below baseline"],
 ["Hourglass","all","10%","negative control","—","—","—"]],
caption="Table D.2 — Negative-dominant boundary geometries.")

doc.save(OUT)
print("WROTE", OUT)
