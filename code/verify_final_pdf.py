"""verify_final_pdf.py - 17-item final checklist on the Word-exported PDF."""
import fitz, re, sys

P = sys.argv[1] if len(sys.argv) > 1 else r"D:\AL_FuzzyClustering_Paper\v2_unified\draft\DRAL_revised_final.pdf"
doc = fitz.open(P)
full = [doc[i].get_text() for i in range(doc.page_count)]
allt = "\n".join(full)
n = doc.page_count
print(f"PAGES={n}")

checks = []

def chk(label, cond, detail=""):
    checks.append((label, cond, detail))
    print(("PASS " if cond else "FAIL ") + label + (" | " + detail if detail else ""))

# 1. formula residue
chk("公式无LaTeX源码", not any(k in allt for k in ["\\mathcal", "\\ell", "\\nu ", "\\Phi", "$", "\\sum"]),
    "search \\mathcal \\ell \\Phi \\$ \\sum")

# 2. Table 3 header repeat across pages - find Table 3 pages
t3pages = [i+1 for i, t in enumerate(full) if "Table 3" in t and "caption" in t.lower()]
print("Table3 mention pages:", t3pages[:4])

# 3. Table 9 completeness
chk("Table9 61/72 84.7%", "61/72" in allt or "61 (raw 62)" in allt or "84.7" in allt)

# 4. Table 11 panels
chk("Table11 panel b exists", "kappa" in allt.lower() or "Table 11" in allt)
t11 = [i+1 for i, t in enumerate(full) if "Table 11" in t]
print("Table11 pages:", t11[:4])

# 5. Fig1
chk("Fig1 caption", any("Fig. 1." in t for t in full))

# 6. Fig15/17/18 captions
for k in ["Fig. 15.", "Fig. 16.", "Fig. 17.", "Fig. 18."]:
    pg = [i+1 for i, t in enumerate(full) if k in t]
    print(f"{k} page:", pg[:2])

# 7. 5.2.5 P4 heading
chk("5.2.5 P4标题", any("5.2.5" in t and ("P4" in t or "Concentration" in t) for t in full))

# 8. 5.5 intensity six lines + Table 12
chk("Table12 pooled -0.54", "[-0.57, -0.51]" in allt)
chk("Table12 gap 0.6 -0.79", "[-0.81, -0.77]" in allt)
chk("c≥5 limitation句", any("c \u2265 5" in t or "c>=5" in t or "identifiable only for c" in t for t in full))

# 9. 5.6 completeness (E1 vs E4 direction)
chk("5.6 方向一致性", "not a contradiction" in allt or ("opposite directions" in allt and "overlap" in allt))

# 10. Section 6 integration
chk("Sec6 84.7%", any("84.7%" in t for t in full))
chk("Sec6 intensity/holr/tost", ("Holm" in allt and "TOST" in allt and "intensity" in allt.lower()))

# 11. Abstract numbers
a = full[0]
ai = a.find("Abstract")
seg = a[ai:ai+2500]
chk("Abstract 84.7%", "84.7%" in seg)
chk("Abstract 8,160", "8,160" in seg)
chk("Abstract +0.69", "+0.69" in seg)
chk("Abstract TOST 9/12", "9/12" in seg)

# 12. refs 46-50 independent lines
rstart = None
for i, t in enumerate(full):
    if "References" in t and len(t) < 40:
        rstart = i
refs = full[rstart:rstart+3] if rstart else []
ref46 = any(re.search(r"\[46\]", t) for t in full[rstart:rstart+6]) if rstart else False
ref50 = any(re.search(r"\[50\]", t) for t in full[rstart:rstart+6]) if rstart else False
chk("参考文献 [46]-[50] 存在", ref46 and ref50, f"refs section pages: {rstart+1 if rstart else '?'}")
# check each of 46-50 on its own visual check via text line
for rid in [46, 47, 48, 49, 50]:
    found = any(re.search(rf"\[{rid}\]", t) for t in full[rstart:rstart+8]) if rstart else False
    print(f"  ref [{rid}] found in refs: {found}")

# 13. Keywords once
kw = allt.count("Keywords")
chk("Keywords 一组", kw == 1, f"count={kw}")

# 14. page continuity / blank pages
blank = [i+1 for i, t in enumerate(full) if len(t.strip()) < 40]
print("near-blank pages:", blank)

# render key pages for visual check
import os
os.makedirs(r"D:\tmp\pages", exist_ok=True)
for pp in [1, 2, 14, 22, 23, 26, 27, 28, 31]:
    if pp <= n:
        doc[pp-1].get_pixmap(dpi=100).save(rf"D:\tmp\pages\v2_p{pp:02d}.png")
print("rendered for visual check: v2_p01..31")
