"""insert_citations_46_50.py - insert [46]-[50] at first-naming locations.
Run-level replacement so OMML math objects are untouched."""
import docx

P = r"D:\AL_FuzzyClustering_Paper\v2_unified\draft\DRAL_revised_final.docx"
d = docx.Document(P)

TASKS = [
    # (paragraph index, search string, replacement)
    (159, "Holm correction", "Holm correction [48]"),
    (195, "DerSimonian-Laird random-effects meta-analysis",
          "DerSimonian-Laird random-effects meta-analysis [47]"),
    (201, "Jonckheere-Terpstra trend", "Jonckheere-Terpstra trend [46]"),
    (201, "Benjamini-Hochberg FDR correction",
          "Benjamini-Hochberg FDR correction [49]"),
    (207, "Benjamini-Hochberg FDR correction",
          "Benjamini-Hochberg FDR correction [49]"),
    (207, "Standardised effect sizes d_z", "Standardised effect sizes d_z [50]"),
    (222, "Jonckheere-Terpstra permutation test",
          "Jonckheere-Terpstra permutation test [46]"),
]

def replace_in_paragraph(para, old, new):
    # try single-run hit first
    for r in para.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    # cross-run: merge text, split at insertion point into runs
    full = "".join(r.text for r in para.runs)
    if old not in full:
        return False
    pos = full.index(old)
    head, tail = full[:pos + len(old)], full[pos + len(old):]
    # rebuild: clear runs, write head, marker run, tail
    for r in para.runs:
        r.text = ""
    runs = para.runs
    # fallback: if no runs at all, add one
    if not runs:
        para.add_run("")
        runs = para.runs
    runs[0].text = head + new + tail
    return True

ok, fail = [], []
for idx, old, new in TASKS:
    para = d.paragraphs[idx]
    if replace_in_paragraph(para, old, new):
        ok.append((idx, old))
    else:
        fail.append((idx, old))

print("inserted:", ok)
print("FAILED:", fail)

d.save(P)

# verify
d2 = docx.Document(P)
alltext = "\n".join(p.text for p in d2.paragraphs)
for n in range(46, 51):
    print(f"[{n}] occurrences now:", alltext.count(f"[{n}]"))
