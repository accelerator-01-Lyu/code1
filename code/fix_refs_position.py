"""fix_refs_position.py - move [46]-[50] from mis-inserted location into the
References block end (just before 'Supplementary material' heading)."""
import copy
import docx
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

P = r"D:\AL_FuzzyClustering_Paper\v2_unified\draft\DRAL_revised_final.docx"
d = docx.Document(P)
ps = d.paragraphs

# 1. collect the 5 misplaced paragraphs (by exact [46..50] prefixes)
mis = []
for p in ps:
    t = p.text.strip()
    if any(t.startswith(f"[{n}]") for n in range(46, 51)):
        mis.append(p)
print("misplaced count:", len(mis))
assert len(mis) == 5

# 2. anchor = 'Supplementary material' Heading 1 (References end)
anchor = None
for p in ps:
    if p.style.name == "Heading 1" and p.text.strip() == "Supplementary material":
        anchor = p
        break
assert anchor is not None
print("anchor:", anchor.text)

# 3. clone [45] formatting template
tpl = None
for p in ps:
    if p.text.strip().startswith("[45]"):
        tpl = p
        break
assert tpl is not None

texts = [p.text for p in mis]

# 4. remove misplaced paragraphs
for p in mis:
    el = p._p
    el.getparent().remove(el)

# 5. re-insert in order before anchor (anchor is the last 'Supplementary material'
#    heading after refs; rebuild relative to current document)
def insert_before(anchor_p, template_p, text):
    new_el = copy.deepcopy(template_p._p)
    for r in new_el.findall(qn("w:r")):
        new_el.remove(r)
    anchor_p._p.addprevious(new_el)
    return Paragraph(new_el, anchor_p._parent)

for t in texts:
    insert_before(anchor, tpl, t)
    print("moved:", t[:50])

d.save(P)

# 6. verify
d2 = docx.Document(P)
ps2 = d2.paragraphs
bib_idx = [i for i, p in enumerate(ps2) if p.style.name == "Bibliography"]
print("bibliography paragraphs:", len(bib_idx))
last_refs = [ps2[i].text.strip()[:60] for i in bib_idx[-6:]]
print("last 6 bib entries:", last_refs)
# check nothing before References heading
refs_h = [i for i, p in enumerate(ps2) if p.style.name == "Heading 1" and p.text.strip() == "References"][0]
early = [i for i in bib_idx if i < refs_h]
print("bib entries before References heading:", len(early))
