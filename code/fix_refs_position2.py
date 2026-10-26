"""fix_refs_position2.py - MOVE the misplaced [46]-[50] paragraph elements
(intact XML) to just before the 'Supplementary material' heading."""
import docx

P = r"D:\AL_FuzzyClustering_Paper\v2_unified\draft\DRAL_revised_final.docx"
d = docx.Document(P)
ps = d.paragraphs

mis = [p for p in ps if p.text.strip().startswith(tuple(f"[{n}]" for n in range(46, 51)))]
print("misplaced:", len(mis))
assert len(mis) == 5

anchor = None
for p in ps:
    if p.style.name == "Heading 1" and p.text.strip() == "Supplementary material":
        anchor = p
        break
assert anchor is not None

# move in reverse order so final document order stays [46..50]
for p in reversed(mis):
    anchor._p.addprevious(p._p)
    print("moved:", p.text.strip()[:55])

d.save(P)

# verify
d2 = docx.Document(P)
ps2 = d2.paragraphs
refs_h = [i for i, p in enumerate(ps2) if p.style.name == "Heading 1" and p.text.strip() == "References"][0]
sup_h = [i for i, p in enumerate(ps2) if p.style.name == "Heading 1" and p.text.strip() == "Supplementary material"][0]
print("refs block:", refs_h, "->", sup_h)
tail = [(i, ps2[i].text.strip()[:70]) for i in range(sup_h - 7, sup_h) if ps2[i].text.strip()]
for i, t in tail:
    print(i, t)
bib_before_refs = sum(1 for i, p in enumerate(ps2) if p.style.name == "Bibliography" and i < refs_h)
bib_total = sum(1 for p in ps2 if p.style.name == "Bibliography")
print("bib before References heading:", bib_before_refs, "| total bib:", bib_total)
