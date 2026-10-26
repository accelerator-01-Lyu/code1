"""audit_docx.py - comprehensive docx-level audit for user P0/P1 items."""
import docx, re

P = r"D:\AL_FuzzyClustering_Paper\v2_unified\draft\DRAL_revised_final.docx"
d = docx.Document(P)
ps = d.paragraphs
texts = [p.text for p in ps]
alltext = "\n".join(texts)

print("=== Keywords occurrences ===")
for i, t in enumerate(texts):
    if "Keywords" in t or "Key words" in t:
        print(i, repr(t[:150]))

print("\n=== LaTeX-source formula counts ===")
for pat, lab in [(r"\$[^$]*\$", "inline $...$"), (r"\\mathcal", r"\mathcal"),
                 (r"\\ell", r"\ell"), (r"\\nu", r"\nu"),
                 (r"\\|", r"\|"), (r"\\sum", r"\sum"),
                 (r"\\langle", r"\langle"), (r"\\Phi", r"\Phi")]:
    n = len(re.findall(pat, alltext))
    print(f"  {lab}: {n}")

print("\n=== Section 6 heading and first 600 chars ===")
i6 = next(i for i, t in enumerate(texts) if t.strip().startswith("6") and "Discussion" in t and len(t) < 60)
print("idx", i6, texts[i6])
i6b = next(i for i, t in enumerate(texts) if i > i6 and t.strip())
print(texts[i6b][:600])

print("\n=== 5.2.5 heading search ===")
hits = [i for i, t in enumerate(texts) if "5.2.5" in t or ("P4" in t and len(t) < 60)]
print(hits[:10], [texts[i][:80] for i in hits[:10]])

print("\n=== 5.6 section content (paragraphs after '5.6') ===")
i56 = next((i for i, t in enumerate(texts) if re.match(r"^5\.6\s", t.strip())), None)
if i56 is not None:
    j = i56 + 1
    while j < len(ps) and ps[j].style.name not in ("Heading 2", "Heading 3") and j < i56 + 12:
        if texts[j].strip():
            print(j, texts[j][:200])
        j += 1
else:
    print("5.6 heading NOT FOUND")

print("\n=== TOST numbers in body ===")
for i, t in enumerate(texts):
    if "TOST" in t and t.strip() and i < 350:
        m = re.findall(r"\d+/12", t)
        print(i, "12-fractions:", m, "|", t[:150])

print("\n=== Table D1/D2 location ===")
for i, t in enumerate(texts):
    if re.match(r"Table D[12]", t.strip()):
        print(i, t[:100])
