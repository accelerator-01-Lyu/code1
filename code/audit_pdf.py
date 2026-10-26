"""audit_pdf.py - verify user's P0/P1 claims on the current PDF."""
import fitz
import re

P = r"D:\AL_FuzzyClustering_Paper\v2_unified\draft\DRAL_revised_final.pdf"
doc = fitz.open(P)
print("pages:", doc.page_count)

def page_text(n):
    return doc[n - 1].get_text()

# 1. Abstract content (page 1)
p1 = page_text(1)
print("\n=== P1 ABSTRACT (first 800 chars) ===")
print(p1[:800])
print("\nABSTRACT has '84.7' ?", "84.7" in p1, "| 'four of six' ?", "four of six" in p1)

# 2. keywords count
kw = p1.count("Keywords")
print("Keywords occurrences on p1:", kw)

# 3. garbage pages
for n in [18, 22, 23, 27, 36, 46, 49, 54]:
    t = page_text(n).strip()
    # count distinct chars - garbage pages are digits only
    digits = re.sub(r"[^0-9.\s]", "", t)
    ratio = len(digits) / max(len(t), 1)
    print(f"p{n}: len={len(t)} digit_ratio={ratio:.2f} head={t[:60]!r}")

# 4. Fig 15/17 existence in text (captions)
for n in range(1, doc.page_count + 1):
    t = page_text(n)
    if re.search(r"Fig\.\s*15\.", t) or re.search(r"Fig\.\s*17\.", t):
        hit = re.findall(r"(Fig\.\s*1[57][^.]{0,80})", t)
        print(f"\nFig 15/17 mention on p{n}:", hit[:3])

# 5. Table 9 content presence
for n in range(1, doc.page_count + 1):
    t = page_text(n)
    if "Table 9" in t and "84.7" in t:
        print(f"\nTable 9 caption+result on p{n}")
        print(t[:500])
        break

# 6. formula escapes
for n, pat in [(2, r"d_\{i,k\}|\\mathcal"), (5, r"Phi"), (9, r"median|m e d i a n")]:
    t = page_text(n)
    hits = re.findall(pat, t)[:3]
    print(f"\np{n} pattern {pat}: {hits}")
