"""audit_omml.py - check OMML formulas, Word availability, Section 6 full text, 5.6 heading."""
import docx, re, os, zipfile

P = r"D:\AL_FuzzyClustering_Paper\v2_unified\draft\DRAL_revised_final.docx"

# 1. OMML sample: dump m:t texts for first 5 math elements
z = zipfile.ZipFile(P)
xml = z.read("word/document.xml").decode("utf-8", "ignore")
print("=== OMML math element count ===")
print("w:oMath occurrences:", xml.count("<m:oMath"))
print("w:oMathPara occurrences:", xml.count("<m:oMathPara"))
# sample first 3 math contents text
import re as _re
mt = _re.findall(r"<m:t[^>]*>(.*?)</m:t>", xml)
print("sample m:t texts:", mt[:40])

# 2. Word availability
try:
    import win32com.client
    print("\nwin32com available: True")
except Exception as e:
    print("\nwin32com available: False", e)

# 3. Section 6 full text
import docx
d = docx.Document(P)
texts = [p.text for p in d.paragraphs]
i6 = 238
print("\n=== Section 6 full (238..242) ===")
for i in range(i6, min(i6 + 6, len(texts))):
    if texts[i].strip():
        print(i, texts[i][:450])
        print("---")

# 4. 5.6 heading existence
hits = [i for i, t in enumerate(texts) if re.match(r"^5\.6\s", t.strip())]
print("\n5.6 headings:", hits, [texts[i][:60] for i in hits])

# 5. laTeX-looking raw text anywhere in document.xml (outside m:t)
plain = _re.sub(r"<[^>]+>", "", xml)
for pat, lab in [(r"\$[^$]{1,80}\$", "$...$"), (r"\\mathcal|\\ell|\\nu|\\|\\sum|\\Phi", "latex cmds")]:
    n = len(_re.findall(pat, plain))
    print(f"raw text {lab}: {n}")
