"""scan_citations.py - find paragraphs mentioning the 5 statistical methods."""
import docx, re

P = r"D:\AL_FuzzyClustering_Paper\v2_unified\draft\DRAL_revised_final.docx"
d = docx.Document(P)
patterns = {
    "JT": r"Jonckheere|jonckheere|JT trend|ordered alternatives|trend test",
    "DL": r"DerSimonian|derSimonian|random-effects meta|DerSimonian-Laird",
    "Holm": r"Holm|holm ",
    "BH": r"Benjamini|benjamini|false discovery|BH-FDR|fdr_bh",
    "dz": r"d_z|Cohen|cohen",
}
for i, pa in enumerate(d.paragraphs):
    t = pa.text
    if pa.style.name.startswith("Heading") or not t.strip():
        continue
    hits = [lab for lab, pat in patterns.items() if re.search(pat, t)]
    if hits:
        print(i, hits, "|", t[:220].replace("\n", " "))
