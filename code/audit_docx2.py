"""quick audit of specific user items in docx."""
import docx, re
d = docx.Document(r"D:\AL_FuzzyClustering_Paper\v2_unified\draft\DRAL_revised_final.docx")
texts = [p.text for p in d.paragraphs]
allt = "\n".join(texts)

print("=== 1. Haolin Liu email occurrences ===")
for i, t in enumerate(texts[:10]):
    if "liuhlin23" in t or "Haolin" in t:
        print(i, repr(t[:120]))

print("\n=== 2. '5.2.5' vs '5.2.2' Concentration heading ===")
hits = [i for i, t in enumerate(texts) if "Concentration sensitivity" in t]
print(hits, [texts[i][:60] for i in hits])

print("\n=== 3. 'theta <= h_i' style / '0' bound in formulas - OMML not in p.text; check raw xml ===")
import zipfile
xml = zipfile.ZipFile(r"D:\AL_FuzzyClustering_Paper\v2_unified\draft\DRAL_revised_final.docx").read("word/document.xml").decode("utf-8", "ignore")
mt = re.findall(r"<m:t[^>]*>(.*?)</m:t>", xml)
print("sample m:t around h_i:", [x for x in mt if x in ("h","i","θ","0","2","≤","Φ","τ","W","w","rob")][:40])

print("\n=== 4. Algorithm 1 duplicate blocks ===")
algo = [i for i, t in enumerate(texts) if "Algorithm" in t]
print("algorithm mentions:", algo, [texts[i][:80] for i in algo])
# count 'for t in' style lines duplicated
steps = [i for i, t in enumerate(texts) if re.match(r"^(Step|for )", t.strip()) and i > 50 and i < 120]
print("step lines count:", len(steps), steps[:6])

print("\n=== 5. '4: concentration' vs 'P4' ===")
for i, t in enumerate(texts):
    if re.search(r"^\s*4:", t.strip()) or "concentration equivalence" in t.lower():
        print(i, repr(t[:80]))

print("\n=== 6. [43] Lyu ===")
refs_start = next(i for i, t in enumerate(texts) if t.strip().startswith("References"))
for i, t in enumerate(texts[refs_start:refs_start+50]):
    if "[43]" in t or "accelerator" in t:
        print(refs_start+i, t[:200])

print("\n=== 7. Table E3 / E1 / C3 / D2 / D3 locations ===")
for pat in ["Table E3", "Table E1", "C.3", "D.2", "D.3", "Table D1", "Table D2", "Table D3", "Fig. C3", "Fig. D2", "S.V.3", "S.V.4", "S.4", "S.12", "5.12 Statistical"]:
    hits = [i for i, t in enumerate(texts) if pat in t]
    print(f"  '{pat}': {hits[:6]}")
