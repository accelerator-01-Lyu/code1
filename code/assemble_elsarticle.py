"""assemble_elsarticle.py v3 - refs from docx directly, body filtered."""
import re, docx

SRC = r"D:\AL_FuzzyClustering_Paper\v2_unified\latex\DRAL_pandoc.tex"
DOCX = r"D:\AL_FuzzyClustering_Paper\v2_unified\draft\DRAL_revised_final.docx"
OUT = r"D:\AL_FuzzyClustering_Paper\v2_unified\latex\DRAL.tex"

t = open(SRC, encoding="utf-8").read()

# ---------- 1. title ----------
m = re.search(r"\\title\{(.*?)\}", t, re.S)
title = m.group(1).strip().replace("\n", " ")

# ---------- 2. author block / abstract / keywords ----------
i_mt = t.find("\\maketitle")
m_abs = re.search(r"(?m)^Abstract ", t)
i_abs = m_abs.start() if m_abs else t.find("Abstract ")
i_kw = t.find("\\textbf{Keywords}")
i_sec = t.find("\\section{1 Introduction}")
author_block = t[i_mt + len("\\maketitle"):i_abs].strip()
abstract = t[i_abs:i_kw].strip()
abstract = re.sub(r"^Abstract\s+", "", abstract, count=1)
keywords = t[i_kw + len("\\textbf{Keywords}"):i_sec].strip()

# ---------- 3. body = Introduction .. before \section{Supplementary material} ----------
i_supp = t.find("\\section{Supplementary material}")
body_raw = t[i_sec:i_supp].strip()

# remove ref-item paragraphs from body (lines starting with {[}N{]})
def is_ref_para(para):
    return bool(re.match(r"^\{\[\}\d+\{\]\}\s", para, re.S))

paras = re.split(r"\n\s*\n", body_raw)
kept = [p for p in paras if not is_ref_para(p)]
body = "\n\n".join(kept)

# ---------- 4. refs from docx ----------
d = docx.Document(DOCX)
refs = []
for p in d.paragraphs:
    txt = p.text.strip()
    mm = re.match(r"^\[(\d+)\]\s+(.*)$", txt, re.S)
    if mm:
        refs.append((int(mm.group(1)), mm.group(2)))
refs.sort(key=lambda x: x[0])

def tex_escape(s):
    s = s.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#").replace("_", "\\_")
    s = re.sub(r"(https?://\S+\.\S+)", r"\\url{\1}", s)
    return s

# ---------- 5. preamble ----------
PRE = r"""%% KBS submission -- elsarticle (Elsevier) -- compiled via pandoc from docx
\documentclass[preprint,12pt]{elsarticle}

\usepackage{amsmath,amssymb}
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\usepackage{unicode-math}
\setmathfont{TeX Gyre Termes Math}
\usepackage{graphicx}
\usepackage{longtable,booktabs,array,calc}
\usepackage{caption}
\captionsetup[figure]{labelformat=empty}
\captionsetup[table]{labelformat=empty}
\usepackage[unicode]{hyperref}
\urlstyle{same}
\usepackage{xurl}
\newcounter{none}
\setlength{\emergencystretch}{3em}

\begin{document}

\begin{frontmatter}
"""

# ---------- 6. authors ----------
lines = [l for l in author_block.split("\n") if l.strip()]
first = lines[0]
rest_lines = lines[1:]

m_auth = re.match(r"(.*?)and(.*)", first, re.S)
a1 = m_auth.group(1).strip() if m_auth else first
a2 = m_auth.group(2).strip() if m_auth else ""

def build_author(name, affmark):
    affs = re.findall(r"\d", affmark)
    corr = "*" in affmark
    return (r"\author[" + ",".join(affs) + "]" + "{" + re.sub(r"\\textsuperscript\{[^}]*\}", "", name).strip()
            + (r"\corref{cor1}" if corr else "") + "}")

author_tex = []
if a1:
    m1 = re.search(r"\\textsuperscript\{([^}]*)\}", a1)
    author_tex.append(build_author(a1, m1.group(1) if m1 else "1"))
if a2:
    m2 = re.search(r"\\textsuperscript\{([^}]*)\}", a2)
    author_tex.append(build_author(a2, m2.group(1) if m2 else "2"))

aff_tex = []
for line in rest_lines:
    mm = re.match(r"^\\textsuperscript\{(\d+)\}\s*(.*)$", line)
    if mm:
        aff_tex.append("\\affiliation[%s]{organization={%s}}" % (mm.group(1), mm.group(2)))

# ---------- 7. abstract / keywords ----------
abs_tex = "\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n\n"
kw_list = [k.strip() for k in keywords.split(";") if k.strip()]
kw_tex = "\\begin{keyword}\n" + " \\sep\n".join(kw_list) + "\n\\end{keyword}\n"

# ---------- 8. body starred ----------
body2 = re.sub(r"\\section\{([^}]*)\}", r"\\section*{\1}", body)
body2 = re.sub(r"\\subsection\{([^}]*)\}", r"\\subsection*{\1}", body2)
body2 = re.sub(r"\\subsubsection\{([^}]*)\}", r"\\subsubsection*{\1}", body2)

# ---------- 9. references ----------
ref_tex = "\\section*{References}\n\\begin{thebibliography}{50}\n"
for num, content in refs:
    ref_tex += "\\bibitem{r%d} %s\n\n" % (num, tex_escape(content))
ref_tex += "\\end{thebibliography}\n"

# ---------- 10. supplementary ----------
supp = t[i_supp:].strip()
supp2 = re.sub(r"\\section\{([^}]*)\}", r"\\section*{\1}", supp)
supp2 = re.sub(r"\\subsection\{([^}]*)\}", r"\\subsection*{\1}", supp2)
supp2 = re.sub(r"\\subsubsection\{([^}]*)\}", r"\\subsubsection*{\1}", supp2)

# ---------- 11. assemble ----------
out = (PRE + "\n".join(author_tex) + "\n" + "\n".join(aff_tex) + "\n"
       + "\\cortext[cor1]{Corresponding author. Email: 3187265823@qq.com}\n\n"
       + abs_tex + kw_tex + "\\end{frontmatter}\n\n"
       + body2 + "\n\n" + ref_tex + "\n\n" + supp2 + "\n\\end{document}\n")
open(OUT, "w", encoding="utf-8").write(out)
print("docx refs:", len(refs), "first:", refs[0][0], "last:", refs[-1][0])
print("body chars:", len(body2), "supp chars:", len(supp2))
