# -*- coding: utf-8 -*-
"""build_additional_docx.py — assemble the 8 supplementary experiment units
(addN_unit.json) into ONE standalone, self-contained PR-style Word file:
  draft\\additional_experiments_units.docx
Each unit is a block the author can paste into the main manuscript. Equations are
native editable OMML (via the same docx_helpers / pandoc pipeline as the main
paper); tables are three-line; figures are embedded with captions below.

Double-spaced, Times New Roman 12, Letter, 1in margins, left-aligned (no
justification), automatic hyphenation off, centered continuous page numbers.
"""
import os
import sys
import re
import json
import glob

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml, OxmlElement

sys.path.insert(0, r'D:\AL_FuzzyClustering_Paper\notes')
import docx_helpers as dh  # noqa: E402

CODE = r'D:\AL_FuzzyClustering_Paper\v2_unified\code'
RESDIR = os.path.join(CODE, 'results', 'additional')
FIGDIR = os.path.join(CODE, 'figures', 'additional')
OUT = r'D:\AL_FuzzyClustering_Paper\v2_unified\draft\additional_experiments_units.docx'
BODY_FONT = 'Times New Roman'

# ---------------------------------------------------------------------------
# Document / page setup (PR)
# ---------------------------------------------------------------------------
doc = Document()
sec = doc.sections[0]
sec.page_width = Inches(8.5)
sec.page_height = Inches(11)
sec.left_margin = Inches(1)
sec.right_margin = Inches(1)
sec.top_margin = Inches(1)
sec.bottom_margin = Inches(1)

# centered continuous footer page number
_ftr = sec.footer
_ftr.is_linked_to_previous = False
_fp = _ftr.paragraphs[0]
_fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
_fr = _fp.add_run()
_fr.font.name = BODY_FONT
_fr.font.size = Pt(10)
_fr._r.append(parse_xml('<w:fldChar %s w:fldCharType="begin"/>' % nsdecls('w')))
_fr._r.append(parse_xml('<w:instrText %s xml:space="preserve"> PAGE </w:instrText>' % nsdecls('w')))
_fr._r.append(parse_xml('<w:fldChar %s w:fldCharType="end"/>' % nsdecls('w')))

# automatic hyphenation off
if doc.settings.element.find(qn('w:autoHyphenation')) is None:
    doc.settings.element.append(
        parse_xml('<w:autoHyphenation %s w:val="0"/>' % nsdecls('w')))

normal = doc.styles['Normal']
normal.font.name = BODY_FONT
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)
normal.paragraph_format.space_after = Pt(0)
normal.paragraph_format.space_before = Pt(0)
normal.paragraph_format.line_spacing = 2.0          # double-spaced (PR)


# ---------------------------------------------------------------------------
# Run / inline-math helpers (mirrors notes/build_PR.py)
# ---------------------------------------------------------------------------
def _set_run_font(run, size=12, bold=False, italic=False, font=BODY_FONT):
    run.font.name = font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = parse_xml('<w:rFonts %s/>' % nsdecls('w'))
        rpr.insert(0, rfonts)
    rfonts.set(qn('w:ascii'), font)
    rfonts.set(qn('w:hAnsi'), font)
    rfonts.set(qn('w:cs'), font)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def strip_underbrace(latex):
    out = []
    i = 0
    while i < len(latex):
        if latex[i:i + 12] == '\\underbrace{':
            depth = 1
            j = i + 12
            while j < len(latex) and depth > 0:
                if latex[j] == '{':
                    depth += 1
                elif latex[j] == '}':
                    depth -= 1
                j += 1
            out.append(latex[i + 12:j - 1])
            i = j
        else:
            out.append(latex[i])
            i += 1
    return ''.join(out)


def norm_latex(latex):
    latex = re.sub(r'\\tag\{[^}]*\}', '', latex)
    latex = re.sub(r'\\rm\s+([A-Za-z]+)', r'\\mathrm{\1}', latex)
    latex = re.sub(r'\\[bB]igg?[lLrR]?\s*', '', latex)
    latex = re.sub(r"'\\^\{-", lambda m: '^{\\prime-', latex)
    latex = strip_underbrace(latex)
    latex = re.sub(r'(?<=[{_^])\s*>', r'\\geq ', latex)
    latex = re.sub(r'^>', r'\\geq ', latex)
    latex = re.sub(r'(?<=[{_^])\s*<', r'\\leq ', latex)
    latex = re.sub(r'<$', r'\\leq ', latex)
    latex = latex.replace(r'\\setminus', r'\\smallsetminus')
    latex = latex.replace(r'^\*', r'^{*}')
    latex = re.sub(r'\\mathrm\{(Cov|KL|E|sign|argmax|argmin|log|exp|min|max|inf|sup)\}',
                   r'\\operatorname{\1}', latex)
    return latex.strip()


def _emit_text_run(p, text, size, bold, italic, font=BODY_FONT):
    if text == '':
        return
    r = p.add_run(text)
    _set_run_font(r, size=size, bold=bold, italic=italic, font=font)


def _parse_inline(text):
    tokens = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '`':
            j = text.find('`', i + 1)
            if j == -1:
                tokens.append(('text', text[i:]))
                break
            tokens.append(('code', text[i + 1:j]))
            i = j + 1
        elif ch == '$':
            j = text.find('$', i + 1)
            if j == -1:
                tokens.append(('text', text[i:]))
                break
            tokens.append(('math', text[i + 1:j]))
            i = j + 1
        elif ch == '*':
            if i + 1 < n and text[i + 1] == '*':
                j = text.find('**', i + 2)
                if j == -1:
                    tokens.append(('text', text[i:]))
                    break
                tokens.append(('bold', _parse_inline(text[i + 2:j])))
                i = j + 2
            else:
                j = text.find('*', i + 1)
                if j == -1:
                    tokens.append(('text', text[i:]))
                    break
                tokens.append(('italic', _parse_inline(text[i + 1:j])))
                i = j + 1
        else:
            j = i
            while j < n and text[j] not in '$*`':
                j += 1
            tokens.append(('text', text[i:j]))
            i = j
    return tokens


def _render_tokens(p, tokens, size, bold, italic, font=BODY_FONT):
    for tok in tokens:
        kind = tok[0]
        if kind == 'text':
            _emit_text_run(p, tok[1], size, bold, italic, font=font)
        elif kind == 'code':
            r = p.add_run(tok[1])
            _set_run_font(r, size=size - 1, bold=bold, italic=italic, font='Consolas')
        elif kind == 'math':
            dh.add_inline_math(p, norm_latex(tok[1]))
        elif kind == 'bold':
            _render_tokens(p, tok[1], size, True, italic, font=font)
        elif kind == 'italic':
            _render_tokens(p, tok[1], size, bold, True, font=font)


def add_runs(p, text, size=12, base_bold=False, base_italic=False, font=BODY_FONT):
    _render_tokens(p, _parse_inline(text), size, base_bold, base_italic, font=font)


# ---------------------------------------------------------------------------
# Paragraph builders
# ---------------------------------------------------------------------------
def _noindent(pf):
    pf.first_line_indent = Inches(0)


def h1(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(12)
    pf.space_after = Pt(4)
    _noindent(pf)
    add_runs(p, text, size=12, base_bold=True)
    return p


def h2(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(2)
    _noindent(pf)
    add_runs(p, text, size=12, base_bold=True, base_italic=True)
    return p


def body(text, size=12, indent=True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_after = Pt(3)
    if indent:
        pf.first_line_indent = Inches(0.25)
    add_runs(p, text, size=size)
    return p


def caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    _noindent(pf)
    pf.space_before = Pt(4)
    pf.space_after = Pt(2)
    pf.line_spacing = 1.0
    add_runs(p, text, size=10.5, base_italic=True)
    return p


def note(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.left_indent = Inches(0.15)
    _noindent(pf)
    pf.space_after = Pt(1)
    pf.line_spacing = 1.0
    add_runs(p, text, size=9.5, base_italic=True)
    return p


def bullet(text, size=11):
    p = doc.add_paragraph(style='List Bullet')
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.space_after = Pt(2)
    add_runs(p, text, size=size)
    return p


def insert_figure(fname, width):
    path = os.path.join(FIGDIR, fname)
    if not os.path.exists(path):
        body('[MISSING FIGURE: %s]' % fname, size=10.5, indent=False)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    p.add_run().add_picture(path, width=Inches(width))


# ---------------------------------------------------------------------------
# Three-line table (no vertical rules, no shading)
# ---------------------------------------------------------------------------
METHOD_ABBREV = {
    'P2_Full': 'Full',
    'P2_NoExploration': 'NoExplo',
    'P2_NoRobust': 'NoRobust',
    'P2_NoRedundancy': 'NoRedund',
    'OriginalP2 (unranked)': 'OrigP2†',
}


def short_header(h):
    s = str(h)
    return METHOD_ABBREV.get(s, s)


def three_line_table(header, rows):
    ncols = max(len(header), max((len(r) for r in rows), default=0))
    wide = ncols >= 8
    header = [short_header(h) for h in header] + [''] * (ncols - len(header))
    rows = [list(r) + [''] * (ncols - len(r)) for r in rows]
    fs = 8 if wide else 9
    # first column (dataset names) wider; remaining columns equal
    if wide:
        col_w = [0.95] + [(6.5 - 0.95) / (ncols - 1)] * (ncols - 1)
    else:
        col_w = [6.5 / ncols] * ncols
    table = doc.add_table(rows=len(rows) + 1, cols=ncols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    tblPr = table._tbl.tblPr

    # tight cell margins (schema order: tblBorders before tblCellMar/tblLook)
    tblCellMar = OxmlElement('w:tblCellMar')
    for side, val in (('top', 26), ('bottom', 26), ('left', 70), ('right', 70)):
        m = OxmlElement('w:' + side)
        m.set(qn('w:w'), str(val))
        m.set(qn('w:type'), 'dxa')
        tblCellMar.append(m)

    tblBorders = OxmlElement('w:tblBorders')
    top_b = OxmlElement('w:top')
    top_b.set(qn('w:val'), 'single')
    top_b.set(qn('w:sz'), '8')
    top_b.set(qn('w:space'), '0')
    top_b.set(qn('w:color'), '000000')
    tblBorders.append(top_b)
    for side in ('left', 'right', 'insideV'):
        b = OxmlElement('w:' + side)
        b.set(qn('w:val'), 'none')
        tblBorders.append(b)
    bot_b = OxmlElement('w:bottom')
    bot_b.set(qn('w:val'), 'single')
    bot_b.set(qn('w:sz'), '8')
    bot_b.set(qn('w:space'), '0')
    bot_b.set(qn('w:color'), '000000')
    tblBorders.append(bot_b)
    insideH = OxmlElement('w:insideH')
    insideH.set(qn('w:val'), 'none')
    tblBorders.append(insideH)
    anchor = None
    for tag in ('w:shd', 'w:tblLayout', 'w:tblCellMar', 'w:tblLook'):
        anchor = tblPr.find(qn(tag))
        if anchor is not None:
            break
    if anchor is not None:
        anchor.addprevious(tblBorders)
    else:
        tblPr.append(tblBorders)
    tblLook = tblPr.find(qn('w:tblLook'))
    if tblLook is not None:
        tblLook.addprevious(tblCellMar)
    else:
        tblPr.append(tblCellMar)

    # header bottom thin rule
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        tcBorders = tcPr.find(qn('w:tcBorders'))
        if tcBorders is None:
            tcBorders = OxmlElement('w:tcBorders')
            tcPr.append(tcBorders)
        bb = tcBorders.find(qn('w:bottom'))
        if bb is None:
            bb = OxmlElement('w:bottom')
            tcBorders.append(bb)
        bb.set(qn('w:val'), 'single')
        bb.set(qn('w:sz'), '4')
        bb.set(qn('w:space'), '0')
        bb.set(qn('w:color'), '000000')

    def fill(cell, text, bold):
        cp = cell.paragraphs[0]
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.first_line_indent = Inches(0)
        cp.paragraph_format.space_after = Pt(0)
        cp.paragraph_format.space_before = Pt(0)
        cp.paragraph_format.line_spacing = 1.0
        add_runs(cp, str(text), size=fs, base_bold=bold)

    for ci, txt in enumerate(header):
        fill(table.rows[0].cells[ci], txt, True)
    for ri, row in enumerate(rows):
        for ci, txt in enumerate(row):
            fill(table.rows[ri + 1].cells[ci], txt, False)

    # fixed equal-width layout (9360 twips text width)
    tblW = tblPr.find(qn('w:tblW'))
    if tblW is None:
        tblW = OxmlElement('w:tblW')
        jc = tblPr.find(qn('w:jc'))
        if jc is not None:
            jc.addnext(tblW)
        else:
            tblPr.append(tblW)
    tblW.set(qn('w:w'), '9360')
    tblW.set(qn('w:type'), 'dxa')
    layout = OxmlElement('w:tblLayout')
    layout.set(qn('w:type'), 'fixed')
    cme = tblPr.find(qn('w:tblCellMar'))
    if cme is not None:
        cme.addprevious(layout)
    else:
        tlk = tblPr.find(qn('w:tblLook'))
        if tlk is not None:
            tlk.addprevious(layout)
        else:
            tblPr.append(layout)
    for ci in range(ncols):
        for row in table.rows:
            row.cells[ci].width = Inches(col_w[ci])
    return wide


# ---------------------------------------------------------------------------
# Protocol summary
# ---------------------------------------------------------------------------
def fmt_protocol(proto):
    if not isinstance(proto, dict):
        return []
    lines = []
    if proto.get('datasets'):
        lines.append('Datasets: ' + ', '.join(str(d) for d in proto['datasets']) + '.')
    if proto.get('methods'):
        lines.append('Methods: ' + ', '.join(str(m) for m in proto['methods']) + '.')
    if proto.get('n_seeds'):
        lines.append('Replicates: %d paired seeds.' % proto['n_seeds'])
    if proto.get('budgets'):
        b = ', '.join('%g%%' % (100 * float(x)) for x in proto['budgets'])
        lines.append('Labeled budget(s): ' + b + '.')
    if proto.get('checkpoint'):
        lines.append('Primary checkpoint: %g%%.' % (100 * proto['checkpoint']))
    fac = proto.get('factors')
    if isinstance(fac, dict) and fac:
        parts = []
        for k, v in fac.items():
            if isinstance(v, float):
                parts.append('%s=%.3g' % (k, v))
            else:
                parts.append('%s=%s' % (k, v))
        if parts:
            lines.append('Frozen factors: ' + '; '.join(parts) + '.')
    return lines


# ---------------------------------------------------------------------------
# Load units
# ---------------------------------------------------------------------------
unit_paths = sorted(glob.glob(os.path.join(RESDIR, 'add*_unit.json')))
units = []
for up in unit_paths:
    with open(up, encoding='utf-8') as f:
        u = json.load(f)
    if u.get('status') in ('complete', 'done'):
        units.append(u)
units.sort(key=lambda u: u.get('exp_id', 99))

# ---- Title block -----------------------------------------------------------
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.paragraph_format.first_line_indent = Inches(0)
t.paragraph_format.space_after = Pt(6)
add_runs(t, 'Additional Experiments for the Pattern Recognition Submission',
         size=14, base_bold=True)
st = doc.add_paragraph()
st.alignment = WD_ALIGN_PARAGRAPH.CENTER
st.paragraph_format.first_line_indent = Inches(0)
st.paragraph_format.space_after = Pt(10)
add_runs(st, 'Standalone self-contained units (S1–S%d) intended to be inserted '
             'into the main manuscript or supplement' % len(units),
         size=11, base_italic=True)

h2('How to read these units')
body('Each unit below is a self-contained supplementary experiment: it states its '
     'motivation, design and frozen protocol, presents its tables and figures, and '
     'reports the observed result together with an honest conclusion. Every number '
     'is computed from the frozen pipeline (the same P2 implementation, the same '
     'initial-labeled-set construction and the same Hungarian-matched ACC used in '
     'the main paper); no result is hand-edited. The pre-registered, multiplicity-'
     'controlled confirmatory claims remain those of the main text. These units are '
     'labelled confirmatory, supportive, or exploratory as appropriate and should be '
     'read as generality, scalability and robustness evidence rather than as new '
     'headline claims.', indent=False)
body('Sign convention. In the main text the double-counting contrast is reported as '
     'NoExploration minus Full (positive favours removing the exploration bonus). '
     'Units S3 and S7 instead regress the gap defined as Full minus NoExploration, '
     'so a negative value there favours NoExploration; the sign is always stated in '
     'the caption/table that uses it. All contrasts are paired by seed and reported '
     'with paired 95% confidence intervals, Holm-adjusted p-values, and the '
     'standardized paired effect $d_z$; non-inferiority uses the pre-registered '
     'margin $\\Delta_{NI}=0.005$.', indent=False)

# ---- Units -----------------------------------------------------------------
for u in units:
    eid = u.get('exp_id')
    h1('S%d. %s' % (eid, u.get('title', '')))
    tag = doc.add_paragraph()
    tag.paragraph_format.first_line_indent = Inches(0)
    tag.paragraph_format.space_after = Pt(3)
    tag.paragraph_format.line_spacing = 1.0
    add_runs(tag, 'Evidence tier: *%s*.' % (u.get('evidence_label', 'n/a')),
             size=10.5)

    if u.get('motivation'):
        h2('Motivation')
        body(u['motivation'])
    if u.get('design'):
        h2('Design and protocol')
        body(u['design'])
    for ln in fmt_protocol(u.get('protocol')):
        bullet(ln, size=11)

    for tb in u.get('tables', []):
        caption(tb.get('caption', ''))
        wide = three_line_table(tb.get('header', []), tb.get('rows', []))
        for nt in tb.get('notes', []):
            note('Note: ' + nt if not nt.lower().startswith('note') else nt)
        if wide:
            note('Method abbreviations: Full=P2_Full, NoExplo=P2_NoExploration, '
                 'NoRobust=P2_NoRobust, NoRedund=P2_NoRedundancy; OrigP2=OriginalP2 '
                 '(archived five-bug control, excluded from the formal ranking).')

    if u.get('results'):
        h2('Results')
    for fg in u.get('figures', []):
        width = 6.4 if fg.get('combo') else 4.6
        insert_figure(fg.get('file', ''), width)
        caption(fg.get('caption', ''))
    if u.get('results'):
        body(u['results'])
    if u.get('conclusion'):
        h2('Conclusion')
        body(u['conclusion'])

    if u.get('annotations'):
        note('Tags: ' + '; '.join(u['annotations']) + '.')
    repro = []
    if u.get('repro_script'):
        repro.append('script ' + u['repro_script'])
    if u.get('csvs'):
        repro.append('data ' + ', '.join(u['csvs']))
    if repro:
        note('Reproducibility: ' + '; '.join(repro) + '.')

doc.save(OUT)
print('WROTE', OUT)
print('units included:', [u.get('exp_id') for u in units])
