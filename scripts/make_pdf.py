"""Render docs/TECHNICAL.md to docs/433.pdf (the technical document for the submission form).

    python scripts/make_pdf.py

A deliberately small Markdown subset -- exactly what TECHNICAL.md uses: # / ## headings,
paragraphs, "- " and "1. " lists, | tables |, ``` code blocks, **bold**, `code`, [links](url).
reportlab (pinned) draws it with the standard PDF fonts; the few symbols those fonts lack are
written in ASCII (SYMBOLS). invariant mode makes the PDF byte-identical between runs.
"""

import re
import sys
from pathlib import Path

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (ListFlowable, ListItem, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "TECHNICAL.md"
TARGET = ROOT / "docs" / "433.pdf"
SYMBOLS = {"→": "->", "↔": "<->", "≤": "<=", "≥": ">=", "≈": "~", "σ": "sigma", "τ": "tau",
           "−": "-"}

rl_config.invariant = 1          # no timestamps / random ids: same bytes every run

STYLES = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=STYLES["BodyText"], fontSize=9.5, leading=12.5)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10)
H1 = ParagraphStyle("h1", parent=STYLES["Title"], fontSize=15, leading=19, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=STYLES["Heading2"], fontSize=11.5, leading=14, spaceBefore=8)
CODE = ParagraphStyle("code", parent=STYLES["Code"], fontSize=8, leading=10)


def inline(text):
    """Markdown inline -> reportlab mini-HTML (escape first, then bold / code / links)."""
    for symbol, ascii_text in SYMBOLS.items():
        text = text.replace(symbol, ascii_text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1 (\2)", text)
    return re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"<i>\1</i>", text)


def table(rows):
    """A list of '| a | b |' lines (header, separator, body) -> a wrapped, gridded Table."""
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    cells = [cells[0]] + cells[2:]                                 # drop the |---| line
    data = [[Paragraph(inline(c), CELL) for c in row] for row in cells]
    width = (A4[0] - 3.6 * cm) / len(cells[0])
    t = Table(data, colWidths=[width] * len(cells[0]), repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def flowables(markdown):
    """Turn the Markdown subset into a list of reportlab flowables."""
    out, lines, i = [], markdown.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):                                  # code block
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append(Preformatted("\n".join(lines[i + 1:j]), CODE))
            i = j + 1
        elif line.startswith("# "):
            out.append(Paragraph(inline(line[2:]), H1))
            i += 1
        elif line.startswith("## "):
            out.append(Paragraph(inline(line[3:]), H2))
            i += 1
        elif line.startswith("|"):                                  # table
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out += [table(lines[i:j]), Spacer(1, 4)]
            i = j
        elif re.match(r"^(- |\d+\. )", line):                       # list
            items, numbered = [], bool(re.match(r"^\d+\. ", line))
            while i < len(lines) and (re.match(r"^(- |\d+\. )", lines[i])
                                      or (lines[i].startswith("   ") and items)):
                if re.match(r"^(- |\d+\. )", lines[i]):
                    items.append(re.sub(r"^(- |\d+\. )", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append(ListFlowable([ListItem(Paragraph(inline(t), BODY)) for t in items],
                                    bulletType="1" if numbered else "bullet", leftIndent=12))
        elif line.strip():                                          # paragraph
            j = i
            while j < len(lines) and lines[j].strip() and not re.match(
                    r"^(#|\||- |\d+\. |```)", lines[j]):
                j += 1
            out.append(Paragraph(inline(" ".join(lines[i:j])), BODY))
            i = j
        else:
            i += 1
    return out


def main():
    doc = SimpleDocTemplate(str(TARGET), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="Team 433 - Ball possession (MTAH 2027 Phase 1)",
                            author="Team 433")
    doc.build(flowables(SOURCE.read_text(encoding="utf-8")))
    print("written {}".format(TARGET.relative_to(ROOT)))


if __name__ == "__main__":
    sys.exit(main())
