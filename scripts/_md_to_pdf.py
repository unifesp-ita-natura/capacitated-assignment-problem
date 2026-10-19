"""Render the nivel report markdown to a PDF using fpdf2 (one-off conversion)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])

MARGIN = 18
FONT = "Helvetica"


def _break_long(text: str, limit: int = 40) -> str:
    """Insert spaces into tokens longer than `limit` so word-wrap has break points."""
    out = []
    for token in text.split(" "):
        while len(token) > limit:
            out.append(token[:limit])
            token = token[limit:]
        out.append(token)
    return " ".join(out)


def _clean(text: str) -> str:
    """Strip inline markdown and map non-latin1 glyphs fpdf's core fonts lack."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    replacements = {
        "—": "-", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
        "…": "...", "→": "->", "×": "x", "≥": ">=", "≤": "<=", "²": "2",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = text.encode("latin-1", "replace").decode("latin-1")
    return _break_long(text)


class Report(FPDF):
    def header(self) -> None:
        pass

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font(FONT, "I", 8)
        self.set_text_color(130)
        self.cell(0, 8, f"{self.page_no()}", align="C")
        self.set_text_color(0)


def _table(pdf: FPDF, rows: list[list[str]]) -> None:
    usable = pdf.w - 2 * MARGIN
    ncol = max(len(r) for r in rows)
    first = usable * 0.40
    rest = (usable - first) / max(ncol - 1, 1)
    widths = [first] + [rest] * (ncol - 1)
    line_h = 5.0
    for i, row in enumerate(rows):
        pdf.set_font(FONT, "B" if i == 0 else "", 8.0)
        fill = i == 0
        if fill:
            pdf.set_fill_color(230)
        x0, y0 = pdf.get_x(), pdf.get_y()
        if y0 + line_h > pdf.h - 15:
            pdf.add_page()
            x0, y0 = pdf.get_x(), pdf.get_y()
        for j in range(ncol):
            cell = _clean(row[j]) if j < len(row) else ""
            pdf.set_xy(x0 + sum(widths[:j]), y0)
            while cell and pdf.get_string_width(cell) > widths[j] - 2:
                cell = cell[:-1]
            pdf.cell(widths[j], line_h, cell, border=1, fill=fill)
        pdf.set_xy(x0, y0 + line_h)
    pdf.ln(2)


def render() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.add_page()

    in_code = False
    code_buf: list[str] = []
    table_buf: list[list[str]] = []

    def flush_table() -> None:
        if table_buf:
            _table(pdf, table_buf)
            table_buf.clear()

    for raw in lines:
        line = raw.rstrip()

        if line.startswith("```"):
            flush_table()
            in_code = not in_code
            if not in_code and code_buf:
                pdf.set_font("Courier", "", 8)
                pdf.set_fill_color(244)
                for c in code_buf:
                    pdf.set_x(MARGIN)
                    pdf.multi_cell(0, 4.5, _clean(c) or " ", fill=True)
                pdf.ln(2)
                code_buf = []
            continue
        if in_code:
            code_buf.append(line)
            continue

        if re.match(r"^\|.*\|$", line):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                continue
            table_buf.append(cells)
            continue
        flush_table()

        pdf.set_x(MARGIN)
        if not line:
            pdf.ln(3)
            continue
        if line.startswith("# "):
            pdf.set_font(FONT, "B", 16)
            pdf.multi_cell(0, 8, _clean(line[2:]))
            pdf.ln(1)
        elif line.startswith("## "):
            pdf.ln(2)
            pdf.set_font(FONT, "B", 13)
            pdf.multi_cell(0, 7, _clean(line[3:]))
            pdf.ln(1)
        elif line.startswith("### "):
            pdf.set_font(FONT, "B", 11)
            pdf.multi_cell(0, 6, _clean(line[4:]))
        elif line.startswith("---"):
            pdf.ln(1)
            y = pdf.get_y()
            pdf.line(MARGIN, y, pdf.w - MARGIN, y)
            pdf.ln(2)
        elif re.match(r"^\d+\.\s", line):
            pdf.set_font(FONT, "", 10)
            pdf.multi_cell(0, 5.5, _clean("  " + line))
        elif line.startswith("- "):
            pdf.set_font(FONT, "", 10)
            pdf.multi_cell(0, 5.5, _clean("  - " + line[2:]))
        else:
            pdf.set_font(FONT, "", 10)
            pdf.multi_cell(0, 5.5, _clean(line))

    flush_table()
    pdf.output(str(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    render()
