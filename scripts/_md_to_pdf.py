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
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
        "→": "->",
        "×": "x",
        "≥": ">=",
        "≤": "<=",
        "²": "2",
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


def _column_widths(pdf: FPDF, ncol: int) -> list[float]:
    usable = pdf.w - 2 * MARGIN
    first = usable * 0.40
    rest = (usable - first) / max(ncol - 1, 1)
    return [first] + [rest] * (ncol - 1)


def _fit(pdf: FPDF, text: str, width: float) -> str:
    """Truncate `text` until it fits in a cell of `width`."""
    while text and pdf.get_string_width(text) > width - 2:
        text = text[:-1]
    return text


def _start_row(pdf: FPDF, line_h: float, header: bool) -> None:
    pdf.set_font(FONT, "B" if header else "", 8.0)
    pdf.set_fill_color(230 if header else 255)
    if pdf.get_y() + line_h > pdf.h - 15:
        pdf.add_page()


def _table_row(pdf: FPDF, row: list[str], widths: list[float], line_h: float, header: bool) -> None:
    _start_row(pdf, line_h, header)
    x0, y0 = pdf.get_x(), pdf.get_y()
    for j, width in enumerate(widths):
        cell = _fit(pdf, _clean(row[j]) if j < len(row) else "", width)
        pdf.set_xy(x0 + sum(widths[:j]), y0)
        pdf.cell(width, line_h, cell, border=1, fill=header)
    pdf.set_xy(x0, y0 + line_h)


def _table(pdf: FPDF, rows: list[list[str]]) -> None:
    widths = _column_widths(pdf, max(len(r) for r in rows))
    for i, row in enumerate(rows):
        _table_row(pdf, row, widths, 5.0, header=i == 0)
    pdf.ln(2)


def _code_block(pdf: FPDF, code: list[str]) -> None:
    pdf.set_font("Courier", "", 8)
    pdf.set_fill_color(244)
    for c in code:
        pdf.set_x(MARGIN)
        pdf.multi_cell(0, 4.5, _clean(c) or " ", fill=True)
    pdf.ln(2)


def _rule(pdf: FPDF) -> None:
    pdf.ln(1)
    y = pdf.get_y()
    pdf.line(MARGIN, y, pdf.w - MARGIN, y)
    pdf.ln(2)


def _text(pdf: FPDF, size: int, height: float, text: str, before: int = 0, after: int = 0) -> None:
    if before:
        pdf.ln(before)
    pdf.set_font(FONT, "B" if size > 10 else "", size)
    pdf.multi_cell(0, height, _clean(text))
    if after:
        pdf.ln(after)


# (matcher, size, line height, text extractor, space before, space after)
_BLOCKS = [
    (lambda ln: ln.startswith("# "), 16, 8, lambda ln: ln[2:], 0, 1),
    (lambda ln: ln.startswith("## "), 13, 7, lambda ln: ln[3:], 2, 1),
    (lambda ln: ln.startswith("### "), 11, 6, lambda ln: ln[4:], 0, 0),
    (lambda ln: bool(re.match(r"^\d+\.\s", ln)), 10, 5.5, lambda ln: "  " + ln, 0, 0),
    (lambda ln: ln.startswith("- "), 10, 5.5, lambda ln: "  - " + ln[2:], 0, 0),
]


def _line(pdf: FPDF, line: str) -> None:
    pdf.set_x(MARGIN)
    if not line:
        pdf.ln(3)
    elif line.startswith("---"):
        _rule(pdf)
    else:
        for match, size, height, extract, before, after in _BLOCKS:
            if match(line):
                _text(pdf, size, height, extract(line), before, after)
                return
        _text(pdf, 10, 5.5, line)


def _table_cells(line: str) -> list[str] | None:
    """Return the cells of a markdown table row, or None if `line` is not one."""
    if not re.match(r"^\|.*\|$", line):
        return None
    return [c.strip() for c in line.strip("|").split("|")]


def _is_separator(cells: list[str]) -> bool:
    return all(re.match(r"^:?-+:?$", c) for c in cells)


class _Converter:
    """Feed markdown lines in order; buffers code blocks and tables until they end."""

    def __init__(self, pdf: FPDF) -> None:
        self.pdf = pdf
        self.in_code = False
        self.code: list[str] = []
        self.table: list[list[str]] = []

    def flush_table(self) -> None:
        if self.table:
            _table(self.pdf, self.table)
            self.table = []

    def fence(self) -> None:
        self.flush_table()
        self.in_code = not self.in_code
        if not self.in_code and self.code:
            _code_block(self.pdf, self.code)
            self.code = []

    def feed(self, line: str) -> None:
        cells = None if self.in_code else _table_cells(line)
        if line.startswith("```"):
            self.fence()
        elif self.in_code:
            self.code.append(line)
        elif cells is not None:
            self.table += [] if _is_separator(cells) else [cells]
        else:
            self.flush_table()
            _line(self.pdf, line)


def render() -> None:
    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.add_page()

    converter = _Converter(pdf)
    for raw in SRC.read_text(encoding="utf-8").splitlines():
        converter.feed(raw.rstrip())
    converter.flush_table()
    pdf.output(str(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    render()
