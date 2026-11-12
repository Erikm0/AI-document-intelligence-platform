"""Builders for real PDF / DOCX / TXT test files."""

from __future__ import annotations

import io
from collections.abc import Sequence


def make_pdf(
    pages: Sequence[Sequence[str]],
    *,
    title: str | None = None,
    footer: str | None = None,
) -> bytes:
    """Each page is a list of lines; headings are lines prefixed with '# '."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    if title:
        pdf.setTitle(title)
    width, height = A4
    for number, lines in enumerate(pages, start=1):
        y = height - 72
        for line in lines:
            if line.startswith("# "):
                pdf.setFont("Helvetica-Bold", 14)
                pdf.drawString(72, y, line[2:])
                y -= 24
            else:
                pdf.setFont("Helvetica", 11)
                pdf.drawString(72, y, line)
                y -= 16
        if footer:
            pdf.setFont("Helvetica", 8)
            pdf.drawString(72, 30, footer)
            pdf.drawRightString(width - 72, 30, f"Page {number}")
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def make_docx(
    pages: Sequence[Sequence[tuple[str, str]]],
    *,
    title: str | None = None,
    table: Sequence[Sequence[str]] | None = None,
) -> bytes:
    """``pages`` -> list of (kind, text) where kind is 'h1', 'h2', 'p' or 'li'."""
    import docx
    from docx.enum.text import WD_BREAK

    document = docx.Document()
    if title:
        document.core_properties.title = title
    for number, blocks in enumerate(pages):
        if number > 0:
            document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        for kind, text in blocks:
            if kind == "h1":
                document.add_heading(text, level=1)
            elif kind == "h2":
                document.add_heading(text, level=2)
            elif kind == "li":
                document.add_paragraph(text, style="List Bullet")
            else:
                document.add_paragraph(text)
    if table:
        t = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                t.cell(r, c).text = value
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


SAMPLE_TXT = """Orion Space Agency Annual Report
================================

1. Mission Overview

The Orion Space Agency launched the Vega-9 lunar orbiter on 12 April 2025. The orbiter carries a
spectrometer designed to map water ice in permanently shadowed craters.

2. Budget

The total programme budget for 2025 was 312 million credits. Personnel costs accounted for 41
percent of the budget.
\f
3. Results

Vega-9 detected water ice deposits in 14 craters near the lunar south pole. The largest deposit,
in Shackleton crater, is estimated at 2.1 million tonnes.

4. Next Steps

The agency plans a sample-return lander called Lyra for 2028.
"""

SAMPLE_DOCX_PAGES = [
    [
        ("h1", "Data Retention Policy"),
        (
            "p",
            "Customer invoices must be retained for seven years after the end of the fiscal year.",
        ),
        ("p", "Server access logs are kept for 90 days and then deleted automatically."),
    ],
    [
        ("h1", "Backups"),
        ("p", "Encrypted backups are stored in two separate regions and tested every quarter."),
        ("li", "Daily incremental backups are kept for 35 days."),
    ],
]
