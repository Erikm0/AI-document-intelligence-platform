"""Synthetic evaluation corpus.

The documents describe *fictional* organisations and facts so that a language
model cannot answer from its pre-training knowledge - any correct answer must
come from retrieval, and any answer to an unanswerable question is a
hallucination. Each document is rendered to a real file format (PDF via
ReportLab, DOCX via python-docx, TXT with form-feed page breaks) so the
evaluation exercises the actual parsers too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Section:
    heading: str
    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class CorpusDocument:
    filename: str
    title: str
    pages: tuple[tuple[Section, ...], ...]  # pages -> sections


AURORA_HANDBOOK = CorpusDocument(
    filename="aurora_employee_handbook.txt",
    title="Aurora Robotics Employee Handbook",
    pages=(
        (
            Section(
                "1. Remote Work Policy",
                (
                    "Aurora Robotics supports hybrid working. Employees may work remotely for up "
                    "to three days per week, provided their manager has approved a hybrid working "
                    "agreement.",
                    "All staff must be reachable during the core collaboration hours of 10:00 to "
                    "15:00 Central European Time, regardless of where they work.",
                    "Working from another country is allowed for a maximum of 20 working days per "
                    "calendar year and requires prior approval from the People Operations team.",
                ),
            ),
            Section(
                "2. Annual Leave",
                (
                    "Full-time employees receive 26 days of paid annual leave per calendar year, "
                    "in addition to public holidays.",
                    "Up to five unused leave days may be carried over into the first quarter of "
                    "the following year; any remaining days expire on 31 March.",
                ),
            ),
        ),
        (
            Section(
                "3. Parental Leave",
                (
                    "Every new parent, regardless of gender, is entitled to 18 weeks of fully paid "
                    "parental leave, which can be taken within the first 12 months after the birth "
                    "or adoption of a child.",
                    "Employees returning from parental leave may work a reduced schedule of 80 "
                    "percent for up to three months at full pay.",
                ),
            ),
            Section(
                "4. Travel and Expenses",
                (
                    "Meal expenses during business travel are reimbursed up to 45 euros per day. "
                    "Alcoholic beverages are never reimbursable.",
                    "Expense claims must be submitted through the Ledgerly portal within 30 days "
                    "of the expense and are approved by the employee's direct manager.",
                    "Economy class is mandatory for flights shorter than six hours; business class "
                    "may be booked for longer flights with director approval.",
                ),
            ),
        ),
        (
            Section(
                "5. Information Security",
                (
                    "Multi-factor authentication is mandatory for every company account. Laptops "
                    "must use full-disk encryption and lock automatically after five minutes of "
                    "inactivity.",
                    "Suspected security incidents, including lost devices and phishing emails, "
                    "must be reported to the security team at security@aurora-robotics.example "
                    "within four hours of discovery.",
                    "Production customer data may only be accessed from managed devices connected "
                    "through the corporate VPN.",
                ),
            ),
        ),
    ),
)

HELIOS_REPORT = CorpusDocument(
    filename="helios7_phase2_report.pdf",
    title="Helios-7 Compact Fusion Prototype - Phase II Test Report",
    pages=(
        (
            Section(
                "Executive Summary",
                (
                    "The Helios-7 compact stellarator completed its Phase II test campaign between "
                    "January and June 2026 at the Kestrel Bay research site.",
                    "The prototype sustained stable plasma confinement for 4.2 seconds, a fourfold "
                    "improvement over Phase I, and reached a peak energy gain factor (Q) of 0.73.",
                    "The campaign consisted of 312 plasma discharges, of which 287 met all "
                    "acceptance criteria.",
                ),
            ),
        ),
        (
            Section(
                "Magnet and Cooling Systems",
                (
                    "The superconducting magnets are made of REBCO tape and operate at 20 kelvin. "
                    "They produced a peak on-axis magnetic field of 9.4 tesla.",
                    "Heat is extracted by a lithium-lead breeding blanket. The blanket coolant "
                    "operates at an outlet temperature of 520 degrees Celsius, which is compatible "
                    "with a conventional steam turbine cycle.",
                ),
            ),
        ),
        (
            Section(
                "Incidents and Downtime",
                (
                    "Two unplanned shutdowns occurred during the campaign, on 14 March 2026 and on "
                    "2 May 2026. Both were caused by a quench in magnet coil C3 after a cryocooler "
                    "pressure drop.",
                    "Total unplanned downtime was 19 days. No personnel were injured and no "
                    "radiation limits were exceeded.",
                ),
            ),
            Section(
                "Budget",
                (
                    "Phase II cost 48.5 million euros, which is 6 percent above the approved "
                    "budget. The overrun was mainly caused by replacement cryocooler components.",
                ),
            ),
        ),
    ),
)

MERIDIAN_CONTRACT = CorpusDocument(
    filename="meridian_master_services_agreement.docx",
    title="Master Services Agreement - Meridian Logistics and Northwind Analytics",
    pages=(
        (
            Section(
                "1. Term",
                (
                    "This Agreement between Meridian Logistics B.V. (the Client) and Northwind "
                    "Analytics Ltd. (the Provider) starts on 1 February 2026 and runs for an "
                    "initial term of 24 months.",
                    "The Agreement renews automatically for successive 12-month periods unless "
                    "either party gives notice of non-renewal.",
                ),
            ),
            Section(
                "2. Fees and Payment",
                (
                    "The Client shall pay a fixed monthly service fee of 18,400 euros, excluding "
                    "VAT. Invoices are payable within 30 days of the invoice date.",
                    "Late payments accrue interest at a rate of 8 percent per annum, calculated "
                    "daily from the due date until payment is received.",
                ),
            ),
        ),
        (
            Section(
                "3. Service Levels",
                (
                    "The Provider guarantees a monthly platform availability of 99.5 percent, "
                    "excluding scheduled maintenance announced at least 72 hours in advance.",
                    "For every 0.5 percentage points of availability below the guarantee, the "
                    "Client receives a service credit of 5 percent of the monthly fee, capped at "
                    "30 percent.",
                ),
            ),
            Section(
                "4. Termination",
                (
                    "Either party may terminate this Agreement for convenience by giving 90 days "
                    "written notice.",
                    "Either party may terminate immediately if the other party commits a material "
                    "breach that is not remedied within 30 days of written notice.",
                ),
            ),
        ),
        (
            Section(
                "5. Liability and Confidentiality",
                (
                    "Each party's total liability under this Agreement is limited to the fees paid "
                    "in the 12 months preceding the claim. The cap does not apply to fraud or "
                    "breaches of confidentiality.",
                    "Confidentiality obligations survive for five years after the termination of "
                    "this Agreement.",
                ),
            ),
            Section(
                "6. Governing Law",
                (
                    "This Agreement is governed by the laws of the Netherlands. Disputes are "
                    "submitted to the competent court in Rotterdam.",
                ),
            ),
        ),
    ),
)

# Distractor: deliberately overlapping vocabulary with *conflicting* values, so a
# retriever that only matches topics (not entities) picks the wrong passage.
BOREALIS_GUIDE = CorpusDocument(
    filename="borealis_contractor_guide.txt",
    title="Borealis Labs Contractor Guide",
    pages=(
        (
            Section(
                "Working Arrangements",
                (
                    "Contractors at Borealis Labs may work remotely five days per week, but they "
                    "must attend the quarterly planning week on site in Tromso.",
                    "Contractor meal expenses are reimbursed up to 30 euros per day when "
                    "travelling on behalf of Borealis Labs.",
                ),
            ),
            Section(
                "Security for Contractors",
                (
                    "Contractors must report security incidents to the Borealis service desk "
                    "within 24 hours. Personal devices may not be used for Borealis work.",
                ),
            ),
        ),
        (
            Section(
                "Invoicing and Termination",
                (
                    "Contractors invoice Borealis Labs monthly and invoices are paid within 45 "
                    "days. Borealis Labs does not pay late payment interest to contractors.",
                    "Either Borealis Labs or the contractor may end an engagement with 14 days "
                    "written notice.",
                ),
            ),
            Section(
                "Laboratory Equipment",
                (
                    "The cryogenic test chamber in Building 4 operates at 77 kelvin and must be "
                    "inspected by a certified engineer every six months.",
                ),
            ),
        ),
    ),
)

CORPUS: tuple[CorpusDocument, ...] = (
    AURORA_HANDBOOK,
    HELIOS_REPORT,
    MERIDIAN_CONTRACT,
    BOREALIS_GUIDE,
)


# ----------------------------------------------------------------------------------
# Renderers
# ----------------------------------------------------------------------------------
def render_txt(doc: CorpusDocument, path: Path) -> Path:
    pages = []
    for page in doc.pages:
        lines: list[str] = []
        for section in page:
            lines.append(section.heading)
            lines.append("")
            for paragraph in section.paragraphs:
                lines.append(paragraph)
                lines.append("")
        pages.append("\n".join(lines).strip() + "\n")
    path.write_text(f"{doc.title}\n{'=' * len(doc.title)}\n\n" + "\f".join(pages), encoding="utf-8")
    return path


def render_docx(doc: CorpusDocument, path: Path) -> Path:
    import docx
    from docx.enum.text import WD_BREAK

    document = docx.Document()
    document.core_properties.title = doc.title
    document.add_heading(doc.title, level=0)
    for page_number, page in enumerate(doc.pages):
        if page_number > 0:
            document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        for section in page:
            document.add_heading(section.heading, level=1)
            for paragraph in section.paragraphs:
                document.add_paragraph(paragraph)
    document.save(str(path))
    return path


def render_pdf(doc: CorpusDocument, path: Path) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()

    def footer(canvas, _doc):  # type: ignore[no-untyped-def]
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(72, 30, f"{doc.title} - Confidential")
        canvas.drawRightString(523, 30, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    story = [Paragraph(doc.title, styles["Title"]), Spacer(1, 12)]
    for page_number, page in enumerate(doc.pages):
        if page_number > 0:
            story.append(PageBreak())
        for section in page:
            story.append(Paragraph(section.heading, styles["Heading2"]))
            for paragraph in section.paragraphs:
                story.append(Paragraph(paragraph, styles["BodyText"]))
                story.append(Spacer(1, 6))
    template = SimpleDocTemplate(str(path), pagesize=A4, title=doc.title)
    template.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def build_corpus(directory: Path) -> list[Path]:
    """Render every corpus document into ``directory`` and return the file paths."""
    directory.mkdir(parents=True, exist_ok=True)
    renderers = {".txt": render_txt, ".docx": render_docx, ".pdf": render_pdf}
    return [renderers[Path(d.filename).suffix](d, directory / d.filename) for d in CORPUS]
