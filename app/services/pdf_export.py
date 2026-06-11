"""Fill an extracted-label record into the COLA Form PDF.

The TTB COLA Form (TTB F 5100.31, Legal size, 612 x 1008 pts) is a flat scan —
no AcroForm fields — so we generate a transparent overlay with reportlab and
merge it onto page 1 of the original PDF with pypdf. Coordinates below were
calibrated visually; nudge them in FIELD_POSITIONS if a field is off-position.

We only fill the six fields we extract from the label:
  3. Source of product (Domestic | Imported) — derived from country_of_origin
  5. Type of product (Wine | Distilled Spirits | Malt Beverages) — class_type
  6. Brand name
  8. Name and address of applicant — producer_name_address
 12. Net contents
 13. Alcohol content

The rest of the form (Plant Registry No, Serial Number, Phone, Email,
Signature, etc.) is application metadata not present on the label and is
left blank for the applicant to fill in.
"""

#NOTE: THIS WAS NOT IMPLEMENTED FULLY
import io
from pathlib import Path
from typing import Optional

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import legal  # 612 x 1008 pts

# Path to the blank COLA template (mounted into the container)
_COLA_TEMPLATE = Path(__file__).resolve().parents[2] / "COLA Form.pdf"

# Coordinates are in PDF points, origin at bottom-left.
# Form sits in the top half of the page; bottom half is blank label-paste area.
# Tweak these if visual placement is off — text is left-aligned, baseline.
FIELD_POSITIONS = {
    # (x, y, max_width) — y is the BASELINE of the first line of text, PDF origin
    # at bottom-left. After visual calibration on the rendered template.
    # Nudge values here if a field renders in the wrong cell on your COLA Form.
    "brand_name":            (20, 780, 290),   # Field 6 — Brand Name content area
    "producer_name_address": (330, 855, 270),  # Field 8 — multi-line address area
    "net_contents":          (20, 695, 110),   # Field 12 — Net Contents content
    "alcohol_content":       (135, 695, 105),  # Field 13 — Alcohol Content content
}
# Checkboxes drawn as a single "X" centered over the small square next to each label.
CHECKBOX_POSITIONS = {
    # Field 3 — Source of product. "Domestic [ ]" and "[ ] Imported" sit on the
    # line just below the "(Required)" sub-label. Square is left of each word.
    "domestic":         (200, 858),
    "imported":         (260, 858),
    # Field 5 — Type of product (checkboxes left of WINE / DISTILLED SPIRITS / MALT BEVERAGES)
    "wine":             (155, 825),
    "distilled_spirits": (155, 812),
    "malt_beverages":   (155, 799),
}

_FONT = "Helvetica"
_FONT_SIZE = 9
_CHECKBOX_FONT_SIZE = 11


def _wrap_text(c: canvas.Canvas, text: str, max_width: int) -> list[str]:
    """Break text into lines that fit within max_width at the canvas's current font."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip() if current else word
        if c.stringWidth(candidate, _FONT, _FONT_SIZE) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _is_us_producer(producer: Optional[str], country: Optional[str]) -> bool:
    """Heuristic: domestic if no foreign country and producer mentions US state/abbreviation."""
    if country:
        # If country_of_origin is populated at this point it survived the
        # _US_LOCATIONS scrub in postprocess, so it's foreign.
        return False
    return True


def _classify_type(class_type: Optional[str]) -> Optional[str]:
    """Map normalized class_type to the COLA Field 5 checkbox key."""
    if not class_type:
        return None
    c = class_type.strip().lower()
    if c == "wine":
        return "wine"
    if c == "distilled spirits":
        return "distilled_spirits"
    if c == "malt beverage":
        return "malt_beverages"
    return None


def generate_filled_cola(extracted: dict) -> bytes:
    """Return a filled COLA PDF as bytes — same template, page 1 overlaid."""
    if not _COLA_TEMPLATE.exists():
        raise FileNotFoundError(f"COLA template not found at {_COLA_TEMPLATE}")

    # Build the overlay on a transparent Legal-size canvas
    overlay_buf = io.BytesIO()
    c = canvas.Canvas(overlay_buf, pagesize=legal)
    c.setFont(_FONT, _FONT_SIZE)

    # Text fields
    for field, (x, y, max_w) in FIELD_POSITIONS.items():
        value = extracted.get(field)
        if not value:
            continue
        lines = _wrap_text(c, str(value), max_w)
        for i, line in enumerate(lines):
            c.drawString(x, y - i * (_FONT_SIZE + 2), line)

    # Checkboxes: source of product (domestic vs imported)
    c.setFont(_FONT, _CHECKBOX_FONT_SIZE)
    src = "imported" if extracted.get("country_of_origin") else "domestic"
    cx, cy = CHECKBOX_POSITIONS[src]
    c.drawString(cx, cy, "X")

    # Checkboxes: type of product
    type_key = _classify_type(extracted.get("class_type"))
    if type_key and type_key in CHECKBOX_POSITIONS:
        cx, cy = CHECKBOX_POSITIONS[type_key]
        c.drawString(cx, cy, "X")

    c.save()
    overlay_buf.seek(0)

    # Merge overlay onto page 1 of the original template
    reader = PdfReader(str(_COLA_TEMPLATE))
    overlay = PdfReader(overlay_buf).pages[0]
    writer = PdfWriter()
    base_page = reader.pages[0]
    base_page.merge_page(overlay)
    writer.add_page(base_page)
    # Keep subsequent pages (instructions / notes) untouched
    for page in reader.pages[1:]:
        writer.add_page(page)

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()
