import re

from backend.ingestion.structure.document_model import (
    Document,
    DocumentElement,
)


# ---------------------------------------------------------------------------
# Statutory section detection
# ---------------------------------------------------------------------------
# Handles:
#   1. Section text
#   3. Section text
#   3A. Section text
#   7A. Section text
#   10A. Section text
#   77A. Section text
#
# Also handles amendment-marker formatting sometimes extracted from PDFs:
#   3[3A. Electronic signature...
#   3[43A. Compensation...
#
# The actual section number is captured in group(1).
# ---------------------------------------------------------------------------

STATUTORY_SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+\[)?(\d+[A-Za-z]?)\.\s+"
)


# ---------------------------------------------------------------------------
# Generic document structures
# ---------------------------------------------------------------------------

BULLET_PATTERN = re.compile(
    r"^\s*(?:[•●▪◦‣\-\*])\s+"
)


NUMBERED_PATTERN = re.compile(
    r"^\s*("
    r"\d+[\.\)]|"
    r"[A-Za-z][\.\)]|"
    r"\([A-Za-z0-9]+\)"
    r")\s+"
)


QUESTION_PATTERN = re.compile(
    r"^\s*"
    r"(who|what|when|where|why|how|"
    r"can|could|is|are|do|does|"
    r"should|may|will)"
    r"\b.*\?\s*$",
    re.IGNORECASE,
)


HEADING_PATTERN = re.compile(
    r"^\s*[A-Z][A-Z0-9\s\-&,.:()]{2,100}\s*$"
)


# ---------------------------------------------------------------------------
# Generic footnote / reference-apparatus detection
# ---------------------------------------------------------------------------
#
# Typeset documents (statutes, contracts, manuals, etc.) commonly place
# small-font reference/annotation text (footnotes, amendment notes,
# citations) at the bottom of a page. This text is visually distinct
# from body content by font size alone, regardless of the document's
# subject matter. Detecting it this way keeps the logic domain-agnostic:
# no keyword lists, no reliance on phrases like "Subs. by Act" or
# "w.e.f.", which are specific to Indian statutory drafting conventions
# and would not generalise to other document types.
#
# A block whose text is dominated by a font size meaningfully smaller
# than the document's dominant (most common) body font size is treated
# as reference apparatus rather than structural content. This prevents
# such blocks from being mistaken for numbered structural items (e.g. a
# footnote marker "1." colliding with an actual top-level section "1.").
# ---------------------------------------------------------------------------

FOOTNOTE_FONT_RATIO = 0.9  # footnote font must be < 90% of dominant body size


def get_block_font_size(block) -> float | None:
    """
    Weighted-average font size for a block, weighted by span text length.
    Returns None if no font-size metadata is available (e.g. non-PDF
    sources), in which case footnote detection is simply skipped.
    """

    lines = (block.metadata or {}).get("lines") or []

    total_weight = 0.0
    weighted_sum = 0.0

    for line in lines:
        for span in line.get("spans", []):
            size = span.get("font_size")
            text = span.get("text", "")

            if size is None or not text.strip():
                continue

            weight = len(text)
            weighted_sum += size * weight
            total_weight += weight

    if total_weight == 0:
        return None

    return weighted_sum / total_weight


def compute_dominant_font_size(pages) -> float | None:
    """
    Document-wide dominant body font size, computed as the size
    accounting for the most total characters across all blocks.
    """

    size_weights: dict[float, float] = {}

    for page in pages:
        for block in page.blocks:
            size = get_block_font_size(block)

            if size is None:
                continue

            weight = len((block.text or "").strip())

            if weight == 0:
                continue

            # Bucket to nearest 0.5pt to absorb minor rendering noise.
            bucket = round(size * 2) / 2
            size_weights[bucket] = size_weights.get(bucket, 0.0) + weight

    if not size_weights:
        return None

    return max(size_weights, key=size_weights.get)


def is_footnote_block(block, dominant_font_size: float | None) -> bool:
    if dominant_font_size is None:
        return False

    size = get_block_font_size(block)

    if size is None:
        return False

    return size < dominant_font_size * FOOTNOTE_FONT_RATIO


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Used only for search/analysis.

    Original extracted text remains in raw_text.
    """

    text = text.replace(
        "\u00a0",
        " "
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ---------------------------------------------------------------------------
# Element type detection
# ---------------------------------------------------------------------------

def detect_element_type(
    text: str,
    block_type: str,
    metadata: dict,
    is_footnote: bool = False,
) -> str:

    stripped = text.strip()

    # Tables
    if block_type == "table_header":
        return "table_header"

    if block_type == "table_row":
        return "table_row"

    # -----------------------------------------------------------------------
    # Footnote / reference apparatus (font-size based, domain-agnostic).
    #
    # Checked BEFORE any numbering pattern so a footnote marker like "1."
    # is never mistaken for a top-level structural item such as an actual
    # "Section 1.". This must run first, otherwise STATUTORY_SECTION_PATTERN
    # or NUMBERED_PATTERN would classify it as structural content.
    # -----------------------------------------------------------------------

    if is_footnote:
        return "footnote"

    # -----------------------------------------------------------------------
    # IMPORTANT:
    # Check statutory sections BEFORE generic numbered items.
    #
    # This allows:
    #   3.
    #   3A.
    #   43A.
    #   77A.
    #
    # to be treated as numbered items that can later be interpreted as
    # legal sections.
    # -----------------------------------------------------------------------

    if STATUTORY_SECTION_PATTERN.match(stripped):
        return "numbered_item"

    # Generic bullets
    if BULLET_PATTERN.match(stripped):
        return "bullet"

    # Questions
    if QUESTION_PATTERN.match(stripped):
        return "question"

    # Generic numbered items
    if NUMBERED_PATTERN.match(stripped):
        return "numbered_item"

    # DOCX heading styles
    style = str(
        metadata.get("style", "")
    ).lower()

    if style.startswith("heading"):
        return "heading"

    # Uppercase headings
    if HEADING_PATTERN.match(stripped):
        return "heading"

    # Default
    return "paragraph"


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def extract_label(text: str) -> str | None:

    patterns = [
        # Normal numeric label:
        # 1.
        # 1)
        r"^\s*(\d+[\.\)])",

        # Alphabetic label:
        # a.
        # a)
        r"^\s*([A-Za-z][\.\)])",

        # Parenthesized label:
        # (1)
        # (a)
        # (iv)
        r"^\s*(\([A-Za-z0-9]+\))",

        # Statutory alphanumeric section, optionally preceded by an
        # amendment-insertion marker such as "3[43A. ...":
        # 3A.
        # 43A.
        # 3[43A.
        r"^\s*(?:\d+\[)?(\d+[A-Za-z]?)\.\s+",
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            text
        )

        if match:
            return match.group(1)

    return None


# ---------------------------------------------------------------------------
# Element extraction
# ---------------------------------------------------------------------------

def detect_elements(
    document: Document,
) -> list[DocumentElement]:

    elements = []

    counter = 0

    dominant_font_size = compute_dominant_font_size(
        document.pages
    )

    for page in document.pages:

        for block in page.blocks:

            counter += 1

            footnote = is_footnote_block(
                block,
                dominant_font_size,
            )

            element_type = detect_element_type(
                block.text,
                block.block_type,
                block.metadata,
                is_footnote=footnote,
            )

            normalized = normalize_text(
                block.text
            )

            element = DocumentElement(
                element_id=f"e{counter}",
                element_type=element_type,
                text=normalized,
                raw_text=block.text,
                page_number=block.page_number,
                label=(
                    None
                    if element_type == "footnote"
                    else extract_label(block.text)
                ),
                metadata={
                    "source_block_id": block.block_id,
                    "bbox": block.bbox,
                    **block.metadata,
                },
            )

            elements.append(element)

    return elements