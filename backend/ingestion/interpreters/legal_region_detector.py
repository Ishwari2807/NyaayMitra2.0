import re
from typing import List

from backend.ingestion.structure.document_model import DocumentElement


ARRANGEMENT_MARKERS = {
    "ARRANGEMENT OF SECTIONS",
    "ARRANGEMENT OF CLAUSES",
}

FRONT_MATTER_MARKERS = {
    "LIST OF ABBREVIATIONS USED",
    "LIST OF AMENDING ACTS",
    "STATEMENT OF OBJECTS AND REASONS",
    "STATEMENT OF OBJECTS AND REASONS",
    "MEMORANDUM REGARDING DELEGATED LEGISLATION",
}

BACK_MATTER_MARKERS = {
    "STATEMENT OF OBJECTS AND REASONS",
    "MEMORANDUM REGARDING DELEGATED LEGISLATION",
    "ANNEXURE",
    "APPENDIX",
}


def clean(text: str) -> str:
    return " ".join((text or "").upper().split())


def is_chapter_heading(element: DocumentElement) -> bool:
    text = clean(element.text)

    return bool(
        re.match(r"^CHAPTER\s+[0-9IVXLCDM]+[A-Z]?$", text)
        or re.match(r"^CHAPTER\s+[0-9]+$", text)
    )


def looks_like_statutory_section(element: DocumentElement) -> bool:
    """
    Conservative test.

    A statutory section must already be structurally identified as a
    numbered_item. We deliberately do not treat arbitrary numbered
    paragraphs as sections.
    """

    if element.element_type != "numbered_item":
        return False

    text = (element.text or "").strip()

    # Must begin with a simple section number.
    match = re.match(r"^(\d+)\.\s+", text)

    if not match:
        return False

    # Amendment notes have very characteristic language.
    upper = text.upper()

    amendment_indicators = (
        "SUBS. BY ACT",
        "SUBSTITUTED BY ACT",
        "INS. BY ACT",
        "INSERTED BY ACT",
        "OMITTED BY ACT",
        "SUBS. BY S.",
        "INS. BY S.",
        "W.E.F.",
        "IBID.",
        "NOTIFICATION NO.",
    )

    if any(indicator in upper for indicator in amendment_indicators):
        return False

    return True


def find_statutory_body_start(elements: List[DocumentElement]) -> int | None:
    """
    Locate the beginning of the actual statutory body.

    We look for a document-title/chapter sequence AFTER the arrangement
    of sections rather than treating the first numbered item as Section 1.
    """

    arrangement_seen = False

    for i, element in enumerate(elements):
        text = clean(element.text)

        if text in ARRANGEMENT_MARKERS:
            arrangement_seen = True
            continue

        if not arrangement_seen:
            continue

        # Look for the document's actual title occurring after the TOC.
        #
        # We intentionally use generic legal-title characteristics rather
        # than hardcoding BNS or IT Act names.
        if element.element_type == "heading":
            title_like = (
                "ACT" in text
                or "CODE" in text
                or "SANHITA" in text
                or "ORDINANCE" in text
            )

            if not title_like:
                continue

            # Search forward for a chapter heading followed by a
            # substantive numbered item.
            found_chapter = False

            for j in range(i + 1, min(i + 15, len(elements))):
                candidate = elements[j]

                if is_chapter_heading(candidate):
                    found_chapter = True
                    continue

                if found_chapter and looks_like_statutory_section(candidate):
                    # A real section usually has substantive text or
                    # children. This also separates it from TOC material.
                    if candidate.children:
                        return j

                    # If there are no children, require reasonably
                    # substantive text.
                    if len(candidate.text.strip()) > 60:
                        return j

    return None


def mark_legal_regions(elements: List[DocumentElement]) -> List[DocumentElement]:
    """
    Add document-region metadata without deleting or restructuring
    elements.
    """

    start = find_statutory_body_start(elements)

    if start is None:
        # Conservative behavior:
        # if we cannot confidently locate the statutory body, do not
        # pretend we know where it starts.
        for element in elements:
            element.metadata["document_region"] = "unknown"

        return elements

    in_back_matter = False

    for i, element in enumerate(elements):

        text = clean(element.text)

        if i < start:
            element.metadata["document_region"] = "front_matter"
            continue

        if text in BACK_MATTER_MARKERS:
            in_back_matter = True

        if in_back_matter:
            element.metadata["document_region"] = "back_matter"
        else:
            element.metadata["document_region"] = "main_content"

    return elements