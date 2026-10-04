import re

from backend.ingestion.structure.document_model import DocumentElement


SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+\[)?(\d+[A-Za-z]?)\.\s+"
)

SUBSECTION_PATTERN = re.compile(
    r"^\s*\(([0-9]+[A-Za-z]?)\)"
)

CLAUSE_PATTERN = re.compile(
    r"^\s*\(([a-z])\)"
)

SUBCLAUSE_PATTERN = re.compile(
    r"^\s*\(([ivxlcdm]+)\)"
)


def is_amendment_note(text: str) -> bool:
    upper = (text or "").upper()

    indicators = (
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

    return any(x in upper for x in indicators)


def interpret_legal_element(element: DocumentElement) -> DocumentElement:

    region = element.metadata.get("document_region")

    # Never interpret front matter / TOC / unknown material as
    # statutory provisions.
    if region != "main_content":
        return element

    text = element.text.strip()

    # Amendment notes must be preserved but not interpreted as sections.
    if element.element_type == "numbered_item" and is_amendment_note(text):
        element.semantic_role = "amendment_note"
        return element

    # Actual statutory section.
    if element.element_type == "numbered_item":

        match = SECTION_PATTERN.match(text)

        if match:
            element.semantic_role = "legal_section"
            element.metadata["section_number"] = match.group(1)
            return element

    # Subsection.
    if element.element_type == "numbered_item":

        match = SUBSECTION_PATTERN.match(text)

        if match:
            element.semantic_role = "legal_subsection"
            element.metadata["subsection_number"] = match.group(1)
            return element

    # Clause.
    if element.element_type == "numbered_item":

        match = CLAUSE_PATTERN.match(text)

        if match:
            element.semantic_role = "legal_clause"
            element.metadata["clause_label"] = match.group(1)
            return element

    # Sub-clause.
    if element.element_type == "numbered_item":

        match = SUBCLAUSE_PATTERN.match(text)

        if match:
            element.semantic_role = "legal_subclause"
            element.metadata["subclause_label"] = match.group(1)
            return element

    # Legal textual structures.
    upper = text.upper()

    if upper.startswith("EXPLANATION"):
        element.semantic_role = "explanation"

    elif upper.startswith("ILLUSTRATION"):
        element.semantic_role = "illustration"

    elif upper.startswith("PROVIDED THAT"):
        element.semantic_role = "proviso"

    return element


def interpret_legal_document(elements):
    """
    Apply legal interpretation conservatively.

    The elements themselves are never removed.
    """

    for element in elements:
        interpret_legal_element(element)

    return elements