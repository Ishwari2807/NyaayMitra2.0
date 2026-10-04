import re

from backend.ingestion.structure.document_model import (
    DocumentElement,
)


# ---------------------------------------------------------
# Patterns
# ---------------------------------------------------------

SECTION_LABEL_PATTERN = re.compile(
    r"^\d+\.$"
)

SUBSECTION_LABEL_PATTERN = re.compile(
    r"^\([0-9A-Za-z]+\)$"
)

CHAPTER_PATTERN = re.compile(
    r"^CHAPTER\s+[IVXLCDM]+[A-Z]*$",
    re.IGNORECASE,
)


def is_section_candidate(
    element: DocumentElement,
) -> bool:

    if element.element_type != "numbered_item":
        return False

    label = element.label or ""

    return bool(
        SECTION_LABEL_PATTERN.match(label)
    )


def is_subsection_candidate(
    element: DocumentElement,
) -> bool:

    if element.element_type != "numbered_item":
        return False

    label = element.label or ""

    return bool(
        SUBSECTION_LABEL_PATTERN.match(label)
    )


def is_chapter_heading(
    element: DocumentElement,
) -> bool:

    if element.element_type != "heading":
        return False

    return bool(
        CHAPTER_PATTERN.match(
            element.text.strip()
        )
    )


def looks_like_statutory_section(
    elements: list[DocumentElement],
    index: int,
) -> bool:
    """
    Determines whether a numbered item is likely to be
    an actual statutory section rather than a table-of-
    contents entry.

    Key idea:
    A TOC entry usually has the section title but does not
    immediately contain the legal provision structure.

    An actual statutory section commonly has content such as
    a subsection, paragraph, proviso, etc. following it.

    This is deliberately conservative.
    """

    element = elements[index]

    if not is_section_candidate(element):
        return False

    # Look ahead a limited number of elements.
    lookahead = elements[
        index + 1:index + 12
    ]

    for candidate in lookahead:

        if is_subsection_candidate(candidate):

            return True

        if candidate.element_type == "paragraph":

            text = candidate.text.strip()

            # Legal sections commonly have meaningful
            # prose immediately following the heading.
            if len(text) > 40:
                return True

    return False


def detect_statutory_body(
    elements: list[DocumentElement],
) -> tuple[int | None, int | None]:

    """
    Detects the approximate beginning and end of the
    statutory body.

    Returns:
        (start_index, end_index)

    The detector is intentionally conservative.

    If the boundary cannot be identified confidently,
    it returns None rather than inventing a boundary.
    """

    start_index = None

    # -----------------------------------------------------
    # Find first strong statutory-section candidate
    # -----------------------------------------------------

    for index, element in enumerate(elements):

        if looks_like_statutory_section(
            elements,
            index,
        ):

            start_index = index
            break

    if start_index is None:
        return None, None

    # -----------------------------------------------------
    # Find possible back matter
    # -----------------------------------------------------

    end_index = len(elements)

    back_matter_markers = [
        "STATEMENT OF OBJECTS AND REASONS",
        "STATEMENT OF OBJECTS",
        "NOTES ON CLAUSES",
        "FINANCIAL MEMORANDUM",
        "MEMORANDUM REGARDING DELEGATED LEGISLATION",
    ]

    for index in range(
        start_index + 1,
        len(elements),
    ):

        element = elements[index]

        text = element.text.strip().upper()

        if (
            element.element_type == "heading"
            and any(
                marker in text
                for marker in back_matter_markers
            )
        ):

            end_index = index
            break

    return start_index, end_index


def mark_regions(
    elements: list[DocumentElement],
) -> list[DocumentElement]:

    """
    Adds a generic region classification to each element.

    Possible regions:

    - front_matter
    - main_content
    - back_matter
    """

    start, end = detect_statutory_body(
        elements
    )

    # If no reliable boundary exists,
    # don't pretend we know one.
    if start is None:

        for element in elements:

            element.metadata[
                "document_region"
            ] = "unknown"

        return elements

    for index, element in enumerate(elements):

        if index < start:

            element.metadata[
                "document_region"
            ] = "front_matter"

        elif index < end:

            element.metadata[
                "document_region"
            ] = "main_content"

        else:

            element.metadata[
                "document_region"
            ] = "back_matter"

    return elements