import re
from backend.ingestion.structure.document_model import (
    DocumentElement,
)


FOOTNOTE_LEVEL = 99  # deliberately deeper than any real structural level


def estimate_level(
    element: DocumentElement,
) -> int:

    # Footnotes/reference apparatus must never anchor the tree or
    # interrupt sibling relationships between real structural items.
    # Giving them the deepest possible level means they simply nest
    # under whatever the current deepest node is, and get popped off
    # the stack (without taking any real content down with them) as
    # soon as the next real structural element arrives.
    if element.element_type == "footnote":
        return FOOTNOTE_LEVEL

    if element.element_type == "heading":
        return 1

    if element.element_type == "question":
        return 1

    if element.element_type == "numbered_item":

        label = element.label or ""

        if re_is_top_level_number(label):
            return 1

        return 2

    if element.element_type == "bullet":
        return 2

    return 2


def re_is_top_level_number(label: str) -> bool:
    if not label:
        return False

    label = label.strip()

    # Section labels are inconsistently emitted with or without a
    # trailing period by upstream detection (e.g. "43." vs "66F").
    # Strip a single trailing period before pattern-matching so both
    # forms are recognised as top-level statutory section labels.
    if label.endswith("."):
        label = label[:-1]

    # Top-level statutory section labels:
    # 1
    # 2
    # 3
    # 3A
    # 7A
    # 43A
    # 77A
    return bool(
        re.fullmatch(
            r"\d+[A-Za-z]?",
            label
        )
    )

def build_hierarchy(
    elements: list[DocumentElement],
) -> list[DocumentElement]:

    roots = []

    stack: list[DocumentElement] = []

    for element in elements:

        element.level = estimate_level(
            element
        )

        # Tables are kept together as flat structures.
        if element.element_type in {
            "table_header",
            "table_row",
        }:

            if roots:
                roots.append(element)
            else:
                roots.append(element)

            continue

        while stack and stack[-1].level >= element.level:
            stack.pop()

        if stack:

            parent = stack[-1]

            element.parent_id = (
                parent.element_id
            )

            parent.children.append(
                element
            )

        else:

            roots.append(element)

        stack.append(element)

    return roots