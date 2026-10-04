from collections import Counter

from backend.ingestion.structure.document_model import (
    Document,
    DocumentElement,
)


def validate_extracted_document(
    document: Document,
) -> list[str]:

    errors = []

    if not document.document_id:
        errors.append(
            "Missing document_id"
        )

    if not document.file_name:
        errors.append(
            "Missing file_name"
        )

    if not document.file_type:
        errors.append(
            "Missing file_type"
        )

    if not document.pages:
        errors.append(
            "Document contains no pages"
        )

    for page in document.pages:

        if page.page_number <= 0:
            errors.append(
                f"Invalid page number: "
                f"{page.page_number}"
            )

        for block in page.blocks:

            if not block.text.strip():
                errors.append(
                    f"Empty block: "
                    f"{block.block_id}"
                )

    return errors


def validate_elements(
    elements: list[DocumentElement],
) -> dict:

    errors = []

    ids = [
        element.element_id
        for element in elements
    ]

    duplicate_ids = [
        element_id
        for element_id, count
        in Counter(ids).items()
        if count > 1
    ]

    if duplicate_ids:

        errors.append(
            "Duplicate element IDs: "
            + ", ".join(
                duplicate_ids
            )
        )

    lookup = {
        element.element_id: element
        for element in elements
    }

    for element in elements:

        if element.parent_id:

            if element.parent_id not in lookup:

                errors.append(
                    f"Element "
                    f"{element.element_id} "
                    f"references missing parent "
                    f"{element.parent_id}"
                )

        if not element.text.strip():

            errors.append(
                f"Element "
                f"{element.element_id} "
                f"has empty text"
            )

    type_counts = Counter(
        element.element_type
        for element in elements
    )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "element_count": len(elements),
        "element_types": dict(
            type_counts
        ),
    }