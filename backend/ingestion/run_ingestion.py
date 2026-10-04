import json
import sys
import os
from pathlib import Path

from backend.ingestion.parsers.pdf_parser import extract_pdf
from backend.ingestion.parsers.docx_parser import extract_docx
from backend.ingestion.parsers.csv_parser import extract_csv
from backend.ingestion.parsers.xlsx_parser import extract_xlsx
from backend.ingestion.parsers.txt_parser import extract_txt
from backend.ingestion.parsers.html_parser import extract_html

from backend.ingestion.structure.element_detector import detect_elements
from backend.ingestion.structure.region_detector import mark_regions
from backend.ingestion.structure.hierarchy_builder import build_hierarchy

from backend.ingestion.interpreters.document_classifier import (
    classify_document,
)
from backend.ingestion.interpreters.legal_region_detector import (
    mark_legal_regions,
)
from backend.ingestion.interpreters.legal_interpreter import (
    interpret_legal_document,
)

from backend.ingestion.chunking.structure_aware_chunker import (
    create_chunks,
)

from backend.ingestion.validation.validate_document import (
    validate_extracted_document,
    validate_elements,
)


# ============================================================
# Directories
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"
STRUCTURED_DIR = BASE_DIR / "data" / "structured"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"


# ============================================================
# Parser selection
# ============================================================

def get_parser(file_path: Path):
    """
    Return the appropriate parser based on file extension.
    """

    extension = file_path.suffix.lower()

    parsers = {
        ".pdf": extract_pdf,
        ".docx": extract_docx,
        ".csv": extract_csv,
        ".xlsx": extract_xlsx,
        ".txt": extract_txt,
        ".html": extract_html,
        ".htm": extract_html,
    }

    return parsers.get(extension)


# ============================================================
# Category detection
# ============================================================

def get_source_category(file_path: Path) -> str:
    """
    Determine the knowledge-base category from the folder
    immediately below data/raw.

    Example:

        data/raw/01_Consumer_Protection/file.pdf

    returns:

        01_Consumer_Protection
    """

    try:

        relative_path = file_path.relative_to(RAW_DIR)

    except ValueError:

        return ""

    if len(relative_path.parts) >= 2:
        return relative_path.parts[0]

    return ""


# ============================================================
# Hierarchy flattening
# ============================================================

def flatten_hierarchy(roots):
    """
    Flatten the hierarchical structure without losing elements.

    The hierarchy is preserved through each element's `children`
    field, while this function provides a complete linear view
    for validation, interpretation and indexing.
    """

    flattened = []

    def collect(nodes):

        for node in nodes:

            flattened.append(node)

            if node.children:
                collect(node.children)

    collect(roots)

    return flattened


# ============================================================
# Structured serialization
# ============================================================

def build_structured_output(
    document,
    classification,
    roots,
    flat_elements,
):
    """
    Build the structured JSON representation.

    `elements` contains ALL elements.

    `roots` contains the hierarchical top-level elements.
    """

    return {
        "document": document.to_dict(),

        "classification": classification,

        "elements": [
            element.to_dict()
            for element in flat_elements
        ],

        "roots": [
            root.to_dict()
            for root in roots
        ],
    }


# ============================================================
# Process one document
# ============================================================

def process_document(file_path: Path):

    print(
        f"\nProcessing: {file_path.name}"
    )

    parser = get_parser(file_path)

    if parser is None:

        print(
            f"Unsupported file type: "
            f"{file_path.suffix}"
        )

        return False

    # --------------------------------------------------------
    # Knowledge-base category
    # --------------------------------------------------------

    source_category = get_source_category(
        file_path
    )

    if source_category:

        print(
            f"Category: "
            f"{source_category}"
        )

    # --------------------------------------------------------
    # 1. Extraction
    # --------------------------------------------------------

    document = parser(
        str(file_path)
    )

    # Preserve source/category information.

    document.metadata = dict(
        document.metadata or {}
    )

    document.metadata.update(
        {
            "source_category": source_category,
            "source_relative_path": str(
                file_path.relative_to(RAW_DIR)
            ),
        }
    )

    extraction_errors = validate_extracted_document(
        document
    )

    if extraction_errors:

        print("\nExtraction errors:")

        for error in extraction_errors:

            print(
                f"  - {error}"
            )

        return False

    EXTRACTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted_path = (
        EXTRACTED_DIR
        / f"{document.document_id}.json"
    )

    with open(
        extracted_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            document.to_dict(),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Extracted pages: "
        f"{len(document.pages)}"
    )

    # --------------------------------------------------------
    # 2. Generic structure detection
    # --------------------------------------------------------

    elements = detect_elements(
        document
    )

    detected_count = len(elements)

    print(
        f"Detected elements: "
        f"{detected_count}"
    )

    elements = mark_regions(
        elements
    )

    # --------------------------------------------------------
    # 3. Hierarchy construction
    # --------------------------------------------------------

    roots = build_hierarchy(
        elements
    )

    root_count = len(roots)

    flat_elements = flatten_hierarchy(
        roots
    )

    flattened_count = len(flat_elements)

    print(
        f"Hierarchy roots: "
        f"{root_count}"
    )

    print(
        f"Flattened elements: "
        f"{flattened_count}"
    )

    # --------------------------------------------------------
    # 3A. Lossless preservation check
    # --------------------------------------------------------

    if flattened_count != detected_count:

        print(
            "\nERROR: Element preservation failure."
        )

        print(
            f"Detected elements: "
            f"{detected_count}"
        )

        print(
            f"Flattened elements: "
            f"{flattened_count}"
        )

        print(
            "Stopping ingestion because "
            "elements were lost during hierarchy "
            "construction."
        )

        return False

    # --------------------------------------------------------
    # 4. Document classification
    # --------------------------------------------------------

    all_text = "\n".join(
        element.text
        for element in flat_elements
        if element.text
    )

    classification = classify_document(
        document.title or document.file_name,
        all_text,
    )

    document_type = classification[
        "document_type"
    ]

    print(
        f"Document type: "
        f"{document_type}"
    )

    # --------------------------------------------------------
    # 5. Legal interpretation
    # --------------------------------------------------------

    if document_type == "legislation":

        flat_elements = mark_legal_regions(
            flat_elements
        )

        flat_elements = interpret_legal_document(
            flat_elements
        )

    # --------------------------------------------------------
    # 6. Validation
    # --------------------------------------------------------

    validation = validate_elements(
        flat_elements
    )

    print(
        f"Elements: "
        f"{validation['element_count']}"
    )

    print(
        f"Element types: "
        f"{validation['element_types']}"
    )

    if not validation["valid"]:

        print(
            "\nValidation errors:"
        )

        for error in validation["errors"]:

            print(
                f"  - {error}"
            )

        return False

    # --------------------------------------------------------
    # 6A. Post-interpretation preservation check
    # --------------------------------------------------------

    if validation["element_count"] != detected_count:

        print(
            "\nERROR: Element count changed "
            "during interpretation."
        )

        print(
            f"Originally detected: "
            f"{detected_count}"
        )

        print(
            f"After interpretation: "
            f"{validation['element_count']}"
        )

        print(
            "Stopping ingestion."
        )

        return False

    # --------------------------------------------------------
    # 7. Structured output
    # --------------------------------------------------------

    STRUCTURED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    structured_output = build_structured_output(
        document=document,
        classification=classification,
        roots=roots,
        flat_elements=flat_elements,
    )

    structured_path = (
        STRUCTURED_DIR
        / f"{document.document_id}.json"
    )

    with open(
        structured_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            structured_output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # 8. Structure-aware chunking
    # --------------------------------------------------------

    configured_limit = os.getenv(
        "NYAAYMITRA_MAX_CHUNK_CHARACTERS"
    )

    max_characters = (
        int(configured_limit)
        if configured_limit
        else None
    )

    chunks = create_chunks(
        roots,
        max_characters=max_characters,
        document_metadata=document.metadata,
    )

    CHUNKS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks_path = (
        CHUNKS_DIR
        / f"{document.document_id}.json"
    )

    with open(
        chunks_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            [
                chunk.to_dict()
                for chunk in chunks
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # 9. Completion
    # --------------------------------------------------------

    print(
        f"Chunks created: "
        f"{len(chunks)}"
    )

    print(
        "Processing completed."
    )

    return True


# ============================================================
# Main
# ============================================================

def main():

    if not RAW_DIR.exists():

        print(
            f"Raw directory not found: "
            f"{RAW_DIR}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Recursive document discovery
    # --------------------------------------------------------

    supported_extensions = {
        ".pdf",
        ".docx",
        ".csv",
        ".xlsx",
        ".txt",
        ".html",
        ".htm",
    }

    files = sorted(
        path
        for path in RAW_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in supported_extensions
        )
    )

    if not files:

        print(
            "No supported documents found in "
            f"{RAW_DIR}"
        )

        return

    print("=" * 70)
    print("NyaayMitra Knowledge Base Ingestion")
    print("=" * 70)

    print(
        f"\nRaw directory: "
        f"{RAW_DIR}"
    )

    print(
        f"Documents discovered: "
        f"{len(files)}"
    )

    # --------------------------------------------------------
    # Process all documents
    # --------------------------------------------------------

    successful = 0
    failed = 0

    for file_path in files:

        try:

            success = process_document(
                file_path
            )

            if success:
                successful += 1
            else:
                failed += 1

        except Exception as error:

            failed += 1

            print(
                f"\nERROR processing "
                f"{file_path}"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)

    print(
        f"Documents discovered: "
        f"{len(files)}"
    )

    print(
        f"Successfully processed: "
        f"{successful}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print("=" * 70)

    if failed:

        print(
            "\nWARNING: Some documents failed "
            "during ingestion."
        )

    else:

        print(
            "\nALL DOCUMENTS PROCESSED SUCCESSFULLY."
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()