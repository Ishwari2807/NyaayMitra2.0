from pathlib import Path

from backend.ingestion.structure.document_model import (
    Document,
    Page,
    RawBlock,
)


def create_document_id(file_path: str) -> str:
    filename = Path(file_path).stem.lower()

    return "".join(
        char if char.isalnum() else "_"
        for char in filename
    ).strip("_")


def extract_txt(file_path: str) -> Document:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if path.suffix.lower() != ".txt":
        raise ValueError(
            "Expected a TXT file."
        )

    text = path.read_text(
        encoding="utf-8"
    )

    blocks = []

    for index, paragraph in enumerate(
        text.split("\n\n"),
        start=1
    ):

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        blocks.append(
            RawBlock(
                block_id=f"b{index}",
                page_number=1,
                text=paragraph,
                block_type="paragraph",
            )
        )

    return Document(
        document_id=create_document_id(file_path),
        file_name=path.name,
        file_type="txt",
        title=path.stem,
        pages=[
            Page(
                page_number=1,
                blocks=blocks,
            )
        ],
    )