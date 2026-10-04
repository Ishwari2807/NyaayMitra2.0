from pathlib import Path

from bs4 import BeautifulSoup

from backend.ingestion.structure.document_model import (
    Document,
    Page,
    RawBlock,
)


def create_document_id(file_path: str) -> str:
    """
    Create a stable document ID from the filename.
    """

    filename = Path(file_path).stem.lower()

    safe_name = "".join(
        char if char.isalnum() else "_"
        for char in filename
    )

    return safe_name.strip("_")


def extract_html(file_path: str) -> Document:
    """
    Extract HTML content into the common NyaayMitra
    Document -> Page -> RawBlock representation.

    The parser preserves:
    - headings
    - paragraphs
    - list items
    - table rows
    - source filename
    - basic HTML element information

    HTML does not have native PDF-style page numbers,
    so the content is represented as page 1.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if path.suffix.lower() not in {".html", ".htm"}:
        raise ValueError(
            f"Expected HTML file, got: {path.suffix}"
        )

    document_id = create_document_id(file_path)

    html = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    blocks = []
    tables = []

    block_counter = 0

    # --------------------------------------------------------
    # Remove non-content elements
    # --------------------------------------------------------

    for tag in soup(
        ["script", "style", "noscript"]
    ):
        tag.decompose()

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title = None

    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(
            " ",
            strip=True,
        )

    if not title:
        title = path.stem

    # --------------------------------------------------------
    # Content extraction
    # --------------------------------------------------------

    content_tags = soup.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "li",
            "blockquote",
            "pre",
        ]
    )

    for tag in content_tags:

        text = tag.get_text(
            " ",
            strip=True,
        )

        if not text:
            continue

        block_counter += 1

        tag_name = tag.name.lower()

        if tag_name.startswith("h"):
            block_type = "heading"

        elif tag_name == "li":
            block_type = "list_item"

        elif tag_name == "blockquote":
            block_type = "blockquote"

        elif tag_name == "pre":
            block_type = "preformatted"

        else:
            block_type = "paragraph"

        blocks.append(
            RawBlock(
                block_id=f"b{block_counter}",
                page_number=1,
                text=text,
                block_type=block_type,
                metadata={
                    "html_tag": tag_name,
                },
            )
        )

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    for table_index, table in enumerate(
        soup.find_all("table"),
        start=1,
    ):

        table_rows = []
        table_headers = []

        rows = table.find_all("tr")

        for row_index, row in enumerate(
            rows,
            start=1,
        ):

            cells = row.find_all(
                ["th", "td"]
            )

            values = [
                cell.get_text(
                    " ",
                    strip=True,
                )
                for cell in cells
            ]

            values = [
                value
                for value in values
                if value
            ]

            if not values:
                continue

            if row_index == 1 and any(
                cell.name == "th"
                for cell in cells
            ):
                table_headers = values

            row_text = " | ".join(values)

            block_counter += 1

            blocks.append(
                RawBlock(
                    block_id=(
                        f"table{table_index}"
                        f"_row{row_index}"
                    ),
                    page_number=1,
                    text=row_text,
                    block_type="table_row",
                    metadata={
                        "table_index": table_index,
                        "row_index": row_index,
                        "columns": values,
                    },
                )
            )

            if table_headers:
                if row_index == 1:
                    continue

                table_rows.append(
                    dict(
                        zip(
                            table_headers,
                            values,
                        )
                    )
                )
            else:
                table_rows.append(
                    {
                        f"column_{i + 1}": value
                        for i, value in enumerate(values)
                    }
                )

        tables.append(
            {
                "table_id": (
                    f"html_table_{table_index}"
                ),
                "columns": table_headers,
                "rows": table_rows,
            }
        )

    # --------------------------------------------------------
    # Create normalized document
    # --------------------------------------------------------

    page = Page(
        page_number=1,
        blocks=blocks,
        metadata={
            "source_format": "html",
        },
    )

    return Document(
        document_id=document_id,
        file_name=path.name,
        file_type="html",
        title=title,
        pages=[page],
        tables=tables,
        metadata={
            "parser": "beautifulsoup4",
        },
    )