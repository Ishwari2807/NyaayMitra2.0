from pathlib import Path

import pandas as pd

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


def extract_csv(file_path: str) -> Document:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            "Expected a CSV file."
        )

    dataframe = pd.read_csv(
        file_path,
        dtype=str,
        keep_default_na=False,
    )

    blocks = []

    # Header
    headers = list(dataframe.columns)

    blocks.append(
        RawBlock(
            block_id="header",
            page_number=1,
            text=" | ".join(headers),
            block_type="table_header",
            metadata={
                "columns": headers
            },
        )
    )

    # Rows
    for index, row in dataframe.iterrows():

        values = [
            str(value)
            for value in row.tolist()
        ]

        blocks.append(
            RawBlock(
                block_id=f"row_{index + 1}",
                page_number=1,
                text=" | ".join(values),
                block_type="table_row",
                metadata={
                    "row_number": index + 1,
                    "columns": dict(
                        zip(headers, values)
                    ),
                },
            )
        )

    return Document(
        document_id=create_document_id(file_path),
        file_name=path.name,
        file_type="csv",
        title=path.stem,
        tables=[{
            "table_id": "csv_table_1",
            "columns": headers,
            "rows": [dict(zip(headers, [str(v) for v in row.tolist()])) for _, row in dataframe.iterrows()],
        }],
        pages=[
            Page(
                page_number=1,
                blocks=blocks,
            )
        ],
        metadata={
            "columns": headers,
            "row_count": len(dataframe),
        },
    )