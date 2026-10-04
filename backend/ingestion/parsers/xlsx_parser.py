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


def extract_xlsx(file_path: str) -> Document:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if path.suffix.lower() not in {
        ".xlsx",
        ".xls",
    }:
        raise ValueError(
            "Expected an XLSX/XLS file."
        )

    workbook = pd.ExcelFile(file_path)

    pages = []
    tables = []

    for sheet_number, sheet_name in enumerate(
        workbook.sheet_names,
        start=1
    ):

        dataframe = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            dtype=str,
        ).fillna("")

        blocks = []

        headers = list(dataframe.columns)

        blocks.append(
            RawBlock(
                block_id=(
                    f"s{sheet_number}_header"
                ),
                page_number=sheet_number,
                text=" | ".join(
                    str(header)
                    for header in headers
                ),
                block_type="table_header",
                metadata={
                    "sheet_name": sheet_name,
                    "columns": headers,
                },
            )
        )

        for index, row in dataframe.iterrows():

            values = [
                str(value)
                for value in row.tolist()
            ]

            blocks.append(
                RawBlock(
                    block_id=(
                        f"s{sheet_number}_row_{index + 1}"
                    ),
                    page_number=sheet_number,
                    text=" | ".join(values),
                    block_type="table_row",
                    metadata={
                        "sheet_name": sheet_name,
                        "row_number": index + 1,
                        "columns": dict(
                            zip(headers, values)
                        ),
                    },
                )
            )

        tables.append({
            "table_id": f"sheet_{sheet_number}",
            "sheet_name": sheet_name,
            "columns": headers,
            "rows": [dict(zip(headers, [str(v) for v in row.tolist()])) for _, row in dataframe.iterrows()],
        })

        pages.append(
            Page(
                page_number=sheet_number,
                blocks=blocks,
                metadata={
                    "sheet_name": sheet_name
                },
            )
        )

    return Document(
        document_id=create_document_id(file_path),
        file_name=path.name,
        file_type="xlsx",
        title=path.stem,
        tables=tables,
        pages=pages,
        metadata={
            "sheet_names": workbook.sheet_names
        },
    )