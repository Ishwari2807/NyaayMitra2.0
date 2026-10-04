from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RawBlock:
    """
    A piece of content extracted from the original document.

    This layer should preserve the source as faithfully as possible.
    No legal interpretation happens here.
    """

    block_id: str
    page_number: int
    text: str

    block_type: str = "text"

    bbox: Optional[list] = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Page:
    """
    Represents one page of the original document.
    """

    page_number: int

    blocks: list[RawBlock] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """
    Universal document representation.

    This model deliberately does NOT assume:
    section, subsection, clause, chapter, etc.
    """

    document_id: str
    file_name: str
    file_type: str

    title: Optional[str] = None

    source: Optional[str] = None

    pages: list[Page] = field(default_factory=list)

    # Normalized table representation. Parsers may populate this when a
    # source format exposes tables; raw table rows remain in page blocks too.
    tables: list[dict[str, Any]] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_langchain_documents(self):
        """Convert pages to LangChain Documents at the integration boundary."""
        from langchain_core.documents import Document as LangChainDocument
        documents = []
        for page in self.pages:
            text = "\n".join(block.text for block in page.blocks if block.text)
            metadata = dict(self.metadata)
            metadata.update({
                "document_id": self.document_id,
                "source_file": self.file_name,
                "page": page.page_number,
                "file_type": self.file_type,
            })
            documents.append(LangChainDocument(page_content=text, metadata=metadata))
        return documents

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "title": self.title,
            "source": self.source,
            "tables": self.tables,
            "pages": [
                {
                    "page_number": page.page_number,
                    "blocks": [
                        {
                            "block_id": block.block_id,
                            "page_number": block.page_number,
                            "text": block.text,
                            "block_type": block.block_type,
                            "bbox": block.bbox,
                            "metadata": block.metadata,
                        }
                        for block in page.blocks
                    ],
                    "metadata": page.metadata,
                }
                for page in self.pages
            ],
            "metadata": self.metadata,
        }


@dataclass
class DocumentElement:
    """
    Generic structural element.

    Examples:
    - heading
    - paragraph
    - bullet
    - numbered_item
    - table
    - footnote
    - other

    'semantic_role' is optional and can later be assigned by
    an interpreter such as legal_interpreter.py.
    """

    element_id: str

    element_type: str

    text: str

    page_number: int

    level: int = 0

    parent_id: Optional[str] = None

    children: list["DocumentElement"] = field(default_factory=list)

    label: Optional[str] = None

    semantic_role: Optional[str] = None

    raw_text: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "text": self.text,
            "raw_text": self.raw_text,
            "page_number": self.page_number,
            "level": self.level,
            "parent_id": self.parent_id,
            "label": self.label,
            "semantic_role": self.semantic_role,
            "metadata": self.metadata,
            "children": [
                child.to_dict()
                for child in self.children
            ],
        }