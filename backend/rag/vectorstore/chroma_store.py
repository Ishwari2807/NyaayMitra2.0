from pathlib import Path
from typing import List, Dict, Any


class ChromaStore:
    """
    Local persistent ChromaDB vector store.
    """

    # ChromaDB has a maximum batch size.
    # Keep this comfortably below the observed limit of 5461.
    BATCH_SIZE = 5000

    def __init__(
        self,
        persist_directory: str = "data/vectorstore",
        collection_name: str = "nyaaymitra_legal",
    ):
        self.persist_directory = str(
            Path(persist_directory)
        )

        Path(self.persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        import chromadb

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": (
                    "NyaayMitra verified civic and legal "
                    "knowledge base"
                )
            },
        )

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ):
        """
        Add chunks and embeddings to ChromaDB.

        Chunks are inserted in batches to stay below
        ChromaDB's maximum batch size.
        """

        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        ids = []
        documents = []
        metadatas = []

        seen_ids = set()

        for chunk in chunks:

            chunk_id = chunk["chunk_id"]

            # ----------------------------------------------------
            # Defensive duplicate check
            # ----------------------------------------------------

            if chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID detected: {chunk_id}"
                )

            seen_ids.add(chunk_id)

            ids.append(chunk_id)

            documents.append(
                chunk["text"]
            )

            metadata = (
                chunk.get("metadata", {})
                .copy()
            )

            # ----------------------------------------------------
            # Chroma-compatible metadata
            # ----------------------------------------------------

            pages = chunk.get("pages", [])

            metadata["pages"] = ",".join(
                str(page)
                for page in pages
            )

            structure_path = (
                chunk.get("structure_path", [])
            )

            metadata["structure_path"] = (
                " > ".join(structure_path)
            )

            element_ids = (
                chunk.get("element_ids", [])
            )

            metadata["element_ids"] = ",".join(
                element_ids
            )

            # section_numbers can be a list,
            # so convert it to a string.
            section_numbers = metadata.get(
                "section_numbers",
                [],
            )

            if isinstance(section_numbers, list):
                metadata["section_numbers"] = ",".join(
                    str(number)
                    for number in section_numbers
                )

            # Keep only Chroma-supported scalar values.
            metadata = {
                key: value
                for key, value in metadata.items()
                if isinstance(
                    value,
                    (str, int, float, bool),
                )
            }

            metadatas.append(metadata)

        # --------------------------------------------------------
        # Final consistency check
        # --------------------------------------------------------

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Internal error: duplicate IDs remain."
            )

        # --------------------------------------------------------
        # Batched insertion into ChromaDB
        # --------------------------------------------------------
        #
        # ChromaDB rejected the previous 10,760-item upsert
        # because its maximum batch size is 5,461.
        #
        # Using 5,000 keeps every request safely below that limit.
        # --------------------------------------------------------

        total = len(ids)

        print(
            f"Adding {total} chunks to ChromaDB "
            f"in batches of {self.BATCH_SIZE}..."
        )

        for start in range(0, total, self.BATCH_SIZE):

            end = min(
                start + self.BATCH_SIZE,
                total,
            )

            print(
                f"  Upserting chunks "
                f"{start + 1}-{end} of {total}..."
            )

            self.collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

        print(
            f"Successfully added {total} chunks to ChromaDB."
        )

    def count(self) -> int:
        return self.collection.count()

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Dict[str, Any] | None = None,
    ):
        """
        Semantic vector search.
        """

        query_arguments = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }

        if where:
            query_arguments["where"] = where

        return self.collection.query(
            **query_arguments
        )