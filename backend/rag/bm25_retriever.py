from pathlib import Path
import json
import re


class BM25Retriever:
    """
    Keyword-based retrieval over the same chunk corpus
    indexed in ChromaDB.

    BM25 is useful for exact legal terminology such as:
        "without permission"
        "electronic signature"
        "section 43"
        "security deposit"
        "Article 21"
        "Rule 12(2)(a)"
    """

    def __init__(
        self,
        chunk_directory: str = "data/chunks",
    ):
        self.chunk_directory = Path(chunk_directory)

        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None

        self._load_chunks()

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Tokenize legal text while preserving useful legal identifiers.

        Examples:
            Section 43
            Section 43A
            Section 43-A
            Article 21
            Rule 12(2)(a)
        """

        text = (text or "").lower()

        # Normalize hyphenated legal identifiers:
        # 43-a -> 43a
        # 43-A -> 43a
        text = re.sub(
            r"\b(\d+)\s*-\s*([a-z]+)\b",
            r"\1\2",
            text,
        )

        # Extract:
        #   43
        #   43a
        #   section
        #   article
        #   rule
        #   electronic
        #   signature
        return re.findall(
            r"\b\d+[a-z]*\b|\b[a-z]+\b",
            text,
        )

    # ------------------------------------------------------------------
    # Load chunks
    # ------------------------------------------------------------------

    def _load_chunks(self):

        for path in sorted(
            self.chunk_directory.glob("*.json")
        ):

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:

                document_chunks = json.load(f)

            # The filename stem identifies the source document.
            source_prefix = path.stem

            for chunk in document_chunks:

                # ------------------------------------------------------
                # Ignore genuinely empty chunks.
                # ------------------------------------------------------

                text = chunk.get("text", "")

                if not text or not text.strip():
                    continue

                # ------------------------------------------------------
                # Create the same global ID used by Chroma.
                # ------------------------------------------------------

                local_chunk_id = chunk.get(
                    "chunk_id"
                )

                if not local_chunk_id:
                    continue

                global_chunk_id = (
                    f"{source_prefix}__{local_chunk_id}"
                )

                chunk["_global_chunk_id"] = (
                    global_chunk_id
                )

                self.chunks.append(chunk)

        # --------------------------------------------------------------
        # Build BM25 corpus
        # --------------------------------------------------------------

        self.tokenized_corpus = [
            self.tokenize(
                chunk.get("text", "")
            )
            for chunk in self.chunks
        ]

        if self.tokenized_corpus:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(
                self.tokenized_corpus
            )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:

        if self.bm25 is None:
            return []

        query_tokens = self.tokenize(
            query
        )

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):

            chunk = self.chunks[index]

            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.get(
                        "_global_chunk_id"
                    ),
                    "text": chunk.get(
                        "text",
                        "",
                    ),
                    "metadata": chunk.get(
                        "metadata",
                        {},
                    ),
                    "bm25_score": float(
                        scores[index]
                    ),
                }
            )

        return results