from typing import Any, Dict, List

from backend.rag.retriever import LegalRetriever
from backend.rag.bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    Hybrid legal retriever for NyaayMitra.

    Combines:
        1. Semantic/vector retrieval
        2. BM25 keyword retrieval

    using Reciprocal Rank Fusion (RRF).

    Pipeline:

        Query
          ↓
        Vector retrieval
          +
        BM25 retrieval
          ↓
        Reciprocal Rank Fusion
          ↓
        Combined ranked evidence
    """

    RRF_K = 60

    def __init__(
        self,
        vector_retriever=None,
        bm25_retriever=None,
    ):

        self.vector_retriever = (
            vector_retriever
            if vector_retriever is not None
            else LegalRetriever()
        )

        self.bm25_retriever = (
            bm25_retriever
            if bm25_retriever is not None
            else BM25Retriever()
        )

    # ------------------------------------------------------------------
    # RRF
    # ------------------------------------------------------------------

    @classmethod
    def _rrf_score(
        cls,
        rank: int,
    ) -> float:
        """
        Reciprocal Rank Fusion score.

        RRF(rank) = 1 / (k + rank)
        """

        if rank is None:
            return 0.0

        return 1.0 / (
            cls.RRF_K + int(rank)
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval using vector search,
        BM25 and Reciprocal Rank Fusion.
        """

        if not query or not query.strip():
            return []

        query = query.strip()

        # --------------------------------------------------------------
        # 1. Vector retrieval
        # --------------------------------------------------------------

        vector_results = (
            self.vector_retriever.retrieve(
                query,
                top_k=retrieval_k,
            )
        )

        # --------------------------------------------------------------
        # 2. BM25 retrieval
        # --------------------------------------------------------------

        bm25_results = (
            self.bm25_retriever.search(
                query,
                top_k=retrieval_k,
            )
        )

        combined: Dict[str, Dict[str, Any]] = {}

        # --------------------------------------------------------------
        # 3. Add vector results
        # --------------------------------------------------------------

        for result in vector_results:

            chunk_id = result.get(
                "chunk_id",
                "",
            )

            if not chunk_id:
                continue

            rank = result.get("rank")

            if chunk_id not in combined:

                combined[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result.get(
                        "text",
                        "",
                    ),
                    "metadata": result.get(
                        "metadata",
                        {},
                    ),
                    "vector_rank": rank,
                    "bm25_rank": None,
                    "vector_distance": result.get(
                        "distance"
                    ),
                    "bm25_score": None,
                    "rrf_score": 0.0,
                }

            else:

                # Keep the vector result if it has
                # richer text/metadata.
                if result.get("text"):
                    combined[chunk_id]["text"] = (
                        result["text"]
                    )

                if result.get("metadata"):
                    combined[chunk_id]["metadata"] = (
                        result["metadata"]
                    )

                combined[chunk_id][
                    "vector_rank"
                ] = rank

                combined[chunk_id][
                    "vector_distance"
                ] = result.get("distance")

            combined[chunk_id][
                "rrf_score"
            ] += self._rrf_score(rank)

        # --------------------------------------------------------------
        # 4. Add BM25 results
        # --------------------------------------------------------------

        for result in bm25_results:

            chunk_id = result.get(
                "chunk_id",
                "",
            )

            if not chunk_id:
                continue

            rank = result.get("rank")

            if chunk_id not in combined:

                combined[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result.get(
                        "text",
                        "",
                    ),
                    "metadata": result.get(
                        "metadata",
                        {},
                    ),
                    "vector_rank": None,
                    "bm25_rank": rank,
                    "vector_distance": None,
                    "bm25_score": result.get(
                        "bm25_score"
                    ),
                    "rrf_score": 0.0,
                }

            else:

                # BM25 can provide stronger exact-match
                # text/metadata information.
                if not combined[chunk_id].get("text"):
                    combined[chunk_id]["text"] = (
                        result.get(
                            "text",
                            "",
                        )
                    )

                if not combined[chunk_id].get("metadata"):
                    combined[chunk_id]["metadata"] = (
                        result.get(
                            "metadata",
                            {},
                        )
                    )

                combined[chunk_id][
                    "bm25_rank"
                ] = rank

                combined[chunk_id][
                    "bm25_score"
                ] = result.get(
                    "bm25_score"
                )

            combined[chunk_id][
                "rrf_score"
            ] += self._rrf_score(rank)

        # --------------------------------------------------------------
        # 5. Sort using RRF score
        # --------------------------------------------------------------

        ranked_results = sorted(
            combined.values(),
            key=lambda item: item[
                "rrf_score"
            ],
            reverse=True,
        )

        # --------------------------------------------------------------
        # 6. Assign final ranks
        # --------------------------------------------------------------

        final_results = []

        for rank, result in enumerate(
            ranked_results[:top_k],
            start=1,
        ):

            result["rank"] = rank

            final_results.append(
                result
            )

        return final_results