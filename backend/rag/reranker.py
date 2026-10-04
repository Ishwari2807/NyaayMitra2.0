from typing import Any, Dict, List

class LegalReranker:
    """
    Cross-encoder reranker for NyaayMitra.

    The reranker evaluates the relevance of each retrieved
    legal chunk against the user's query.

    Pipeline:

        Query + Retrieved Chunk
                ↓
          Cross Encoder
                ↓
          Relevance Score
                ↓
          Reranked Evidence
    """

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
    ):

        from sentence_transformers import CrossEncoder

        print(
            f"Loading reranker model: {model_name}"
        )

        self.model = CrossEncoder(
            model_name
        )

        print("Reranker model loaded.")

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        if not query or not query.strip():
            return []

        if not results:
            return []

        query = query.strip()

        # --------------------------------------------------------------
        # Create query-document pairs
        # --------------------------------------------------------------

        pairs = [
            (
                query,
                result.get("text", "")
            )
            for result in results
        ]

        # --------------------------------------------------------------
        # Calculate relevance scores
        # --------------------------------------------------------------

        scores = self.model.predict(
            pairs
        )

        # --------------------------------------------------------------
        # Attach scores
        # --------------------------------------------------------------

        reranked = []

        for result, score in zip(
            results,
            scores
        ):

            updated_result = result.copy()

            updated_result[
                "reranker_score"
            ] = float(score)

            reranked.append(
                updated_result
            )

        # --------------------------------------------------------------
        # Sort by cross-encoder score
        # --------------------------------------------------------------

        reranked.sort(
            key=lambda result: result[
                "reranker_score"
            ],
            reverse=True,
        )

        # --------------------------------------------------------------
        # Assign new rank
        # --------------------------------------------------------------

        final_results = []

        for rank, result in enumerate(
            reranked[:top_k],
            start=1,
        ):

            result["rerank_rank"] = rank

            final_results.append(
                result
            )

        return final_results