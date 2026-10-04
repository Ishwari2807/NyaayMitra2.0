from collections import defaultdict
from typing import Any, Dict, List


class DocumentRouter:
    """
    Identifies candidate verified documents from retrieved evidence.

    The router never creates a document name.
    A document can only be selected if it exists in the
    retrieved verified knowledge-base results.
    """

    def route(
        self,
        retrieval_results: List[Dict[str, Any]],
        top_documents: int = 3,
    ) -> List[Dict[str, Any]]:

        if not retrieval_results:
            return []

        grouped = defaultdict(list)

        for result in retrieval_results:
            metadata = result.get("metadata", {})

            source_file = metadata.get(
                "source_file"
            )

            if not source_file:
                continue

            grouped[source_file].append(result)

        document_candidates = []

        for source_file, results in grouped.items():

            scores = []

            for result in results:
                score = result.get(
                    "reranker_score"
                )

                if score is not None:
                    scores.append(float(score))

            best_score = max(scores) if scores else None

            document_candidates.append(
                {
                    "source_file": source_file,
                    "document_title": self._document_title(
                        results
                    ),
                    "best_score": best_score,
                    "supporting_chunks": len(results),
                }
            )

        document_candidates.sort(
            key=lambda x: (
                x["best_score"]
                if x["best_score"] is not None
                else float("-inf")
            ),
            reverse=True,
        )

        return document_candidates[:top_documents]

    @staticmethod
    def _document_title(
        results: List[Dict[str, Any]]
    ) -> str:

        for result in results:
            metadata = result.get("metadata", {})

            title = metadata.get("title")

            if title:
                return title

            source_file = metadata.get(
                "source_file"
            )

            if source_file:
                return source_file

        return "Unknown document"