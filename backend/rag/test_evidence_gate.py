from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.reranker import LegalReranker
from backend.rag.evidence_gate import EvidenceGate


def main():

    hybrid = HybridRetriever()

    reranker = LegalReranker()

    gate = EvidenceGate()

    queries = [
        "What is the definition of a child?",
        "What is the punishment for murder?",
        "What is an electronic signature?",
        "What does the law say about electronic records?",
        "Someone accessed a computer without permission.",
    ]

    for query in queries:

        print("\n")
        print("=" * 100)
        print("QUERY")
        print("=" * 100)
        print(query)

        # --------------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------------

        candidates = hybrid.search(
            query,
            top_k=10,
            retrieval_k=10,
        )

        # --------------------------------------------------------------
        # Reranking
        # --------------------------------------------------------------

        reranked = reranker.rerank(
            query,
            candidates,
            top_k=5,
        )

        # --------------------------------------------------------------
        # Evidence gate
        # --------------------------------------------------------------

        decision = gate.evaluate(
            query,
            reranked,
        )

        print("\n" + "-" * 100)

        print(
            "STATUS:",
            decision["status"]
        )

        print(
            "PROCEED:",
            decision["proceed"]
        )

        print(
            "REASON:",
            decision["reason"]
        )

        print(
            "Top Score:",
            decision.get(
                "top_score"
            )
        )

        print(
            "Score Margin:",
            decision.get(
                "score_margin"
            )
        )

        print(
            "Query Overlap:",
            round(
                decision.get(
                    "query_term_overlap",
                    0.0
                ),
                3
            )
        )

        print("\nTop Evidence:")

        for result in reranked:

            print(
                "\nSection:",
                result.get(
                    "metadata",
                    {}
                ).get(
                    "section_numbers"
                )
            )

            print(
                "Reranker:",
                round(
                    result.get(
                        "reranker_score",
                        0.0
                    ),
                    4
                )
            )

            print(
                "Text:",
                result.get(
                    "text",
                    ""
                )[:300]
            )


if __name__ == "__main__":
    main()