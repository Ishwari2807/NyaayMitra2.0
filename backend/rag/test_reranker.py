from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.reranker import LegalReranker


def main():

    print("\nInitializing hybrid retriever...")
    hybrid = HybridRetriever()

    print("\nInitializing reranker...")
    reranker = LegalReranker()

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
        # Step 1: Hybrid retrieval
        # --------------------------------------------------------------

        candidates = hybrid.search(
            query,
            top_k=10,
            retrieval_k=10,
        )

        print("\nHybrid candidates:", len(candidates))

        # --------------------------------------------------------------
        # Step 2: Cross-encoder reranking
        # --------------------------------------------------------------

        results = reranker.rerank(
            query,
            candidates,
            top_k=5,
        )

        # --------------------------------------------------------------
        # Step 3: Display final evidence
        # --------------------------------------------------------------

        for result in results:

            metadata = result.get(
                "metadata",
                {}
            )

            print("\n" + "-" * 100)

            print(
                "Final Rank:",
                result["rerank_rank"]
            )

            print(
                "Reranker Score:",
                round(
                    result["reranker_score"],
                    4
                )
            )

            print(
                "Chunk:",
                result["chunk_id"]
            )

            print(
                "Section:",
                metadata.get(
                    "section_numbers"
                )
            )

            print(
                "RRF Score:",
                round(
                    result["rrf_score"],
                    6
                )
            )

            print(
                "Vector Rank:",
                result.get(
                    "vector_rank"
                )
            )

            print(
                "BM25 Rank:",
                result.get(
                    "bm25_rank"
                )
            )

            print(
                "Text:",
                result["text"][:600]
            )


if __name__ == "__main__":
    main()