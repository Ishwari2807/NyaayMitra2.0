from backend.rag.hybrid_retriever import HybridRetriever


def main():

    retriever = HybridRetriever()

    queries = [
        "What is the definition of a child?",
        "What is the punishment for murder?",
        "What is an electronic signature?",
        "What does the law say about electronic records?",
        "Someone accessed a computer without permission.",
    ]

    for query in queries:

        print("\n")
        print("=" * 90)
        print("QUERY")
        print("=" * 90)
        print(query)

        results = retriever.search(
            query,
            top_k=5,
            retrieval_k=10,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            metadata = result["metadata"]

            print("\n" + "-" * 90)

            print(
                "Final Rank:",
                rank
            )

            print(
                "RRF Score:",
                round(
                    result["rrf_score"],
                    6
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
                "Vector Rank:",
                result["vector_rank"]
            )

            print(
                "BM25 Rank:",
                result["bm25_rank"]
            )

            print(
                "Text:",
                result["text"][:500]
            )


if __name__ == "__main__":
    main()