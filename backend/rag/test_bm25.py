from backend.rag.bm25_retriever import BM25Retriever


def main():

    retriever = BM25Retriever()

    queries = [
        "What is the definition of a child?",
        "What is the punishment for murder?",
        "What is an electronic signature?",
        "What does the law say about electronic records?",
        "Someone accessed a computer without permission.",
    ]

    for query in queries:

        print("\n")
        print("=" * 80)
        print("QUERY")
        print("=" * 80)
        print(query)

        results = retriever.search(
            query,
            top_k=5,
        )

        for result in results:

            metadata = result["metadata"]

            print("\n" + "-" * 80)

            print(
                "Rank:",
                result["rank"]
            )

            print(
                "BM25 score:",
                result["bm25_score"]
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
                "Text:",
                result["text"][:500]
            )


if __name__ == "__main__":
    main()