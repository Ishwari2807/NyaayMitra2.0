from backend.rag.retriever import LegalRetriever


def print_results(
    query: str,
    results,
):
    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)
    print(query)

    if not results:
        print("\nNo results found.")
        return

    for result in results:

        print("\n" + "-" * 80)

        print(
            f"Rank: {result['rank']}"
        )

        print(
            f"Chunk ID: {result['chunk_id']}"
        )

        print(
            f"Distance: {result['distance']}"
        )

        metadata = result["metadata"]

        print(
            f"Source: {metadata.get('source_file')}"
        )

        print(
            f"Section: {metadata.get('section_numbers')}"
        )

        print(
            f"Pages: {metadata.get('pages')}"
        )

        print(
            f"Structure: "
            f"{metadata.get('structure_path')}"
        )

        print("\nTEXT:")
        print(
            result["text"][:2500]
        )


def main():

    retriever = LegalRetriever()

    test_queries = [

        "What is the definition of a child?",

        "What is the punishment for murder?",

        "What is an electronic signature?",

        "What does the law say about electronic records?",

        "Someone accessed a computer without permission.",

    ]

    for query in test_queries:

        results = retriever.retrieve(
            query,
            top_k=5,
        )

        print_results(
            query,
            results,
        )


if __name__ == "__main__":
    main()