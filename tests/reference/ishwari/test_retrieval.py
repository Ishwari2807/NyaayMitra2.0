from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaStore


def main():
    print("========================================")
    print("          RETRIEVAL TEST")
    print("========================================")

    # ---------------------------------------------------------
    # STEP 1: Load the embedding model
    # ---------------------------------------------------------

    embedding_model = EmbeddingModel()

    # ---------------------------------------------------------
    # STEP 2: Connect to our test ChromaDB
    # ---------------------------------------------------------

    chroma_store = ChromaStore(
        persist_directory="data/test_chroma_db",
        collection_name="test_collection"
    )

    print(f"\nStored chunks: {chroma_store.count()}")

    # ---------------------------------------------------------
    # STEP 3: Create a user query
    # ---------------------------------------------------------

    query = (
        "The seller is refusing to refund my defective product."
    )

    print(f"\nUser query:\n{query}")

    # ---------------------------------------------------------
    # STEP 4: Convert the query into an embedding
    # ---------------------------------------------------------

    query_embedding = embedding_model.embed_query(query)

    print("\nQuery embedding created.")
    print(f"Vector dimension: {len(query_embedding)}")

    # ---------------------------------------------------------
    # STEP 5: Search ChromaDB
    # ---------------------------------------------------------

    results = chroma_store.collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    # ---------------------------------------------------------
    # STEP 6: Get matching chunks
    # ---------------------------------------------------------

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # ---------------------------------------------------------
    # STEP 7: Print matching chunks
    # ---------------------------------------------------------

    print("\n========================================")
    print("          TOP MATCHING CHUNKS")
    print("========================================")

    for index, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):
        print(f"\n--- Result {index} ---")

        print(f"Distance: {distance:.4f}")

        print("\nChunk:")
        print(document)

        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    print("\n========================================")
    print("✓ RETRIEVAL TEST COMPLETED")
    print("========================================")


if __name__ == "__main__":
    main()