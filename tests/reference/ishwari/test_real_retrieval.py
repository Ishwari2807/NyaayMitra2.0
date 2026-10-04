from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaStore


def main():
    print("========================================")
    print("       REAL RETRIEVAL TEST")
    print("========================================")

    # Connect to the real CivicGuides database
    embedding_model = EmbeddingModel()

    chroma_store = ChromaStore(
        persist_directory="data/chroma_db",
        collection_name="civic_guides"
    )

    print(f"\nStored chunks: {chroma_store.count()}")

    # Test user question
    query = "I bought a defective product online and the seller is refusing to give me a refund."

    print(f"\nUser query:\n{query}")

    # Create embedding for the user query
    query_embedding = embedding_model.embed_query(query)

    print("\nQuery embedding created.")
    print(f"Vector dimension: {len(query_embedding)}")

    # Search ChromaDB
    results = chroma_store.collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("\n========================================")
    print("       TOP 3 MATCHING CHUNKS")
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
        print(f"  Domain: {metadata.get('domain')}")
        print(f"  Source: {metadata.get('source_file')}")
        print(f"  Page: {metadata.get('page', 'N/A')}")
        print(f"  Chunk ID: {metadata.get('chunk_id')}")

    print("\n========================================")
    print("✓ REAL RETRIEVAL TEST COMPLETED")
    print("========================================")


if __name__ == "__main__":
    main()