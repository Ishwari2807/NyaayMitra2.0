from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaStore
from langchain_core.documents import Document


def main():
    print("========================================")
    print("       CHROMADB STORAGE TEST")
    print("========================================")

    # ---------------------------------------------------------
    # STEP 1: Create 3 sample chunks
    # ---------------------------------------------------------

    documents = [
        Document(
            page_content="A consumer has the right to seek redressal.",
            metadata={
                "domain": "consumer",
                "source_file": "CP Act 2019.pdf",
                "page": 10,
                "chunk_id": "test_1"
            }
        ),
        Document(
            page_content="An e-commerce entity must provide information about refund policies.",
            metadata={
                "domain": "consumer",
                "source_file": "E-Commerce Rules 2020.pdf",
                "page": 9,
                "chunk_id": "test_2"
            }
        ),
        Document(
            page_content="Customers should report unauthorized electronic transactions promptly.",
            metadata={
                "domain": "banking",
                "source_file": "Customer Protection.pdf",
                "page": 5,
                "chunk_id": "test_3"
            }
        )
    ]

    print(f"\nSample chunks created: {len(documents)}")

    # ---------------------------------------------------------
    # STEP 2: Create embeddings for the chunks
    # ---------------------------------------------------------

    embedding_model = EmbeddingModel()

    texts = [
        document.page_content
        for document in documents
    ]

    embeddings = embedding_model.embed_documents(texts)

    print(f"Embeddings created: {len(embeddings)}")

    # ---------------------------------------------------------
    # STEP 3: Create ChromaDB store
    # ---------------------------------------------------------

    chroma_store = ChromaStore(
        persist_directory="data/test_chroma_db",
        collection_name="test_collection"
    )

    # ---------------------------------------------------------
    # STEP 4: Store chunks + embeddings + metadata
    # ---------------------------------------------------------

    chroma_store.add_documents(
        documents=documents,
        embeddings=embeddings
    )

    # ---------------------------------------------------------
    # STEP 5: Check stored count
    # ---------------------------------------------------------

    count = chroma_store.count()

    print(f"\nStored count: {count}")

    assert count == 3

    print("\n✓ ChromaDB storage test passed")
    print("========================================")


if __name__ == "__main__":
    main()