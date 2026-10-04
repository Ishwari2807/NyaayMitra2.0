import numpy as np

from src.embeddings.embedding_model import EmbeddingModel


def test_single_text_embedding():
    print("\n--- Test 1: Single Text ---")

    embedding_model = EmbeddingModel()

    text = "A consumer has the right to seek redressal."

    embedding = embedding_model.embed_text(text)

    print(f"Vector type: {type(embedding)}")
    print(f"Vector length: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    assert isinstance(embedding, list)
    assert len(embedding) > 0

    print("✓ Single text embedding passed")


def test_multiple_document_embeddings():
    print("\n--- Test 2: Multiple Chunks ---")

    embedding_model = EmbeddingModel()

    chunks = [
        "A consumer has the right to seek redressal.",
        "An e-commerce entity shall provide information about refund policies.",
        "Consumers should keep evidence of online transactions."
    ]

    embeddings = embedding_model.embed_documents(chunks)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Number of vectors: {len(embeddings)}")

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(chunks)

    for embedding in embeddings:
        assert isinstance(embedding, list)
        assert len(embedding) > 0

    print("✓ Multiple document embeddings passed")


def test_vector_dimension():
    print("\n--- Test 3: Vector Dimension ---")

    embedding_model = EmbeddingModel()

    text = "A consumer can file a complaint against an e-commerce entity."

    embedding = embedding_model.embed_text(text)

    expected_dimension = 384

    print(f"Vector dimension: {len(embedding)}")
    print(f"Expected dimension: {expected_dimension}")

    assert len(embedding) == expected_dimension

    print("✓ Vector dimension test passed")


def test_normalized_output():
    print("\n--- Test 4: Normalized Output ---")

    embedding_model = EmbeddingModel()

    text = "The seller must provide a refund when applicable."

    embedding = embedding_model.embed_text(text)

    vector = np.array(embedding)

    vector_norm = np.linalg.norm(vector)

    print(f"Vector norm: {vector_norm}")

    assert np.isclose(vector_norm, 1.0, atol=1e-5)

    print("✓ Normalization test passed")


if __name__ == "__main__":
    print("========================================")
    print("       EMBEDDING MODEL TESTS")
    print("========================================")

    test_single_text_embedding()
    test_multiple_document_embeddings()
    test_vector_dimension()
    test_normalized_output()

    print("\n========================================")
    print("✓ ALL EMBEDDING TESTS PASSED")
    print("========================================")