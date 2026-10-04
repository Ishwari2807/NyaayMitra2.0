import json
from pathlib import Path

from backend.rag.embeddings.embedding_model import EmbeddingModel
from backend.rag.vectorstore.chroma_store import ChromaStore


CHUNKS_DIR = Path("data/chunks")


def load_chunk_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    print("=" * 70)
    print("NyaayMitra — Building Vector Index")
    print("=" * 70)

    chunk_files = sorted(
        CHUNKS_DIR.glob("*.json")
    )

    if not chunk_files:
        raise FileNotFoundError(
            "No chunk files found in data/chunks/"
        )

    print(f"\nChunk files found: {len(chunk_files)}")

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    all_chunks = []

    for path in chunk_files:

        print(f"\nLoading: {path.name}")

        chunks = load_chunk_file(path)

        print(f"Chunks: {len(chunks)}")

        # ----------------------------------------------------
        # Prepare chunks
        # ----------------------------------------------------

        for chunk in chunks:

            text = chunk.get("text", "").strip()

            # Skip empty chunks.
            if not text:
                continue

            metadata = chunk.setdefault(
                "metadata",
                {},
            )

            # Add document-level identifiers.
            metadata["source_file"] = path.stem

            # ChromaDB requires globally unique IDs.
            # Chunk IDs are currently unique only within
            # each document.
            original_chunk_id = chunk["chunk_id"]

            chunk["chunk_id"] = (
                f"{path.stem}__{original_chunk_id}"
            )

            metadata["document_id"] = path.stem
            metadata["original_chunk_id"] = original_chunk_id

            all_chunks.append(chunk)

    print(
        f"\nTotal non-empty chunks to index: "
        f"{len(all_chunks)}"
    )

    if not all_chunks:
        raise ValueError(
            "No non-empty chunks available for indexing."
        )

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    embedding_model = EmbeddingModel()

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    print("\nGenerating embeddings...")

    embeddings = embedding_model.encode(texts)

    print(
        f"Generated embeddings: {len(embeddings)}"
    )

    # --------------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------------

    store = ChromaStore()

    print("\nAdding chunks to ChromaDB...")

    store.add_chunks(
        chunks=all_chunks,
        embeddings=embeddings,
    )

    print("\n" + "=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Indexed chunks: {store.count()}"
    )


if __name__ == "__main__":
    main()