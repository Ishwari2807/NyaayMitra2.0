from pathlib import Path

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import HybridLegalChunker


# ---------------------------------------------------------
# 1. Load documents
# ---------------------------------------------------------

knowledge_base_path = Path("data/knowledge_base")

loader = DocumentLoader(str(knowledge_base_path))
documents = loader.load_all_documents()

print(f"\nLoaded documents: {len(documents)}")


# ---------------------------------------------------------
# 2. Create chunks
# ---------------------------------------------------------

chunker = HybridLegalChunker()

chunks = chunker.chunk_documents(documents)

print(f"Created chunks: {len(chunks)}")


# ---------------------------------------------------------
# 3. Test a real CivicGuides-style query
# ---------------------------------------------------------

test_query = (
    "I bought a defective product online and the seller "
    "is refusing to give me a refund."
)

print("\n" + "=" * 70)
print("CHUNK QUALITY TEST")
print("=" * 70)

print(f"\nTest situation:\n{test_query}")


# ---------------------------------------------------------
# 4. Simple keyword-based inspection
# ---------------------------------------------------------

keywords = [
    "defect",
    "refund",
    "consumer",
    "unfair trade",
    "e-commerce",
    "goods"
]

matching_chunks = []

for chunk in chunks:

    text = chunk.page_content.lower()

    score = sum(
        1 for keyword in keywords
        if keyword in text
    )

    if score > 0:
        matching_chunks.append((score, chunk))


# Highest keyword match first
matching_chunks.sort(
    key=lambda item: item[0],
    reverse=True
)


# ---------------------------------------------------------
# 5. Display top matching chunks
# ---------------------------------------------------------

print("\nTop matching chunks:\n")

for rank, (score, chunk) in enumerate(
    matching_chunks[:5],
    start=1
):

    print(f"\nRESULT {rank}")
    print("-" * 70)

    print(f"Keyword matches: {score}")

    print("\nContent:")
    print(chunk.page_content[:1000])

    print("\nMetadata:")
    print({
        "domain": chunk.metadata.get("domain"),
        "source_file": chunk.metadata.get("source_file"),
        "page": chunk.metadata.get("page"),
        "chunk_id": chunk.metadata.get("chunk_id"),
        "chunk_size": chunk.metadata.get("chunk_size")
    })