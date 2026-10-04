from pathlib import Path

from src.ingestion.document_loader import DocumentLoader


# Path to our legal knowledge base
knowledge_base_path = Path("data/knowledge_base")


# Create the document loader
loader = DocumentLoader(str(knowledge_base_path))


# Load all supported documents
documents = loader.load_all_documents()


# Display a small sample
print("\n--- SAMPLE DOCUMENT ---")

if documents:
    document = documents[0]

    print("Content:")
    print(document.page_content[:500])

    print("\nMetadata:")
    print(document.metadata)
else:
    print("No documents were loaded.")