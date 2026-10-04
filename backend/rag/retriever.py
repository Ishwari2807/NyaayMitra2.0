from typing import Any, Dict, List

from backend.rag.embeddings.embedding_model import (
    EmbeddingModel,
)

from backend.rag.vectorstore.chroma_store import (
    ChromaStore,
)


class LegalRetriever:
    """
    Semantic retriever for NyaayMitra.

    Responsibilities:
        User query
            ↓
        Query embedding
            ↓
        ChromaDB search
            ↓
        Retrieved evidence
    """

    def __init__(
        self,
        persist_directory: str = "data/vectorstore",
        collection_name: str = "nyaaymitra_legal",
        embedding_model_name: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
    ):

        self.embedding_model = EmbeddingModel(
            model_name=embedding_model_name
        )

        self.store = ChromaStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most semantically relevant chunks.
        """

        if not query or not query.strip():
            return []

        query = query.strip()

        query_embedding = (
            self.embedding_model.encode_one(query)
        )

        results = self.store.search(
            query_embedding=query_embedding,
            n_results=top_k,
            where={
                "is_retrievable": True
            },
        )

        return self._format_results(results)

    def _format_results(
        self,
        results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert Chroma's nested response into a clean
        application-level structure.
        """

        if not results:
            return []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted = []

        for index in range(len(ids)):

            formatted.append({
                "rank": index + 1,
                "chunk_id": ids[index],
                "text": documents[index],
                "metadata": metadatas[index],
                "distance": distances[index],
            })

        return formatted