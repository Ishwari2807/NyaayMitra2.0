class EmbeddingModel:
    """
    Wrapper around the embedding model.

    Keeping this behind a class means we can change the model
    later without changing the vector-store or retrieval code.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_name = model_name

        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded.")

    def encode(self, texts):
        """
        Convert a list of texts into embedding vectors.
        """

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def encode_one(self, text):
        """
        Encode one text.
        """

        return self.encode([text])[0]