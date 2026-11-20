from __future__ import annotations

import numpy as np

from . import config


class HashingEmbedder:
    """Deterministic, offline bag-of-words embedder. Used for unit tests and quick local runs only;
    report evaluation results with the sentence-transformers backend."""

    def __init__(self, dim: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._vec = HashingVectorizer(
            n_features=dim, alternate_sign=False, norm="l2", ngram_range=(1, 2), stop_words="english"
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._vec.transform(texts).toarray().astype("float32")


class SBertEmbedder:
    def __init__(self, model_name: str = config.SBERT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. Run: pip install -r requirements-embeddings.txt "
                "(or set EMBED_BACKEND=hashing for an offline demo)."
            ) from e
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")


def get_embedder(backend: str | None = None):
    backend = backend or config.EMBED_BACKEND
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sbert":
        return SBertEmbedder()
    raise ValueError(f"unknown embedding backend: {backend}")
