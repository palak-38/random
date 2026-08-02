from __future__ import annotations

import numpy as np

from config import EMBEDDING_MODEL

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Returns L2-normalised embeddings so cosine similarity is a plain dot product."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)


class SimilarityIndex:
    """Brute-force cosine similarity. At ~400 historical messages an ANN index
    would add dependencies without measurable benefit."""

    def __init__(self, ids: list[str], vectors: np.ndarray):
        self.ids = ids
        self.vectors = vectors

    @classmethod
    def build(cls, items: list[tuple[str, str]]) -> "SimilarityIndex":
        ids = [i for i, _ in items]
        vectors = embed([t for _, t in items])
        return cls(ids, vectors)

    def search(self, query: str, allowed_ids: set[str] | None = None, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.ids or not query.strip():
            return []
        qvec = embed([query])[0]
        scores = self.vectors @ qvec
        order = np.argsort(-scores)
        results: list[tuple[str, float]] = []
        for idx in order:
            mid = self.ids[idx]
            if allowed_ids is not None and mid not in allowed_ids:
                continue
            results.append((mid, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results
