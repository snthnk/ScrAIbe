"""Эмбеддеры для RAG.

* ``SentenceTransformerEmbedder`` — боевой: ``intfloat/multilingual-e5-large`` (префиксы «query: »/«passage: »).
* ``HashingEmbedder`` — офлайн: хеширование символьных n-грамм в вектор фиксированной длины. Не требует
  скачивания модели и нужен для тестов и проверки конвейера; качество семантического поиска у него ниже.
"""

import hashlib
import os
from typing import List, Optional, Protocol, Sequence

import numpy as np

DEFAULT_E5_MODEL = "intfloat/multilingual-e5-large"


class Embedder(Protocol):
    name: str
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


class HashingEmbedder:
    """Детерминированный «мешок» символьных 3–5-грамм слов, свёрнутый хешем в ``dim`` координат."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    @staticmethod
    def _grams(text: str) -> List[str]:
        grams: List[str] = []
        for word in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split():
            padded = f"^{word}$"
            for n in (3, 4, 5):
                grams.extend(padded[i : i + n] for i in range(max(len(padded) - n + 1, 0)))
        return grams

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for gram in self._grams(text):
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, idx] += sign
        return _normalize(out)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_E5_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Для e5-эмбеддингов нужен пакет sentence-transformers (pip install sentence-transformers)") from exc
        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def _encode(self, prefix: str, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode([prefix + t for t in texts], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode("passage: ", texts)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode("query: ", texts)


def build_embedder(kind: Optional[str] = None) -> Embedder:
    """``kind``: ``e5`` (по умолчанию) или ``hash``. Берётся из RAG_EMBEDDER; модель — из RAG_EMBEDDER_MODEL."""
    kind = (kind or os.getenv("RAG_EMBEDDER") or "e5").strip().lower()
    if kind in {"hash", "hashing", "offline"}:
        return HashingEmbedder()
    if kind in {"e5", "sentence-transformers", "st"}:
        return SentenceTransformerEmbedder(os.getenv("RAG_EMBEDDER_MODEL") or DEFAULT_E5_MODEL)
    raise ValueError(f"Неизвестный RAG_EMBEDDER: {kind!r} (допустимо: e5, hash)")
