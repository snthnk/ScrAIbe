"""Гибридный поиск: ближайшие по эмбеддингу аргументы (FAISS) + фильтр по пересечению тегов (>= 1).

Режимы ``mode``:
    vector  — только косинусная близость (базовая линия для оценки);
    hybrid  — из кандидатов по вектору остаются те, у кого есть хотя бы один общий тег с запросом;
              если таких нет (или у запроса нет тегов), результат совпадает с ``vector``; если их меньше k,
              выдача добирается лучшими по вектору.
"""

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union


from agent_module.rag.base import DATA_DIR, DEFAULT_ARGUMENTS_PATH, Argument, load_arguments
from agent_module.rag.embedder import Embedder, build_embedder
from agent_module.rag.tags import extract_tags, normalize_tags
from agent_module.utils import setup_logger

logger = setup_logger("ege_ai_agent.rag")

MODE_VECTOR = "vector"
MODE_HYBRID = "hybrid"
DEFAULT_INDEX_DIR = DATA_DIR / "rag_index"
CANDIDATES = 30


@dataclass
class SearchHit:
    argument: Argument
    score: float
    matched_tags: List[str] = field(default_factory=list)


@dataclass
class SearchOutcome:
    hits: List[SearchHit]
    query_tags: List[str]
    mode: str
    tag_filter_applied: bool


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HybridRetriever:
    def __init__(
        self,
        arguments: Sequence[Argument],
        embedder: Embedder,
        index: Any,
    ) -> None:
        self.arguments = list(arguments)
        self.embedder = embedder
        self.index = index

    # ---- построение / загрузка индекса ----
    @classmethod
    def build(
        cls,
        arguments: Sequence[Argument],
        embedder: Embedder,
    ) -> "HybridRetriever":
        import faiss

        vectors = embedder.embed_passages([a.passage for a in arguments])
        index = faiss.IndexFlatIP(vectors.shape[1])  # векторы нормированы -> скалярное произведение = косинус
        index.add(vectors)
        return cls(arguments, embedder, index)

    @classmethod
    def from_csv(
        cls,
        csv_path: Optional[Union[str, Path]] = None,
        embedder: Optional[Embedder] = None,
        index_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
    ) -> "HybridRetriever":
        import faiss

        path = Path(csv_path) if csv_path else DEFAULT_ARGUMENTS_PATH
        arguments = load_arguments(path)
        embedder = embedder or build_embedder()
        if not use_cache:
            return cls.build(arguments, embedder)

        cache_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
        meta_path, index_path = cache_dir / "meta.json", cache_dir / "faiss.index"
        meta = {"embedder": embedder.name, "dim": embedder.dim, "csv_sha256": _file_sha256(path), "size": len(arguments)}
        if meta_path.exists() and index_path.exists():
            try:
                if json.loads(meta_path.read_text(encoding="utf-8")) == meta:
                    logger.info("RAG: загружен индекс из %s", cache_dir)
                    return cls(arguments, embedder, faiss.read_index(str(index_path)))
            except Exception as exc:
                logger.warning("RAG: кэш индекса повреждён (%s), пересобираем", exc)
        retriever = cls.build(arguments, embedder)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(retriever.index, str(index_path))
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("RAG: индекс (%d аргументов, %s) сохранён в %s", len(arguments), embedder.name, cache_dir)
        except OSError as exc:
            logger.warning("RAG: не удалось сохранить индекс: %s", exc)
        return retriever

    # ---- поиск ----
    def search(
        self,
        query: str,
        k: int = 3,
        mode: str = MODE_HYBRID,
        tags: Optional[Sequence[str]] = None,
    ) -> SearchOutcome:
        if mode not in {MODE_VECTOR, MODE_HYBRID}:
            raise ValueError(f"mode должен быть {MODE_VECTOR!r} или {MODE_HYBRID!r}")
        query = (query or "").strip()
        if not query:
            return SearchOutcome([], [], mode, False)

        query_tags = normalize_tags(tags) if tags else extract_tags(query)
        vector = self.embedder.embed_queries([query])
        n_candidates = min(max(CANDIDATES, k), len(self.arguments))
        scores, ids = self.index.search(vector, n_candidates)
        candidates = [
            SearchHit(self.arguments[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0
        ]

        applied = False
        if mode == MODE_HYBRID and query_tags:
            wanted = set(query_tags)
            filtered = []
            for hit in candidates:
                common = [t for t in hit.argument.tags if t in wanted]
                if common:
                    hit.matched_tags = common
                    filtered.append(hit)
            if filtered:
                if len(filtered) < k:  # тегов совпало мало — добираем лучшими по вектору, чтобы не отдавать пустую выдачу
                    kept = {hit.argument.id for hit in filtered}
                    filtered += [hit for hit in candidates if hit.argument.id not in kept]
                candidates, applied = filtered, True
        return SearchOutcome(candidates[:k], query_tags, mode, applied)


_singleton: Optional[HybridRetriever] = None
_lock = threading.Lock()


def get_retriever() -> HybridRetriever:
    """Ленивый общий экземпляр для инструмента ``literature_rag``."""
    global _singleton
    with _lock:
        if _singleton is None:
            csv_path = os.getenv("RAG_ARGUMENTS_CSV") or None
            _singleton = HybridRetriever.from_csv(csv_path)
        return _singleton


def reset_retriever() -> None:
    global _singleton
    with _lock:
        _singleton = None
