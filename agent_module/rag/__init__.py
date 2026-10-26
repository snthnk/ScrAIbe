"""Литературный RAG: база аргументов (CSV) -> эмбеддинги -> FAISS -> гибридный поиск (вектор + теги)."""

from agent_module.rag.base import Argument, load_arguments
from agent_module.rag.retriever import HybridRetriever, SearchHit, get_retriever

__all__ = ["Argument", "HybridRetriever", "SearchHit", "get_retriever", "load_arguments"]
