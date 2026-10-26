from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from agent_module.models import RagItem, RagResult
from agent_module.rag import get_retriever
from agent_module.rag.retriever import MODE_HYBRID
from agent_module.utils.logger import setup_logger

logger = setup_logger("ege_ai_agent.tools")

TOP_K = 3


@tool
def literature_rag(query: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """Поиск литературных аргументов для сочинения ЕГЭ по локальной базе (эмбеддинги + FAISS + фильтр по темам).

    query — тема или проблема своими словами, например «роль книги в жизни человека».
    tags — необязательный список тем из словаря (например ["война", "милосердие"]); если не задан,
    темы определяются по тексту запроса.
    """
    logger.info("Tool: literature_rag (query='%s', tags=%s)", query, tags)
    try:
        outcome = get_retriever().search(query, k=TOP_K, mode=MODE_HYBRID, tags=tags)
    except Exception as exc:  # нет модели эмбеддингов, сломан индекс и т.п.
        logger.warning("Tool: literature_rag unavailable: %s", exc)
        result = RagResult(status="unavailable", query=query, message=f"База аргументов недоступна: {exc}")
        return result.model_dump()

    retrieval = {"mode": outcome.mode, "query_tags": outcome.query_tags, "tag_filter_applied": outcome.tag_filter_applied}
    if not outcome.hits:
        result = RagResult(status="empty", query=query, message="Подходящих аргументов не найдено.", retrieval=retrieval)
    else:
        items = [
            RagItem(
                id=hit.argument.id,
                title=hit.argument.title,
                author=hit.argument.author,
                argument=hit.argument.argument,
                tags=hit.argument.tags,
                score=round(hit.score, 4),
            )
            for hit in outcome.hits
        ]
        result = RagResult(status="ok", query=query, results=items, retrieval=retrieval)
    logger.debug("Tool: literature_rag returned status=%s, hits=%d", result.status, len(result.results))
    return result.model_dump()
