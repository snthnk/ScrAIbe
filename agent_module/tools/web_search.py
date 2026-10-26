"""``web_fact_checker``: fact verification through the Tavily search API.

The tool only fetches evidence (title / link / snippet). Interpreting the evidence is left to the
agent — there are no canned answers for particular works or authors here.
"""

import json
import os
import re
from typing import Any, Dict, List
from urllib import request

from langchain_core.tools import tool

from agent_module.models import FactCheckResult, FactEvidence
from agent_module.utils.logger import setup_logger

logger = setup_logger("ege_ai_agent.tools")

TAVILY_URL = "https://api.tavily.com/search"
_AUTHOR_INTENT = ("автор", "написал", "писатель", "кто создал")
_GENRE_INTENT = ("жанр", "рассказ или повесть", "это рассказ", "это повесть", "это роман", "это поэма")


def _tokenize(text: str) -> List[str]:
    normalized = re.sub(r"([0-9A-Za-zА-Яа-яЁё])[-–—]([0-9A-Za-zА-Яа-яЁё])", r"\1\2", text.lower())
    return [t for t in re.split(r"[^0-9A-Za-zА-Яа-яЁё]+", normalized) if len(t) >= 2]


def _tavily_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        logger.warning("Tool: web_fact_checker - TAVILY_API_KEY не задан")
        return {"error": "TAVILY_API_KEY не задан"}
    logger.info("Tool: web_fact_checker sending request to Tavily (query='%s')", query)
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
    }
    req = request.Request(
        TAVILY_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=15) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
        logger.debug("Tool: web_fact_checker received %d results from Tavily", len(parsed.get("results", [])))
        return parsed
    except Exception as exc:
        logger.error("Tool: web_fact_checker Tavily request failed: %s", exc)
        return {"error": str(exc)}


def _extract_title(text: str) -> str | None:
    """First quoted work title («…» or "…") found in the text, if any."""
    matches = re.findall(r"[«\"]([^»\"]{2,80})[»\"]", text)
    return matches[0] if matches else None


def _normalize_query(query: str) -> str:
    """Short queries pass through. A long one (a whole pasted message) is reduced to «title» + intent."""
    raw = query.strip()
    if not raw:
        return raw
    if len(raw) > 200 or "\n" in raw:
        title = _extract_title(raw)
        if title:
            lowered = raw.lower()
            if any(marker in lowered for marker in _GENRE_INTENT):
                intent = "жанр"
            else:
                intent = "автор"
            return f"\"{title}\" {intent}"
        return raw.splitlines()[0][:200] if "\n" in raw else raw[:200]
    return raw


def _must_have_tokens(query: str) -> set:
    """Tokens at least one of which a result must contain (only for «who wrote …» questions)."""
    lowered = query.lower()
    if any(marker in lowered for marker in _AUTHOR_INTENT):
        return {"автор", "написал", "писатель"}
    return set()


def _to_evidence(item: Dict[str, Any]) -> FactEvidence:
    return FactEvidence(title=item.get("title"), href=item.get("url"), snippet=item.get("content"))


@tool
def web_fact_checker(query: str) -> Dict[str, Any]:
    """Проверка фактов (автор, жанр, год, содержание произведения) через веб-поиск. Принимает краткий запрос: название + автор/жанр. Возвращает ссылки и фрагменты-доказательства."""
    normalized_query = _normalize_query(query)
    keyword_set = set(_tokenize(normalized_query))
    must_have = _must_have_tokens(normalized_query)

    search = _tavily_search(normalized_query, max_results=5)
    if search.get("error"):
        return FactCheckResult(query=normalized_query, status="error", error=str(search["error"])).model_dump()

    results = search.get("results") if isinstance(search.get("results"), list) else []
    evidence: List[FactEvidence] = []
    for item in results:
        blob = f"{item.get('title', '')} {item.get('content', '')} {item.get('url', '')}".lower()
        blob_tokens = set(_tokenize(blob))
        if must_have and not (blob_tokens & must_have):
            continue
        if keyword_set and len(blob_tokens & keyword_set) < max(1, min(2, len(keyword_set) // 6)):
            continue
        evidence.append(_to_evidence(item))
    if not evidence and results:
        evidence = [_to_evidence(item) for item in results[:3]]

    status = "found" if evidence else "not_found"
    return FactCheckResult(query=normalized_query, status=status, evidence=evidence).model_dump()
