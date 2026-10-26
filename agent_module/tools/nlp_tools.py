"""Agent tools that wrap the NLP module (``nlp_module.analyzer``)."""

from typing import Any, Dict, List

from langchain_core.tools import tool

from agent_module.utils.logger import setup_logger
from nlp_module import analyzer as nlp_module

logger = setup_logger("ege_ai_agent.tools")


@tool
def nlp_analyzer(essay_text: str) -> Dict[str, Any]:
    """Полный анализ сочинения: структура, семантика, орфография, пунктуация, грамматика, баллы ФИПИ (K1–K10) и список ошибок."""
    logger.info("Tool: nlp_analyzer (text_len=%d)", len(essay_text))
    res = nlp_module.full_analysis(essay_text)
    logger.debug("Tool: nlp_analyzer completed (score=%s)", res.get("score", {}).get("total_score"))
    return res


@tool
def nlp_structure_tool(essay_text: str) -> Dict[str, Any]:
    """Структура сочинения: число слов, предложений, абзацев и роли абзацев."""
    logger.info("Tool: nlp_structure_tool (text_len=%d)", len(essay_text))
    res = nlp_module.analyze_structure(essay_text)
    logger.debug("Tool: nlp_structure_tool result: %s", res)
    return res


@tool
def nlp_semantics_tool(essay_text: str) -> Dict[str, Any]:
    """Смысловой разбор сочинения: введение, примеры-иллюстрации, связь между ними, собственный аргумент, заключение."""
    logger.info("Tool: nlp_semantics_tool (text_len=%d)", len(essay_text))
    res = nlp_module.analyze_semantics(essay_text)
    logger.debug("Tool: nlp_semantics_tool completed")
    return res


@tool
def nlp_orthography_tool(essay_text: str) -> Dict[str, Any]:
    """Орфографический анализ сочинения (критерий К7)."""
    logger.info("Tool: nlp_orthography_tool (text_len=%d)", len(essay_text))
    res = nlp_module.analyze_orthography(essay_text)
    logger.debug("Tool: nlp_orthography_tool found %d errors", len(res.get("errors", [])))
    return res


@tool
def nlp_punctuation_tool(essay_text: str) -> Dict[str, Any]:
    """Пунктуационный анализ сочинения (критерий К8)."""
    logger.info("Tool: nlp_punctuation_tool (text_len=%d)", len(essay_text))
    res = nlp_module.analyze_punctuation(essay_text)
    logger.debug("Tool: nlp_punctuation_tool status=%s", res.get("status"))
    return res


@tool
def nlp_grammar_tool(essay_text: str) -> Dict[str, Any]:
    """Грамматический анализ сочинения (критерий К9)."""
    logger.info("Tool: nlp_grammar_tool (text_len=%d)", len(essay_text))
    res = nlp_module.analyze_grammar(essay_text)
    logger.debug("Tool: nlp_grammar_tool completed")
    return res


@tool
def nlp_scoring_tool(essay_text: str) -> Dict[str, Any]:
    """Подсчёт первичных баллов по критериям ФИПИ К1–К10 (максимум 22)."""
    logger.info("Tool: nlp_scoring_tool (text_len=%d)", len(essay_text))
    res = nlp_module.score_fipi(essay_text)
    logger.debug("Tool: nlp_scoring_tool score=%s/22", res.get("total_score"))
    return res


@tool
def nlp_errors_tool(essay_text: str) -> List[Dict[str, Any]]:
    """Список найденных ошибок сочинения (орфографические, пунктуационные, грамматические, речевые, фактические, логические) с исправлениями."""
    logger.info("Tool: nlp_errors_tool (text_len=%d)", len(essay_text))
    res = nlp_module.extract_errors(essay_text)
    logger.debug("Tool: nlp_errors_tool extracted %d errors", len(res))
    return res
