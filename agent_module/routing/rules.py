"""Keyword rules for tool selection — the fallback when the LLM is unavailable, and the eval baseline.

Priority (first match wins): criterion question -> error check -> fact question -> argument request
-> generic "проверь" with an essay present. Only the *student's last message* is looked at; the plan
step never forces a tool (the old ``_force_tool_call`` fired ``literature_rag`` for any message such as
«Спасибо!» whenever the plan step mentioned a literary example).
"""

import re
from typing import Any, Dict, List

from agent_module.constants import (
    EXPERT_REASONING,
    LITERATURE_RAG,
    NLP_ERRORS,
    WEB_FACT_CHECKER,
)
from agent_module.messages import last_user_text
from agent_module.routing.args import build_expert_payload, criterion_from_text, essay_for_nlp
from agent_module.routing.decision import DECIDED_BY_RULES, ToolDecision

_CRITERION_RE = re.compile(r"\b[кk]\s*\d+\b")

ERROR_KEYWORDS = ("ошибк", "орфограф", "пунктуац", "грамматик", "речевые", "исправ", "опечатк")
# A request that mixes error words with these is probably about literature, not about the essay's errors.
LITERATURE_KEYWORDS = ("литератур", "подбери", "найди аргумент", "приведи пример из литературы", "книга", "автор")
FACT_KEYWORDS = ("кто написал", "кто автор", "кто такой автор", "это рассказ", "это повесть", "жанр", "писатель")
ARGUMENT_VERBS = ("подбери", "найди", "подскажи", "предложи")
ARGUMENT_NOUNS = ("аргумент", "пример", "литературный")
ARGUMENT_CONTEXT = ("литератур", "по теме", "к теме", "сочинени", "приведи")


def _contains(text: str, words) -> bool:
    return any(w in text for w in words)


def _is_criterion_question(text: str) -> bool:
    return "критер" in text or bool(_CRITERION_RE.search(text))


def _is_error_request(text: str) -> bool:
    if "ошибк" in text or "фактич" in text:
        return True
    return _contains(text, ERROR_KEYWORDS) and not _contains(text, LITERATURE_KEYWORDS)


def _is_fact_question(text: str) -> bool:
    return _contains(text, FACT_KEYWORDS)


def _is_argument_request(text: str) -> bool:
    asks = _contains(text, ARGUMENT_VERBS) and _contains(text, ARGUMENT_NOUNS)
    asks_in_context = _contains(text, ("аргумент", "пример")) and _contains(text, ARGUMENT_CONTEXT)
    return asks or asks_in_context


def _none(reason: str) -> ToolDecision:
    return ToolDecision(tool=None, decided_by=DECIDED_BY_RULES, reason=reason)


def select_tool_by_rules(messages: List[Any], state: Dict[str, Any], step_text: str = "") -> ToolDecision:
    raw_text = last_user_text(messages)
    if not raw_text.strip():
        return _none("no user message")
    text = raw_text.lower()

    if _is_criterion_question(text):
        criterion = criterion_from_text(text)
        payload = build_expert_payload(state, raw_text, step_text, criterion)
        return ToolDecision(
            tool=EXPERT_REASONING, args={"payload": payload}, decided_by=DECIDED_BY_RULES, reason="rule:criterion_question"
        )

    if _is_error_request(text):
        essay = essay_for_nlp(state, messages)
        if not essay:
            return _none("rule:error_request but no essay text is available")
        return ToolDecision(tool=NLP_ERRORS, args={"essay_text": essay}, decided_by=DECIDED_BY_RULES, reason="rule:error_request")

    if _is_fact_question(text):
        return ToolDecision(tool=WEB_FACT_CHECKER, args={"query": raw_text}, decided_by=DECIDED_BY_RULES, reason="rule:fact_question")

    if _is_argument_request(text):
        return ToolDecision(tool=LITERATURE_RAG, args={"query": raw_text}, decided_by=DECIDED_BY_RULES, reason="rule:argument_request")

    if "провер" in text:
        essay = essay_for_nlp(state, messages)
        if essay:
            return ToolDecision(tool=NLP_ERRORS, args={"essay_text": essay}, decided_by=DECIDED_BY_RULES, reason="rule:check_essay")

    return _none("no rule matched")
