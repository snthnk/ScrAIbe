import json
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent_module.constants import LITERATURE_RAG, NLP_ERRORS
from agent_module.routing import select_tool
from agent_module.routing.selector import InvalidToolChoice, decide_with_llm, tool_selection_mode
from tests.fakes import FakeLLM, tool_call_message
from tests.fixtures import ESSAY, STATES

PROMPT = "system"


def history(text):
    return [HumanMessage(content=text)]


def test_llm_choice_is_used_and_marked_as_llm():
    llm = FakeLLM([tool_call_message(LITERATURE_RAG, {"query": "про дружбу"})])
    decision = select_tool(llm, history("нужен пример про дружбу"), dict(STATES["essay"]), "шаг", PROMPT, mode="llm")
    assert decision.tool == LITERATURE_RAG
    assert decision.decided_by == "llm"
    assert decision.fallback_reason is None
    assert decision.args["query"]


def test_llm_picked_errors_tool_gets_the_essay_from_state_not_from_the_model():
    llm = FakeLLM([tool_call_message(NLP_ERRORS, {"essay_text": "обрезано моделью"})])
    decision = select_tool(llm, history("проверь"), dict(STATES["essay"]), "шаг", PROMPT, mode="llm")
    assert decision.tool == NLP_ERRORS
    assert decision.args["essay_text"] == ESSAY


def test_llm_may_answer_without_a_tool_even_if_a_rule_would_fire():
    answer = AIMessage(content="Подумай, что именно ты хочешь проверить.")
    llm = FakeLLM([answer])
    decision = select_tool(llm, history("Какие у меня ошибки?"), dict(STATES["essay"]), "шаг", PROMPT, mode="llm")
    assert decision.tool is None
    assert decision.decided_by == "llm"
    assert decision.llm_message is answer


def test_provider_failure_falls_back_to_rules_and_says_so():
    llm = FakeLLM([RuntimeError("429 quota")])
    decision = select_tool(llm, history("Какие у меня ошибки?"), dict(STATES["essay"]), "шаг", PROMPT, mode="llm")
    assert decision.tool == NLP_ERRORS
    assert decision.decided_by == "rules"
    assert "RuntimeError" in decision.fallback_reason


def test_unknown_tool_from_llm_falls_back_to_rules():
    llm = FakeLLM([tool_call_message("made_up_tool", {})])
    decision = select_tool(llm, history("Какие у меня ошибки?"), dict(STATES["essay"]), "шаг", PROMPT, mode="llm")
    assert decision.decided_by == "rules"
    assert decision.tool == NLP_ERRORS
    assert "made_up_tool" in decision.fallback_reason


def test_failure_without_a_matching_rule_is_reraised_for_the_retry_logic():
    llm = FakeLLM([RuntimeError("boom")])
    with pytest.raises(RuntimeError, match="boom"):
        select_tool(llm, history("Привет"), dict(STATES["essay"]), "шаг", PROMPT, mode="llm")


def test_allow_fallback_false_exposes_llm_errors_for_benchmarks():
    llm = FakeLLM([tool_call_message("made_up_tool", {})])
    with pytest.raises(InvalidToolChoice):
        decide_with_llm(llm, history("x"), dict(STATES["essay"]), "шаг", PROMPT)
    with pytest.raises(RuntimeError):
        select_tool(FakeLLM([RuntimeError("x")]), history("Какие у меня ошибки?"), dict(STATES["essay"]), "", PROMPT, mode="llm", allow_fallback=False)


def test_rules_mode_does_not_call_the_llm():
    llm = FakeLLM([RuntimeError("must not be called")])
    decision = select_tool(llm, history("Какие у меня ошибки?"), dict(STATES["essay"]), "шаг", PROMPT, mode="rules")
    assert llm.calls == 0
    assert decision.decided_by == "rules"
    assert decision.tool == NLP_ERRORS


def test_mode_comes_from_environment(monkeypatch):
    monkeypatch.delenv("TOOL_SELECTION_MODE", raising=False)
    assert tool_selection_mode() == "llm"
    monkeypatch.setenv("TOOL_SELECTION_MODE", "RULES")
    assert tool_selection_mode() == "rules"
    monkeypatch.setenv("TOOL_SELECTION_MODE", "что-то")
    assert tool_selection_mode() == "llm"


def test_every_decision_is_logged_with_who_made_it(caplog):
    logger = logging.getLogger("ege_ai_agent.routing")
    logger.propagate = True
    with caplog.at_level(logging.INFO, logger="ege_ai_agent.routing"):
        select_tool(FakeLLM([tool_call_message(LITERATURE_RAG, {"query": "x"})]), history("пример"), dict(STATES["essay"]), "", PROMPT, mode="llm")
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("tool_decision "))
    payload = json.loads(line.removeprefix("tool_decision "))
    assert payload["decided_by"] == "llm"
    assert payload["tool"] == LITERATURE_RAG
    assert "essay_text" not in json.dumps(payload, ensure_ascii=False)  # текст сочинения в логи не попадает
