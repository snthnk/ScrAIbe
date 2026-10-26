"""Граф целиком, без сети: LLM подменён, RAG работает на офлайн-эмбеддере."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent_module.graph import build_workflow
from agent_module.rag import HybridRetriever, load_arguments
from agent_module.rag.embedder import HashingEmbedder
from agent_module.tools import rag as rag_tool
from tests.fakes import FakeLLM, tool_call_message


@pytest.fixture(autouse=True)
def offline_rag(monkeypatch):
    retriever = HybridRetriever.build(load_arguments(), HashingEmbedder())
    monkeypatch.setattr(rag_tool, "get_retriever", lambda: retriever)


def run(llm, text, mode, monkeypatch):
    monkeypatch.setenv("TOOL_SELECTION_MODE", mode)
    state = {
        "messages": [HumanMessage(content=text)],
        "global_plan": ["Подобрать литературный аргумент через literature_rag"],
        "current_step": 0,
        "task_condition": "Какова роль природы в жизни человека?",
        "essay_text": "",
        "source_text": "",
        "show_reasoning": True,
    }
    return build_workflow(llm).invoke(state)


def test_llm_chooses_rag_and_answer_comes_from_the_knowledge_base(monkeypatch):
    llm = FakeLLM([tool_call_message("literature_rag", {"query": "аргумент про дружбу"})])
    result = run(llm, "Мне нужен пример про дружбу", "llm", monkeypatch)
    answer = result["messages"][-1].content
    assert isinstance(result["messages"][-1], AIMessage)
    assert "Дружб" in answer or "дружб" in answer
    decision = result["tool_decisions"][0]
    assert decision["decided_by"] == "llm" and decision["tool"] == "literature_rag"


def test_rules_mode_makes_the_same_kind_of_call_without_llm(monkeypatch):
    llm = FakeLLM([RuntimeError("LLM must not be called")])
    result = run(llm, "подбери аргумент по теме дружба", "rules", monkeypatch)
    assert llm.calls == 0
    assert result["tool_decisions"][0]["decided_by"] == "rules"
    assert result["tool_decisions"][0]["tool"] == "literature_rag"


def test_thanks_gets_a_direct_llm_answer_and_no_tool_call(monkeypatch):
    llm = FakeLLM([AIMessage(content="Пожалуйста! Что разберём дальше?")])
    result = run(llm, "Спасибо!", "llm", monkeypatch)
    assert result["messages"][-1].content == "Пожалуйста! Что разберём дальше?"
    assert result["tool_decisions"][0]["tool"] is None
