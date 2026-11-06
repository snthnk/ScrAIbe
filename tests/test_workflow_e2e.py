"""Граф целиком, без сети: LLM подменён, RAG работает на офлайн-эмбеддере."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent_module.graph import build_workflow
from agent_module.messages import is_human, message_content
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


def test_recheck_uses_edited_essay_instead_of_cached_analysis(monkeypatch):
    monkeypatch.setenv("TOOL_SELECTION_MODE", "llm")
    old_essay = "Это предыдущая версия."
    new_essay = "Здесь теперь ровно пять слов."
    monkeypatch.setattr(
        "agent_module.graph.nodes.nlp_analyzer",
        SimpleNamespace(invoke=lambda text: {"structure": {"words": 999}}),
    )
    llm = FakeLLM([
        AIMessage(content="Проверить структуру\nПроверить аргументы"),
        tool_call_message("nlp_structure_tool", {"essay_text": old_essay}),
    ])
    workflow = build_workflow(llm)
    first = workflow.invoke({
        "essay_text": old_essay,
        "messages": [HumanMessage(content=f"Сочинение: {old_essay}"), HumanMessage(content="Давай начнём")],
    })
    assert "Количество слов: 999" in first["messages"][-1].content

    result = workflow.invoke({
        **first,
        "essay_text": new_essay,
        "messages": [*first["messages"], HumanMessage(content="Проверь структуру после правки")],
    })
    assert "Количество слов: 5" in result["messages"][-1].content
    assert any(new_essay in message_content(message) for message in result["messages"] if is_human(message))
    assert all(old_essay not in message_content(message) for message in result["messages"])
    assert result["current_step"] == first["current_step"]


def test_plan_advances_after_a_complete_answer_in_natural_language(monkeypatch):
    monkeypatch.setenv("TOOL_SELECTION_MODE", "llm")
    workflow = build_workflow(FakeLLM([
        AIMessage(content="Как именно примеры дополняют друг друга?"),
        AIMessage(content="Пожалуйста!"),
        AIMessage(content="Уточни, какую сторону проблемы раскрывает каждый пример."),
        AIMessage(content='{"completed": false}'),
        AIMessage(content="Да, ты объяснил вклад обоих примеров."),
        AIMessage(content='{"completed": true, "next_question": "Какой аргумент поддержит твою позицию?"}'),
    ]))
    state = workflow.invoke({
        "global_plan": ["Объяснить связь примеров", "Обсудить аргумент ученика"],
        "current_step": 0,
        "messages": [HumanMessage(content="Не понимаю, как объяснить связь.")],
    })
    state = workflow.invoke({
        **state,
        "messages": [*state["messages"], HumanMessage(content="Спасибо!")],
    })
    assert state["current_step"] == 0

    state = workflow.invoke({
        **state,
        "messages": [*state["messages"], HumanMessage(content="Они дополняют друг друга.")],
    })
    assert state["current_step"] == 0

    result = workflow.invoke({
        **state,
        "messages": [*state["messages"], HumanMessage(content="Первый пример показывает причину, а второй — последствия этого решения.")],
    })
    assert result["current_step"] == 1
    assert "Какой аргумент поддержит твою позицию?" in message_content(result["messages"][-1])


def test_plan_advances_after_rechecking_a_revised_essay(monkeypatch):
    monkeypatch.setenv("TOOL_SELECTION_MODE", "llm")
    workflow = build_workflow(FakeLLM([
        AIMessage(content="Вынеси второй пример в отдельный абзац."),
        tool_call_message("nlp_structure_tool", {}),
        AIMessage(content='{"completed": true, "next_question": "Что связывает эти примеры?"}'),
    ]))
    old_essay = "Первый пример. Второй пример."
    state = workflow.invoke({
        "essay_text": old_essay,
        "global_plan": ["Разделить примеры по абзацам", "Объяснить связь примеров"],
        "messages": [HumanMessage(content="Здравствуйте!")],
    })
    assert state["current_step"] == 0

    result = workflow.invoke({
        **state,
        "essay_text": "Первый пример.\nВторой пример.",
        "messages": [*state["messages"], HumanMessage(content="Посмотри, я вынес второй пример отдельно.")],
    })
    answer = message_content(result["messages"][-1])
    assert "Количество абзацев: 2" in answer
    assert "Что связывает эти примеры?" in answer
    assert result["current_step"] == 1
