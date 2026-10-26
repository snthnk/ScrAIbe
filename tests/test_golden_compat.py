"""Рефакторинг не должен менять разбор и форматирование: сравниваем с «слепком» поведения до рефакторинга.

Намеренные отличия (в слепке они остались от старой версии):
    * ``nlp_analyzer_mock`` переименован в ``nlp_analyzer``;
    * формат ответа literature_rag/web_fact_checker поменялся (настоящая база, нет заготовки под «Муму») — см. test_formatting.py.
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_module.graph.formatting import format_full_analysis, format_nlp_errors, generate_k2_no_source_fallback
from agent_module.graph.nodes import normalize_plan, should_run_global_planner
from agent_module.graph.workflow import route_from_local_reasoning
from agent_module.messages import parse_raw_tool_calls, sanitize_message_history, tool_calls_from_text
from tests.fixtures import ANALYSIS_OK, ERR_SETS, ESSAY, STATES

GOLDEN = json.loads((Path(__file__).parent / "golden" / "legacy_behavior.json").read_text(encoding="utf-8"))


def test_format_nlp_errors_unchanged():
    expected = GOLDEN["format_nlp_errors"]
    for name, payload in ERR_SETS.items():
        assert format_nlp_errors(payload) == expected[name], name
    assert format_nlp_errors(json.dumps(ERR_SETS["mixed"], ensure_ascii=False)) == expected["json_string"]
    assert format_nlp_errors("Что-то пошло не так") == expected["plain_string"]


def test_format_full_analysis_unchanged():
    expected = GOLDEN["format_full_analysis"]
    refused = json.loads(json.dumps(ANALYSIS_OK))
    refused["structure"]["words"] = 50
    assert format_full_analysis(ANALYSIS_OK) == expected["ok_dict"]
    assert format_full_analysis(json.dumps(ANALYSIS_OK, ensure_ascii=False)) == expected["ok_json_string"]
    assert format_full_analysis(refused) == expected["refused"]
    assert format_full_analysis(None) == expected["none"]
    assert format_full_analysis([1]) == expected["list"]
    assert format_full_analysis("просто текст") == expected["plain_str"]
    assert format_full_analysis("{not json") == expected["bad_json"]
    assert format_full_analysis({}) == expected["empty_dict"]


def test_k2_no_source_fallback_unchanged():
    assert generate_k2_no_source_fallback(ESSAY) == GOLDEN["k2_no_source_fallback"]


@pytest.mark.parametrize("case", GOLDEN["tool_calls_from_text"])
def test_tool_calls_from_text_unchanged(case):
    result = tool_calls_from_text(case["text"].replace("nlp_analyzer_mock", "nlp_analyzer"))
    expected = json.loads(json.dumps(case["value"]).replace("nlp_analyzer_mock", "nlp_analyzer"))
    assert result == expected


@pytest.mark.parametrize("case", GOLDEN["parse_raw_tool_calls"])
def test_parse_raw_tool_calls_unchanged(case):
    assert parse_raw_tool_calls(case["raw"]) == case["value"]


@pytest.mark.parametrize("case", GOLDEN["normalize_plan"])
def test_normalize_plan_unchanged(case):
    assert normalize_plan(case["text"]) == case["value"]


def test_sanitize_message_history_unchanged():
    msgs = [
        HumanMessage(content="q1"),
        AIMessage(content="", tool_calls=[{"id": "t1", "name": "web_fact_checker", "args": {"query": "x"}}]),
        ToolMessage(content="res", tool_call_id="t1"),
        AIMessage(content="answer"),
        AIMessage(content="", tool_calls=[{"id": "t2", "name": "literature_rag", "args": {"query": "x"}}]),
        ToolMessage(content="orphan", tool_call_id="zzz"),
        HumanMessage(content="q2"),
        {"role": "assistant", "content": "dict-ai"},
        {"type": "human", "content": "dict-h"},
    ]
    got = [{"cls": type(m).__name__, "content": m.content} for m in sanitize_message_history(msgs)]
    assert got == GOLDEN["sanitize_message_history"]


def test_should_run_global_planner_unchanged():
    states = {**STATES, "planned": {"essay_text": ESSAY, "global_plan": ["x"]}}
    for case in GOLDEN["should_run_global_planner"]:
        assert should_run_global_planner(case["text"], dict(states[case["state"]])) is case["value"], case


def test_route_from_local_reasoning_unchanged():
    expected = GOLDEN["route_from_local_reasoning"]
    call = AIMessage(content="", tool_calls=[{"id": "1", "name": "literature_rag", "args": {"query": "x"}}])
    assert route_from_local_reasoning({"messages": []}) == expected["empty"]
    assert route_from_local_reasoning({"messages": [HumanMessage(content="q"), ToolMessage(content="r", tool_call_id="1"), AIMessage(content="a")]}) == expected["after_tool"]
    assert route_from_local_reasoning({"messages": [HumanMessage(content="q"), call]}) == expected["tool_call"]
    assert route_from_local_reasoning({"messages": [HumanMessage(content="q"), AIMessage(content="ответ")]}) == expected["plain"]
