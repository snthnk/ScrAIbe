"""LangGraph nodes: CheckContext, GlobalPlanner, LocalReasoning, ToolExecutor, ResponseGenerator."""

import json
import re
import uuid
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_module.constants import ESSAY_PREFIXES, NLP_TOOL_CACHE_SECTION
from agent_module.graph.formatting import auto_answer_from_tool, format_full_analysis, format_nlp_errors
from agent_module.messages import (
    coerce_messages,
    is_human,
    last_tool_call,
    message_content,
    sanitize_message_history,
    state_to_dict,
    tool_calls_from_message,
    with_raw_tool_calls,
)
from agent_module.prompts import (
    DIRECT_ANSWER_ADDENDUM,
    GLOBAL_PLANNER_SYSTEM_PROMPT,
    LOCAL_REASONING_SYSTEM_PROMPT,
)
from agent_module.routing import select_tool
from agent_module.tools import TOOLS_BY_NAME, nlp_analyzer
from agent_module.utils import setup_logger

logger = setup_logger()

_EXPLICIT_REQUEST_ROOTS = ("провер", "ошибк", "критер", "разбор", "оцен", "анализ")


def _append_reasoning(state: Dict[str, Any], entry: str) -> Dict[str, Any]:
    logs = list(state.get("reasoning_log", []))
    logs.append(entry)
    return {"reasoning_log": logs}


def _current_step_text(state: Dict[str, Any]) -> str:
    plan = state.get("global_plan", [])
    step_index = state.get("current_step", 0)
    return plan[step_index] if step_index < len(plan) else "Финальное уточнение"


def _split_essay_prefix(user_text: str) -> str | None:
    """If the message starts with 'Сочинение:' / 'Эссе:' return the essay text after the colon."""
    lowered = user_text.strip().lower()
    for prefix in ESSAY_PREFIXES:
        if lowered.startswith(prefix):
            return user_text.split(":", 1)[1].strip()
    return None


# ---------------------------------------------------------------------------
# CheckContext / GlobalPlanner
# ---------------------------------------------------------------------------


def normalize_plan(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if cleaned.startswith("["):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip().lstrip("-•").strip()
        if line:
            lines.append(line)
    return lines


def should_run_global_planner(user_text: str, state: Dict[str, Any]) -> bool:
    lowered = user_text.strip().lower()
    submitted_essay = _split_essay_prefix(user_text)

    if state.get("global_plan"):
        return submitted_essay is not None

    essay_text = submitted_essay if submitted_essay is not None else state.get("essay_text", "")
    if essay_text.strip():
        is_explicit = any(root in lowered for root in _EXPLICIT_REQUEST_ROOTS) or bool(re.search(r"\b[кk]\s*\d+\b", lowered))
        if is_explicit:
            return False
    return True


def check_context(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Node: CheckContext")
    state = state_to_dict(state)
    messages = state.get("messages", [])
    user_text = message_content(messages[-1]) if messages else ""

    submitted_essay = _split_essay_prefix(user_text)
    is_new = not state.get("global_plan") or submitted_essay is not None
    logger.debug(f"CheckContext: messages_count={len(messages)}, is_new={is_new}, has_source={bool(state.get('source_text'))}")
    if not is_new:
        return {}

    essay_text = submitted_essay if submitted_essay is not None else state.get("essay_text", "")
    if not essay_text:
        essay_text = user_text
    logger.debug(f"CheckContext: registered essay_text length={len(essay_text)} chars")
    return {"essay_text": essay_text, "current_step": 0, "nlp_analysis": {}}


def global_planner(state: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    logger.info("Node: GlobalPlanner")
    state = state_to_dict(state)
    essay_text = state.get("essay_text", "")
    logger.debug(f"GlobalPlanner: invoking nlp_analyzer on essay ({len(essay_text)} chars)")
    nlp_analysis = nlp_analyzer.invoke(essay_text)
    user_prompt = f"NLP-анализ:\n{json.dumps(nlp_analysis, ensure_ascii=False)}\nСформируй план."
    response = llm.invoke([SystemMessage(content=GLOBAL_PLANNER_SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
    plan = normalize_plan(message_content(response))
    if not plan:
        plan = [
            "Определить сильные и слабые стороны по критериям",
            "Сформировать рекомендации по улучшению содержания и структуры",
            "Подобрать аргументы и исправить ошибки",
        ]
    logger.info(f"GlobalPlanner: generated plan ({len(plan)} steps): {plan}")
    return {
        "nlp_analysis": nlp_analysis,
        "global_plan": plan,
        "current_step": 0,
        **_append_reasoning(state, f"Global plan: {plan}"),
    }


# ---------------------------------------------------------------------------
# LocalReasoning
# ---------------------------------------------------------------------------


def local_reasoning(state: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """Pick the next action. After a tool result: render it. Otherwise: select a tool (LLM first, rules fallback)."""
    logger.info("Node: LocalReasoning")
    state = state_to_dict(state)
    history = coerce_messages(state.get("messages", []))
    step_text = _current_step_text(state)
    last_message = history[-1] if history else None

    if isinstance(last_message, ToolMessage):
        last_call = last_tool_call(history)
        tool_name = str(last_call.get("name")) if last_call else None
        tool_args = last_call.get("args", {}) if last_call else {}
        answer = auto_answer_from_tool(tool_name, tool_args, message_content(last_message), state.get("essay_text", ""))
        logger.info(f"LocalReasoning rendered tool result of '{tool_name}'")
        reasoning = _append_reasoning(state, f"Local step: {step_text}; rendered result of {tool_name}")
        return {"messages": state.get("messages", []) + [AIMessage(content=answer)], **reasoning}

    system_prompt = f"{LOCAL_REASONING_SYSTEM_PROMPT} Текущий шаг плана: {step_text}."
    decision = select_tool(llm, history, state, step_text, system_prompt)

    if decision.tool:
        call_id = f"{decision.decided_by}_{decision.tool}_{uuid.uuid4().hex}"
        response = AIMessage(content="", tool_calls=[{"id": call_id, "name": decision.tool, "args": decision.args}])
    elif decision.llm_message is not None:
        response = decision.llm_message
    else:
        # Rules found nothing to call; let the LLM answer the student directly, without tools.
        direct_prompt = f"{system_prompt}\n{DIRECT_ANSWER_ADDENDUM}"
        response = llm.invoke([SystemMessage(content=direct_prompt)] + sanitize_message_history(history))

    decisions = list(state.get("tool_decisions", [])) + [decision.to_log_dict(_last_human_text(history))]
    reasoning = _append_reasoning(state, f"Local step: {step_text}")
    reasoning["reasoning_log"].append(decision.summary())
    return {"messages": state.get("messages", []) + [response], "tool_decisions": decisions, **reasoning}


def _last_human_text(history: List[Any]) -> str:
    for msg in reversed(history):
        if is_human(msg):
            return message_content(msg)
    return ""


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------


def _cached_nlp_section(name: str, state: Dict[str, Any]) -> Any:
    """Return the part of the already computed NLP analysis that this tool would recompute, if any."""
    section = NLP_TOOL_CACHE_SECTION.get(name)
    analysis = state.get("nlp_analysis")
    if not section or not analysis:
        return None
    cached = analysis if section == "full" else analysis.get(section)
    if cached is None or cached == {} or cached == []:
        return None
    return cached


def tool_executor(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Node: ToolExecutor")
    state = state_to_dict(state)
    messages = state.get("messages", [])
    if not messages:
        return {}
    last_message = messages[-1]
    tool_calls = tool_calls_from_message(last_message)
    if not tool_calls:
        return {}

    if isinstance(tool_calls, list) and (
        (isinstance(last_message, AIMessage) and not getattr(last_message, "tool_calls", None)) or isinstance(last_message, dict)
    ):
        # The calls were parsed out of raw provider kwargs / plain text: store them natively on the message.
        messages = messages[:-1] + [with_raw_tool_calls(last_message, tool_calls)]

    tool_messages = []
    called = []
    tool_outputs = []
    for call in tool_calls:
        name = call.get("name")
        args = call.get("args", {})
        call_id = call.get("id") or call.get("tool_call_id") or call.get("toolCallId") or f"call_{uuid.uuid4().hex}"

        cached = _cached_nlp_section(name, state) if name and name.startswith("nlp_") else None
        if cached is not None:
            called.append(name)
            logger.info(f"Using cached NLP analysis for tool {name}.")
            content = json.dumps(cached, ensure_ascii=False) if isinstance(cached, (dict, list)) else str(cached)
            if state.get("show_reasoning"):
                tool_outputs.append(f"Tool output ({name}) [cached]: {content}")
        else:
            tool = TOOLS_BY_NAME.get(name)
            if tool is None:
                content = f"Unknown tool: {name}"
            else:
                called.append(name)
                logger.info(f"Tool call: {name}")
                try:
                    result = tool.invoke(args)
                    logger.debug(f"Tool '{name}' execution succeeded")
                except Exception as exc:
                    logger.error(f"Tool '{name}' execution failed: {exc}", exc_info=True)
                    result = f"Tool error: {exc}"
                content = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
                if state.get("show_reasoning"):
                    tool_outputs.append(f"Tool output ({name}): {content}")
        tool_messages.append(ToolMessage(content=content, tool_call_id=str(call_id)))

    logs = list(state.get("reasoning_log", []))
    logs.append(f"Tool calls: {called}" if called else "Tool calls: none")
    logs.extend(tool_outputs)
    return {"messages": messages + tool_messages, "reasoning_log": logs}


# ---------------------------------------------------------------------------
# ResponseGenerator
# ---------------------------------------------------------------------------

_EMPTY_MARKERS = ("None", "null", "[]", "{}")
NO_ANSWER_MESSAGE = "Не получилось сформулировать ответ. Попробуй переформулировать вопрос или уточнить, что именно нужно проверить."


def _tool_message_this_turn(messages: List[Any]) -> Any:
    """Latest ToolMessage after the student's last message (results of earlier turns must not leak in)."""
    for msg in reversed(messages):
        coerced = coerce_messages([msg])
        item = coerced[0] if coerced else msg
        if isinstance(item, ToolMessage):
            return item
        if is_human(item):
            return None
    return None


def response_generator(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Node: ResponseGenerator")
    state = state_to_dict(state)
    current_step = state.get("current_step", 0)
    plan = state.get("global_plan", [])
    if plan and current_step < len(plan) - 1:
        current_step += 1
    logger.debug(f"ResponseGenerator: current_step={current_step}/{len(plan)}")

    messages = list(state.get("messages", []))
    updates: Dict[str, Any] = {"current_step": current_step}
    if not messages:
        return updates

    last = coerce_messages([messages[-1]])
    last_msg = last[0] if last else messages[-1]
    content = message_content(last_msg).strip()
    is_tool = isinstance(last_msg, ToolMessage)
    is_raw_data = content.startswith("[") or content.startswith("{") or content in _EMPTY_MARKERS
    if not (is_tool or not content or is_raw_data):
        return updates

    tool_message = _tool_message_this_turn(messages)
    if tool_message is None:
        if not content:
            messages.append(AIMessage(content=NO_ANSWER_MESSAGE))
            updates["messages"] = messages
        return updates

    last_call = last_tool_call(messages)
    tool_name = str(last_call.get("name")) if last_call else None
    tool_args = last_call.get("args", {}) if last_call else {}
    answer = auto_answer_from_tool(tool_name, tool_args, tool_message.content, state.get("essay_text", ""))
    messages.append(AIMessage(content=answer))
    updates["messages"] = messages
    return updates


__all__ = [
    "check_context",
    "global_planner",
    "local_reasoning",
    "tool_executor",
    "response_generator",
    "should_run_global_planner",
    "normalize_plan",
    "format_full_analysis",
    "format_nlp_errors",
]
