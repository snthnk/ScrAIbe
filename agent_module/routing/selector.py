"""Tool selection: the LLM decides via function calling, the keyword rules are the fallback.

    TOOL_SELECTION_MODE=llm    (default) the LLM picks the tool; if it fails or answers nonsense, rules take over
    TOOL_SELECTION_MODE=rules  only the keyword rules pick the tool (no LLM call for the decision)

Every decision is logged with who made it (``decided_by``) — see ``ToolDecision.log_line``.
"""

import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage

from agent_module.messages import last_user_text, sanitize_message_history, tool_calls_from_message
from agent_module.routing.args import ground_args
from agent_module.routing.decision import DECIDED_BY_LLM, ToolDecision
from agent_module.routing.rules import select_tool_by_rules
from agent_module.tools import ALL_TOOLS, TOOLS_BY_NAME
from agent_module.utils import setup_logger

logger = setup_logger("ege_ai_agent.routing")

MODE_LLM = "llm"
MODE_RULES = "rules"


class InvalidToolChoice(RuntimeError):
    """The model asked for a tool that does not exist."""


def tool_selection_mode() -> str:
    mode = (os.getenv("TOOL_SELECTION_MODE") or MODE_LLM).strip().lower()
    return mode if mode in (MODE_LLM, MODE_RULES) else MODE_LLM


def decide_with_llm(
    llm: Any,
    history: List[Any],
    state: Dict[str, Any],
    step_text: str,
    system_prompt: str,
) -> ToolDecision:
    """One function-calling round. Raises on provider errors and on an unknown tool name."""
    messages = [SystemMessage(content=system_prompt)] + sanitize_message_history(history)
    started = time.perf_counter()
    response = llm.bind_tools(ALL_TOOLS).invoke(messages)
    latency_ms = (time.perf_counter() - started) * 1000.0

    calls = tool_calls_from_message(response)
    if not calls:
        return ToolDecision(
            tool=None, decided_by=DECIDED_BY_LLM, reason="llm:answered_without_tool", latency_ms=latency_ms, llm_message=response
        )

    first = calls[0]
    name = first.get("name")
    if name not in TOOLS_BY_NAME:
        raise InvalidToolChoice(f"model requested unknown tool {name!r}")
    if len(calls) > 1:
        logger.warning("LLM requested %d tool calls in one step; only the first (%s) is executed", len(calls), name)
    args = ground_args(name, first.get("args") or {}, state, last_user_text(history), step_text, history)
    return ToolDecision(
        tool=name, args=args, decided_by=DECIDED_BY_LLM, reason="llm:function_call", latency_ms=latency_ms, llm_message=response
    )


def select_tool(
    llm: Any,
    history: List[Any],
    state: Dict[str, Any],
    step_text: str,
    system_prompt: str,
    mode: Optional[str] = None,
    allow_fallback: bool = True,
) -> ToolDecision:
    """Pick the next tool (or none). ``allow_fallback=False`` is for measuring the LLM on its own."""
    mode = mode or tool_selection_mode()
    user_text = last_user_text(history)

    if mode == MODE_RULES:
        started = time.perf_counter()
        decision = select_tool_by_rules(history, state, step_text)
        decision.latency_ms = (time.perf_counter() - started) * 1000.0
        logger.info(decision.log_line(user_text))
        return decision

    try:
        decision = decide_with_llm(llm, history, state, step_text, system_prompt)
    except Exception as exc:
        if not allow_fallback:
            raise
        logger.warning("LLM tool selection failed (%s: %s); falling back to keyword rules", type(exc).__name__, str(exc)[:200])
        decision = select_tool_by_rules(history, state, step_text)
        if decision.tool is None:
            # No rule applies and the LLM is the only thing that could answer: surface the original error so
            # the caller's retry / model-fallback logic can deal with it (rate limits carry a retry delay).
            raise
        decision.fallback_reason = f"{type(exc).__name__}: {str(exc)[:160]}"
    logger.info(decision.log_line(user_text))
    return decision
