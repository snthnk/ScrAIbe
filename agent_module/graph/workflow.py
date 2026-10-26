"""The LangGraph state machine.

    check_context ──┬─> global_planner ─┐
                    └───────────────────┴─> local_reasoning ──┬─> tool_executor ─┐
                                                 ▲            │                  │
                                                 └────────────┼──────────────────┘
                                                              └─> response_generator ─> END
"""

from functools import partial
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from agent_module.graph.nodes import (
    check_context,
    global_planner,
    local_reasoning,
    response_generator,
    should_run_global_planner,
    tool_executor,
)
from agent_module.graph.state import AgentState
from agent_module.messages import (
    is_ai_message,
    is_tool_message,
    message_content,
    state_to_dict,
    tool_calls_from_message,
)
from agent_module.utils import setup_logger

logger = setup_logger()


def route_from_check_context(state: Dict[str, Any]) -> str:
    state = state_to_dict(state)
    messages = state.get("messages", [])
    user_text = message_content(messages[-1]) if messages else ""
    should_plan = should_run_global_planner(user_text, state)
    dest = "global_planner" if should_plan else "local_reasoning"
    logger.debug(f"Routing from CheckContext -> {dest} (should_run_global_planner={should_plan})")
    return dest


def route_from_local_reasoning(state: Dict[str, Any]) -> str:
    state = state_to_dict(state)
    messages = state.get("messages", [])
    if not messages:
        logger.debug("Routing from LocalReasoning -> response_generator (messages empty)")
        return "response_generator"
    if len(messages) >= 2 and is_tool_message(messages[-2]) and is_ai_message(messages[-1]):
        logger.debug("Routing from LocalReasoning -> response_generator (tool execution completed)")
        return "response_generator"
    tool_calls = tool_calls_from_message(messages[-1])
    if tool_calls:
        logger.debug(f"Routing from LocalReasoning -> tool_executor ({len(tool_calls)} tool calls requested)")
        return "tool_executor"
    logger.debug("Routing from LocalReasoning -> response_generator (no tool calls)")
    return "response_generator"


def build_workflow(llm: Any) -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("check_context", check_context)
    graph.add_node("global_planner", partial(global_planner, llm=llm))
    graph.add_node("local_reasoning", partial(local_reasoning, llm=llm))
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("response_generator", response_generator)

    graph.set_entry_point("check_context")
    graph.add_conditional_edges(
        "check_context",
        route_from_check_context,
        {"global_planner": "global_planner", "local_reasoning": "local_reasoning"},
    )
    graph.add_edge("global_planner", "local_reasoning")
    graph.add_conditional_edges(
        "local_reasoning",
        route_from_local_reasoning,
        {"tool_executor": "tool_executor", "response_generator": "response_generator"},
    )
    graph.add_edge("tool_executor", "local_reasoning")
    graph.add_edge("response_generator", END)
    logger.info("Workflow: build")
    return graph.compile()
