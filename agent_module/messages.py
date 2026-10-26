"""Message and tool-call helpers shared by the graph nodes and the workflow routers.

These used to be copy-pasted in both ``nodes.py`` and ``workflow.py``.
"""

import ast
import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_module.constants import ALL_TOOL_NAMES, EXPERT_REASONING, LITERATURE_RAG, WEB_FACT_CHECKER


def state_to_dict(state: Any) -> Dict[str, Any]:
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return dict(state)


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content) if content is not None else ""


def message_content(message: Any) -> str:
    if hasattr(message, "content"):
        return _content_to_text(message.content)
    if isinstance(message, dict):
        return _content_to_text(message.get("content", ""))
    return ""


def coerce_messages(messages: List[Any]) -> List[Any]:
    """Turn dict-style messages into LangChain message objects (unknown roles are dropped)."""
    coerced = []
    for message in messages:
        if hasattr(message, "content"):
            coerced.append(message)
            continue
        if not isinstance(message, dict):
            continue
        role = message.get("type") or message.get("role")
        content = message.get("content", "")
        if role in {"human", "user"}:
            coerced.append(HumanMessage(content=content))
        elif role in {"ai", "assistant"}:
            coerced.append(AIMessage(content=content))
        elif role == "system":
            coerced.append(SystemMessage(content=content))
        elif role == "tool":
            coerced.append(ToolMessage(content=content, tool_call_id=message.get("tool_call_id")))
    return coerced


def is_human(message: Any) -> bool:
    if isinstance(message, HumanMessage):
        return True
    return isinstance(message, dict) and (message.get("type") in {"human", "user"} or message.get("role") == "user")


def is_tool_message(message: Any) -> bool:
    if isinstance(message, ToolMessage):
        return True
    return isinstance(message, dict) and (message.get("type") or message.get("role")) == "tool"


def is_ai_message(message: Any) -> bool:
    if isinstance(message, AIMessage):
        return True
    return isinstance(message, dict) and (message.get("type") or message.get("role")) in {"ai", "assistant"}


def last_user_text(messages: List[Any]) -> str:
    for msg in reversed(messages):
        if is_human(msg):
            return message_content(msg)
    return ""


# ---------------------------------------------------------------------------
# Tool-call extraction
# ---------------------------------------------------------------------------


def tool_calls_from_message(message: Any) -> Any:
    """Return the tool calls requested by a message (native, raw provider format, or ``name(...)`` text)."""
    if hasattr(message, "tool_calls"):
        calls = message.tool_calls
        if calls:
            return calls
        additional = getattr(message, "additional_kwargs", {}) or {}
        raw_calls = additional.get("tool_calls") or additional.get("function_call")
        parsed = parse_raw_tool_calls(raw_calls)
        return parsed if parsed else calls
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls")
        if tool_calls:
            return tool_calls
        parsed = tool_calls_from_text(str(message.get("content", "")))
        return parsed if parsed else None
    parsed = tool_calls_from_text(message_content(message))
    return parsed if parsed else None


def parse_raw_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    if not raw:
        return []
    if isinstance(raw, dict) and raw.get("name") and raw.get("arguments") is not None:
        args = raw.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        return [{"id": f"raw_{raw.get('name')}", "name": raw.get("name"), "args": args if isinstance(args, dict) else {}}]
    if not isinstance(raw, list):
        return []
    parsed: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        call_id = item.get("id")
        fn = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = fn.get("name") or item.get("name")
        args = fn.get("arguments") or item.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not name:
            continue
        parsed.append({"id": call_id or f"raw_{name}", "name": name, "args": args if isinstance(args, dict) else {}})
    return parsed


def tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse a pseudo-call such as ``web_fact_checker(query='...')`` that a weak model wrote as plain text."""
    if not text:
        return []
    pattern = r"(" + "|".join(re.escape(name) for name in ALL_TOOL_NAMES) + r")\s*\("
    match = re.search(pattern, text)
    if not match:
        return []
    name = match.group(1)
    open_paren_index = match.end() - 1
    close_paren_index = find_matching_paren(text, open_paren_index)
    if close_paren_index is None:
        return []
    args_text = text[open_paren_index + 1 : close_paren_index].strip()
    args: Dict[str, Any] = {}
    if name in {WEB_FACT_CHECKER, LITERATURE_RAG}:
        args = parse_single_argument(args_text, key="query")
    elif name.startswith("nlp_"):
        args = parse_single_argument(args_text, key="essay_text")
    elif name == EXPERT_REASONING:
        payload = parse_payload_argument(args_text)
        if payload is not None:
            args = {"payload": payload}
    if not args:
        return []
    return [{"id": f"parsed_{name}", "name": name, "args": args}]


def find_matching_paren(text: str, open_index: int) -> Optional[int]:
    depth = 0
    for i in range(open_index, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def parse_single_argument(args_text: str, key: str) -> Dict[str, Any]:
    raw = args_text.strip().rstrip(",")
    if not raw:
        return {}
    if "=" in raw:
        k, v = raw.split("=", 1)
        if k.strip() != key:
            return {}
        return {key: literal_or_string(v.strip())}
    return {key: literal_or_string(raw)}


def parse_payload_argument(args_text: str) -> Optional[Dict[str, Any]]:
    raw = args_text.strip().rstrip(",")
    if not raw or "payload" not in raw:
        return None
    _, v = raw.split("payload", 1)
    parsed = literal_or_string(v.lstrip().lstrip("=").strip())
    return parsed if isinstance(parsed, dict) else None


def literal_or_string(value_text: str) -> Any:
    try:
        return ast.literal_eval(value_text)
    except Exception:
        return value_text.strip().strip('"').strip("'")


def last_tool_call(history: List[Any]) -> Optional[Dict[str, Any]]:
    for msg in reversed(history):
        calls = tool_calls_from_message(msg)
        if calls and isinstance(calls, list):
            first = calls[0]
            if isinstance(first, dict) and first.get("name"):
                return first
    return None


def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def with_raw_tool_calls(message: Any, tool_calls: List[Dict[str, Any]]) -> AIMessage:
    """Rebuild an AIMessage so the calls are available both natively and in provider-raw form."""
    raw_tool_calls = []
    for call in tool_calls:
        name = call.get("name")
        call_id = call.get("id")
        if not name or not call_id:
            continue
        raw_tool_calls.append(
            {
                "id": str(call_id),
                "type": "function",
                "function": {"name": str(name), "arguments": json.dumps(call.get("args", {}), ensure_ascii=False)},
            }
        )
    return AIMessage(
        content=message_content(message),
        tool_calls=tool_calls,
        additional_kwargs={"tool_calls": raw_tool_calls} if raw_tool_calls else {},
    )


def sanitize_message_history(messages: List[Any]) -> List[Any]:
    """Drop tool calls without a matching tool result (and orphan tool results).

    Providers reject histories where an assistant tool call has no answer; this can happen after an
    interrupted run or a retry.
    """
    coerced = coerce_messages(messages)

    tool_message_ids = set()
    for msg in coerced:
        if isinstance(msg, ToolMessage):
            tid = str(msg.tool_call_id or "").strip()
            if tid:
                tool_message_ids.add(tid)

    valid_ai_message_indices = set()
    ai_msg_to_calls: Dict[int, List[str]] = {}
    for idx, msg in enumerate(coerced):
        if not isinstance(msg, AIMessage):
            continue
        calls = tool_calls_from_message(msg)
        if not calls:
            valid_ai_message_indices.add(idx)
            continue
        matching = [str(c.get("id") or "").strip() for c in calls if str(c.get("id") or "").strip() in tool_message_ids]
        if matching:
            valid_ai_message_indices.add(idx)
            ai_msg_to_calls[idx] = matching

    sanitized = []
    kept_tool_call_ids = set()
    for idx, msg in enumerate(coerced):
        if isinstance(msg, (SystemMessage, HumanMessage)):
            sanitized.append(msg)
        elif isinstance(msg, AIMessage):
            if idx in valid_ai_message_indices:
                sanitized.append(msg)
                kept_tool_call_ids.update(ai_msg_to_calls.get(idx, []))
        elif isinstance(msg, ToolMessage):
            if str(msg.tool_call_id or "").strip() in kept_tool_call_ids:
                sanitized.append(msg)
        else:
            sanitized.append(msg)
    return sanitized
