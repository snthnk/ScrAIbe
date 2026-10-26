"""Argument grounding: whoever picks the tool (LLM or rules), the heavy arguments come from the graph state.

The LLM decides *which* tool to call. It is not trusted to copy a 300-word essay or the task condition
verbatim, so those are filled in from the state — deterministic and cheap.
"""

import re
from typing import Any, Dict, List, Optional

from agent_module.constants import (
    EXPERT_REASONING,
    LITERATURE_RAG,
    WEB_FACT_CHECKER,
    normalize_criterion_id,
)
from agent_module.messages import is_human, message_content

_CRITERION_IN_TEXT = re.compile(r"\b[кk]\s*(\d+)\b", re.IGNORECASE)
LONG_ESSAY_MIN_CHARS = 200


def criterion_from_text(text: str, default: str = "K2") -> str:
    match = _CRITERION_IN_TEXT.search(text.lower())
    if match:
        return normalize_criterion_id(f"K{match.group(1)}") or default
    return default


def fragment_from_text(text: str) -> str:
    """'Разбери К2: <фрагмент>' -> '<фрагмент>'; without a colon the whole message is the fragment."""
    return text.split(":", 1)[1].strip() if ":" in text else text


def essay_for_nlp(state: Dict[str, Any], history: Optional[List[Any]] = None) -> str:
    """The essay the NLP tools should work on: state first, otherwise the last long student message."""
    essay = (state.get("essay_text") or "").strip()
    if essay:
        return state["essay_text"]
    for msg in reversed(history or []):
        if is_human(msg):
            content = message_content(msg)
            if len(content) > LONG_ESSAY_MIN_CHARS:
                return content
    return ""


def build_expert_payload(
    state: Dict[str, Any],
    user_text: str,
    step_text: str,
    criterion: str,
    fragment: Optional[str] = None,
) -> Dict[str, Any]:
    source_text = state.get("source_text", "") or ""
    return {
        "criterion": criterion,
        "text": fragment if fragment is not None else fragment_from_text(user_text),
        "task_condition": state.get("task_condition", "") or "",
        "essay_text": state.get("essay_text", "") or "",
        "focus": step_text,
        "source_text": source_text,
        "source_text_provided": bool(source_text.strip()),
    }


def ground_args(
    tool_name: str,
    args: Dict[str, Any],
    state: Dict[str, Any],
    user_text: str,
    step_text: str,
    history: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Return the final, valid arguments for ``tool_name``."""
    args = dict(args or {})
    if tool_name in (LITERATURE_RAG, WEB_FACT_CHECKER):
        query = args.get("query")
        return {"query": query if isinstance(query, str) and query.strip() else user_text}

    if tool_name == EXPERT_REASONING:
        payload = args.get("payload") if isinstance(args.get("payload"), dict) else args
        criterion = (
            normalize_criterion_id(payload.get("criterion"))
            or criterion_from_text(user_text)
        )
        fragment = payload.get("text") if isinstance(payload.get("text"), str) and payload.get("text").strip() else None
        built = build_expert_payload(state, user_text, step_text, criterion, fragment)
        if isinstance(payload.get("focus"), str) and payload["focus"].strip():
            built["focus"] = payload["focus"]
        return {"payload": built}

    if tool_name.startswith("nlp_"):
        essay = essay_for_nlp(state, history) or (args.get("essay_text") or "")
        return {"essay_text": essay}

    return args
