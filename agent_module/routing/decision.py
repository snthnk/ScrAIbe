"""ToolDecision: the outcome of one tool-selection step, including *who* made the decision."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

DECIDED_BY_LLM = "llm"
DECIDED_BY_RULES = "rules"


@dataclass
class ToolDecision:
    """What the agent will do next: call ``tool`` with ``args``, or answer directly (``tool is None``).

    ``decided_by``:
      * ``"llm"``   — the model picked the tool (or chose to answer without one);
      * ``"rules"`` — the keyword rules decided: either rules mode is configured, or the LLM failed.
    ``fallback_reason`` is set only when the rules took over because the LLM failed / answered nonsense.
    """

    tool: Optional[str]
    args: Dict[str, Any] = field(default_factory=dict)
    decided_by: str = DECIDED_BY_RULES
    reason: str = ""
    fallback_reason: Optional[str] = None
    latency_ms: float = 0.0
    # The raw AIMessage when the LLM answered directly (kept out of the logged / serialised form).
    llm_message: Any = field(default=None, repr=False, compare=False)

    @property
    def is_fallback(self) -> bool:
        return self.fallback_reason is not None

    def to_log_dict(self, user_text: str = "") -> Dict[str, Any]:
        data = asdict(self)
        data.pop("llm_message", None)
        data["args"] = sorted(self.args.keys())  # argument names only: essays are long and private
        if user_text:
            data["user_text"] = user_text[:120]
        return data

    def log_line(self, user_text: str = "") -> str:
        return "tool_decision " + json.dumps(self.to_log_dict(user_text), ensure_ascii=False)

    def summary(self) -> str:
        """Short human line for the reasoning panel in the UI."""
        who = "LLM" if self.decided_by == DECIDED_BY_LLM else "правила"
        what = self.tool or "без инструмента"
        extra = f" (запасной вариант: {self.fallback_reason})" if self.fallback_reason else ""
        return f"Выбор инструмента: {what} — решил: {who}{extra}"
