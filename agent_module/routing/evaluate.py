"""Бенчмарк выбора инструмента: шаг плана + реплика ученика -> ожидаемый инструмент (или ``none``).

Метрики:
    Tool Selection Accuracy   — доля сценариев, где выбранный инструмент совпал с ожидаемым (``none`` — не звать инструмент);
    Accuracy на «не нужен инструмент» — доля сценариев с ``none``, где агент справедливо промолчал (ложные вызовы — это минус);
    Accuracy на «нужен инструмент»   — доля сценариев с ожидаемым инструментом, где выбран именно он;
    p50/p95 латентности решения.
"""

import csv
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from langchain_core.messages import AIMessage, HumanMessage

from agent_module.constants import ALL_TOOL_NAMES
from agent_module.prompts import LOCAL_REASONING_SYSTEM_PROMPT
from agent_module.rag.base import DATA_DIR
from agent_module.routing.rules import select_tool_by_rules
from agent_module.routing.selector import decide_with_llm

DEFAULT_SCENARIOS_PATH = DATA_DIR / "tool_selection_scenarios.csv"
NO_TOOL = "none"

# Короткое сочинение-заглушка: его текст нужен, чтобы у инструментов «для сочинения» были входные данные.
SAMPLE_ESSAY = (
    "Какова роль природы в жизни человека? Над этим вопросом размышляет К.Г. Паустовский.\n\n"
    "Автор рассказывает, как он ушёл в лес, чтобы побыть наедине с собой. Этот пример показывает, "
    "что природа восхищает человека своей красотой.\n\n"
    "Позиция автора ясна: природа лечит душу человека. Я согласен с автором."
)
SAMPLE_TASK = "Какова роль природы в жизни человека?"


@dataclass(frozen=True)
class Scenario:
    id: str
    step: str
    user_text: str
    essay_present: bool
    expected_tool: str
    assistant_prev: str = ""  # необязательно: реплика наставника перед сообщением ученика


def load_scenarios(path: Optional[Union[str, Path]] = None) -> List[Scenario]:
    csv_path = Path(path) if path else DEFAULT_SCENARIOS_PATH
    with csv_path.open(encoding="utf-8", newline="") as fh:
        return [
            Scenario(
                id=row["id"],
                step=row["step"].strip(),
                user_text=row["user_text"].strip(),
                essay_present=row["essay_present"].strip() == "1",
                expected_tool=row["expected_tool"].strip(),
                assistant_prev=(row.get("assistant_prev") or "").strip(),
            )
            for row in csv.DictReader(fh)
        ]


# Для абляции: без таблицы правил, только роль и описания инструментов (они приходят через bind_tools).
MINIMAL_SYSTEM_PROMPT = (
    "You are a tutoring agent helping a student improve an EGE Russian essay. You must respond in Russian. "
    "Call a tool only if it is needed to answer the student's latest message; otherwise answer directly."
)


def mask_tool_names(text: str) -> str:
    """Убирает названия инструментов из шага плана («через literature_rag» -> «через нужный инструмент»)."""
    for name in sorted(ALL_TOOL_NAMES, key=len, reverse=True):
        text = text.replace(name, "нужный инструмент")
    return text


def build_case(scenario: Scenario, prompt: str = "full", mask_step: bool = False) -> Dict[str, Any]:
    """Состояние и история диалога, как их увидел бы узел local_reasoning."""
    state: Dict[str, Any] = {"task_condition": SAMPLE_TASK, "source_text": "", "essay_text": "", "messages": []}
    history: List[Any] = []
    if scenario.essay_present:
        state["essay_text"] = SAMPLE_ESSAY
        history.append(HumanMessage(content=f"Сочинение: {SAMPLE_ESSAY}"))
    if scenario.assistant_prev:
        history.append(AIMessage(content=scenario.assistant_prev))
    history.append(HumanMessage(content=scenario.user_text))
    state["messages"] = history
    base = MINIMAL_SYSTEM_PROMPT if prompt == "minimal" else LOCAL_REASONING_SYSTEM_PROMPT
    step = mask_tool_names(scenario.step) if mask_step else scenario.step
    return {"state": state, "history": history, "system_prompt": f"{base} Текущий шаг плана: {step}."}


def predict_rules(scenario: Scenario) -> Dict[str, Any]:
    case = build_case(scenario)
    decision = select_tool_by_rules(case["history"], case["state"], scenario.step)
    return {"predicted": decision.tool or NO_TOOL, "latency_ms": decision.latency_ms, "error": None}


def make_llm_predictor(llm: Any, prompt: str = "full", mask_step: bool = False) -> Callable[[Scenario], Dict[str, Any]]:
    """Решение принимает только LLM: без запасного варианта на правилах. prompt/mask_step — режимы абляции."""

    def predict(scenario: Scenario) -> Dict[str, Any]:
        case = build_case(scenario, prompt=prompt, mask_step=mask_step)
        try:
            decision = decide_with_llm(llm, case["history"], case["state"], scenario.step, case["system_prompt"])
        except Exception as exc:
            logging.getLogger("ege_ai_agent.eval").error("сценарий %s: %s", scenario.id, exc, exc_info=True)
            return {"predicted": "error", "latency_ms": 0.0, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
        return {"predicted": decision.tool or NO_TOOL, "latency_ms": decision.latency_ms, "error": None}

    return predict


def run(
    scenarios: Sequence[Scenario],
    predictor: Callable[[Scenario], Dict[str, Any]],
    pause: Callable[[], None] = lambda: None,
    on_row: Optional[Callable[[dict, List[dict]], None]] = None,
) -> List[dict]:
    rows: List[dict] = []
    for scenario in scenarios:
        result = predictor(scenario)
        rows.append(
            {
                "id": scenario.id,
                "user_text": scenario.user_text,
                "expected": scenario.expected_tool,
                "predicted": result["predicted"],
                "correct": result["predicted"] == scenario.expected_tool,
                "latency_ms": round(result["latency_ms"], 1),
                "error": result["error"],
            }
        )
        if on_row:
            on_row(rows[-1], rows)
        pause()
    return rows


def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def summarize(rows: Sequence[dict]) -> Dict[str, Any]:
    total = len(rows)
    needs_tool = [r for r in rows if r["expected"] != NO_TOOL]
    needs_none = [r for r in rows if r["expected"] == NO_TOOL]
    latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"] > 0)
    per_tool: Dict[str, dict] = {}
    for tool in sorted({r["expected"] for r in rows}):
        subset = [r for r in rows if r["expected"] == tool]
        per_tool[tool] = {"n": len(subset), "accuracy": _ratio(sum(r["correct"] for r in subset), len(subset))}
    return {
        "n": total,
        "tool_selection_accuracy": _ratio(sum(r["correct"] for r in rows), total),
        "accuracy_when_tool_needed": _ratio(sum(r["correct"] for r in needs_tool), len(needs_tool)),
        "accuracy_when_no_tool_needed": _ratio(sum(r["correct"] for r in needs_none), len(needs_none)),
        "false_tool_calls": sum(1 for r in needs_none if r["predicted"] != NO_TOOL),
        "errors": sum(1 for r in rows if r["error"]),
        "latency_ms_p50": round(statistics.median(latencies), 1) if latencies else None,
        "latency_ms_p95": round(latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)], 1) if latencies else None,
        "per_expected_tool": per_tool,
    }
