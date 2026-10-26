"""Shared constants for the agent: FIPI scale and tool names.

Single source of truth — previously the FIPI maxima were copy-pasted three times in nodes.py.
"""

import re
from typing import Dict, Optional, Tuple

# Official maximum primary score per criterion (task 27, FIPI 2025/2026 scale used by the NLP module).
OFFICIAL_MAX: Dict[str, int] = {
    "K1": 1,
    "K2": 3,
    "K3": 2,
    "K4": 1,
    "K5": 2,
    "K6": 1,
    "K7": 3,
    "K8": 3,
    "K9": 3,
    "K10": 3,
}
MAX_TOTAL_SCORE: int = sum(OFFICIAL_MAX.values())  # 22

# Tool identifiers as the LLM sees them (function-calling names).
NLP_ANALYZER = "nlp_analyzer"
NLP_STRUCTURE = "nlp_structure_tool"
NLP_SEMANTICS = "nlp_semantics_tool"
NLP_ORTHOGRAPHY = "nlp_orthography_tool"
NLP_PUNCTUATION = "nlp_punctuation_tool"
NLP_GRAMMAR = "nlp_grammar_tool"
NLP_SCORING = "nlp_scoring_tool"
NLP_ERRORS = "nlp_errors_tool"
LITERATURE_RAG = "literature_rag"
WEB_FACT_CHECKER = "web_fact_checker"
EXPERT_REASONING = "expert_reasoning_tool"

ALL_TOOL_NAMES: Tuple[str, ...] = (
    WEB_FACT_CHECKER,
    LITERATURE_RAG,
    EXPERT_REASONING,
    NLP_ANALYZER,
    NLP_STRUCTURE,
    NLP_SEMANTICS,
    NLP_ORTHOGRAPHY,
    NLP_PUNCTUATION,
    NLP_GRAMMAR,
    NLP_SCORING,
    NLP_ERRORS,
)

# Which section of the cached NLP analysis (state["nlp_analysis"]) each NLP tool can be served from.
NLP_TOOL_CACHE_SECTION: Dict[str, str] = {
    NLP_ERRORS: "errors",
    NLP_SCORING: "score",
    NLP_STRUCTURE: "structure",
    NLP_SEMANTICS: "semantics",
    NLP_ORTHOGRAPHY: "orthography",
    NLP_PUNCTUATION: "punctuation",
    NLP_GRAMMAR: "grammar",
    NLP_ANALYZER: "full",
}

# Prefixes with which the student submits (or replaces) the essay text in a chat message.
ESSAY_PREFIXES: Tuple[str, ...] = ("новое сочинение:", "сочинение:", "эссе:")


_CRITERION_RE = re.compile(r"^\s*[КKkк]\s*(\d{1,2})\b")


def normalize_criterion_id(value: object) -> Optional[str]:
    """``'К2'`` (Cyrillic), ``'k2'``, ``'K 2 (комментарий)'`` -> ``'K2'``; unknown -> ``None``.

    LLMs happily emit the Cyrillic letter, which would silently bypass ``criterion == "K2"`` guards.
    """
    match = _CRITERION_RE.match(str(value or ""))
    if not match:
        return None
    key = f"K{int(match.group(1))}"
    return key if key in OFFICIAL_MAX else None
