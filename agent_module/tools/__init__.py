from typing import Dict, List

from langchain_core.tools import BaseTool

from agent_module.tools.expert import expert_reasoning_tool
from agent_module.tools.nlp_tools import (
    nlp_analyzer,
    nlp_errors_tool,
    nlp_grammar_tool,
    nlp_orthography_tool,
    nlp_punctuation_tool,
    nlp_scoring_tool,
    nlp_semantics_tool,
    nlp_structure_tool,
)
from agent_module.tools.rag import literature_rag
from agent_module.tools.web_search import web_fact_checker

# Order matters a little: this is the order in which the tools are shown to the LLM.
ALL_TOOLS: List[BaseTool] = [
    nlp_analyzer,
    nlp_structure_tool,
    nlp_semantics_tool,
    nlp_orthography_tool,
    nlp_punctuation_tool,
    nlp_grammar_tool,
    nlp_scoring_tool,
    nlp_errors_tool,
    literature_rag,
    web_fact_checker,
    expert_reasoning_tool,
]
TOOLS_BY_NAME: Dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}

__all__ = [
    "ALL_TOOLS",
    "TOOLS_BY_NAME",
    "expert_reasoning_tool",
    "literature_rag",
    "web_fact_checker",
    "nlp_analyzer",
    "nlp_structure_tool",
    "nlp_semantics_tool",
    "nlp_orthography_tool",
    "nlp_punctuation_tool",
    "nlp_grammar_tool",
    "nlp_scoring_tool",
    "nlp_errors_tool",
]
