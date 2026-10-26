from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CriteriaItem(BaseModel):
    score: Optional[int] = None
    max: Optional[int] = None
    comment: Optional[str] = None
    status: Optional[str] = None
    issue: Optional[str] = None


class ScoreBreakdown(BaseModel):
    total_score: int
    max_score: int
    criteria: Dict[str, CriteriaItem]


class StructureAnalysis(BaseModel):
    words: int
    sentences: int
    paragraphs: int
    paragraph_roles: List[str]


class ParagraphArgument(BaseModel):
    thesis: str
    example: str
    micro_conclusion: str


class SemanticsAnalysis(BaseModel):
    intro: str
    argument_1: str
    argument_2: str
    link: str
    personal_argument: str
    conclusion: str
    paragraph_analysis: Dict[str, ParagraphArgument]


class ErrorItem(BaseModel):
    type: str
    text: str
    fix: str


class NLPAnalysis(BaseModel):
    topic: str
    score: ScoreBreakdown
    structure: StructureAnalysis
    semantics: SemanticsAnalysis
    orthography: Dict[str, Any]
    punctuation: Dict[str, Any]
    grammar: Dict[str, Any]
    errors: List[ErrorItem]
    agent_hint: str


class ExpertReasoningInput(BaseModel):
    criterion: str
    text: Optional[str] = ""
    task_condition: Optional[str] = ""
    essay_text: Optional[str] = ""
    focus: Optional[str] = ""
    source_text: Optional[str] = ""
    source_text_provided: Optional[bool] = None


class RagItem(BaseModel):
    id: str = ""
    title: str
    author: str = ""
    argument: str
    tags: List[str] = Field(default_factory=list)
    score: float = 0.0


class RagResult(BaseModel):
    status: str  # "ok" | "empty" | "unavailable"
    query: str
    message: str = ""
    results: List[RagItem] = Field(default_factory=list)
    # How the hits were obtained: retrieval mode, query tags, whether the tag filter was applied.
    retrieval: Dict[str, Any] = Field(default_factory=dict)


class FactEvidence(BaseModel):
    title: Optional[str] = None
    href: Optional[str] = None
    snippet: Optional[str] = None


class FactCheckResult(BaseModel):
    query: str
    status: str
    evidence: List[FactEvidence] = Field(default_factory=list)
    error: Optional[str] = None


class AgentStateModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: List[Any] = Field(default_factory=list)
    essay_text: str = ""
    source_text: str = ""
    task_condition: str = ""
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    nlp_analysis: Dict[str, Any] = Field(default_factory=dict)
    global_plan: List[str] = Field(default_factory=list)
    current_step: int = 0
    show_reasoning: bool = False
    reasoning_log: List[str] = Field(default_factory=list)
    essay_sent: bool = False
    # One entry per tool-selection step: who decided (llm / rules), which tool, fallback reason, latency.
    tool_decisions: List[Dict[str, Any]] = Field(default_factory=list)
