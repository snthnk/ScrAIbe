"""NLP analysis module for ScrAIbe EGE essay assessment."""

from nlp_module.analyzer import (
    TEST_ESSAY,
    TEST_TOPIC,
    analyze_grammar,
    analyze_orthography,
    analyze_punctuation,
    analyze_semantics,
    analyze_structure,
    extract_errors,
    extract_topic,
    full_analysis,
    get_current_ortho,
    get_current_punct,
    get_current_semantics,
    score_fipi,
)

__all__ = [
    "TEST_TOPIC",
    "TEST_ESSAY",
    "full_analysis",
    "analyze_structure",
    "analyze_semantics",
    "analyze_orthography",
    "analyze_punctuation",
    "analyze_grammar",
    "score_fipi",
    "extract_errors",
    "extract_topic",
    "get_current_semantics",
    "get_current_ortho",
    "get_current_punct",
]
