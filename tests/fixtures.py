"""Общие тестовые данные (часть входов совпадает с теми, на которых снят tests/golden/legacy_behavior.json)."""

from nlp_module.analyzer import TEST_ESSAY, TEST_TOPIC

ESSAY = TEST_ESSAY
TOPIC = TEST_TOPIC
SOURCE = "(1) Человек должен жить в окружении природы. (2) Природа лечит душу."

ERR_SETS = {
    "empty_list": [],
    "empty_str_none": "None",
    "dict_empty": {"errors": []},
    "ortho": [{"type": "orthography", "text": "пошол", "fix": "Возможно, вы имели в виду: пошёл. Контекст: ...я пошол в лес..."}],
    "mixed": [
        {"type": "orthography", "text": "пошол", "fix": "пошёл"},
        {"type": "punctuation", "text": "Автор считает что", "fix": "Поставьте запятую перед «что»."},
        {"type": "fact", "text": "Пушкин «Муму»", "fix": "Автор — Тургенев."},
        {"type": "grammar", "text": "по приезду", "fix": "по приезде"},
        {"type": "speech", "text": "очень огромный", "fix": "огромный"},
        {"type": "logic", "text": "Но потому что", "fix": "Исправьте связку."},
        {"type": "author_position", "text": "Автор считает что книга это очень важно", "fix": "Позиция..."},
    ],
    "orth_raw_dict": [{"type": "orthography", "word": "кросавчик", "suggestions": ["красавчик"], "context": "...кросавчик..."}],
}

ANALYSIS_OK = {
    "topic": "Какова роль природы в жизни человека?",
    "score": {
        "total_score": 99,
        "max_score": 99,
        "criteria": {
            "K1": {"score": 1, "max": 1, "comment": "Позиция верна"},
            "K2": {"score": 5, "max": 6, "comment": "Связь не пояснена"},
            "K3": {"score": 1, "max": 2, "comment": "Аргумент слабый"},
            "K4": {"score": 0, "max": 1, "comment": "Фактическая ошибка"},
            "K7 (орфография)": {"score": 3, "max": 3, "comment": ""},
            "Other": {"score": 2, "max": 2, "comment": "x"},
        },
    },
    "structure": {"words": 120, "sentences": 9, "paragraphs": 4, "paragraph_roles": ["intro", "arg", "arg", "conclusion"]},
    "semantics": {"general_feedback": "Общий отзыв."},
    "errors": ERR_SETS["mixed"],
}

STATES = {
    "essay": {"essay_text": ESSAY, "task_condition": TOPIC, "source_text": "", "global_plan": ["Шаг 1", "Шаг 2"], "current_step": 0},
    "no_essay": {"essay_text": "", "task_condition": "", "source_text": "", "global_plan": [], "current_step": 0},
}
