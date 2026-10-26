"""Deterministic formatting of tool results into the text the student sees.

NLP results (errors, scores, structure) are rendered by code, not by the LLM: the numbers the student
sees are exactly the numbers the NLP module computed.
"""

import json
import re
from typing import Any, Dict, List

from agent_module.constants import (
    EXPERT_REASONING,
    LITERATURE_RAG,
    MAX_TOTAL_SCORE,
    NLP_ANALYZER,
    NLP_ERRORS,
    NLP_SCORING,
    NLP_SEMANTICS,
    NLP_STRUCTURE,
    OFFICIAL_MAX,
    WEB_FACT_CHECKER,
    normalize_criterion_id,
)
from agent_module.messages import safe_json_loads
from nlp_module import analyzer as nlp_module

NO_ERRORS_MESSAGE = (
    "Я проверил сочинение и не нашёл никаких орфографических, пунктуационных, грамматических или речевых ошибок. "
    "Отличная работа!"
)

_CATEGORY_BY_TYPE = {
    "orthography": "Орфография",
    "ortho": "Орфография",
    "punctuation": "Пунктуация",
    "punct": "Пунктуация",
    "grammar": "Грамматика",
    "fact": "Фактология",
    "factual": "Фактология",
    "logic": "Логика",
    "speech": "Речевые ошибки",
    "style": "Речевые ошибки",
}
_CATEGORY_ORDER = ["Орфография", "Пунктуация", "Грамматика", "Фактология", "Логика", "Речевые ошибки", "Другие замечания"]
_TYPE_LABEL = {
    "orthography": "Орфографическая ошибка",
    "ortho": "Орфографическая ошибка",
    "punctuation": "Пунктуационная ошибка",
    "punct": "Пунктуационная ошибка",
    "grammar": "Грамматическая ошибка",
    "speech": "Речевая ошибка",
    "style": "Речевая ошибка",
    "fact": "Фактическая ошибка",
    "factual": "Фактическая ошибка",
    "logic": "Логическая ошибка",
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def parse_error_item(err: Any) -> Dict[str, Any] | None:
    if not isinstance(err, dict):
        return None
    err_type = str(err.get("type") or err.get("category") or "other").strip().lower()
    fragment = str(err.get("text") or err.get("quote") or err.get("word") or err.get("fragment") or "").strip()
    explanation = str(err.get("explanation") or err.get("comment") or err.get("info") or "").strip()
    fix = err.get("fix") or err.get("suggestions") or err.get("correction") or ""
    if isinstance(fix, list):
        fix = ", ".join(str(f) for f in fix)
    fix = str(fix).strip()

    if err_type == "orthography":
        if not fix and not explanation:
            suggs = err.get("suggestions")
            if isinstance(suggs, list) and suggs:
                fix = ", ".join(str(s) for s in suggs)
            elif isinstance(suggs, str) and suggs:
                fix = suggs
        if not explanation:
            context = err.get("context") or ""
            explanation = f"Контекст: {context}" if context else "Нарушение орфографической нормы."
    elif not explanation:
        explanation = "Нарушение языковой нормы."

    return {"type": err_type, "fragment": fragment, "explanation": explanation, "fix": fix}


def _load_errors_payload(tool_content: Any) -> Any:
    """Best-effort decode of an errors payload (JSON / python-literal string, list or dict)."""
    import ast

    if not isinstance(tool_content, str):
        return tool_content
    content_str = tool_content.strip()
    if content_str in ("None", "null", "", "[]", "{}"):
        return []
    try:
        return json.loads(content_str)
    except Exception:
        pass
    try:
        return ast.literal_eval(content_str)
    except Exception:
        return content_str if not (content_str.startswith("[") or content_str.startswith("{")) else None


def format_nlp_errors(tool_content: Any) -> str:
    data = _load_errors_payload(tool_content)
    if isinstance(data, str):
        stripped = data.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            data = []
        else:
            return stripped

    errors_list: List[Any] = []
    if isinstance(data, list):
        errors_list = data
    elif isinstance(data, dict):
        errors_list = data.get("errors") or data.get("results") or data.get("items") or []
        if not isinstance(errors_list, list):
            errors_list = [errors_list]

    parsed_errors = []
    for item in errors_list:
        parsed = parse_error_item(item)
        if parsed and (parsed["fragment"] or parsed["explanation"] or parsed["fix"]):
            parsed_errors.append(parsed)
    if not parsed_errors:
        return NO_ERRORS_MESSAGE

    grouped: Dict[str, List[Dict[str, Any]]] = {name: [] for name in _CATEGORY_ORDER}
    for err in parsed_errors:
        grouped[_CATEGORY_BY_TYPE.get(err["type"], "Другие замечания")].append(err)

    lines = ["# Анализ ошибок в сочинении"]
    for cat_name in _CATEGORY_ORDER:
        errs = grouped[cat_name]
        if not errs:
            continue
        lines.append(f"\n## {cat_name}")
        for idx, err in enumerate(errs, 1):
            lines.append(f"{idx}. Фрагмент: «{err['fragment']}»")
            lines.append(f"   - Тип: {_TYPE_LABEL.get(err['type'], 'Ошибка')}")
            lines.append(f"   - Проблема/Объяснение: {err['explanation']}")
            if err["fix"]:
                lines.append(f"   - Исправление: {err['fix']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def _criterion_key(name: str) -> str | None:
    for official in OFFICIAL_MAX:
        if name == official or name.startswith((official + " ", official + "(")):
            return official
    return None


def normalize_criteria(criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp every scored criterion to the official maximum; ``'K7 (орфография)'`` -> ``'K7'``."""
    normalized: Dict[str, Any] = {}
    for name, value in criteria.items():
        key = _criterion_key(name)
        if key is None:
            normalized[name] = value
            continue
        val = dict(value) if isinstance(value, dict) else {"score": value}
        max_val = OFFICIAL_MAX[key]
        val["score"] = min(val.get("score", 0) or 0, max_val)
        val["max"] = max_val
        normalized[key] = val
    return normalized


def total_score(criteria: Dict[str, Any]) -> int:
    return sum(v.get("score", 0) if isinstance(v, dict) else v for v in criteria.values())


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        value = safe_json_loads(value)
    return value if isinstance(value, dict) else {}


def format_full_analysis(content: Any) -> str:
    try:
        if isinstance(content, str):
            content_str = content.strip()
            if content_str.startswith("{") or content_str.startswith("["):
                try:
                    content = json.loads(content_str)
                except Exception:
                    return "Ошибка: некорректный формат данных анализа."
            else:
                return content_str
        if content is None:
            return "Данные анализа отсутствуют."
        if isinstance(content, list):
            return "Результаты анализа в формате списка не поддерживаются."
        if not isinstance(content, dict):
            return "Некорректный формат результатов анализа."

        score_data = _as_dict(content.get("score"))
        criteria = normalize_criteria(_as_dict(score_data.get("criteria")))
        total = total_score(criteria)

        topic = content.get("topic") or "не определена"
        struct = _as_dict(content.get("structure"))
        words = struct.get("words", 0)
        sentences = struct.get("sentences", 0)
        paragraphs = struct.get("paragraphs", 0)
        roles = struct.get("paragraph_roles", [])
        if not isinstance(roles, list):
            roles = []

        lines = ["## Результаты анализа", f"Тема: {topic}", f"Оценка: {total} из {MAX_TOTAL_SCORE}"]
        if words <= 99:
            lines.append(f"Статус: Отказ в проверке. Объем сочинения ({words} слов) 99 слов или менее. Проверка невозможна.")
        else:
            roles_str = ", ".join(roles) if roles else "не определены"
            lines.append(f"Структура: предложений — {sentences}, слов — {words}, абзацев — {paragraphs} (роли: {roles_str})")

        formatted_errors = format_nlp_errors(content.get("errors") or [])
        if formatted_errors:
            formatted_errors = formatted_errors.replace("# Анализ ошибок в сочинении", "").strip() if formatted_errors.startswith("# Анализ ошибок в сочинении") else formatted_errors
            lines.append("\n### Ошибки:")
            lines.append(formatted_errors)

        lines.append("\n### Рекомендации:")
        recs = []
        feedback = _as_dict(content.get("semantics")).get("general_feedback")
        if feedback:
            recs.append(str(feedback))
        for crit_name, crit_val in criteria.items():
            if isinstance(crit_val, dict):
                comment = crit_val.get("comment")
                if comment and crit_val.get("score", 0) < crit_val.get("max", 0):
                    recs.append(f"По критерию {crit_name}: {comment}")
        if not recs:
            recs.append("Рекомендуется обратить внимание на логику изложения и следить за соблюдением орфографических и пунктуационных норм.")
        lines.extend(f"- {r}" for r in recs)
        return "\n".join(lines)
    except Exception:
        return "Ошибка: некорректная структура данных анализа."


def _format_scoring(tool_args: Dict[str, Any], tool_content: str, state_essay_text: str) -> str:
    data = safe_json_loads(tool_content)
    if not (isinstance(data, dict) and isinstance(data.get("criteria"), dict)):
        data = nlp_module.score_fipi(state_essay_text or tool_args.get("essay_text") or "")
    criteria = normalize_criteria(data.get("criteria", {}))
    lines = [f"Оценка сочинения по критериям ФИПИ: **{total_score(criteria)} из {MAX_TOTAL_SCORE} баллов**."]
    if criteria:
        lines.append("Детализация по критериям:")
        for crit_name, crit_val in criteria.items():
            if isinstance(crit_val, dict):
                lines.append(f"- **{crit_name}**: {crit_val.get('score', 0)}/{crit_val.get('max', 0)} ({crit_val.get('comment', '')})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Expert reasoning guardrails
# ---------------------------------------------------------------------------

_WAR_ROOTS = ["войн", "военн", "солдат", "боевых", "сражен", "окоп", "фронт"]
_ESSAY_STOPWORDS = {
    "автор", "текст", "сочинение", "критерий", "пример", "иллюстрация",
    "проблема", "первый", "второй", "связь", "смысловая", "пояснение",
    "отношение", "позиция", "исходный", "который", "чтобы", "однако",
    "также", "например", "потому", "почему", "анализ", "разбор", "слово",
}
_HALLUCINATION_MARKERS = [
    "например, автор описывает",
    "автор описывает, как",
    "в исходном тексте",
    "в тексте автора",
    "автор иллюстрирует",
    "пример-иллюстрация из текста",
]
K2_NO_SOURCE_DISCLOSURE = "исходный текст для анализа к2 не предоставлен"


def is_generic_expert_output(output: str, essay_text: str) -> bool:
    """Heuristic: does the expert answer talk about something that is not in the student's essay?"""
    if not output:
        return True
    lowered = output.lower()
    if "недостаточно данных" in lowered:
        return False

    output_mentions_war = any(w in lowered for w in ["война", "войне", "войны", "военн", "о войне"])
    essay_mentions_war = bool(essay_text) and any(w in essay_text.lower() for w in _WAR_ROOTS)
    if output_mentions_war and not essay_mentions_war:
        return True
    if "шаблон" in lowered and "исходный текст" in lowered and "два примера" in lowered:
        return True

    if essay_text:
        essay_words = {w for w in re.findall(r"[а-яА-ЯёЁa-zA-Z]{4,}", essay_text.lower()) if w not in _ESSAY_STOPWORDS}
        output_words = set(re.findall(r"[а-яА-ЯёЁa-zA-Z]{4,}", lowered))
        if essay_words and not (essay_words & output_words):
            return True
    return False


def _essay_facts(essay_text: str) -> tuple[int, str]:
    paragraphs = [p.strip() for p in essay_text.split("\n") if p.strip()]
    quotes = re.findall(r'(?:«[^»]+»|"[^"]+")', essay_text)
    quotes_str = f"обнаружены цитаты: {', '.join(quotes[:2])}" if quotes else "цитаты в явном виде не обнаружены"
    return len(paragraphs), quotes_str


def generate_expert_fallback(essay_text: str, criterion: str) -> str:
    if not essay_text:
        return f"Разбор по критерию {criterion}: недостаточно данных сочинения для детального анализа."
    words = essay_text.split()
    snippet = " ".join(words[:15]) + "..." if len(words) > 15 else essay_text
    paragraph_count, quotes_str = _essay_facts(essay_text)

    message = (
        f"Рекомендации по критерию {criterion} (Сгенерировано автоматически):\n"
        f"Анализируемый фрагмент текста сочинения: «{snippet}» (всего слов: {len(words)}, абзацев: {paragraph_count}).\n"
        f"Конкретные признаки: {quotes_str}.\n"
    )
    if criterion == "K2":
        message += (
            "По критерию К2 (комментарий): для получения максимального балла убедитесь, что в сочинении "
            "выделены два конкретных примера-иллюстрации из текста, даны пояснения к каждому из них, "
            "а также указана и проанализирована смысловая связь между ними.\n"
        )
        if paragraph_count < 3:
            message += "Замечание: структура сочинения содержит мало абзацев, примеры могут быть смешаны."
        else:
            message += "Рекомендация: проверьте логическое разделение примеров по абзацам."
    else:
        message += f"Пожалуйста, проработайте структуру сочинения в соответствии с требованиями критерия {criterion}."
    return message


def generate_k2_no_source_fallback(essay_text: str) -> str:
    paragraph_count, quotes_str = _essay_facts(essay_text)
    fallback = (
        "исходный текст для анализа К2 не предоставлен\n\n"
        "Анализ проводится исключительно на основе текста сочинения.\n"
        f"Наблюдения по сочинению (длина: {len(essay_text.split())} слов, абзацев: {paragraph_count}):\n"
        f"- В тексте сочинения {quotes_str}.\n"
    )
    if paragraph_count < 3:
        fallback += "- Замечание: структура сочинения содержит мало абзацев, что может затруднить выделение отдельных примеров-иллюстраций.\n"
    else:
        fallback += "- Структурно текст разделен на абзацы, что хорошо для выделения смысловых блоков.\n"
    fallback += (
        "\nОбратите внимание: без исходного текста невозможно точно проверить правильность выбора примеров-иллюстраций "
        "и их соответствие авторской позиции."
    )
    return fallback


def _format_expert(tool_args: Dict[str, Any], tool_content: str, state_essay_text: str) -> str:
    payload = tool_args.get("payload", {}) if isinstance(tool_args.get("payload"), dict) else {}
    essay_text = state_essay_text or payload.get("essay_text") or ""
    criterion = normalize_criterion_id(payload.get("criterion")) or payload.get("criterion") or "K2"

    source_text_provided = payload.get("source_text_provided")
    if source_text_provided is None:
        source_text_provided = bool((payload.get("source_text") or "").strip())

    if criterion == "K2" and not source_text_provided:
        lowered = tool_content.lower()
        disclosed = K2_NO_SOURCE_DISCLOSURE in lowered
        hallucinated = any(phrase in lowered for phrase in _HALLUCINATION_MARKERS)
        if not disclosed or hallucinated or is_generic_expert_output(tool_content, essay_text):
            return generate_k2_no_source_fallback(essay_text)

    answer = tool_content
    if is_generic_expert_output(tool_content, essay_text):
        answer = generate_expert_fallback(essay_text, criterion)

    # The LLM must not invent its own score: replace any "X из 3/5/6" with the deterministic NLP score.
    official = nlp_module.score_fipi(essay_text)["criteria"].get(criterion)
    if official is not None and criterion in OFFICIAL_MAX:
        det_max = OFFICIAL_MAX[criterion]
        det_score = min(official["score"], det_max)
        answer = re.sub(r"\b\d+\s*(?:баллов|балла|балл)?\s*(?:из|/)\s*(?:3|5|6)\b", f"{det_score} из {det_max}", answer)
        answer = re.sub(r"\b[0-6]\s*(?:баллов|балла|балл)\b", f"{det_score} баллов", answer)
    return answer


# ---------------------------------------------------------------------------
# Other tools
# ---------------------------------------------------------------------------


def _format_rag(data: Any, tool_content: str) -> str:
    if isinstance(data, dict) and data.get("status") in ("stub", "unavailable", "error"):
        return data.get("message") or "Поиск аргументов временно недоступен."
    items: List[Any] = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("results") or data.get("items") or []
    if not items:
        return "Не нашёл релевантный литературный аргумент по запросу. Сформулируй тему чуть конкретнее."

    lines = []
    for item in items[:2]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        author = str(item.get("author", "")).strip()
        argument = str(item.get("argument", "")).strip()
        if title and argument:
            who = f" ({author})" if author else ""
            lines.append(f"«{title}»{who}: {argument}")
        elif argument:
            lines.append(argument)
    if lines:
        return "Вот подходящие аргументы:\n" + "\n".join(lines)
    return f"Подойдёт такой аргумент:\n{tool_content}"


def _format_web(data: Any) -> str:
    if isinstance(data, dict):
        evidence = data.get("evidence") if isinstance(data.get("evidence"), list) else []
        if str(data.get("status") or "") == "found" and evidence:
            top = evidence[0] if isinstance(evidence[0], dict) else {}
            parts = [str(top.get(k, "")).strip() for k in ("snippet", "title", "href")]
            parts = [p for p in parts if p]
            if parts:
                return "Проверил по веб-выдаче. Вот релевантная зацепка:\n" + "\n".join(parts[:3])
        return "По текущему запросу в веб-выдаче не нашлось надежного подтверждения. Переформулируй запрос (название + автор/жанр)."
    return "Получен результат веб-поиска. Сформулируй, что именно нужно подтвердить (название + автор/жанр)."


def _format_structure(data: Any, tool_content: str) -> str:
    if not isinstance(data, dict):
        return f"Structure: {tool_content}"
    roles = data.get("paragraph_roles", [])
    roles_str = ", ".join(roles) if roles else "не определены"
    return (
        "Анализ структуры сочинения:\n"
        f"- Количество слов: {data.get('words', 0)}\n"
        f"- Количество предложений: {data.get('sentences', 0)}\n"
        f"- Количество абзацев: {data.get('paragraphs', 0)}\n"
        f"- Роли абзацев: {roles_str}"
    )


def _format_semantics(data: Any, tool_content: str) -> str:
    if not isinstance(data, dict):
        return f"Semantics: {tool_content}"
    return (
        "Семантический анализ сочинения:\n"
        f"- **Введение**: {data.get('intro', '')}\n"
        f"- **Пример 1**: {data.get('argument_1', '')}\n"
        f"- **Пример 2**: {data.get('argument_2', '')}\n"
        f"- **Связь**: {data.get('link', '')}\n"
        f"- **Заключение**: {data.get('conclusion', '')}"
    )


def auto_answer_from_tool(tool_name: str | None, tool_args: Dict[str, Any], tool_content: str, state_essay_text: str = "") -> str:
    """Turn the last tool result into the assistant's reply (no LLM involved)."""
    if not tool_content or tool_content.strip() in ("None", "null", ""):
        return "Проверка завершена. Замечаний или ошибок не обнаружено."

    if tool_name in (NLP_ANALYZER, "full"):
        return format_full_analysis(tool_content)
    if tool_name == NLP_ERRORS:
        return format_nlp_errors(tool_content)
    if tool_name == NLP_SCORING:
        return _format_scoring(tool_args, tool_content, state_essay_text)
    if tool_name == EXPERT_REASONING:
        return _format_expert(tool_args, tool_content, state_essay_text)

    data = safe_json_loads(tool_content)
    if tool_name == NLP_STRUCTURE:
        return _format_structure(data, tool_content)
    if tool_name == NLP_SEMANTICS:
        return _format_semantics(data, tool_content)
    if tool_name == LITERATURE_RAG:
        return _format_rag(data, tool_content)
    if tool_name == WEB_FACT_CHECKER:
        return _format_web(data)

    if tool_content.strip() in ("[]", "{}", "None", "null"):
        return "Обработка завершена. Результаты не содержат замечаний."
    return tool_content
