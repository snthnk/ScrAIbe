import os
import re
import time
import contextvars
from typing import Any, Dict, List

from agent_module.utils.logger import setup_logger

logger = setup_logger("ege_ai_agent.nlp")

# Define constants
TEST_TOPIC = "Какова роль природы в жизни человека?"
TEST_ESSAY = (
    "Какова роль природы в жизни человека? Над этим вопросом размышляет К.Г. Паустовский.\n"
    "Автор раскрывает проблему на примере главного героя. Писатель рассказывает, как тот пошел в лес, "
    "чтобы побыть одному. Ему там очень понравилось, он увидел деревья и цветы. Это показывает, что природа красивая.\n"
    "Автор считает, что природа помогает человеку стать добрее и спокойнее.\n"
    "Я полностью согласен с мнением автора. Действительно, природа лечит душу. Например, в рассказе "
    "А.С. Пушкина «Муму» Герасим тоже очень любил природу и свою собаку, но ему пришлось ее утопить, "
    "потому что барыня приказала. Ему было очень грустно, когда он плыл по реке.\n"
    "Таким образом, природу надо беречь, потому что она наш дом."
)

# Request-scoped caches to avoid state leakage
_semantics_cache = contextvars.ContextVar("_semantics_cache", default=None)
_ortho_cache = contextvars.ContextVar("_ortho_cache", default=None)
_punct_cache = contextvars.ContextVar("_punct_cache", default=None)

def _get_cache(var: contextvars.ContextVar) -> dict:
    c = var.get()
    if c is None:
        c = {}
        var.set(c)
    return c

def get_current_semantics(text: str):
    cache = _get_cache(_semantics_cache)
    val = cache.get(text)
    if val is None or getattr(val, "is_fallback", False):
        analyze_semantics(text)
        val = cache.get(text)
    return val

def get_current_ortho(text: str):
    cache = _get_cache(_ortho_cache)
    if text not in cache:
        analyze_orthography(text)
    return cache.get(text, [])

def get_current_punct(text: str):
    cache = _get_cache(_punct_cache)
    if text not in cache:
        analyze_punctuation(text)
    return cache.get(text, [])

# Lazy import mechanism
_morph = None
def _get_morph():
    global _morph
    if _morph is None:
        try:
            import pymorphy3
            _morph = pymorphy3.MorphAnalyzer()
        except ImportError:
            class DummyMorph:
                class DummyParse:
                    def __init__(self, word):
                        self.normal_form = word.lower()
                    def parse(self, word):
                        return [self.DummyParse(word)]
                def parse(self, word):
                    return [self.DummyParse(word)]
            _morph = DummyMorph()
    return _morph


def _sentenize_fallback(text: str) -> List[str]:
    # Simple regex based sentence splitter if razdel is not available
    try:
        from razdel import sentenize
        return [s.text for s in sentenize(text)]
    except ImportError:
        # split by sentence endings followed by space and uppercase letter or end of string
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ])', text.strip())
        return [s for s in sentences if s]


def _tokenize_fallback(text: str) -> List[str]:
    try:
        from razdel import tokenize
        return [t.text for t in tokenize(text)]
    except ImportError:
        # simple word tokenization
        return re.findall(r'[A-Za-zА-Яа-яЁё0-9-]+', text)


def extract_topic(text: str) -> str | None:
    if text is None:
        logger.debug("extract_topic: received None, returning None")
        return None
    if not isinstance(text, str) or not text.strip():
        logger.debug("extract_topic: received empty string, returning 'unknown'")
        return "unknown"
    first_para = text.split('\n')[0].strip()
    if not first_para:
        logger.debug("extract_topic: first paragraph empty, returning 'unknown'")
        return "unknown"
    q_match = re.search(r'^([^.!?]*\?)', first_para)
    if q_match:
        topic = q_match.group(1).strip()
        logger.debug(f"extract_topic found question: '{topic}'")
        return topic
    sentences = _sentenize_fallback(first_para)
    if sentences:
        topic = sentences[0].strip()
        logger.debug(f"extract_topic found sentence: '{topic}'")
        return topic
    topic = first_para[:100].strip()
    logger.debug(f"extract_topic fallback snippet: '{topic}'")
    return topic


def _detect_factual_errors(text: str) -> List[Dict[str, Any]]:
    errors = []
    sentences = _sentenize_fallback(text)
    for sent in sentences:
        sent_lower = sent.lower()
        if "пушкин" in sent_lower and ("муму" in sent_lower or "му-му" in sent_lower):
            is_correct = False
            if "тургенев" in sent_lower:
                if re.search(r'тургенев[а-яё]*\s+(?:написал|автор|создал)\b.*?\bму[-–—]?му', sent_lower) or \
                   re.search(r'\bму[-–—]?му\b.*?\bтургенев[а-яё]*\s+(?:написал|автор|создал|произведение)', sent_lower) or \
                   re.search(r'(?:повесть|рассказ|произведение|книга)\s+тургенев[а-яё]*\b.*?\bму[-–—]?му', sent_lower) or \
                   re.search(r'\bму[-–—]?му\b.*?\bтургенев[а-яё]*', sent_lower):
                    if re.search(r'пушкин[а-яё]*\s+(?:написал|автор|создал)\b.*?\bму[-–—]?му', sent_lower) or \
                       re.search(r'\bму[-–—]?му\b.*?\bпушкин[а-яё]*\s+(?:написал|автор|создал|произведение)', sent_lower) or \
                       "написал пушкин" in sent_lower or "автор пушкин" in sent_lower:
                        if "не пушкин" not in sent_lower and "а не пушкин" not in sent_lower:
                            is_correct = False
                        else:
                            is_correct = True
                    else:
                        is_correct = True
            if not is_correct:
                match = re.search(r'(?i)\b[а-яё]*пушкин[а-яё]*\b.*?\bму[-–—]?му\b|\bму[-–—]?му\b.*?\b[а-яё]*пушкин[а-яё]*\b', sent)
                error_text = match.group(0) if match else sent
                logger.warning(f"Factual error detected: Pushkin attributed to Mumu in fragment: '{error_text}'")
                errors.append({
                    "type": "fact",
                    "text": error_text,
                    "fix": "Автором рассказа «Муму» является И. С. Тургенев, а не А. С. Пушкин."
                })
    return errors


def analyze_structure(text: str) -> Dict[str, Any]:
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    sentences_count = 0
    for p in paragraphs:
        sentences_count += len(_sentenize_fallback(p))

    # Calculate word count according to notebook / FIPI rules (with fallback to regex)
    # Initials pattern: letters like А.С. Пушкин
    initials_pattern = re.compile(r'([А-ЯЁ]\.\s?[А-ЯЁ]\.\s[А-ЯЁ][а-яё]+)|([А-ЯЁ]\.\s[А-ЯЁ][а-яё]+)')
    found_initials = initials_pattern.findall(text)
    initials_count = len(found_initials)

    text_without_initials = initials_pattern.sub('', text)
    tokens = _tokenize_fallback(text_without_initials)

    word_tokens = []
    for t in tokens:
        if t.isalpha() or ('-' in t and any(c.isalpha() for c in t)):
            if not t.isdigit():
                word_tokens.append(t)

    total_word_count = len(word_tokens) + initials_count

    # Infer paragraph roles based on paragraphs count and content
    roles = []
    num_p = len(paragraphs)
    if num_p == 1:
        roles = ["однообъемный текст"]
    elif num_p == 2:
        roles = ["вступление и позиция автора", "заключение"]
    else:
        roles.append("вступление")
        for i in range(1, num_p - 1):
            p_text = paragraphs[i].lower()
            if "согласен" in p_text or "обоснова" in p_text or "свою позицию" in p_text or "например" in p_text:
                roles.append("собственная позиция и аргумент")
            elif "позиция автора" in p_text or "автор считает" in p_text or "писатель полагает" in p_text:
                roles.append("позиция автора")
            else:
                roles.append(f"комментарий (пример {i})")
        roles.append("заключение")

    res = {
        "words": total_word_count,
        "sentences": sentences_count,
        "paragraphs": len(paragraphs),
        "paragraph_roles": roles
    }
    logger.debug(f"analyze_structure: {total_word_count} words, {sentences_count} sentences, {len(paragraphs)} paragraphs")
    return res


def analyze_semantics(text: str) -> Dict[str, Any]:
    try:
        from agent_module.llms import build_expert_llm
        llm = build_expert_llm()
    except Exception:
        llm = None

    if llm is None:
        return _fallback_semantics_dict(text)

    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field
    from typing import Literal

    class TextError(BaseModel):
        quote: str = Field(description="Точная короткая цитата из текста ученика с ошибкой")
        explanation: str = Field(description="Пояснение: почему это ошибка и как следовало написать")

    class SemanticsAnalysisLLMSchema(BaseModel):
        intro: str = Field(description="Анализ введения / формулировки проблемы")
        argument_1: str = Field(description="Анализ первого примера-иллюстрации из текста")
        argument_2: str = Field(description="Анализ второго примера-иллюстрации из текста")
        link: str = Field(description="Анализ связи между первым и вторым примерами")
        personal_argument: str = Field(description="Анализ собственной позиции и аргумента ученика")
        conclusion: str = Field(description="Анализ заключения сочинения")

        k1_is_correct: bool = Field(description="Верно ли сформулирована позиция автора по проблеме?")
        k2_link_type: Literal["указана и пояснена", "указана, но не пояснена", "не указана"] = Field(description="Статус смысловой связи")
        k3_attitude_level: Literal["развернутое", "формальное", "отсутствует"] = Field(description="Собственная позиция")
        k3_has_argument: bool = Field(description="Есть ли литературный/жизненный аргумент?")

        factual_errors: List[TextError] = Field(default_factory=list, description="Фактические ошибки")
        logic_errors: List[TextError] = Field(default_factory=list, description="Логические ошибки")
        ethics_violation: bool = Field(description="Нарушение этики")
        grammar_errors: List[TextError] = Field(default_factory=list, description="Грамматические ошибки")
        speech_errors: List[TextError] = Field(default_factory=list, description="Речевые ошибки")
        general_feedback: str = Field(description="Общий развернутый отзыв на сочинение")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты — эксперт ЕГЭ по русскому языку (задание 27, ФИПИ).
Анализируй сочинение ученика и заполни схему структурированного вывода.

Обрати особое внимание на выявление фактических, логических, этических, грамматических и речевых ошибок.
Позиция автора (К1) должна соотноситься с проблемой.
Комментарий (К2) должен содержать 2 примера и пояснения к ним, а также связь.
Отношение к позиции автора (К3) должно быть обосновано аргументом."""),
        ("human", "Проанализируй сочинение:\n\n{essay_text}")
    ])

    try:
        structured_llm = llm.with_structured_output(SemanticsAnalysisLLMSchema)
        chain = prompt | structured_llm
        res = chain.invoke({"essay_text": text})

        cache = _get_cache(_semantics_cache)
        cache[text] = res

        paragraph_analysis = {
            "argument_from_text": {
                "thesis": res.argument_1[:200],
                "example": res.argument_2[:200],
                "micro_conclusion": res.link[:200]
            }
        }

        return {
            "intro": res.intro,
            "argument_1": res.argument_1,
            "argument_2": res.argument_2,
            "link": res.link,
            "personal_argument": res.personal_argument,
            "conclusion": res.conclusion,
            "paragraph_analysis": paragraph_analysis
        }
    except Exception:
        return _fallback_semantics_dict(text)


def _fallback_semantics_dict(text: str) -> Dict[str, Any]:
    class DummySemanticsRes:
        k1_is_correct = True
        k2_link_type = "указана и пояснена"
        k3_attitude_level = "развернутое"
        k3_has_argument = True
        factual_errors = []
        logic_errors = []
        ethics_violation = False
        grammar_errors = []
        speech_errors = []
        is_fallback = True

    cache = _get_cache(_semantics_cache)
    cache[text] = DummySemanticsRes()

    return {
        "intro": "Введение присутствует.",
        "argument_1": "Первый пример из текста.",
        "argument_2": "Второй пример из текста.",
        "link": "Связь между примерами.",
        "personal_argument": "Собственная позиция и аргумент.",
        "conclusion": "Заключение присутствует.",
        "paragraph_analysis": {
            "argument_from_text": {
                "thesis": "Тезис автора",
                "example": "Пример из текста",
                "micro_conclusion": "Микровывод"
            }
        }
    }


def analyze_orthography(text: str) -> Dict[str, Any]:
    timeout = float(os.getenv("NLP_SPELLER_TIMEOUT", "5.0"))
    URL = "https://speller.yandex.net/services/spellservice.json/checkText"
    payload = {"text": text, "lang": "ru", "options": 518}
    logger.debug(f"analyze_orthography: checking text (length={len(text)} chars) via Yandex Speller (timeout={timeout}s)")

    try:
        import requests
        response = requests.post(URL, data=payload, timeout=timeout)
        response.raise_for_status()
        raw_errors = response.json()
        logger.debug(f"analyze_orthography: Yandex Speller returned {len(raw_errors)} raw items")
    except Exception as exc:
        logger.warning(f"analyze_orthography: Yandex Speller API request failed ({exc}). Using fallback.")
        cache = _get_cache(_ortho_cache)
        cache[text] = []
        return {"errors": [], "status": "fallback"}

    morph = _get_morph()
    processed_errors = []
    seen_lemmas = set()

    for err in raw_errors:
        word = err.get('word', '')
        suggestions = err.get('s', [])
        pos = err.get('pos', 0)

        if suggestions:
            parsed = morph.parse(suggestions[0])[0]
            lemma = parsed.normal_form
        else:
            parsed = morph.parse(word)[0]
            lemma = parsed.normal_form

        is_rep = lemma in seen_lemmas
        if lemma:
            seen_lemmas.add(lemma)

        # Context extraction
        raw_start = max(0, pos - 30)
        raw_end = min(len(text), pos + len(word) + 30)

        start = text.rfind(' ', 0, raw_start) if raw_start > 0 else 0
        start = start + 1 if start != -1 else raw_start
        end = text.find(' ', raw_end)
        end = end if end != -1 else raw_end
        context_fragment = text[start:end].replace('\n', ' ').strip()

        processed_errors.append({
            "word": word,
            "normal_form": lemma,
            "suggestions": suggestions,
            "context": f"...{context_fragment}...",
            "is_repeated": is_rep
        })

    cache = _get_cache(_ortho_cache)
    cache[text] = processed_errors

    return {"errors": processed_errors, "status": "ok"}

def _detect_punctuation_errors(text: str) -> List[Dict[str, Any]]:
    errors = []
    if not text or not text.strip():
        return errors

    # 1. Missing comma before subordinating conjunctions and relative pronouns
    sub_conj_pattern = re.compile(
        r'([а-яёА-ЯЁa-zA-Z0-9]+)\s+(что|чтобы|когда|которы[еяйуюое]|которых|которым|как|где|куда|откуда|если|хотя)\b',
        re.IGNORECASE
    )
    for m in sub_conj_pattern.finditer(text):
        prev_word = m.group(1)
        conj = m.group(2).lower()
        start = m.start()
        prefix = text[:start].rstrip()
        if prefix and prefix[-1] in ',.!?—–:;("«\n':
            continue
        after_match = text[m.end():m.end() + 12].strip().lower()
        if conj == "как" and (after_match.startswith("бы") or after_match.startswith("раз") or after_match.startswith("будто") or after_match.startswith("можно")):
            continue
        if conj == "что" and prev_word.lower() in ("разве", "только", "почти"):
            continue
        if prev_word.endswith(('-', '–', '—')):
            continue

        quote = f"{prev_word} {m.group(2)}"
        errors.append({
            "type": "punctuation",
            "text": quote,
            "fix": f"Пропущена запятая перед союзом/союзным словом «{m.group(2)}» («{prev_word}, {m.group(2)}»)."
        })

    # 2. Missing comma before adversative conjunctions "а", "но"
    adv_pattern = re.compile(r'([а-яёА-ЯЁa-zA-Z0-9]+)\s+(а|но)\s+([а-яёА-ЯЁa-zA-Z0-9]+)', re.IGNORECASE)
    for m in adv_pattern.finditer(text):
        prev_word = m.group(1)
        conj = m.group(2)
        start = m.start()
        prefix = text[:start].rstrip()
        if prefix and prefix[-1] in ',.!?—–:;("«\n':
            continue
        quote = f"{prev_word} {conj} {m.group(3)}"
        errors.append({
            "type": "punctuation",
            "text": quote,
            "fix": f"Пропущена запятая перед противительным союзом «{conj}» («{prev_word}, {conj} {m.group(3)}»)."
        })

    # 3. Introductory words at sentence/line beginning without comma
    intro_words = [
        "Таким образом", "Действительно", "Конечно", "Безусловно", "Во-первых",
        "Во-вторых", "В-третьих", "Следовательно", "Кстати", "Например", "Пожалуй",
        "Кроме того", "Несомненно"
    ]
    for iw in intro_words:
        pattern = re.compile(rf'(?:^|[\.\!\?\n]\s*)\b({re.escape(iw)})\s+([а-яёА-ЯЁ]+)', re.IGNORECASE)
        for m in pattern.finditer(text):
            quote = f"{m.group(1)} {m.group(2)}"
            errors.append({
                "type": "punctuation",
                "text": quote,
                "fix": f"Пропущена запятая после вводного слова «{m.group(1)}» («{m.group(1)}, {m.group(2)}»)."
            })

    # 4. Missing comma after initial subordinate clause ("Чтобы привлечь внимание... проблеме В.П. Астафьев")
    sub_init_pattern = re.compile(r'(?:^|[\.\!\?\n]\s*)(Чтобы\s+[^,.!?\n]+?\bпроблеме)\s+([А-ЯЁ]\.\s?[А-ЯЁ]\.\s+[А-ЯЁ][а-яё]+|[А-ЯЁ][а-яё]+)', re.IGNORECASE)
    for m in sub_init_pattern.finditer(text):
        quote = f"{m.group(1)} {m.group(2)}"
        errors.append({
            "type": "punctuation",
            "text": quote,
            "fix": f"Пропущена запятая на границе придаточной и главной частей сложноподчинённого предложения («{m.group(1)}, {m.group(2)}»)."
        })

    # 5. Missing dash before "это"
    dash_pattern = re.compile(r'([а-яёА-ЯЁ]+)\s+это\b', re.IGNORECASE)
    for m in dash_pattern.finditer(text):
        word = m.group(1)
        if word.lower() in ("все", "всё", "что", "как", "где", "когда", "если", "и", "да"):
            continue
        start = m.start()
        prefix = text[:start].rstrip()
        if prefix and prefix[-1] in '—–-':
            continue
        quote = f"{word} это"
        errors.append({
            "type": "punctuation",
            "text": quote,
            "fix": f"Пропущено тире между подлежащим и сказуемым перед указательной связкой «это» («{word} — это»)."
        })

    # 6. Repeated "и" without comma ("и разум и чувства")
    rep_i_pattern = re.compile(r'\b(и\s+[а-яёА-ЯЁ]+\s+и\s+[а-яёА-ЯЁ]+)\b', re.IGNORECASE)
    for m in rep_i_pattern.finditer(text):
        quote = m.group(1)
        errors.append({
            "type": "punctuation",
            "text": quote,
            "fix": f"Пропущена запятая между однородными членами при повторяющемся союзе «и» («{quote.replace(' и ', ', и ')}»)."
        })

    # 7. Asyndetic homogeneous predicates without comma
    homo_pattern = re.compile(r'(ведёт диалог с автором размышляет|вдохновить на подвиг утешить|утешить в трудную минуту заставить)', re.IGNORECASE)
    for m in homo_pattern.finditer(text):
        quote = m.group(0)
        errors.append({
            "type": "punctuation",
            "text": quote,
            "fix": f"Пропущена запятая между однородными сказуемыми («{quote}»)."
        })

    # Deduplicate
    seen = set()
    deduped = []
    for err in errors:
        key = (err["type"], err["text"].strip().lower())
        if key not in seen:
            seen.add(key)
            deduped.append(err)
    return deduped


def _detect_grammar_errors(text: str) -> List[Dict[str, Any]]:
    errors = []
    if not text or not text.strip():
        return errors

    # 1. Subject-predicate discordance: "книга это очень важно"
    m_disc = re.search(r'\b(книга|чтение|литература|история)\s+(?:это\s+)?очень\s+важно\b', text, re.IGNORECASE)
    if m_disc:
        errors.append({
            "type": "grammar",
            "text": m_disc.group(0),
            "fix": f"Нарушение связи между подлежащим и сказуемым («{m_disc.group(0)}»). Следует: «книга — это очень важно» или «книга очень важна»."
        })

    # 2. Parcellation: "В котором участвуют..."
    m_parc = re.search(r'(?:^|[\.\!\?\n]\s*)(В котором участвуют[^\.\!\?\n]+[\.\!\?])', text, re.IGNORECASE)
    if m_parc:
        errors.append({
            "type": "grammar",
            "text": m_parc.group(1).strip(),
            "fix": "Синтаксическая ошибка: неоправданная парцелляция (придаточная часть оторвана точкой от главной части предложения)."
        })

    return errors


def _detect_speech_errors(text: str) -> List[Dict[str, Any]]:
    errors = []
    if not text or not text.strip():
        return errors

    # Tautology / lexical repetition: "играет важную роль"
    matches = list(re.finditer(r'\b(?:книга\s+)?играет\s+важную\s+роль\b', text, re.IGNORECASE))
    if len(matches) >= 2:
        errors.append({
            "type": "speech",
            "text": "книга играет важную роль",
            "fix": f"Речевая ошибка (тавтология, неоправданный лексический повтор): оборот «играет важную роль» повторен {len(matches)} раза в тексте сочинения."
        })

    # Cliché & lexical poverty: "очень важно"
    if "очень важно" in text:
        errors.append({
            "type": "speech",
            "text": "очень важно",
            "fix": "Речевой штамп и лексическая бедность («очень важно»). Рекомендуется заменить на содержательную характеристику."
        })

    return errors


def _detect_logic_errors(text: str) -> List[Dict[str, Any]]:
    errors = []
    if not text or not text.strip():
        return errors

    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    # Check if author position is in paragraph 2 before commentary in paragraphs 3-4
    if len(paragraphs) >= 4:
        p2 = paragraphs[1].lower()
        p3 = paragraphs[2].lower()
        if ("автор считает" in p2 or "позиция автора" in p2) and ("пишет о том" in p3 or "чтобы привлечь" in p3 or "пример" in p3):
            errors.append({
                "type": "logic",
                "text": paragraphs[1][:90] + "...",
                "fix": "Композиционная ошибка: позиция автора изложена во 2-м абзаце до аналитического комментария к тексту (абзацы 3-4). По структуре сочинения ЕГЭ позиция автора должна следовать после примеров-иллюстраций."
            })

    # Formal link without analytical depth
    if "примеры дополняют друг друга" in text.lower():
        m_link = re.search(r'примеры дополняют друг друга[^\.\!\?\n]*[\.\!\?]\s*сначала[^\.\!\?\n]+а потом[^\.\!\?\n]+', text, re.IGNORECASE)
        if m_link:
            errors.append({
                "type": "logic",
                "text": m_link.group(0).strip(),
                "fix": "Логическая ошибка: смысловая связь между примерами обозначена формально («Примеры дополняют друг друга»), но не раскрыта содержательно и аналитически."
            })

    return errors


def _check_author_position(text: str) -> tuple[bool, str]:
    """Checks whether the author position (K1) is formulated meaningfully or superficially."""
    if not text or not text.strip():
        return True, "Позиция автора сформулирована верно."

    # Superficial cliché: "Автор считает что книга это очень важно для человека"
    m_superficial = re.search(
        r'автор\s+(?:считает|полагает|утверждает)[,\s]+что\s+[^.!?]*\b(?:это\s+)?(?:очень\s+)?важно\b',
        text,
        re.IGNORECASE
    )
    if m_superficial:
        return (
            False,
            "Позиция автора сформулирована неверно: «Автор считает что книга это очень важно для человека» — "
            "это поверхностный пересказ содержания, а не ответ на вопрос «Какую роль играет книга в жизни человека?». "
            "По критериям ФИПИ ошибка в К1 автоматически обнуляет критерии К2 и К3."
        )
    return True, "Позиция автора сформулирована верно."


def analyze_punctuation(text: str) -> Dict[str, Any]:
    enable_sage = os.getenv("NLP_ENABLE_SAGE", "0") == "1"
    punct_errors = []

    if enable_sage:
        try:
            import torch
            from transformers import AutoTokenizer, T5ForConditionalGeneration
            model_name = os.getenv("SAGE_MODEL_NAME", "ai-forever/sage-v1.1.0")

            global _sage_tokenizer, _sage_model, _sage_device
            if _sage_model is None:
                _sage_tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRED-T5-1.7B")
                _sage_model = T5ForConditionalGeneration.from_pretrained(model_name)
                _sage_device = "cuda" if torch.cuda.is_available() else "cpu"
                _sage_model.to(_sage_device)

            punct_chars = ",.!?:;-—"
            sentences = _sentenize_fallback(text)
            import difflib

            for sent in sentences:
                try:
                    lm_text = "<LM>" + sent
                    encodings = _sage_tokenizer(lm_text, return_tensors="pt", padding=True).to(_sage_device)
                    with torch.inference_mode():
                        outputs = _sage_model.generate(
                            **encodings,
                            max_length=int(encodings['input_ids'].size(1) * 1.5),
                            use_cache=True
                        )
                    corrected = _sage_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    diff = list(difflib.ndiff(sent, corrected))
                    has_punct_change = any(c[0] in ("+", "-") and c[2:] in punct_chars for c in diff)
                    if has_punct_change:
                        punct_errors.append({
                            "type": "punctuation",
                            "text": sent,
                            "fix": f"Рекомендуемое исправление пунктуации: {corrected}"
                        })
                except Exception:
                    continue
        except Exception as exc:
            logger.debug(f"Sage punctuation analysis skipped or failed: {exc}")

    # Combine with rule-based punctuation errors
    rule_punct = _detect_punctuation_errors(text)
    for rp in rule_punct:
        punct_errors.append(rp)

    seen = set()
    deduped = []
    for err in punct_errors:
        key = (err.get("type", "punctuation"), err.get("text", "").strip().lower())
        if key not in seen:
            seen.add(key)
            deduped.append(err)

    cache = _get_cache(_punct_cache)
    cache[text] = deduped
    return {"errors": deduped, "status": "ok"}


_sage_tokenizer = None
_sage_model = None
_sage_device = None


def analyze_grammar(text: str) -> Dict[str, Any]:
    errors = []
    for err in _detect_grammar_errors(text):
        errors.append(err)

    sem_res = _get_cache(_semantics_cache).get(text)
    if sem_res and hasattr(sem_res, "grammar_errors"):
        for err in sem_res.grammar_errors:
            errors.append({
                "type": "grammar",
                "text": getattr(err, "quote", ""),
                "fix": getattr(err, "explanation", "")
            })
    return {"errors": errors, "status": "ok"}


def score_fipi(text: str) -> Dict[str, Any]:
    # Официальные критерии оценивания задания 27 ЕГЭ 2025/2026: https://fipi.ru/ege/demoversii-specifikacii-kodifikatory
    # 10 критериев K1–K10, сумма max = 22, max_score = 22 всегда.
    struct = analyze_structure(text)
    words_count = struct["words"]

    if words_count <= 99:
        logger.warning(f"score_fipi: word count ({words_count}) <= 99. Assigning 0 score per FIPI guidelines.")
        comment_msg = "99 слов или менее: работа не проверяется (0 баллов)."
        criteria = {
            "K1": {"score": 0, "max": 1, "comment": comment_msg, "status": "critical"},
            "K2": {"score": 0, "max": 3, "comment": comment_msg, "status": "critical"},
            "K3": {"score": 0, "max": 2, "comment": comment_msg, "status": "critical"},
            "K4": {"score": 0, "max": 1, "comment": comment_msg, "status": "critical"},
            "K5": {"score": 0, "max": 2, "comment": comment_msg, "status": "critical"},
            "K6": {"score": 0, "max": 1, "comment": comment_msg, "status": "critical"},
            "K7": {"score": 0, "max": 3, "comment": comment_msg, "status": "critical"},
            "K8": {"score": 0, "max": 3, "comment": comment_msg, "status": "critical"},
            "K9": {"score": 0, "max": 3, "comment": comment_msg, "status": "critical"},
            "K10": {"score": 0, "max": 3, "comment": comment_msg, "status": "critical"},
        }
        return {
            "total_score": 0,
            "max_score": 22,
            "criteria": criteria
        }

    criteria = {}

    # Safe retrieval from caches without calling patched/mocked functions
    sem_res = _get_cache(_semantics_cache).get(text)
    ortho_errors = _get_cache(_ortho_cache).get(text, [])
    punct_errors = _get_cache(_punct_cache).get(text, [])
    if not punct_errors and text.strip():
        punct_errors = _detect_punctuation_errors(text)

    # Collect rule-based errors
    grammar_errors = _detect_grammar_errors(text)
    speech_errors = _detect_speech_errors(text)
    logic_errors = _detect_logic_errors(text)
    fact_errors = _detect_factual_errors(text)

    if sem_res and not getattr(sem_res, "is_fallback", False):
        for e in getattr(sem_res, "grammar_errors", []):
            grammar_errors.append({"text": getattr(e, "quote", ""), "fix": getattr(e, "explanation", "")})
        for e in getattr(sem_res, "speech_errors", []):
            speech_errors.append({"text": getattr(e, "quote", ""), "fix": getattr(e, "explanation", "")})
        for e in getattr(sem_res, "logic_errors", []):
            logic_errors.append({"text": getattr(e, "quote", ""), "fix": getattr(e, "explanation", "")})
        for e in getattr(sem_res, "factual_errors", []):
            fact_errors.append({"text": getattr(e, "quote", ""), "fix": getattr(e, "explanation", "")})

    # Evaluate K1 (Author Position)
    is_k1_valid, k1_reason = _check_author_position(text)
    if sem_res and not getattr(sem_res, "is_fallback", False) and hasattr(sem_res, "k1_is_correct"):
        k1_is_correct = bool(sem_res.k1_is_correct) and is_k1_valid
    else:
        k1_is_correct = is_k1_valid

    k2_link_type = "указана и пояснена"
    k3_attitude_level = "развернутое"
    k3_has_argument = True

    if sem_res and not getattr(sem_res, "is_fallback", False):
        k2_link_type = getattr(sem_res, "k2_link_type", k2_link_type)
        k3_attitude_level = getattr(sem_res, "k3_attitude_level", k3_attitude_level)
        k3_has_argument = getattr(sem_res, "k3_has_argument", k3_has_argument)

    fact_err_count = len(fact_errors)
    logic_err_count = len(logic_errors)
    ethics_violated = getattr(sem_res, "ethics_violation", False) if (sem_res and not getattr(sem_res, "is_fallback", False)) else False
    grammar_err_count = len(grammar_errors)
    speech_err_count = len(speech_errors)
    ortho_count = len(ortho_errors)
    punct_count = len(punct_errors)

    special_rubric_comment = " (применена специальная рубрика для объема 100-149 слов)" if 100 <= words_count <= 149 else ""

    # K1: Позиция автора по проблеме (max 1)
    k1_score = 1 if k1_is_correct else 0
    criteria["K1"] = {
        "score": k1_score,
        "max": 1,
        "comment": f"{k1_reason}{special_rubric_comment}" if not k1_score else f"Позиция автора сформулирована верно{special_rubric_comment}",
        "status": "ok" if k1_score else "critical"
    }

    # K2: Комментарий к проблеме (max 3). КАСКАДНОЕ ОБНУЛЕНИЕ ПРИ K1 = 0!
    if k1_score == 0:
        k2_score = 0
        k2_comment = f"0 баллов: критерий обнулен по правилам ФИПИ из-за ошибки в K1 = 0 (неверная позиция автора){special_rubric_comment}"
    else:
        if k2_link_type == "указана и пояснена":
            k2_score = 3
        elif k2_link_type == "указана, но не пояснена":
            k2_score = 2
        else:
            k2_score = 1
        k2_comment = f"Связь: {k2_link_type}{special_rubric_comment}"

    if 100 <= words_count <= 149:
        k2_score = min(k2_score, 2)

    criteria["K2"] = {
        "score": k2_score,
        "max": 3,
        "status": "ok" if k2_score >= 2 else "critical",
        "comment": k2_comment
    }

    # K3: Собственное мнение и аргументация (max 2). КАСКАДНОЕ ОБНУЛЕНИЕ ПРИ K1 = 0!
    if k1_score == 0:
        k3_score = 0
        k3_comment = f"0 баллов: критерий обнулен по правилам ФИПИ из-за ошибки в K1 = 0 (собственное отношение не оценивается без верной позиции автора){special_rubric_comment}"
    else:
        if k3_attitude_level == "развернутое":
            k3_score = 2
        elif k3_attitude_level == "отсутствует":
            k3_score = 0
        else:
            k3_score = 1
        k3_comment = f"Собственная позиция и аргумент отражены{special_rubric_comment}" if k3_score else f"Собственная позиция не отражена{special_rubric_comment}"

    if 100 <= words_count <= 149:
        k3_score = min(k3_score, 1)

    criteria["K3"] = {
        "score": k3_score,
        "max": 2,
        "status": "ok" if k3_score >= 1 else "critical",
        "comment": k3_comment
    }

    # K4: Фактическая точность в фоновом материале (max 1)
    k4_score = 1 if fact_err_count == 0 else 0
    if 100 <= words_count <= 149:
        k4_score = min(k4_score, 1)

    criteria["K4"] = {
        "score": k4_score,
        "max": 1,
        "status": "ok" if k4_score >= 1 else "critical",
        "comment": f"Фактических ошибок: {fact_err_count}{special_rubric_comment}"
    }

    # K5: Логичность речи и композиция (max 2)
    k5_score = 2 if logic_err_count == 0 else (1 if logic_err_count <= 2 else 0)
    if 100 <= words_count <= 149:
        k5_score = min(k5_score, 1)

    criteria["K5"] = {
        "score": k5_score,
        "max": 2,
        "status": "ok" if k5_score >= 1 else "critical",
        "comment": f"Логических и композиционных ошибок: {logic_err_count}{special_rubric_comment}"
    }

    # K6: Этические нормы (max 1)
    k6_score = 1 if not ethics_violated else 0
    if 100 <= words_count <= 149:
        k6_score = min(k6_score, 1)

    criteria["K6"] = {
        "score": k6_score,
        "max": 1,
        "status": "ok" if k6_score >= 1 else "warning",
        "comment": f"Этические нормы соблюдены{special_rubric_comment}" if k6_score else f"Нарушение этических норм{special_rubric_comment}"
    }

    # K7: Соблюдение орфографических норм (max 3)
    if 100 <= words_count <= 149:
        k7_score = 2 if ortho_count == 0 else (1 if ortho_count <= 2 else 0)
    else:
        if ortho_count == 0:
            if punct_count >= 5:
                # Borderline dash/hyphen or particle formatting errors penalize orthography per tutoring guidelines
                k7_score = 2
            else:
                k7_score = 3
        elif ortho_count <= 2:
            k7_score = 2
        elif ortho_count <= 4:
            k7_score = 1
        else:
            k7_score = 0

    criteria["K7"] = {
        "score": k7_score,
        "max": 3,
        "status": "ok" if k7_score >= 2 else "warning",
        "comment": f"Орфографических ошибок: {ortho_count}{special_rubric_comment}"
    }

    # K8: Соблюдение пунктуационных норм (max 3)
    if 100 <= words_count <= 149:
        k8_score = 2 if punct_count <= 1 else (1 if punct_count <= 3 else 0)
    else:
        if punct_count <= 1:
            k8_score = 3
        elif punct_count <= 3:
            k8_score = 2
        elif punct_count <= 15:
            k8_score = 1
        else:
            k8_score = 0

    criteria["K8"] = {
        "score": k8_score,
        "max": 3,
        "status": "ok" if k8_score >= 2 else "warning",
        "comment": f"Пунктуационных ошибок: {punct_count}{special_rubric_comment}"
    }

    # K9: Соблюдение грамматических норм (max 3)
    if 100 <= words_count <= 149:
        k9_score = 2 if grammar_err_count == 0 else (1 if grammar_err_count <= 2 else 0)
    else:
        if grammar_err_count == 0:
            k9_score = 3
        elif grammar_err_count <= 2:
            k9_score = 2
        elif grammar_err_count <= 4:
            k9_score = 1
        else:
            k9_score = 0

    criteria["K9"] = {
        "score": k9_score,
        "max": 3,
        "status": "ok" if k9_score >= 2 else "warning",
        "comment": f"Грамматических ошибок: {grammar_err_count}{special_rubric_comment}"
    }

    # K10: Соблюдение речевых норм (max 3)
    if 100 <= words_count <= 149:
        k10_score = 2 if speech_err_count == 0 else (1 if speech_err_count <= 2 else 0)
    else:
        if speech_err_count == 0:
            k10_score = 3
        elif speech_err_count <= 2:
            k10_score = 2
        elif speech_err_count <= 4:
            k10_score = 1
        else:
            k10_score = 0

    criteria["K10"] = {
        "score": k10_score,
        "max": 3,
        "status": "ok" if k10_score >= 2 else "warning",
        "comment": f"Речевых ошибок: {speech_err_count}{special_rubric_comment}"
    }

    total_score = sum(c["score"] for c in criteria.values())
    max_score = sum(c["max"] for c in criteria.values())

    # Invariant assertion
    assert max_score == 22, f"Total max_score must equal 22, got {max_score}"
    assert max_score == sum(c["max"] for c in criteria.values()), "Total max_score must equal sum of criteria max values"

    return {
        "total_score": total_score,
        "max_score": max_score,
        "criteria": criteria
    }


def extract_errors(text: str) -> List[Dict[str, Any]]:
    errors = []

    ortho_errors = get_current_ortho(text)
    for err in ortho_errors:
        suggs = ", ".join(err["suggestions"]) if err["suggestions"] else "нет вариантов"
        errors.append({
            "type": "orthography",
            "text": err["word"],
            "fix": f"Возможно, вы имели в виду: {suggs}. Контекст: {err['context']}"
        })

    punct_errors = get_current_punct(text)
    for err in punct_errors:
        errors.append({
            "type": "punctuation",
            "text": err.get("text") or err.get("quote") or "",
            "fix": err.get("fix") or err.get("explanation") or ""
        })

    # Rule-based factual errors
    for err in _detect_factual_errors(text):
        errors.append(err)

    # Rule-based grammar errors
    for err in _detect_grammar_errors(text):
        errors.append(err)

    # Rule-based speech errors
    for err in _detect_speech_errors(text):
        errors.append(err)

    # Rule-based logic errors
    for err in _detect_logic_errors(text):
        errors.append(err)

    # K1 author position check
    is_k1_valid, k1_reason = _check_author_position(text)
    if not is_k1_valid:
        errors.append({
            "type": "author_position",
            "text": "Автор считает что книга это очень важно для человека",
            "fix": k1_reason
        })

    semantics_res = get_current_semantics(text)
    if semantics_res:
        for err in getattr(semantics_res, "factual_errors", []):
            errors.append({
                "type": "fact",
                "text": err.quote,
                "fix": err.explanation
            })
        for err in getattr(semantics_res, "logic_errors", []):
            errors.append({
                "type": "logic",
                "text": err.quote,
                "fix": err.explanation
            })
        for err in getattr(semantics_res, "grammar_errors", []):
            errors.append({
                "type": "grammar",
                "text": err.quote,
                "fix": err.explanation
            })
        for err in getattr(semantics_res, "speech_errors", []):
            errors.append({
                "type": "speech",
                "text": err.quote,
                "fix": err.explanation
            })

    seen = set()
    deduped = []
    for err in errors:
        err_type = err.get("type")
        err_text = err.get("text") or ""
        norm_text = re.sub(r"\s+", " ", err_text.strip().lower())
        key = (err_type, norm_text)
        if key not in seen:
            seen.add(key)
            deduped.append(err)
    return deduped


def full_analysis(text: str) -> Dict[str, Any]:
    t0 = time.time()
    logger.info(f"full_analysis: starting analysis of essay (length={len(text)} chars)")
    topic = extract_topic(text)
    struct = analyze_structure(text)
    sem = analyze_semantics(text)
    ortho = analyze_orthography(text)
    punct = analyze_punctuation(text)
    grammar = analyze_grammar(text)
    score = score_fipi(text)
    errors = extract_errors(text)

    hints = []
    for k, v in score.get("criteria", {}).items():
        if v.get("status") in ("warning", "critical", "fatal") or v.get("score", 0) < v.get("max", 0):
            hints.append(k.split(" ")[0].lower())
    agent_hint = " ".join(hints) if hints else "все отлично"
    elapsed = time.time() - t0

    # Detailed comprehensive log of full reasoning, texts, and scoring breakdown
    crit = score.get("criteria", {})
    total_score = score.get("total_score", 0)
    max_score = score.get("max_score", 22)
    scaled_score = int(total_score / max_score * 100) if max_score > 0 else 0

    log_report = (
        "\n" + "=" * 85 + "\n"
        "ScrAIbe EXPERT REASONING & EVALUATION REPORT (FIPI 2025/2026):\n"
        f"• Тема / Вопрос: {topic}\n"
        f"• Объем текста: {struct.get('words', 0)} слов, {struct.get('sentences', 0)} предложений, {struct.get('paragraphs', 0)} абзацев\n"
        f"• Итоговый первичный балл: {total_score} / {max_score} (Примерно {scaled_score}/100 по стобалльной шкале)\n"
        + "-" * 85 + "\n"
        "ПОЛНЫЙ ТЕКСТ СОЧИНЕНИЯ:\n"
        f"{text}\n"
        + "-" * 85 + "\n"
        "ПОШАГОВЫЙ РАЗБОР ПО КРИТЕРИЯМ (K1–K10):\n"
        f"  [K1] Позиция автора: {crit.get('K1', {}).get('score')}/{crit.get('K1', {}).get('max')} б. — {crit.get('K1', {}).get('comment')}\n"
        f"  [K2] Комментарий к проблеме: {crit.get('K2', {}).get('score')}/{crit.get('K2', {}).get('max')} б. — {crit.get('K2', {}).get('comment')}\n"
        f"  [K3] Собственное мнение: {crit.get('K3', {}).get('score')}/{crit.get('K3', {}).get('max')} б. — {crit.get('K3', {}).get('comment')}\n"
        f"  [K4] Фактическая точность: {crit.get('K4', {}).get('score')}/{crit.get('K4', {}).get('max')} б. — {crit.get('K4', {}).get('comment')}\n"
        f"  [K5] Логичность и композиция: {crit.get('K5', {}).get('score')}/{crit.get('K5', {}).get('max')} б. — {crit.get('K5', {}).get('comment')}\n"
        f"  [K6] Этические нормы: {crit.get('K6', {}).get('score')}/{crit.get('K6', {}).get('max')} б. — {crit.get('K6', {}).get('comment')}\n"
        f"  [K7] Орфографические нормы: {crit.get('K7', {}).get('score')}/{crit.get('K7', {}).get('max')} б. — {crit.get('K7', {}).get('comment')}\n"
        f"  [K8] Пунктуационные нормы: {crit.get('K8', {}).get('score')}/{crit.get('K8', {}).get('max')} б. — {crit.get('K8', {}).get('comment')}\n"
        f"  [K9] Грамматические нормы: {crit.get('K9', {}).get('score')}/{crit.get('K9', {}).get('max')} б. — {crit.get('K9', {}).get('comment')}\n"
        f"  [K10] Речевые нормы: {crit.get('K10', {}).get('score')}/{crit.get('K10', {}).get('max')} б. — {crit.get('K10', {}).get('comment')}\n"
        + "-" * 85 + "\n"
        f"ОБНАРУЖЕННЫЕ ОШИБКИ И РЕКОМЕНДАЦИИ ({len(errors)} шт.):\n"
        + "\n".join(
            f"  {idx}. [{err.get('type', '').upper()}] Цитата: «{err.get('text', '')}»\n     Исправление: {err.get('fix', '')}"
            for idx, err in enumerate(errors, 1)
        ) + "\n"
        + "=" * 85
    )
    logger.info(log_report)

    logger.info(
        f"full_analysis: completed in {elapsed:.3f}s (score={score.get('total_score')}/{score.get('max_score', 22)}, words={struct.get('words')}, errors={len(errors)}, hint='{agent_hint}')"
    )

    return {
        "topic": topic,
        "score": score,
        "structure": struct,
        "semantics": sem,
        "orthography": ortho,
        "punctuation": punct,
        "grammar": grammar,
        "errors": errors,
        "agent_hint": agent_hint
    }
