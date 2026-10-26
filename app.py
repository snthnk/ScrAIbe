"""
ScrAIbe — AI-наставник по подготовке к сочинению ЕГЭ (Задание 27).
Интерфейс на Streamlit:
- Проверка по критериям ФИПИ (K1–K10, шкала 2025/2026, макс 22 балла)
- Мгновенный экспресс-анализ NLP (орфография, пунктуация, логика, факты)
- Сократический диалог с наставником (LangGraph / Gemini / Mistral / офлайн-режим)
- Прозрачные логи рассуждений (Reasoning)
"""

import re
from typing import List

import streamlit as st
from langchain_core.messages import HumanMessage

from agent_module.constants import OFFICIAL_MAX
from agent_module.runner import WorkflowRunner, extract_last_ai_message
from agent_module.utils import setup_logger
from nlp_module.analyzer import TEST_ESSAY, TEST_TOPIC, full_analysis

logger = setup_logger("ege_ai_agent.ui")

LIVE_MODE_LABEL = "🚀 ИИ-наставник (основной)"
OFFLINE_MODE_LABEL = "🧪 Офлайн-режим (без LLM)"

# ==============================================================================
# Presets & Default Data
# ==============================================================================

PAUSTOVSKY_SOURCE_TEXT = (
    "(1) Человек должен жить в окружении природы, среди её вечной и благородной красоты. "
    "(2) Я помню один тихий осенний день в глухом мещёрском лесу. "
    "(3) Я ушёл туда ранним утром, чтобы побыть наедине с самим собой, отдохнуть от городской суеты и шума. "
    "(4) Лес встретил меня торжественной тишиной, прозрачным воздухом и запахом прелых листьев. "
    "(5) На полянах ещё цвели запоздалые синие колокольчики, а берёзы роняли на мхи золотые листья. "
    "(6) Я шёл медленно, разглядывая каждое дерево, и вдруг почувствовал, как тревога и раздражение, "
    "накопленные за долгие недели работы, уходят без следа. "
    "(7) В душе воцарились необыкновенный покой и тихая радость. "
    "(8) Природа обладает удивительной способностью лечить человеческую душу, возвращая человеку доброту, "
    "равновесие и ясность взгляда на мир. "
    "(9) Общение с природой делает нас чище и мягче."
)

PERFECT_ESSAY = (
    "Какова роль природы в жизни человека? Над этим вопросом размышляет К.Г. Паустовский.\n\n"
    "Автор раскрывает проблему на примере собственных переживаний. Писатель рассказывает, как он ранней "
    "осенью ушел в мещерский лес, чтобы побыть наедине со своими мыслями. В глуши он увидел золотые березы и "
    "поздние цветы, ощутил торжественную тишину. Этот пример показывает, что первозданная природа восхищает "
    "человека своей гармонией и красотой.\n\n"
    "Далее автор обращает внимание на душевное состояние героя: гуляя по осеннему лесу, он почувствовал, как "
    "тревога и раздражение исчезают, а в душе воцаряются покой и радость. Становится очевидно, что общение с "
    "природой исцеляет человека от городской суеты и душевной усталости.\n\n"
    "Оба примера-иллюстрации, дополняя друг друга, показывают: природа не просто радует взор человека, но и "
    "оказывает глубокое целебное воздействие на его внутреннее состояние.\n\n"
    "Позиция автора ясна: К.Г. Паустовский убежден, что природа лечит человеческую душу, возвращая ей спокойствие, "
    "доброту и душевное равновесие.\n\n"
    "Я полностью согласен с мнением автора. Действительно, живая природа дарит человеку силы и помогает преодолеть "
    "жизненные невзгоды. Например, в романе-эпопее Л.Н. Толстого «Война и мир» встреча с весенним возродившимся "
    "дубом помогает князю Андрею Болконскому оправиться от тяжелой душевной депрессии и вновь обрести веру в жизнь.\n\n"
    "Таким образом, природа — это наш общий дом и неиссякаемый источник душевных сил, который нужно беречь."
)

ASTAFYEV_TOPIC = "Какую роль играет книга в жизни человека?"
ASTAFYEV_SOURCE_TEXT = (
    "(1)Книга — это величайшее изобретение человечества. (2)Она хранит знания, накопленные веками, и передаёт их из поколения в поколение. "
    "(3)Без книги невозможно представить развитие цивилизации.\n"
    "(4)Чтение книги — это не просто механическое восприятие текста. (5)Это сложный процесс, в котором участвуют и разум, и чувства человека. "
    "(6)Читатель как бы ведёт диалог с автором, размышляет над его идеями, спорит или соглашается.\n"
    "(7)Хорошая книга способна изменить человека. (8)Она может вдохновить на подвиг, утешить в трудную минуту, заставить задуматься о смысле жизни. "
    "(9)Многие люди вспоминают книги, которые оказали влияние на их судьбу.\n"
    "(10)В наше время, когда вокруг так много информации, роль книги не уменьшается. (11)Напротив, она становится ещё более важной. "
    "(12)Книга учит отличать главное от второстепенного, помогает разобраться в сложном мире.\n(По В.П. Астафьеву)"
)
ASTAFYEV_FLAWED_ESSAY = (
    "Какую роль играет книга в жизни человека? Именно над этим вопросом размышляет В.П. Астафьев в предложенном тексте.\n"
    "Автор считает что книга это очень важно для человека. Она хранит знания и передаёт их людям. Без книги нельзя представить развитие цивилизации.\n"
    "Чтобы привлечь внимание читателя к поставленной проблеме В.П. Астафьев пишет о том что чтение книги это сложный процесс. "
    "В котором участвуют и разум и чувства человека. Читатель ведёт диалог с автором размышляет над его идеями.\n"
    "Далее автор отмечает что хорошая книга способна изменить человека. Она может вдохновить на подвиг утешить в трудную минуту "
    "заставить задуматься о смысле жизни. Многие люди вспоминают книги которые оказали влияние на их судьбу.\n"
    "Примеры дополняют друг друга. Сначала автор говорит о процессе чтения а потом о том как книга влияет на человека.\n"
    "Я согласен с позицией автора. Действительно книга играет важную роль в жизни. Мой друг читал книгу «Война и мир». "
    "После этого он стал лучше понимать людей и задумался о том как жить правильно.\n"
    "Таким образом книга играет важную роль в жизни человека."
)

CRITERIA_INFO = {
    "K1": {
        "title": "Позиция автора по проблеме",
        "max": 1,
        "desc": "Позиция автора исходного текста сформулирована чётко и соотносится с поставленной проблемой (при 0 баллов обнуляются K2 и K3).",
    },
    "K2": {
        "title": "Комментарий к сформулированной проблеме",
        "max": 3,
        "desc": "2 примера-иллюстрации из текста, пояснения к каждому примеру и анализ смысловой связи между ними.",
    },
    "K3": {
        "title": "Собственное мнение и аргументация",
        "max": 2,
        "desc": "Выражено собственное согласие/несогласие и приведено развёрнутое обоснование с литературным или жизненным примером.",
    },
    "K4": {
        "title": "Фактическая точность в фоновом материале",
        "max": 1,
        "desc": "Отсутствие фактических ошибок в понимании текста автора и фоновом материале (авторство, факты произведений).",
    },
    "K5": {
        "title": "Смысловая цельность, логика и композиция",
        "max": 2,
        "desc": "Логика изложения, отсутствие композиционных инверсий, формальных связок и смысловых нарушений.",
    },
    "K6": {
        "title": "Соблюдение этических норм",
        "max": 1,
        "desc": "Соблюдение этических норм речи, корректность формулировок и уважительное отношение.",
    },
    "K7": {
        "title": "Соблюдение орфографических норм",
        "max": 3,
        "desc": "Соблюдение орфографических норм (3 б. — 0 или 1 негрубая ошибка, 2 б. — 2 ошибки).",
    },
    "K8": {
        "title": "Соблюдение пунктуационных норм",
        "max": 3,
        "desc": "Соблюдение пунктуационных норм (3 б. — 0-1 ошибка, 2 б. — 2-3 ошибки, 1 б. — 4-15 ошибок).",
    },
    "K9": {
        "title": "Соблюдение грамматических норм",
        "max": 3,
        "desc": "Отсутствие грамматических ошибок (согласование, управление, парцелляция, построение предложений).",
    },
    "K10": {
        "title": "Соблюдение речевых норм",
        "max": 3,
        "desc": "Точность словоупотребления, отсутствие тавтологии, плеоназмов и лексических повторов.",
    },
}

# ==============================================================================
# Page Setup & Styling
# ==============================================================================

st.set_page_config(
    page_title="ScrAIbe — AI-наставник по сочинению ЕГЭ",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Main typography and headers */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.15s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F2937;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }

    /* Criteria badges */
    .badge-ok {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-warn {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-crit {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Reasoning box */
    .reasoning-box {
        background: #F9FAFB;
        border-left: 4px solid #3B82F6;
        border-radius: 4px 8px 8px 4px;
        padding: 0.75rem 1rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.88rem;
        color: #374151;
        margin-bottom: 0.8rem;
    }

    /* Chat message enhancements */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# State Initialization (Empty by default)
# ==============================================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "task_condition" not in st.session_state:
    st.session_state["task_condition"] = ""

if "source_text" not in st.session_state:
    st.session_state["source_text"] = ""

if "essay_text" not in st.session_state:
    st.session_state["essay_text"] = ""

if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

if "workflow_state" not in st.session_state:
    st.session_state["workflow_state"] = {
        "messages": [],
        "global_plan": [],
        "current_step": 0,
        "user_profile": {},
        "task_condition": "",
        "source_text": "",
        "essay_text": "",
        "essay_sent": False,
        "show_reasoning": True,
        "reasoning_log": [],
    }

if "history_reasoning" not in st.session_state:
    st.session_state["history_reasoning"] = []


# ==============================================================================
# Helper Functions
# ==============================================================================

def reset_session():
    logger.info("UI: reset_session called")
    st.session_state["messages"] = []
    st.session_state["analysis_result"] = None
    st.session_state["history_reasoning"] = []
    st.session_state["workflow_state"] = {
        "messages": [],
        "global_plan": [],
        "current_step": 0,
        "user_profile": {},
        "task_condition": st.session_state.get("task_condition", ""),
        "source_text": st.session_state.get("source_text", ""),
        "essay_text": st.session_state.get("essay_text", ""),
        "essay_sent": False,
        "show_reasoning": True,
        "reasoning_log": [],
    }


def load_preset(name: str):
    logger.info(f"UI: loading preset '{name}'")
    if name == "astafyev":
        st.session_state["task_condition"] = ASTAFYEV_TOPIC
        st.session_state["source_text"] = ASTAFYEV_SOURCE_TEXT
        st.session_state["essay_text"] = ASTAFYEV_FLAWED_ESSAY
    elif name == "flawed":
        st.session_state["task_condition"] = TEST_TOPIC
        st.session_state["source_text"] = PAUSTOVSKY_SOURCE_TEXT
        st.session_state["essay_text"] = TEST_ESSAY
    elif name == "perfect":
        st.session_state["task_condition"] = TEST_TOPIC
        st.session_state["source_text"] = PAUSTOVSKY_SOURCE_TEXT
        st.session_state["essay_text"] = PERFECT_ESSAY
    elif name == "empty":
        st.session_state["task_condition"] = ""
        st.session_state["source_text"] = ""
        st.session_state["essay_text"] = ""
    reset_session()


def perform_fast_analysis(essay_text: str):
    logger.info("=" * 80)
    logger.info(f"UI: perform_fast_analysis called (essay_len={len(essay_text)} chars)")
    with st.spinner("Анализируем текст сочинения по критериям ФИПИ..."):
        result = full_analysis(essay_text)
        st.session_state["analysis_result"] = result

    score_data = result.get("score", {})
    total_score = score_data.get("total_score", 0)
    max_score = score_data.get("max_score", 22)
    crit = score_data.get("criteria", {})
    errors = result.get("errors", [])

    logger.info(f"UI: fast analysis completed. TOTAL SCORE: {total_score}/{max_score}")
    logger.info(f"UI: Criteria breakdown: {crit}")
    logger.info(f"UI: Detected {len(errors)} error(s):")
    for idx, err in enumerate(errors, 1):
        logger.info(f"  {idx}. [{err.get('type')}] {err.get('text')} -> {err.get('fix')}")
    logger.info("=" * 80)
    return result


def offline_tutor_reply(user_msg: str, essay_text: str, source_text: str) -> tuple[str, List[str]]:
    """Офлайн-режим без LLM: сократический ответ строится только по результату детерминированного NLP-анализа.

    Никаких заготовок под конкретное сочинение: вопрос ученику зависит от первой найденной ошибки и от
    слабейшего критерия. Для полноценного диалога (инструменты, RAG, поиск) нужен режим «ИИ-наставник».
    """
    user_lower = user_msg.lower()
    reasoning = ["Офлайн-режим: LLM не используется, ответ строится по результатам NLP-анализа."]

    analysis = full_analysis(essay_text) if essay_text.strip() else {}
    errors = analysis.get("errors", [])
    criteria = analysis.get("score", {}).get("criteria", {})
    weak = sorted(
        ((cid, int(v)) for cid, v in criteria.items() if isinstance(v, (int, float)) and v < OFFICIAL_MAX.get(cid, 0)),
        key=lambda item: item[1] - OFFICIAL_MAX.get(item[0], 0),
    )
    asks_for_corrections = any(w in user_lower for w in ["исправ", "предложи", "как лучше", "что изменить", "перепиши", "помоги"])
    first_message = not st.session_state["workflow_state"].get("essay_sent")

    if not essay_text.strip():
        reasoning.append("Текст сочинения пуст.")
        return "Вставь текст сочинения в поле выше — тогда я смогу его разобрать.", reasoning

    if first_message or asks_for_corrections:
        if errors:
            err = errors[0]
            reasoning.append(f"Первая найденная ошибка: тип «{err.get('type')}».")
            reply = (
                "Я прочитал работу. Начнём с фрагмента:\n\n"
                f"> *«{err.get('text', '')}»*\n\n"
                f"💡 {err.get('fix', '')}\n\n"
                "Как бы ты переформулировал его, чтобы не потерять баллы?"
            )
            if asks_for_corrections and len(errors) > 1:
                rest = "\n".join(f"* {e.get('text', '')} → {e.get('fix', '')}" for e in errors[1:6])
                reply += f"\n\nЕщё замечания:\n{rest}"
        elif weak:
            cid, value = weak[0]
            reasoning.append(f"Грубых ошибок нет; слабейший критерий — {cid} ({value}/{OFFICIAL_MAX.get(cid)}).")
            reply = f"Явных ошибок я не вижу, но по критерию **{cid}** набрано {value} из {OFFICIAL_MAX.get(cid)}. Что, по-твоему, можно усилить?"
        else:
            reasoning.append("Ошибок не найдено, все критерии на максимуме.")
            reply = "По результатам автоматического анализа работа выглядит сильной. Хочешь проверить конкретный критерий?"
        return reply, reasoning

    reasoning.append("Общий ответ: напоминание структуры задания 27.")
    reply = (
        f"Я услышал твой ответ: *«{user_msg}»*.\n\n"
        "Помни структуру задания 27: проблема → два примера с пояснениями → связка и её анализ → "
        "позиция автора → твоя позиция с обоснованием → вывод. "
        "Пришли исправленный фрагмент или нажми **«Экспресс-анализ ФИПИ»** для полной проверки."
    )
    return reply, reasoning


@st.cache_resource(show_spinner=False)
def get_runner() -> WorkflowRunner:
    return WorkflowRunner()


def _log_reply(tag: str, reasoning: List[str], reply: str) -> None:
    logger.info(f"UI: [{tag} REASONING] ({len(reasoning)} steps):")
    for step in reasoning:
        logger.info(f"  • {step}")
    logger.info(f"UI: [{tag} TUTOR REPLY]:\n{reply}")
    logger.info("=" * 80)


def execute_agent_step(user_input: str, is_demo: bool) -> tuple[str, List[str]]:
    """Один шаг диалога: LangGraph-агент с LLM из .env либо офлайн-режим без LLM."""
    mode_str = "OFFLINE" if is_demo else "LIVE_LLM"
    logger.info("=" * 80)
    logger.info(f"UI DIALOGUE STEP START | MODE={mode_str} | INPUT_LEN={len(user_input)}")
    logger.info(f"UI: [USER MESSAGE]: {user_input}")
    logger.info("=" * 80)

    essay, source = st.session_state["essay_text"], st.session_state["source_text"]
    if is_demo:
        reply, reasoning = offline_tutor_reply(user_input, essay, source)
        st.session_state["workflow_state"]["essay_sent"] = True
        _log_reply("OFFLINE", reasoning, reply)
        return reply, reasoning

    try:
        runner = get_runner()
    except Exception as exc:
        logger.warning(f"Failed to build live LLM ({exc}). Falling back to offline mode.")
        reply, reasoning = offline_tutor_reply(user_input, essay, source)
        reasoning.insert(0, f"⚠️ Не удалось инициализировать LLM из .env ({exc}). Включён офлайн-режим.")
        _log_reply("FALLBACK", reasoning, reply)
        return reply, reasoning

    state = st.session_state["workflow_state"]
    state["task_condition"] = st.session_state["task_condition"]
    state["source_text"] = source
    state["essay_text"] = essay
    state["show_reasoning"] = True

    if not state.get("essay_sent"):
        first_content = f"Сочинение: {essay}"
        if source:
            first_content = f"Исходный текст: {source}\n\n" + first_content
        state["messages"] = state.get("messages", []) + [HumanMessage(content=first_content)]
        state["essay_sent"] = True
    state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]

    try:
        result = runner.invoke(state)
    except Exception as exc:
        logger.error(f"Live workflow invocation error: {exc}")
        reply, reasoning = offline_tutor_reply(user_input, essay, source)
        reasoning.insert(0, f"⚠️ Ошибка вызова API ({exc}). Ответ сформирован офлайн-модулем.")
        _log_reply("ERROR-FALLBACK", reasoning, reply)
        return reply, reasoning

    state.update(result)
    reasoning_logs = list(state.get("reasoning_log", []))
    state["reasoning_log"] = []
    answer = extract_last_ai_message(state.get("messages", []))
    logger.info(f"UI: live workflow completed (answer_len={len(answer)}, reasoning_items={len(reasoning_logs)})")
    _log_reply("LIVE", reasoning_logs, answer)
    return answer, reasoning_logs


# ==============================================================================
# Sidebar (User-facing only)
# ==============================================================================

with st.sidebar:
    st.markdown("## ✍️ **ScrAIbe**")
    st.caption("AI-наставник по подготовке к сочинению ЕГЭ (Задание 27)")
    st.divider()

    st.markdown("### ⚙️ Режим наставника")
    mode_option = st.radio(
        "Выберите режим:",
        [LIVE_MODE_LABEL, OFFLINE_MODE_LABEL],
        index=0,
        help="Основной режим работает через модель, настроенную в конфигурации (.env). Офлайн-режим работает без LLM и без расхода токенов: только детерминированный NLP-анализ.",
    )
    is_demo_mode = mode_option == OFFLINE_MODE_LABEL

    st.divider()
    st.markdown("### 🧠 Настройки диалога")
    show_reasoning_toggle = st.checkbox(
        "Показывать ход мыслей наставника (Reasoning)",
        value=True,
        help="Отображает пошаговые размышления агента перед генерацией ответа.",
    )

    st.divider()
    st.markdown("### 📚 Примеры для проверки")
    st.caption("Вы можете загрузить готовый пример или ввести своё сочинение:")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("📖 О книге (ошибки)", use_container_width=True, help="Загрузить сочинение по Астафьеву с ошибками K1, K2, K8 (10/22 баллов)"):
            load_preset("astafyev")
            st.rerun()
    with col_p2:
        if st.button("📝 О природе (с ошибками)", use_container_width=True, help="Загрузить учебное сочинение по Паустовскому с ошибками"):
            load_preset("flawed")
            st.rerun()

    if st.button("🏆 Идеальное сочинение (22 балла)", use_container_width=True, help="Загрузить эталонное сочинение на 22 балла"):
        load_preset("perfect")
        st.rerun()

    if st.button("🗑️ Очистить поля", use_container_width=True):
        load_preset("empty")
        st.rerun()

    st.divider()
    if st.button("🔄 Сбросить сессию диалога", use_container_width=True):
        reset_session()
        st.rerun()


# ==============================================================================
# Header Section
# ==============================================================================

st.markdown('<div class="hero-title">ScrAIbe ✍️ | Экспертный наставник сочинения ЕГЭ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Интеллектуальная проверка по критериям ФИПИ 2025/2026 (K1–K10, макс. 22 балла), '
    'мгновенный поиск ошибок и сократический диалог с наставником.</div>',
    unsafe_allow_html=True,
)


# ==============================================================================
# Input Section
# ==============================================================================

with st.expander("📌 **Тема и исходный текст автора**", expanded=True):
    col_top1, col_top2 = st.columns([1, 1])
    with col_top1:
        topic_val = st.text_input(
            "Тема или проблемный вопрос (task_condition):",
            value=st.session_state["task_condition"],
            placeholder="Например: Какова роль природы в жизни человека?",
        )
        st.session_state["task_condition"] = topic_val

    with col_top2:
        st.caption("📖 Исходный текст автора необходим для верификации цитат и критерия K2.")
        has_source = bool(st.session_state["source_text"].strip())
        status_text = "✅ Исходный текст загружен" if has_source else "⚪ Текст не указан (опционально)"
        st.write(f"**Статус:** {status_text}")

    source_val = st.text_area(
        "Исходный текст автора (source_text):",
        value=st.session_state["source_text"],
        height=130,
        placeholder="Вставьте исходный текст автора из КИМ для проверки критерия K2 (необязательно)...",
    )
    st.session_state["source_text"] = source_val

# Essay text area
essay_val = st.text_area(
    "📝 **Текст сочинения ученика:**",
    value=st.session_state["essay_text"],
    height=240,
    placeholder="Вставьте сюда текст вашего сочинения для проверки...",
)
st.session_state["essay_text"] = essay_val

# Text metrics & word count feedback
words_count = len(re.findall(r"\b\w+(?:-\w+)*\b", essay_val)) if essay_val.strip() else 0
chars_count = len(essay_val)
paragraphs_count = len([p for p in essay_val.split("\n") if p.strip()])

if not essay_val.strip():
    word_badge = '<span style="color: #6B7280;">Поле сочинения пока пусто</span>'
elif words_count < 100:
    word_badge = f'<span class="badge-crit">⚠️ {words_count} слов (Менее 100 слов — 0 баллов по ФИПИ)</span>'
elif words_count < 150:
    word_badge = f'<span class="badge-warn">🟡 {words_count} слов (100–149 слов — применяются понижающие критерии)</span>'
else:
    word_badge = f'<span class="badge-ok">🟢 {words_count} слов (Рекомендуемый объём &ge; 150)</span>'

st.markdown(
    f"Объём текста: {word_badge} &nbsp;|&nbsp; "
    f"Символов: **{chars_count}** &nbsp;|&nbsp; "
    f"Абзацев: **{paragraphs_count}**",
    unsafe_allow_html=True,
)

# Action buttons
st.write("")
col_btn1, col_btn2, col_spacer = st.columns([1.2, 1.4, 2])
with col_btn1:
    btn_analyze = st.button("📊 **Экспресс-анализ ФИПИ**", type="primary", use_container_width=True)
with col_btn2:
    btn_start_dialog = st.button("💬 **Отправить наставнику в диалог**", use_container_width=True)

if btn_analyze:
    if not essay_val.strip():
        st.warning("Пожалуйста, введите текст сочинения для анализа.")
    else:
        perform_fast_analysis(essay_val)

if btn_start_dialog:
    if not essay_val.strip():
        st.warning("Пожалуйста, введите текст сочинения перед началом диалога.")
    else:
        perform_fast_analysis(essay_val)
        with st.spinner("Наставник изучает ваше сочинение..."):
            reply, reasoning = execute_agent_step(
                user_input="Здравствуйте! Я написал сочинение и хочу его проверить.",
                is_demo=is_demo_mode,
            )
            st.session_state["messages"].append({"role": "user", "content": "Здравствуйте! Проверьте, пожалуйста, моё сочинение."})
            st.session_state["messages"].append({"role": "assistant", "content": reply, "reasoning": reasoning})
        st.rerun()

st.divider()

# ==============================================================================
# Tabs: Chat, Scoring, Errors, Handbook
# ==============================================================================

tab_chat, tab_score, tab_errors, tab_guide = st.tabs([
    "💬 **Диалог с наставником**",
    "📊 **Критерии ФИПИ (K1–K10)**",
    "🔍 **Детальный анализ ошибок**",
    "📖 **Справка и правила ЕГЭ**",
])

# ------------------------------------------------------------------------------
# TAB 1: Chat with Socratic Tutor
# ------------------------------------------------------------------------------
with tab_chat:
    st.markdown("### 💬 Сократический диалог с наставником")
    st.caption("Наставник не переписывает текст за вас, а помогает обнаружить слабые места и довести сочинение до максимума.")

    # Display chat history
    if not st.session_state["messages"]:
        st.info("💡 Введите сочинение выше и нажмите кнопку **«Отправить наставнику в диалог»**, чтобы начать диалог.")
    else:
        for msg in st.session_state["messages"]:
            role = msg["role"]
            content = msg["content"]
            with st.chat_message(role, avatar="🤖" if role == "assistant" else "👤"):
                if role == "assistant" and show_reasoning_toggle and msg.get("reasoning"):
                    with st.expander("🧠 **Ход мыслей наставника (Reasoning)**", expanded=False):
                        for step in msg["reasoning"]:
                            st.markdown(f"• `{step}`")
                st.markdown(content)

    # Chat input box
    chat_prompt = st.chat_input("Напишите ответ наставнику или задайте вопрос...")
    if chat_prompt:
        if not st.session_state["essay_text"].strip():
            st.error("Текст сочинения пуст! Введите текст сочинения в блоке выше.")
        else:
            st.session_state["messages"].append({"role": "user", "content": chat_prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(chat_prompt)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Наставник думает над ответом..."):
                    reply, reasoning = execute_agent_step(
                        user_input=chat_prompt,
                        is_demo=is_demo_mode,
                    )
                    if show_reasoning_toggle and reasoning:
                        with st.expander("🧠 **Ход мыслей наставника (Reasoning)**", expanded=False):
                            for step in reasoning:
                                st.markdown(f"• `{step}`")
                    st.markdown(reply)

            st.session_state["messages"].append({"role": "assistant", "content": reply, "reasoning": reasoning})
            st.rerun()

# ------------------------------------------------------------------------------
# TAB 2: Scoring K1–K10
# ------------------------------------------------------------------------------
with tab_score:
    st.markdown("### 📊 Оценка по 10 критериям ФИПИ (2025/2026)")

    analysis = st.session_state.get("analysis_result")
    if analysis:
        score_data = analysis.get("score", {})
        total = score_data.get("total_score", 0)
        max_s = score_data.get("max_score", 22)
        criteria = score_data.get("criteria", {})
        percentage = int((total / max_s) * 100) if max_s > 0 else 0

        # Score Summary Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{total} / {max_s}</div>'
                f'<div class="metric-label">Первичных баллов ({percentage}%)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{words_count}</div>'
                f'<div class="metric-label">Слов в сочинении</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c3:
            errors_cnt = len(analysis.get("errors", []))
            err_color = "#10B981" if errors_cnt == 0 else "#EF4444"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color: {err_color};">{errors_cnt}</div>'
                f'<div class="metric-label">Обнаружено замечаний</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.progress(percentage / 100)
        st.write("")

        # Criteria Grid
        st.markdown("#### Детализация по каждому критерию:")
        for code, info in CRITERIA_INFO.items():
            c_data = criteria.get(code, {})
            c_score = c_data.get("score", 0)
            c_max = c_data.get("max", info["max"])
            c_comment = c_data.get("comment", "")
            c_status = c_data.get("status", "ok")

            if c_status == "ok":
                badge_html = f'<span class="badge-ok">🟢 {c_score} / {c_max} б.</span>'
            elif c_status == "warning":
                badge_html = f'<span class="badge-warn">🟡 {c_score} / {c_max} б.</span>'
            else:
                badge_html = f'<span class="badge-crit">🔴 {c_score} / {c_max} б.</span>'

            with st.expander(f"**{code}**: {info['title']} — {c_score}/{c_max} б.", expanded=(c_status != "ok")):
                col_info, col_status = st.columns([3, 1])
                with col_info:
                    st.write(f"**Описание:** {info['desc']}")
                    if c_comment:
                        st.info(f"**Вердикт:** {c_comment}")
                with col_status:
                    st.markdown(f"**Балл:** {badge_html}", unsafe_allow_html=True)
    else:
        st.info("Введите текст сочинения выше и нажмите кнопку **«📊 Экспресс-анализ ФИПИ»**, чтобы рассчитать баллы.")

# ------------------------------------------------------------------------------
# TAB 3: Errors Breakdown
# ------------------------------------------------------------------------------
with tab_errors:
    st.markdown("### 🔍 Карта замечаний и структура текста")

    analysis = st.session_state.get("analysis_result")
    if analysis:
        errors = analysis.get("errors", [])
        struct = analysis.get("structure", {})

        col_err, col_str = st.columns([3, 2])

        with col_err:
            st.markdown("#### 🚨 Найденные ошибки и неточности:")
            if not errors:
                st.success("🎉 Грубых орфографических или фактических ошибок не обнаружено!")
            else:
                for idx, err in enumerate(errors, 1):
                    err_type = err.get("type", "Замечание")
                    err_text = err.get("text", "")
                    err_msg = err.get("fix") or err.get("message") or err.get("explanation") or ""
                    st.warning(f"**#{idx} [{err_type.upper()}]** {err_text}\n\n{err_msg}")

            # Orthography & Punctuation detail
            ortho_data = analysis.get("orthography", {})
            ortho_list = ortho_data.get("errors", []) if isinstance(ortho_data, dict) else (ortho_data if isinstance(ortho_data, list) else [])
            if ortho_list:
                st.markdown("#### 🔤 Орфографические замечания:")
                for item in ortho_list:
                    if isinstance(item, dict):
                        word = item.get('word', '')
                        suggs = item.get('suggestions') or item.get('s') or []
                        st.write(f"• Слово: `{word}` $\\to$ Варианты: `{', '.join(suggs)}`")

        with col_str:
            st.markdown("#### 📐 Структурный состав:")
            st.markdown(f"• **Предложений:** {struct.get('sentences', 0)}")
            st.markdown(f"• **Слов:** {struct.get('words', 0)}")
            st.markdown(f"• **Абзацев:** {struct.get('paragraphs', 0)}")

            topic_val = analysis.get("topic")
            if isinstance(topic_val, dict):
                detected_prob = topic_val.get("detected_problem", "Не определена")
                kw_str = ", ".join(topic_val.get("keywords", []))
            elif isinstance(topic_val, str) and topic_val.strip():
                detected_prob = topic_val.strip()
                kw_str = "извлечена из текста"
            else:
                detected_prob = "Не определена"
                kw_str = "—"
            st.markdown(f"• **Определённая проблема:** *{detected_prob}*")
            st.markdown(f"• **Ключевые слова темы:** `{kw_str}`")

            st.divider()
            st.markdown("#### 💡 Подсказка для агента:")
            st.caption(f"Фокус доработки: `{analysis.get('agent_hint', 'все отлично')}`")
    else:
        st.info("Введите текст сочинения выше и нажмите кнопку **«📊 Экспресс-анализ ФИПИ»**, чтобы увидеть полный отчёт по ошибкам.")

# ------------------------------------------------------------------------------
# TAB 4: Handbook & Guidelines
# ------------------------------------------------------------------------------
with tab_guide:
    st.markdown("### 📖 Памятка по написанию сочинения ЕГЭ (Задание 27)")
    st.markdown(
        """
        В 2025/2026 году максимальный первичный балл за сочинение составляет **22 балла**.

        #### 🏗️ Идеальная 7-абзацная композиция:
        1. **Вступление и формулировка проблемы (K1)**:
           - Проблемный вопрос или тезис («Какова роль природы в жизни человека? Над этим вопросом размышляет...»).
        2. **Первый пример-иллюстрация из текста (K2)**:
           - Цитата или сжатый пересказ фрагмента + обязательное **пояснение** («Этот пример убедительно доказывает, что...»).
        3. **Второй пример-иллюстрация из текста (K2)**:
           - Второй эпизод текста + **пояснение** («Писатель подчеркивает...»).
        4. **Смысловая связь между примерами и её анализ (K2)**:
           - Обязательно укажите вид связи: *дополнение*, *противопоставление*, *причинно-следственная связь*.
           - Обязательно проанализируйте связь: как оба примера вместе раскрывают мысль автора.
        5. **Позиция автора исходного текста (K3)**:
           - Четкий ответ на вопрос, поставленный в K1.
        6. **Своя позиция и обоснование (K4)**:
           - Согласие/несогласие с автором.
           - Обоснование: литературный пример, исторический факт, научный факт или жизненный опыт с размышлением.
        7. **Заключение**:
           - Обобщающий вывод по всей работе.

        #### ⚠️ Частые ловушки:
        - **Фактическая ошибка в аргументе (K10/K12)**: например, назвать Пушкина автором «Муму» вместо Тургенева.
        - **Формальная связка без анализа (K2)**: просто написать «примеры дополняют друг друга», не объяснив, в чём именно заключается дополнение.
        - **Объём менее 150 слов**: работа проверяется со снижением максимальных баллов по грамотности, а при объеме менее 100 слов ставится **0 баллов** за всё сочинение.
        """
    )

# Footer
st.divider()
st.caption("ScrAIbe © 2026 | Разработано с использованием LangGraph, LangChain, Streamlit и современных NLP-алгоритмов.")
