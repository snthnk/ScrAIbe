"""LLM factory: VseLLM (OpenAI-compatible gateway), Mistral or Gemini as the primary model.

Environment (see ``.env.example``):
    LLM_PROVIDER              vsellm | mistral | gemini
    VSELLM_MODEL_NAME / MISTRAL_MODEL_NAME / GEMINI_MODEL_NAME / MODEL_NAME
    VSELLM_EXPERT_MODEL_NAME / MISTRAL_EXPERT_MODEL_NAME / EXPERT_MODEL_NAME   model for ``expert_reasoning_tool`` and semantic NLP analysis
    VSELLM_API_KEY, VSELLM_BASE_URL, MISTRAL_API_KEY, GOOGLE_API_KEY or GEMINI_API_KEY
    LLM_TEMPERATURE           optional; for vsellm it is sent only when set (some models reject a custom temperature)
    LLM_FALLBACKS=0           disable the fallback chain
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv

from agent_module.utils import setup_logger

logger = setup_logger("ege_ai_agent.llm")

PROVIDER_MISTRAL = "mistral"
PROVIDER_GEMINI = "gemini"
PROVIDER_VSELLM = "vsellm"

# The model name on VseLLM is whatever the gateway lists for the account; set VSELLM_MODEL_NAME in .env.
DEFAULT_MODELS = {PROVIDER_MISTRAL: "ministral-8b-latest", PROVIDER_GEMINI: "gemini-3.6-flash", PROVIDER_VSELLM: "claude-haiku-5.5"}
DEFAULT_VSELLM_BASE_URL = "https://api.vsellm.ru/v1"
# Tried (in order) when the chosen model of this provider is rate-limited or unavailable.
MISTRAL_FALLBACK_MODELS: Tuple[str, ...] = ("ministral-8b-latest", "open-mistral-7b")
GEMINI_FALLBACK_MODELS: Tuple[str, ...] = ("gemini-2.5-flash", "gemini-2.0-flash")
TEMPERATURE = 0.2

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _is_placeholder(value: str) -> bool:
    """Значение из .env.example («your_..._here») — это не ключ."""
    value = (value or "").strip().lower()
    return not value or value.startswith("your_") or value.endswith("_here")


def fallbacks_enabled() -> bool:
    """LLM_FALLBACKS=0 отключает запасные модели (нужно для оценки: иначе непонятно, какая модель ответила)."""
    return os.getenv("LLM_FALLBACKS", "1").strip().lower() not in {"0", "false", "no", "off"}


def provider() -> str:
    return (os.getenv("LLM_PROVIDER") or PROVIDER_GEMINI).strip().lower()


def _load_env() -> None:
    # override=False: a key exported in the shell / CI wins over a stale value in .env.
    load_dotenv(dotenv_path=_ENV_PATH, override=False)


def resolve_model_name(provider_name: str, override: Optional[str] = None, expert: bool = False) -> str:
    if override:
        return override
    if expert:
        value = os.getenv(f"{provider_name.upper()}_EXPERT_MODEL_NAME") or os.getenv("EXPERT_MODEL_NAME")
        if value:
            return value
    value = os.getenv(f"{provider_name.upper()}_MODEL_NAME") or os.getenv("MODEL_NAME")
    return value or DEFAULT_MODELS.get(provider_name, DEFAULT_MODELS[PROVIDER_GEMINI])


def fallback_model_names(provider_name: str) -> List[str]:
    """Model names of the *same provider* to retry with. Never mixes providers (a Gemini name sent to Mistral fails)."""
    if provider_name == PROVIDER_VSELLM:
        models: Tuple[str, ...] = ()  # the gateway's model ids are account-specific: only FALLBACK_MODEL_NAME
    else:
        models = MISTRAL_FALLBACK_MODELS if provider_name == PROVIDER_MISTRAL else GEMINI_FALLBACK_MODELS
    extra = os.getenv("FALLBACK_MODEL_NAME")
    return ([extra] if extra else []) + list(models)


def _make_mistral(model: str, api_key: str) -> Any:
    try:
        from langchain_mistralai import ChatMistralAI
    except Exception as exc:
        raise ImportError("Для провайдера mistral нужен пакет langchain-mistralai") from exc
    return ChatMistralAI(model=model, mistral_api_key=api_key, temperature=TEMPERATURE)


def _make_gemini(model: str, api_key: str) -> Any:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as exc:
        raise ImportError("Для провайдера gemini нужен пакет langchain-google-genai") from exc
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=TEMPERATURE)


def _make_vsellm(model: str, api_key: str) -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        raise ImportError("Для провайдера vsellm нужен пакет langchain-openai") from exc
    base_url = (os.getenv("VSELLM_BASE_URL") or "").strip() or DEFAULT_VSELLM_BASE_URL
    kwargs: dict = {"model": model, "api_key": api_key, "base_url": base_url, "max_retries": 2, "timeout": 120}
    temperature = (os.getenv("LLM_TEMPERATURE") or "").strip()
    if temperature:
        kwargs["temperature"] = float(temperature)
    return ChatOpenAI(**kwargs)


def _fallback_chain(provider_name: str, chosen: str, gemini_key: str, mistral_key: str) -> List[Any]:
    fallbacks: List[Any] = []
    if provider_name == PROVIDER_VSELLM:
        return fallbacks
    if provider_name == PROVIDER_MISTRAL:
        fallbacks += [_make_mistral(m, mistral_key) for m in MISTRAL_FALLBACK_MODELS if m != chosen]
        if gemini_key:
            fallbacks.append(_make_gemini(DEFAULT_MODELS[PROVIDER_GEMINI], gemini_key))
    elif mistral_key:
        try:
            fallbacks += [_make_mistral(m, mistral_key) for m in MISTRAL_FALLBACK_MODELS]
        except ImportError:
            logger.debug("langchain-mistralai is not installed: no Mistral fallbacks")
    return fallbacks


def build_llm(model_override: Optional[str] = None, expert: bool = False) -> Any:
    _load_env()
    provider_name = provider()
    if provider_name not in DEFAULT_MODELS:
        raise ValueError(f"Неизвестный LLM_PROVIDER={provider_name!r}; допустимо: {', '.join(DEFAULT_MODELS)}")
    chosen = resolve_model_name(provider_name, model_override, expert=expert)
    role = "expert" if expert else "main"
    logger.info(f"Building {role} LLM: provider='{provider_name}', model='{chosen}'")

    gemini_key = next((v for v in (os.getenv("GOOGLE_API_KEY"), os.getenv("GEMINI_API_KEY")) if not _is_placeholder(v or "")), "")
    mistral_key = os.getenv("MISTRAL_API_KEY") or ""
    if _is_placeholder(mistral_key):
        mistral_key = ""

    vsellm_key = os.getenv("VSELLM_API_KEY") or ""
    if _is_placeholder(vsellm_key):
        vsellm_key = ""

    if provider_name == PROVIDER_VSELLM:
        if not vsellm_key:
            logger.error("VSELLM_API_KEY не задан")
            raise ValueError("API key required for VseLLM (VSELLM_API_KEY)")
        primary = _make_vsellm(chosen, vsellm_key)
    elif provider_name == PROVIDER_MISTRAL:
        if not mistral_key:
            logger.error("MISTRAL_API_KEY не задан")
            raise ValueError("API key required for Mistral")
        primary = _make_mistral(chosen, mistral_key)
    else:
        if not gemini_key:
            logger.error("GOOGLE_API_KEY или GEMINI_API_KEY не задан")
            raise ValueError("API key required for Gemini")
        primary = _make_gemini(chosen, gemini_key)

    fallbacks = _fallback_chain(provider_name, chosen, gemini_key, mistral_key) if fallbacks_enabled() else []
    logger.debug(f"Initialized {provider_name} {role} LLM(model={chosen}) with {len(fallbacks)} fallbacks")
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def build_main_llm(model_name: Optional[str] = None) -> Any:
    return build_llm(model_name, expert=False)


def build_expert_llm(model_name: Optional[str] = None) -> Any:
    return build_llm(model_name, expert=True)
