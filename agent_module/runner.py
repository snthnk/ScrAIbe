"""Running the compiled workflow with retries and model fallback — shared by the CLI and the Streamlit UI.

Previously this lived in ``main.py`` and ``app.py`` imported it from there. The fallback candidates also
used to be hard-coded Gemini names that were then passed to the Mistral client when LLM_PROVIDER=mistral.
"""

import re
import time
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage

from agent_module.graph import build_workflow
from agent_module.llms import build_main_llm, fallback_model_names, provider
from agent_module.messages import message_content
from agent_module.utils import setup_logger

logger = setup_logger("ege_ai_agent.runner")

_EMPTY_CONTENT = ("[]", "{}", "None", "null")
FALLBACK_ANSWER = "Извините, не удалось сформировать ответ. Попробуйте еще раз."


def extract_last_ai_message(messages: List[Any]) -> str:
    """Last non-empty assistant message; otherwise the last non-empty message of any role."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) or (
            isinstance(message, dict) and (message.get("type") or message.get("role")) in {"ai", "assistant"}
        ):
            content = message_content(message)
            if content.strip() and content.strip() not in _EMPTY_CONTENT:
                return content
    for message in reversed(messages):
        content = message_content(message)
        if content.strip() and content.strip() not in _EMPTY_CONTENT:
            return content
    return FALLBACK_ANSWER


def retry_delay_seconds(exc: Exception) -> Optional[float]:
    text = str(exc)
    for pattern in (r"Please retry in\s+([0-9.]+)s", r"'retryDelay':\s*'(\d+)s'"):
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def looks_like_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "Quota exceeded" in text or "429" in text


def looks_like_model_not_found(exc: Exception) -> bool:
    text = str(exc)
    return "NOT_FOUND" in text and "models/" in text


class WorkflowRunner:
    """Owns a compiled workflow; retries rate-limit errors and switches to a fallback model of the same provider."""

    def __init__(
        self,
        llm_factory: Callable[..., Any] = build_main_llm,
        workflow_factory: Callable[[Any], Any] = build_workflow,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._llm_factory = llm_factory
        self._workflow_factory = workflow_factory
        self._max_attempts = max_attempts
        self._sleep = sleep
        self.model_in_use: Optional[str] = None  # None = the configured default
        self.workflow = workflow_factory(llm_factory())

    def _switch_to_fallback_model(self) -> Optional[str]:
        for candidate in fallback_model_names(provider()):
            try:
                self.workflow = self._workflow_factory(self._llm_factory(model_name=candidate))
            except Exception as exc:
                logger.debug(f"Fallback model '{candidate}' could not be built: {exc}")
                continue
            self.model_in_use = candidate
            logger.info(f"Switched to fallback model '{candidate}'")
            return candidate
        return None

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        attempts = 0
        used_fallback = False
        last_exc: Optional[Exception] = None
        while attempts < self._max_attempts:
            try:
                logger.debug(f"Invoking workflow, attempt {attempts + 1}/{self._max_attempts}")
                return self.workflow.invoke(state)
            except Exception as exc:
                last_exc = exc
                delay = retry_delay_seconds(exc)
                if delay is not None and "PerDay" not in str(exc):
                    logger.warning(f"Retryable error ({exc}), sleeping {delay + 1}s")
                    self._sleep(delay + 1)
                    attempts += 1
                    continue
                if not used_fallback and (looks_like_quota_error(exc) or looks_like_model_not_found(exc)):
                    logger.warning(f"Quota or model-not-found error ({exc}). Trying a fallback model...")
                    used_fallback = True
                    if self._switch_to_fallback_model():
                        attempts += 1
                        continue
                break
        assert last_exc is not None
        logger.error(f"Workflow execution failed: {last_exc}")
        raise last_exc
