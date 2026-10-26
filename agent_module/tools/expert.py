from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from agent_module.constants import normalize_criterion_id
from agent_module.llms import build_expert_llm
from agent_module.models import ExpertReasoningInput
from agent_module.prompts import EXPERT_REASONING_SYSTEM_PROMPT
from agent_module.utils import setup_logger


logger = setup_logger()


@tool
def expert_reasoning_tool(payload: Dict[str, Any]) -> str:
    """LLM-инструмент для экспертного разбора по критерию ЕГЭ."""
    data = ExpertReasoningInput.model_validate(payload)
    criterion = normalize_criterion_id(data.criterion) or data.criterion.strip()
    fragment = data.text.strip() if data.text else ""
    essay_text = data.essay_text.strip() if data.essay_text else ""
    source_text = data.source_text.strip() if data.source_text else ""
    source_text_provided = data.source_text_provided
    if source_text_provided is None:
        source_text_provided = bool(source_text)

    if not criterion or (not fragment and not essay_text):
        return "Недостаточно данных: укажите criterion и text или essay_text."
    llm = build_expert_llm()
    logger.info("Tool: expert_reasoning_tool")
    logger.debug(f"expert_reasoning_tool: criterion='{criterion}', source_provided={source_text_provided}, fragment_len={len(fragment)}, essay_len={len(essay_text)}")
    system_prompt = EXPERT_REASONING_SYSTEM_PROMPT

    if criterion == "K2" and not source_text_provided:
        logger.warning("expert_reasoning_tool: K2 evaluated without source text! Guardrail activated.")
        system_prompt += (
            "\n\nВНИМАНИЕ (ВАЖНЫЙ GUARD): Исходный текст для анализа К2 не предоставлен. "
            "Вы ОБЯЗАНЫ явно написать в начале ответа фразу: «исходный текст для анализа К2 не предоставлен». "
            "Анализируйте исключительно предоставленный текст сочинения. "
            "Категорически запрещены выдуманные цитаты из исходного текста, гипотетические эпизоды и "
            "фразы вида «например, автор описывает…»."
        )

    user_prompt_parts = [f"Критерий: {criterion}"]
    if fragment:
        user_prompt_parts.append(f"Фрагмент сочинения для анализа: {fragment}")
    if essay_text:
        user_prompt_parts.append(f"Полный текст сочинения: {essay_text}")
    if source_text:
        user_prompt_parts.append(f"Исходный текст: {source_text}")
    if data.task_condition:
        user_prompt_parts.append(f"Условие задания: {data.task_condition}")
    if data.focus:
        user_prompt_parts.append(f"Фокус разбора (текущий шаг): {data.focus}")
    user_prompt_parts.append("Дай разбор и рекомендации.")

    user_prompt = "\n".join(user_prompt_parts)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    content = response.content if hasattr(response, "content") else str(response)
    logger.debug(f"expert_reasoning_tool completed: output_length={len(content)} chars")
    return content
