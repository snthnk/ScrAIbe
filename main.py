"""CLI for the ScrAIbe tutor: ``python main.py``."""

import os

from langchain_core.messages import HumanMessage

from agent_module.runner import WorkflowRunner, extract_last_ai_message
from agent_module.utils import setup_logger
from nlp_module.analyzer import TEST_ESSAY, TEST_TOPIC


def _print_reasoning(state) -> None:
    if not state.get("show_reasoning"):
        state["reasoning_log"] = []
        return
    logs = state.get("reasoning_log", [])
    if not logs:
        return
    print("Ассистент: [Reasoning]")
    for entry in logs:
        print(f"Ассистент: {entry}")
    state["reasoning_log"] = []


def main() -> None:
    logger = setup_logger("ege_ai_agent.cli")
    logger.info("Initializing CLI main LLM and workflow...")
    runner = WorkflowRunner()
    state = {
        "messages": [],
        "global_plan": [],
        "current_step": 0,
        "user_profile": {},
        "task_condition": TEST_TOPIC,
        "essay_text": TEST_ESSAY,
        "essay_sent": False,
        "show_reasoning": os.getenv("SHOW_REASONING", "0") == "1",
        "reasoning_log": [],
    }

    print("Ассистент: Привет! Я наставник по сочинению ЕГЭ.")
    print(f"Ассистент: Тема сочинения: {TEST_TOPIC}")
    logger.info(f"CLI session started with topic: '{TEST_TOPIC}'")

    while True:
        user_input = input("Пользователь: ").strip()
        if not user_input:
            continue
        logger.debug(f"CLI: user input: '{user_input}'")
        if user_input.lower().startswith("/reasoning"):
            parts = user_input.lower().split()
            if len(parts) > 1 and parts[1] in ("on", "off"):
                state["show_reasoning"] = parts[1] == "on"
                print(f"Ассистент: Режим отображения reasoning {'включен' if state['show_reasoning'] else 'выключен'}.")
            else:
                status = "on" if state.get("show_reasoning") else "off"
                print(f"Ассистент: Текущий режим reasoning: {status}. Используй /reasoning on|off.")
            continue
        if user_input.lower() in {"exit", "quit", "выход"}:
            logger.info("CLI: Завершение сессии по запросу пользователя")
            break
        if not state.get("essay_sent"):
            state["messages"] = state.get("messages", []) + [HumanMessage(content=f"Сочинение: {TEST_ESSAY}")]
            state["essay_sent"] = True
        state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]
        state.update(runner.invoke(state))
        _print_reasoning(state)
        print(f"Ассистент: {extract_last_ai_message(state.get('messages', []))}")


if __name__ == "__main__":
    main()
