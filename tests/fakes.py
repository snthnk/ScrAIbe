from typing import Any, List, Union

from langchain_core.messages import AIMessage


class FakeLLM:
    """Подставной LLM: отдаёт заранее заданные ответы (или бросает исключение) и считает вызовы."""

    def __init__(self, responses: List[Union[AIMessage, Exception]]):
        self.responses = list(responses)
        self.calls = 0
        self.bound_tools: List[Any] = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages, *args, **kwargs):
        self.calls += 1
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


def tool_call_message(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args}])
