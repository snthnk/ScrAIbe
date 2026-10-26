from langchain_core.messages import HumanMessage

from agent_module.constants import EXPERT_REASONING, LITERATURE_RAG, NLP_ERRORS, WEB_FACT_CHECKER
from agent_module.routing.rules import select_tool_by_rules
from tests.fixtures import ESSAY, STATES, TOPIC


def pick(text, state="essay", step=""):
    return select_tool_by_rules([HumanMessage(content=text)], dict(STATES[state]), step)


def test_thanks_never_triggers_a_tool_even_if_the_plan_step_mentions_rag():
    # В старой версии _force_tool_call вызывал literature_rag на любое сообщение, если шаг плана говорил о примере.
    decision = pick("Спасибо!", step="Подобрать пример-иллюстрацию через literature_rag")
    assert decision.tool is None
    assert decision.decided_by == "rules"


def test_criterion_question_goes_to_expert_with_grounded_payload():
    decision = pick("Почему К2 получил 2 балла?")
    assert decision.tool == EXPERT_REASONING
    payload = decision.args["payload"]
    assert payload["criterion"] == "K2"
    assert payload["essay_text"] == ESSAY
    assert payload["task_condition"] == TOPIC


def test_cyrillic_criterion_letter_is_understood():
    assert pick("что исправить по К4").args["payload"]["criterion"] == "K4"


def test_error_question_goes_to_errors_tool_with_essay():
    decision = pick("Какие у меня ошибки?")
    assert decision.tool == NLP_ERRORS
    assert decision.args == {"essay_text": ESSAY}


def test_error_question_without_essay_is_not_guessed():
    assert pick("Какие у меня ошибки?", state="no_essay").tool is None


def test_fact_question_goes_to_web_checker():
    decision = pick("Кто автор «Белой гвардии»?")
    assert decision.tool == WEB_FACT_CHECKER
    assert decision.args["query"] == "Кто автор «Белой гвардии»?"


def test_argument_request_goes_to_rag():
    decision = pick("подбери аргумент по теме природа")
    assert decision.tool == LITERATURE_RAG
    assert "природ" in decision.args["query"]


def test_chit_chat_and_own_answers_need_no_tool():
    for text in ["Привет", "Я думаю, автор считает что природа лечит", "помоги с введением"]:
        assert pick(text).tool is None, text
