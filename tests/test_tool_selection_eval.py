from agent_module.constants import ALL_TOOL_NAMES
from agent_module.routing.evaluate import NO_TOOL, load_scenarios, make_llm_predictor, predict_rules, run, summarize
from tests.fakes import FakeLLM, tool_call_message


def test_dataset_is_well_formed():
    scenarios = load_scenarios()
    assert len(scenarios) == 100
    assert len({s.id for s in scenarios}) == 100
    allowed = set(ALL_TOOL_NAMES) | {NO_TOOL}
    assert {s.expected_tool for s in scenarios} <= allowed
    assert sum(s.expected_tool == NO_TOOL for s in scenarios) >= 10  # есть сценарии «инструмент не нужен»
    assert len({s.expected_tool for s in scenarios}) == len(allowed)  # встречаются все инструменты


def test_rules_baseline_runs_offline():
    rows = run(load_scenarios(), predict_rules)
    summary = summarize(rows)
    assert summary["n"] == 100
    assert 0 < summary["tool_selection_accuracy"] < 1
    assert summary["errors"] == 0


def test_summarize_counts_false_calls():
    rows = [
        {"id": "1", "expected": "none", "predicted": "none", "correct": True, "latency_ms": 5, "error": None},
        {"id": "2", "expected": "none", "predicted": "literature_rag", "correct": False, "latency_ms": 7, "error": None},
        {"id": "3", "expected": "literature_rag", "predicted": "literature_rag", "correct": True, "latency_ms": 9, "error": None},
        {"id": "4", "expected": "nlp_errors_tool", "predicted": "error", "correct": False, "latency_ms": 0, "error": "boom"},
    ]
    s = summarize(rows)
    assert s["tool_selection_accuracy"] == 0.5
    assert s["accuracy_when_no_tool_needed"] == 0.5
    assert s["accuracy_when_tool_needed"] == 0.5
    assert s["false_tool_calls"] == 1
    assert s["errors"] == 1


def test_llm_predictor_maps_answers_and_errors():
    scenarios = load_scenarios()[:3]
    ok = make_llm_predictor(FakeLLM([tool_call_message("literature_rag", {"query": "x"})]))
    assert ok(scenarios[0])["predicted"] == "literature_rag"
    err = make_llm_predictor(FakeLLM([RuntimeError("quota")]))
    result = err(scenarios[0])
    assert result["predicted"] == "error" and "quota" in result["error"]


def test_run_reports_progress_for_every_scenario():
    from agent_module.routing.evaluate import load_scenarios, predict_rules, run

    scenarios = load_scenarios()[:5]
    seen = []
    rows = run(scenarios, predict_rules, on_row=lambda row, all_rows: seen.append((row["id"], len(all_rows))))
    assert [s for s, _ in seen] == [s.id for s in scenarios]
    assert [n for _, n in seen] == [1, 2, 3, 4, 5]
    assert len(rows) == 5


def test_holdout_scenarios_are_valid_and_disjoint_from_main_set():
    from agent_module.routing.evaluate import DEFAULT_SCENARIOS_PATH, load_scenarios

    holdout = load_scenarios(DEFAULT_SCENARIOS_PATH.with_name("tool_selection_holdout.csv"))
    main = load_scenarios()
    assert len(holdout) >= 50
    assert sum(1 for s in holdout if s.assistant_prev) >= 8
    assert any(not s.essay_present for s in holdout)
    assert not {s.user_text for s in holdout} & {s.user_text for s in main}


def test_assistant_prev_is_placed_before_student_message():
    from agent_module.routing.evaluate import Scenario, build_case

    case = build_case(Scenario("x", "шаг", "Да, давайте", True, "literature_rag", assistant_prev="Подобрать аргумент?"))
    assert [type(m).__name__ for m in case["history"]] == ["HumanMessage", "AIMessage", "HumanMessage"]
    assert case["history"][-1].content == "Да, давайте"


def test_ablation_modes_change_prompt_and_step():
    from agent_module.routing.evaluate import MINIMAL_SYSTEM_PROMPT, Scenario, build_case, mask_tool_names

    scenario = Scenario("x", "Подобрать аргумент через literature_rag", "Спасибо!", True, "none")
    full = build_case(scenario)["system_prompt"]
    assert "Tool decision rules" in full and "literature_rag" in full.split("Текущий шаг плана:")[1]
    masked = build_case(scenario, prompt="minimal", mask_step=True)["system_prompt"]
    assert masked.startswith(MINIMAL_SYSTEM_PROMPT)
    assert "literature_rag" not in masked.split("Текущий шаг плана:")[1]
    assert mask_tool_names("nlp_errors_tool и nlp_analyzer") == "нужный инструмент и нужный инструмент"
