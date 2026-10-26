import pytest

from agent_module import llms
from agent_module.runner import WorkflowRunner, extract_last_ai_message, looks_like_quota_error, retry_delay_seconds


def test_fallback_models_never_cross_providers():
    for name in llms.fallback_model_names(llms.PROVIDER_MISTRAL):
        assert "gemini" not in name
    for name in llms.fallback_model_names(llms.PROVIDER_GEMINI):
        assert "mistral" not in name.lower()


def test_model_name_resolution(monkeypatch):
    for var in ("MODEL_NAME", "MISTRAL_MODEL_NAME", "GEMINI_MODEL_NAME", "EXPERT_MODEL_NAME", "MISTRAL_EXPERT_MODEL_NAME"):
        monkeypatch.delenv(var, raising=False)
    assert llms.resolve_model_name("mistral") == llms.DEFAULT_MODELS["mistral"]
    monkeypatch.setenv("MISTRAL_MODEL_NAME", "open-mistral-7b")
    assert llms.resolve_model_name("mistral") == "open-mistral-7b"
    assert llms.resolve_model_name("mistral", override="x") == "x"
    monkeypatch.setenv("MISTRAL_EXPERT_MODEL_NAME", "big")
    assert llms.resolve_model_name("mistral", expert=True) == "big"


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr(llms, "_load_env", lambda: None)
    with pytest.raises(ValueError, match="Mistral"):
        llms.build_llm()


class FlakyWorkflow:
    def __init__(self, errors, result=None):
        self.errors = list(errors)
        self.result = result or {"messages": []}
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def make_runner(workflows, sleeps):
    it = iter(workflows)
    built = []

    def workflow_factory(llm):
        wf = next(it)
        built.append((llm, wf))
        return wf

    runner = WorkflowRunner(llm_factory=lambda model_name=None: model_name or "default", workflow_factory=workflow_factory, sleep=sleeps.append)
    return runner, built


def test_rate_limit_with_delay_is_retried_after_sleeping():
    sleeps = []
    wf = FlakyWorkflow([RuntimeError("429 Please retry in 2.5s")], {"messages": ["ok"]})
    runner, _ = make_runner([wf], sleeps)
    assert runner.invoke({}) == {"messages": ["ok"]}
    assert sleeps == [3.5] and wf.calls == 2


def test_quota_error_switches_to_same_provider_fallback_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.delenv("FALLBACK_MODEL_NAME", raising=False)
    first = FlakyWorkflow([RuntimeError("RESOURCE_EXHAUSTED PerDay")])
    second = FlakyWorkflow([], {"messages": ["from fallback"]})
    runner, built = make_runner([first, second], [])
    assert runner.invoke({}) == {"messages": ["from fallback"]}
    assert built[1][0] in llms.MISTRAL_FALLBACK_MODELS
    assert runner.model_in_use == built[1][0]


def test_unrecoverable_error_is_raised():
    runner, _ = make_runner([FlakyWorkflow([ValueError("bad input")])], [])
    with pytest.raises(ValueError):
        runner.invoke({})


def test_helpers():
    assert retry_delay_seconds(RuntimeError("Please retry in 7s")) == 7.0
    assert retry_delay_seconds(RuntimeError("nothing")) is None
    assert looks_like_quota_error(RuntimeError("429 Too Many"))
    assert extract_last_ai_message([]).startswith("Извините")


def test_placeholder_keys_do_not_create_fallbacks(monkeypatch):
    monkeypatch.setattr(llms, "_load_env", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "real-looking-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "your_mistral_api_key_here")
    monkeypatch.delenv("LLM_FALLBACKS", raising=False)
    monkeypatch.setattr(llms, "_make_gemini", lambda model, key: object())
    monkeypatch.setattr(llms, "_make_mistral", lambda model, key: pytest.fail("Mistral не должен создаваться по заглушке"))
    llms.build_llm()


def test_fallbacks_can_be_disabled(monkeypatch):
    monkeypatch.setattr(llms, "_load_env", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "real-looking-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "real-mistral-key")
    monkeypatch.setenv("LLM_FALLBACKS", "0")
    monkeypatch.setattr(llms, "_make_gemini", lambda model, key: object())
    monkeypatch.setattr(llms, "_make_mistral", lambda model, key: pytest.fail("запасные модели отключены"))
    assert llms.fallbacks_enabled() is False
    llms.build_llm()


def test_vsellm_provider_uses_base_url_model_and_no_fallbacks(monkeypatch):
    monkeypatch.setattr(llms, "_load_env", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "vsellm")
    monkeypatch.setenv("VSELLM_API_KEY", "sk-test")
    monkeypatch.setenv("VSELLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("VSELLM_MODEL_NAME", "my-model")
    monkeypatch.setenv("MISTRAL_API_KEY", "real-mistral-key")
    monkeypatch.delenv("VSELLM_EXPERT_MODEL_NAME", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    llm = llms.build_llm()
    assert llm.model_name == "my-model"
    assert str(llm.openai_api_base) == "https://example.test/v1"
    assert llm.temperature is None or llm.temperature == 1  # не навязываем температуру
    assert llms.build_llm(expert=True).model_name == "my-model"
    monkeypatch.setenv("VSELLM_EXPERT_MODEL_NAME", "big-model")
    assert llms.build_llm(expert=True).model_name == "big-model"
    monkeypatch.delenv("FALLBACK_MODEL_NAME", raising=False)
    assert llms.fallback_model_names("vsellm") == []


def test_vsellm_requires_key(monkeypatch):
    monkeypatch.setattr(llms, "_load_env", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "vsellm")
    monkeypatch.setenv("VSELLM_API_KEY", "your_vsellm_api_key_here")
    with pytest.raises(ValueError, match="VSELLM_API_KEY"):
        llms.build_llm()


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setattr(llms, "_load_env", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "vsellmm")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        llms.build_llm()
