import pytest

from fpl_predictor.analyst.provider import (
    DisabledProvider, FakeProvider, OpenAIResponsesProvider, ProviderError, provider_from_config,
)


def test_provider_is_disabled_without_key(monkeypatch):
    for key in ("FPL_ANALYST_PROVIDER", "FPL_ANALYST_API_KEY", "FPL_ANALYST_MODEL"):
        monkeypatch.delenv(key, raising=False)
    assert isinstance(provider_from_config({}), DisabledProvider)


def test_openai_provider_rejects_malformed_output(monkeypatch):
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"output": [{"type": "message", "content": []}]}
    monkeypatch.setattr("fpl_predictor.analyst.provider.requests.post", lambda *args, **kwargs: Response())
    with pytest.raises(ProviderError, match="malformed"):
        OpenAIResponsesProvider("secret", "test-model").generate([{"role": "user", "content": "hello"}])


def test_fake_provider_supports_success_and_timeout_without_network():
    assert FakeProvider("grounded").generate([]) == "grounded"
    with pytest.raises(TimeoutError):
        FakeProvider(error=TimeoutError()).generate([])

