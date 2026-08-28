from types import SimpleNamespace

from fpl_predictor.health import assess_health
from fpl_predictor.ui.reliability import record_analyst_response


def test_health_levels():
    assert assess_health(live_available=True, stale=False, schema_passed=True,
                         artifacts_passed=True).level == "HEALTHY"
    assert assess_health(live_available=True, stale=True, schema_passed=True,
                         artifacts_passed=True).level == "DEGRADED"
    assert assess_health(live_available=True, stale=False, schema_passed=False,
                         artifacts_passed=True).level == "UNAVAILABLE"
    assert assess_health(live_available=True, stale=False, schema_passed=True,
                         artifacts_passed=False).level == "UNAVAILABLE"


def test_disabled_provider_is_not_degraded():
    result = assess_health(live_available=True, stale=False, schema_passed=True,
                           artifacts_passed=True, provider_enabled=False, provider_healthy=True)
    assert result.level == "HEALTHY"


def test_analyst_success_failure_fallback_and_grounding_metrics():
    state = {}
    record_analyst_response(state, SimpleNamespace(provider="openai", fallback_used=False,
                            validation_passed=True, latency_ms=100, failure_categories=()))
    record_analyst_response(state, SimpleNamespace(provider="openai", fallback_used=True,
                            validation_passed=True, latency_ms=300,
                            failure_categories=("provider_failure",)))
    record_analyst_response(state, SimpleNamespace(provider="openai", fallback_used=True,
                            validation_passed=False, latency_ms=200,
                            failure_categories=("unsupported_player",)))
    metrics = state["runtime_analyst_metrics"]
    assert metrics["calls"] == 3 and metrics["provider_successes"] == 1
    assert metrics["fallbacks"] == 2 and metrics["average_latency_ms"] == 200
    assert metrics["grounding_validations"] == 2 and metrics["grounding_failures"] == 1
    assert metrics["failure_categories"]["provider_failure"] == 1
    assert len(state["runtime_event_log"]) == 2
