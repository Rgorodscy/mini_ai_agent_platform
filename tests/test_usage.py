"""
Usage accounting: the accumulator, persistence on the execution row, and
the aggregated /usage report.
"""

import pytest

from app.config import GROQ_MODEL
from app.core.usage import Usage, record, track_usage
from tests.conftest import make_llm_response


@pytest.fixture
def priced(monkeypatch):
    """Prices in USD per million tokens, for cost assertions."""
    pricing = {GROQ_MODEL: {"input": 1.0, "output": 2.0}}
    monkeypatch.setattr("app.core.usage.MODEL_PRICING", pricing)
    return pricing


# --- The accumulator ---


def test_tracks_tokens_from_a_recorded_call():
    with track_usage() as usage:
        record("m", make_llm_response(prompt_tokens=10, completion_tokens=5))

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15
    assert usage.llm_calls == 1


def test_accumulates_across_calls():
    with track_usage() as usage:
        record("m", make_llm_response(prompt_tokens=10, completion_tokens=5))
        record("m", make_llm_response(prompt_tokens=20, completion_tokens=1))

    assert usage.total_tokens == 36
    assert usage.llm_calls == 2


def test_breaks_down_by_model():
    with track_usage() as usage:
        record("a", make_llm_response(prompt_tokens=10, completion_tokens=1))
        record("b", make_llm_response(prompt_tokens=20, completion_tokens=2))

    assert usage.by_model["a"].total_tokens == 11
    assert usage.by_model["b"].total_tokens == 22


def test_records_latency():
    with track_usage() as usage:
        record("m", make_llm_response())

    assert usage.latency_ms >= 0


def test_handles_a_provider_that_reports_no_usage():
    with track_usage() as usage:
        record(
            "m",
            make_llm_response(prompt_tokens=None, completion_tokens=None),
        )

    assert usage.total_tokens == 0
    assert usage.llm_calls == 1


def test_recording_outside_a_scope_is_a_noop():
    """Library code calls record() unconditionally; nothing should break."""
    record("m", make_llm_response())


def test_scope_does_not_leak_after_an_exception():
    with pytest.raises(RuntimeError):
        with track_usage():
            raise RuntimeError("boom")

    # If the ContextVar had leaked, this would accrue to the dead scope.
    record("m", make_llm_response())

    with track_usage() as fresh:
        record("m", make_llm_response(prompt_tokens=7, completion_tokens=0))

    assert fresh.prompt_tokens == 7


def test_nested_scopes_do_not_double_count():
    with track_usage() as outer:
        with track_usage() as inner:
            record(
                "m", make_llm_response(prompt_tokens=5, completion_tokens=0)
            )

    assert inner.prompt_tokens == 5
    assert outer.prompt_tokens == 0


# --- Cost ---


def test_cost_is_none_without_configured_prices(monkeypatch):
    """
    Null, not zero: a zero would read as "this run was free" on a billing
    page.
    """
    monkeypatch.setattr("app.core.usage.MODEL_PRICING", {})

    usage = Usage()
    usage.add("unpriced-model", 1000, 1000)

    assert usage.cost_usd is None


def test_cost_is_computed_per_million_tokens(monkeypatch):
    monkeypatch.setattr(
        "app.core.usage.MODEL_PRICING",
        {"m": {"input": 1.0, "output": 2.0}},
    )

    usage = Usage()
    usage.add("m", 1_000_000, 500_000)

    # 1M input at $1 + 0.5M output at $2 = $2.00
    assert usage.cost_usd == pytest.approx(2.0)


def test_cost_is_none_when_any_model_is_unpriced(monkeypatch):
    monkeypatch.setattr(
        "app.core.usage.MODEL_PRICING", {"priced": {"input": 1, "output": 1}}
    )

    usage = Usage()
    usage.add("priced", 1000, 1000)
    usage.add("unpriced", 1000, 1000)

    assert usage.cost_usd is None


# --- Persisted on the execution row ---


@pytest.fixture
def agent(client):
    return client.post(
        "/agents",
        json={
            "name": "A",
            "role": "assistant",
            "description": "d",
            "tools": [],
        },
    ).json()


def run(client, agent_id, task="hello"):
    return client.post(
        f"/agents/{agent_id}/run",
        json={"task": task, "model": GROQ_MODEL},
    )


def test_run_records_usage(client, agent, fake_llm):
    fake_llm.queue(
        make_llm_response(
            content="Done.", prompt_tokens=120, completion_tokens=30
        )
    )

    body = run(client, agent["id"]).json()

    assert body["prompt_tokens"] == 120
    assert body["completion_tokens"] == 30
    assert body["total_tokens"] == 150
    assert body["llm_calls"] == 1
    assert body["latency_ms"] >= 0


def test_run_records_usage_across_multiple_turns(client, agent, fake_llm):
    fake_llm.queue(
        make_llm_response(
            tool_calls=[("web_search", {"query": "x"})],
            prompt_tokens=100,
            completion_tokens=10,
        ),
        make_llm_response(
            content="Done.", prompt_tokens=150, completion_tokens=20
        ),
    )

    body = run(client, agent["id"]).json()

    assert body["llm_calls"] == 2
    assert body["total_tokens"] == 280


def test_run_records_the_model_breakdown(client, agent, fake_llm):
    fake_llm.queue(make_llm_response(content="Done."))

    body = run(client, agent["id"]).json()

    assert GROQ_MODEL in body["usage_by_model"]
    assert body["usage_by_model"][GROQ_MODEL]["calls"] == 1


def test_run_records_cost_when_priced(client, agent, fake_llm, priced):
    fake_llm.queue(
        make_llm_response(
            content="Done.",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
    )

    body = run(client, agent["id"]).json()

    assert body["cost_usd"] == pytest.approx(3.0)


def test_run_cost_is_null_when_unpriced(client, agent, fake_llm, monkeypatch):
    monkeypatch.setattr("app.core.usage.MODEL_PRICING", {})
    fake_llm.queue(make_llm_response(content="Done."))

    assert run(client, agent["id"]).json()["cost_usd"] is None


# --- The /usage report ---


def test_usage_is_empty_for_a_tenant_with_no_runs(client):
    body = client.get("/usage").json()

    assert body["executions"] == 0
    assert body["total_tokens"] == 0
    assert body["by_model"] == []
    assert body["by_agent"] == []


def test_usage_reports_the_calling_tenant(client):
    assert client.get("/usage").json()["tenant_id"] == "test_tenant"


def test_usage_sums_across_runs(client, agent, fake_llm):
    fake_llm.queue(
        make_llm_response(
            content="Done.", prompt_tokens=100, completion_tokens=10
        )
    )
    for _ in range(3):
        run(client, agent["id"])

    body = client.get("/usage").json()

    assert body["executions"] == 3
    assert body["total_tokens"] == 330
    assert body["llm_calls"] == 3


def test_usage_breaks_down_by_model(client, agent, fake_llm):
    fake_llm.queue(make_llm_response(content="Done."))
    run(client, agent["id"])

    breakdown = client.get("/usage").json()["by_model"]

    assert len(breakdown) == 1
    assert breakdown[0]["model"] == GROQ_MODEL
    assert breakdown[0]["executions"] == 1


def test_usage_breaks_down_by_agent(client, agent, fake_llm):
    fake_llm.queue(make_llm_response(content="Done."))
    run(client, agent["id"])

    breakdown = client.get("/usage").json()["by_agent"]

    assert len(breakdown) == 1
    assert breakdown[0]["agent_id"] == agent["id"]


def test_usage_reports_average_latency(client, agent, fake_llm):
    fake_llm.queue(make_llm_response(content="Done."))
    run(client, agent["id"])

    assert client.get("/usage").json()["avg_latency_ms"] is not None


def test_usage_cost_is_null_when_nothing_is_priced(client, agent, fake_llm):
    fake_llm.queue(make_llm_response(content="Done."))
    run(client, agent["id"])

    assert client.get("/usage").json()["cost_usd"] is None


def test_usage_sums_cost_when_priced(client, agent, fake_llm, priced):
    fake_llm.queue(
        make_llm_response(
            content="Done.",
            prompt_tokens=1_000_000,
            completion_tokens=0,
        )
    )
    run(client, agent["id"])
    run(client, agent["id"])

    assert client.get("/usage").json()["cost_usd"] == pytest.approx(2.0)


# --- Time filtering ---


def test_usage_respects_a_since_filter(client, agent, fake_llm):
    fake_llm.queue(make_llm_response(content="Done."))
    run(client, agent["id"])

    body = client.get("/usage?since=2099-01-01T00:00:00").json()

    assert body["executions"] == 0


def test_usage_respects_an_until_filter(client, agent, fake_llm):
    fake_llm.queue(make_llm_response(content="Done."))
    run(client, agent["id"])

    body = client.get("/usage?until=2000-01-01T00:00:00").json()

    assert body["executions"] == 0


def test_usage_includes_runs_inside_the_window(client, agent, fake_llm):
    fake_llm.queue(make_llm_response(content="Done."))
    run(client, agent["id"])

    body = client.get(
        "/usage?since=2000-01-01T00:00:00&until=2099-01-01T00:00:00"
    ).json()

    assert body["executions"] == 1


def test_usage_rejects_a_malformed_date(client):
    assert client.get("/usage?since=not-a-date").status_code == 422


# --- Tenant isolation ---


def test_usage_does_not_leak_across_tenants(client, agent, fake_llm):
    from app.main import app
    from app.middleware.auth import get_tenant
    from fastapi.testclient import TestClient

    fake_llm.queue(make_llm_response(content="Done."))
    run(client, agent["id"])

    app.dependency_overrides[get_tenant] = lambda: "other_tenant"
    other = TestClient(app)

    body = other.get("/usage").json()

    assert body["tenant_id"] == "other_tenant"
    assert body["executions"] == 0
    assert body["total_tokens"] == 0


def test_usage_requires_authentication(raw_client):
    assert raw_client.get("/usage").status_code == 422


def test_usage_rejects_an_invalid_api_key(raw_client):
    response = raw_client.get("/usage", headers={"x-api-key": "nope"})

    assert response.status_code == 401
