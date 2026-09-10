import pytest

from app.config import GROQ_MODEL
from app.core import llm
from app.core.llm import (
    Provider,
    UnsupportedModel,
    chat_completion,
    resolve_provider,
    supported_models,
)
from tests.conftest import make_llm_response


def fake_provider(name, models, api_key="key"):
    return Provider(
        name=name,
        models=models,
        client_factory=lambda: None,
        api_key=api_key,
    )


# --- Advertised models reflect what is configured ---


def test_default_model_is_supported():
    assert GROQ_MODEL in supported_models()


def test_supported_models_are_sorted_and_unique():
    models = supported_models()

    assert models == sorted(set(models))


def test_unconfigured_provider_advertises_nothing(monkeypatch):
    """A provider without an API key must not appear in the catalogue."""
    monkeypatch.setattr(
        llm,
        "PROVIDERS",
        (
            fake_provider("configured", ("model-a",)),
            fake_provider("no-key", ("model-b",), api_key=None),
        ),
    )

    assert supported_models() == ["model-a"]


def test_provider_with_no_models_advertises_nothing(monkeypatch):
    monkeypatch.setattr(llm, "PROVIDERS", (fake_provider("empty", ()),))

    assert supported_models() == []


# --- Routing ---


def test_resolves_the_default_model_to_groq():
    assert resolve_provider(GROQ_MODEL).name == "groq"


def test_unknown_model_is_rejected():
    with pytest.raises(UnsupportedModel, match="Unknown model"):
        resolve_provider("totally-made-up-model")


def test_known_model_from_an_unconfigured_provider_is_rejected(monkeypatch):
    """
    The distinction matters for debugging: the model exists, but this
    deployment has no credentials for the provider that serves it.
    """
    monkeypatch.setattr(
        llm,
        "PROVIDERS",
        (fake_provider("somewhere", ("model-x",), api_key=None),),
    )

    with pytest.raises(UnsupportedModel, match="not configured"):
        resolve_provider("model-x")


def test_routes_each_model_to_its_own_provider(monkeypatch):
    monkeypatch.setattr(
        llm,
        "PROVIDERS",
        (
            fake_provider("first", ("model-a", "model-b")),
            fake_provider("second", ("model-c",)),
        ),
    )

    assert resolve_provider("model-b").name == "first"
    assert resolve_provider("model-c").name == "second"


# --- The request reaches the provider ---


def test_requested_model_is_passed_to_the_provider(fake_llm):
    fake_llm.queue(make_llm_response(content="ok"))

    chat_completion(model=GROQ_MODEL, messages=[{"role": "user", "c": "hi"}])

    assert fake_llm.calls[0]["model"] == GROQ_MODEL


def test_tools_are_omitted_when_empty(fake_llm):
    """Providers reject an empty tool list, so it must not be sent."""
    fake_llm.queue(make_llm_response(content="ok"))

    chat_completion(model=GROQ_MODEL, messages=[], tools=[])

    assert "tools" not in fake_llm.calls[0]
    assert "tool_choice" not in fake_llm.calls[0]


def test_tools_are_forwarded_when_present(fake_llm):
    fake_llm.queue(make_llm_response(content="ok"))
    tools = [{"type": "function", "function": {"name": "x"}}]

    chat_completion(
        model=GROQ_MODEL, messages=[], tools=tools, tool_choice="none"
    )

    assert fake_llm.calls[0]["tools"] == tools
    assert fake_llm.calls[0]["tool_choice"] == "none"


def test_unsupported_model_never_reaches_a_provider(fake_llm):
    with pytest.raises(UnsupportedModel):
        chat_completion(model="made-up", messages=[])

    assert fake_llm.calls == []


# --- End to end through the API ---


def test_run_records_the_model_that_actually_ran(client, fake_llm):
    """
    Regression: the API used to accept and record gpt-4o while always
    calling a Groq model, so the execution row named a model that never
    ran.
    """
    fake_llm.queue(make_llm_response(content="Done."))
    agent = client.post(
        "/agents",
        json={
            "name": "A",
            "role": "assistant",
            "description": "d",
            "tools": [],
        },
    ).json()

    body = client.post(
        f"/agents/{agent['id']}/run",
        json={"task": "hello", "model": GROQ_MODEL},
    ).json()

    assert body["model"] == GROQ_MODEL
    assert fake_llm.calls[0]["model"] == body["model"]


def test_api_rejects_a_model_it_cannot_serve(client, fake_llm):
    agent = client.post(
        "/agents",
        json={
            "name": "A",
            "role": "assistant",
            "description": "d",
            "tools": [],
        },
    ).json()

    response = client.post(
        f"/agents/{agent['id']}/run",
        json={"task": "hello", "model": "gpt-4o"},
    )

    assert response.status_code == 400
    assert fake_llm.calls == []
