"""
Configuration parsing and startup validation.

These cover a class of failure the rest of the suite cannot see: the app
imports and answers /health, but is configured such that every request
fails. The test suite sets its own environment, so only explicit tests
exercise what a real deployment passes in.
"""

import pytest

from app.config import _model_list, validate_settings

# --- Model list parsing ---


def test_reads_a_comma_separated_list(monkeypatch):
    monkeypatch.setenv("TEST_MODELS", "model-a,model-b,model-c")

    assert _model_list("TEST_MODELS", "fallback") == (
        "model-a",
        "model-b",
        "model-c",
    )


def test_strips_whitespace_around_entries(monkeypatch):
    monkeypatch.setenv("TEST_MODELS", " model-a , model-b ")

    assert _model_list("TEST_MODELS", "fallback") == ("model-a", "model-b")


def test_drops_empty_entries(monkeypatch):
    monkeypatch.setenv("TEST_MODELS", "model-a,,model-b,")

    assert _model_list("TEST_MODELS", "fallback") == ("model-a", "model-b")


def test_unset_variable_uses_the_default(monkeypatch):
    monkeypatch.delenv("TEST_MODELS", raising=False)

    assert _model_list("TEST_MODELS", "fallback") == ("fallback",)


def test_empty_variable_uses_the_default(monkeypatch):
    """
    Regression: docker compose's `GROQ_MODELS: ${GROQ_MODELS:-}` sets the
    variable to an empty string rather than leaving it unset, so os.getenv
    returned "" and the default never applied. The container booted
    healthy and then rejected every model with a 400.
    """
    monkeypatch.setenv("TEST_MODELS", "")

    assert _model_list("TEST_MODELS", "fallback") == ("fallback",)


def test_whitespace_only_variable_uses_the_default(monkeypatch):
    monkeypatch.setenv("TEST_MODELS", "   ")

    assert _model_list("TEST_MODELS", "fallback") == ("fallback",)


# --- Startup validation ---


def test_validate_settings_passes_with_the_test_environment():
    validate_settings()


def test_validate_settings_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        validate_settings()


def test_validate_settings_refuses_to_start_with_no_serveable_model(
    monkeypatch,
):
    """
    The failure this is guarding against: /health answers, so an
    orchestrator marks the container healthy, while every run returns
    "Unsupported model ... Supported: []".
    """
    from app.core import llm

    monkeypatch.setattr(llm, "PROVIDERS", ())

    with pytest.raises(RuntimeError, match="No LLM model is serveable"):
        validate_settings()
