"""
Provider-agnostic chat completion.

The platform used to advertise OpenAI model names while always calling one
hardcoded Groq model, so the `model` recorded on every execution row was
wrong. This module makes the advertised list real: the set of callable
models comes from the providers that are actually configured, and the
model the caller asks for is the model that runs.

Only Groq is implemented. Adding a provider means adding an adapter
function and a PROVIDERS entry — the execution loop and the RAG pipeline
call `chat_completion` and never learn which provider answered. Note that
the adapter is responsible for translating to and from the OpenAI
chat-completions shape used here; not every provider SDK speaks it
natively.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from app.config import GROQ_API_KEY, GROQ_MODELS, _require
from app.logger import get_logger

logger = get_logger(__name__)


class UnsupportedModel(ValueError):
    """Raised for a model id no configured provider serves."""


class ProviderRejectedModel(RuntimeError):
    """
    Raised when the provider itself does not recognise the model.

    Distinct from UnsupportedModel: the deployment believes it can serve
    this model, but the provider disagrees — almost always a GROQ_MODELS
    entry that has since been retired from the provider's catalogue.
    Without this, a stale config surfaced as a bare 500 and had to be
    diagnosed from the container log.
    """


@dataclass(frozen=True)
class Provider:
    name: str
    models: tuple[str, ...]
    client_factory: Callable[[], Any]
    api_key: str | None

    @property
    def configured(self) -> bool:
        """A provider with no credentials cannot serve anything."""
        return bool(self.api_key) and bool(self.models)


def _groq_client():
    from groq import Groq

    return Groq(api_key=_require("GROQ_API_KEY"))


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="groq",
        models=GROQ_MODELS,
        client_factory=_groq_client,
        api_key=GROQ_API_KEY,
    ),
)


def supported_models() -> list[str]:
    """
    Every model this deployment can actually serve.

    Derived from configuration, not hardcoded: a provider with no API key
    contributes nothing, so the API never advertises a model that would
    fail when called.
    """
    return sorted(
        model
        for provider in PROVIDERS
        if provider.configured
        for model in provider.models
    )


def resolve_provider(model: str) -> Provider:
    for provider in PROVIDERS:
        if model in provider.models:
            if not provider.configured:
                raise UnsupportedModel(
                    f"Model '{model}' is served by {provider.name}, which "
                    f"is not configured on this deployment."
                )
            return provider

    raise UnsupportedModel(f"Unknown model '{model}'.")


@lru_cache(maxsize=None)
def _client_for(provider_name: str):
    provider = next(p for p in PROVIDERS if p.name == provider_name)
    logger.info(f"Building LLM client | provider={provider_name}")
    return provider.client_factory()


def chat_completion(
    model: str,
    messages: list,
    tools: list | None = None,
    tool_choice: str = "auto",
):
    """
    Calls the provider that serves `model`.

    Returns the provider's response in the OpenAI chat-completions shape;
    callers read `.choices[0].message`. `tools` is omitted from the request
    entirely when empty — providers reject an empty tool list.
    """
    provider = resolve_provider(model)
    client = _client_for(provider.name)

    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        if _is_model_not_found(e):
            logger.error(
                f"Provider does not serve this model | "
                f"provider={provider.name} model={model}"
            )
            raise ProviderRejectedModel(
                f"Provider '{provider.name}' does not serve model "
                f"'{model}'. The configured model list is out of date with "
                f"the provider's catalogue."
            ) from e
        raise


def _is_model_not_found(error: Exception) -> bool:
    """
    Detects a provider's "no such model" rejection.

    Matched on the response payload rather than the SDK's exception class,
    so this does not need a separate branch per provider.
    """
    status = getattr(error, "status_code", None)
    if status not in (400, 404):
        return False

    text = str(error).lower()
    return "model_not_found" in text or "does not exist" in text
