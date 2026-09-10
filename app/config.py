import json
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Raises an error if a required env variable is missing."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def _optional(key: str, default: str) -> str:
    """
    Reads an optional setting, treating an empty value as absent.

    os.getenv returns "" for a variable that is set but empty, which is what
    a compose file's `${VAR:-}` produces. Taking that literally would let an
    unset compose variable override the default defined here, so the default
    would have to be duplicated in every deployment manifest.
    """
    return (os.getenv(key) or "").strip() or default


DATABASE_URL = _require("DATABASE_URL")
MAX_EXECUTION_STEPS = int(os.getenv("MAX_EXECUTION_STEPS", "5"))
API_KEYS: dict[str, str] = {
    _require("API_KEY_TENANT_A"): "tenant_a",
    _require("API_KEY_TENANT_B"): "tenant_b",
    _require("API_KEY_TENANT_C"): "tenant_c",
}

# --- LLM settings ---

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# The default model, used wherever the platform calls an LLM on its own
# behalf (query expansion, RAG answer generation) rather than on behalf of
# a caller who named one.
#
# Providers retire models, so any default here has a shelf life — this is
# why the list below is configuration. To see what your account serves:
#   python -c "from app.core.llm import _groq_client; \
#       print([m.id for m in _groq_client().models.list().data])"
GROQ_MODEL = _optional("GROQ_MODEL", "openai/gpt-oss-120b")


def _model_list(key: str, default: str) -> tuple[str, ...]:
    """
    Reads a comma-separated model list from the environment.

    The catalogue a provider serves changes over time, so which models this
    deployment offers is configuration, not a hardcoded guess.

    An empty value falls back to the default rather than meaning "no
    models": os.getenv returns "" for a variable that is set but empty,
    which is what a compose file passing `${GROQ_MODELS:-}` produces, and
    taking that literally leaves the deployment unable to serve anything.
    """
    raw = _optional(key, default)
    return tuple(m.strip() for m in raw.split(",") if m.strip())


GROQ_MODELS = _model_list("GROQ_MODELS", GROQ_MODEL)


def _model_pricing() -> dict[str, dict[str, float]]:
    """
    Per-model prices in USD per million tokens, as JSON:
      {"openai/gpt-oss-120b": {"input": 0.15, "output": 0.6}}

    Deliberately empty by default. Bundling a price table would mean
    shipping numbers that go stale silently and bill tenants wrongly —
    prices are a deployment's own responsibility. A model with no entry
    reports cost as null rather than zero.
    """
    raw = _optional("MODEL_PRICING", "")
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"MODEL_PRICING is not valid JSON: {e}") from e

    pricing: dict[str, dict[str, float]] = {}
    for model, price in parsed.items():
        if not isinstance(price, dict) or {"input", "output"} - set(price):
            raise RuntimeError(
                f"MODEL_PRICING['{model}'] must have 'input' and 'output' "
                f"keys, in USD per million tokens."
            )
        pricing[model] = {
            "input": float(price["input"]),
            "output": float(price["output"]),
        }
    return pricing


MODEL_PRICING = _model_pricing()

# --- RAG settings ---

EMBEDDING_MODEL = _optional("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = _optional(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
CHROMA_PATH = _optional("CHROMA_PATH", "./chroma_db")


def _flag(key: str, default: bool) -> bool:
    raw = _optional(key, "true" if default else "false").lower()
    return raw in ("1", "true", "yes", "on")


# Both stages cost real latency and, for expansion, an extra LLM call per
# search. Which ones pay for themselves depends on the corpus, so they are
# switches rather than assumptions — measure with evals/retrieval_eval.py
# against your own documents before trusting either default.
RAG_USE_EXPANSION = _flag("RAG_USE_EXPANSION", True)
RAG_USE_RERANK = _flag("RAG_USE_RERANK", True)
RAG_CANDIDATE_POOL = int(_optional("RAG_CANDIDATE_POOL", "10"))
RAG_TOP_K = int(_optional("RAG_TOP_K", "3"))


def validate_settings() -> None:
    """
    Fails fast on missing configuration at application startup.

    Import-time validation is deliberately limited to the settings above;
    anything that would reach out to the network (LLM client, embedding
    models) is validated here instead, so the package stays importable in
    test and tooling contexts.
    """
    _require("GROQ_API_KEY")

    # A deployment that can serve no model answers /health but rejects
    # every run with a 400. Refuse to start instead: the misconfiguration
    # belongs in the boot log, not in a confusing response body.
    from app.core.llm import supported_models

    if not supported_models():
        raise RuntimeError(
            "No LLM model is serveable: every provider is missing either "
            "credentials or a model list. Check GROQ_API_KEY and GROQ_MODELS."
        )
