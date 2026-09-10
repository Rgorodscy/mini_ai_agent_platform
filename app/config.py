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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _model_list(key: str, default: str) -> tuple[str, ...]:
    """
    Reads a comma-separated model list from the environment.

    The catalogue a provider serves changes over time, so which models this
    deployment offers is configuration, not a hardcoded guess.
    """
    raw = os.getenv(key, default)
    return tuple(m.strip() for m in raw.split(",") if m.strip())


GROQ_MODELS = _model_list("GROQ_MODELS", GROQ_MODEL)

# --- RAG settings ---

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")


def validate_settings() -> None:
    """
    Fails fast on missing configuration at application startup.

    Import-time validation is deliberately limited to the settings above;
    anything that would reach out to the network (LLM client, embedding
    models) is validated here instead, so the package stays importable in
    test and tooling contexts.
    """
    _require("GROQ_API_KEY")
