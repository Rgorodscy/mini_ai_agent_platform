"""
Test configuration.

Two rules hold for the whole suite:

1. No test reaches the network. The Groq client, the embedding model and
   the cross-encoder are replaced with fakes by an autouse fixture, so a
   test that forgets to stub the LLM fails loudly instead of spending
   money.
2. No test shares state. Each test gets its own SQLite file and its own
   ChromaDB directory under tmp_path.
"""

import os

import pytest

# Must be set before app.config is imported: it calls _require() at import
# time. Values are deliberately obvious fakes.
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("API_KEY_TENANT_A", "test-key-tenant-a")
os.environ.setdefault("API_KEY_TENANT_B", "test-key-tenant-b")
os.environ.setdefault("API_KEY_TENANT_C", "test-key-tenant-c")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key-not-real")

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.config import API_KEYS, GROQ_MODEL  # noqa: E402
from app.database import Base, get_db, get_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.middleware.auth import get_tenant  # noqa: E402

TEST_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine
)

TENANT_A_KEY = "test-key-tenant-a"
TENANT_B_KEY = "test-key-tenant-b"
TENANT_C_KEY = "test-key-tenant-c"


# --- Fakes ---


class FakeEmbedder:
    """
    Deterministic stand-in for SentenceTransformer.

    Maps text to a vector by hashing characters into buckets, so identical
    text always embeds identically and different text usually does not.
    That is enough for chunking and retrieval tests; it is not meant to be
    semantically meaningful.
    """

    DIMENSIONS = 16

    def encode(self, texts):
        if isinstance(texts, str):
            return self._vector(texts)
        return np.array([self._vector(t) for t in texts])

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.DIMENSIONS, dtype=np.float32)
        for i, char in enumerate(text):
            vec[(ord(char) + i) % self.DIMENSIONS] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


class FakeReranker:
    """Scores by overlap of lowercase words — no model, no download."""

    def predict(self, pairs):
        scores = []
        for query, chunk in pairs:
            query_words = set(query.lower().split())
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words)
            scores.append(float(overlap))
        return np.array(scores)


class FakeGroqClient:
    """
    Records the calls made to it and replays queued responses.

    Queue responses with `client.queue(...)`; the last queued response is
    reused once the queue runs dry, so a test only has to describe the
    turns it cares about.
    """

    def __init__(self):
        self.calls = []
        self._responses = []
        self.chat = self  # mirrors groq_client.chat.completions.create
        self.completions = self

    def queue(self, *responses):
        self._responses.extend(responses)
        return self

    def reset(self):
        """
        Clears queued responses and recorded calls.

        Needed when one test drives two runs: the last queued response is
        reused rather than popped, so re-queuing without clearing leaves it
        in front of the new expectations.
        """
        self._responses.clear()
        self.calls.clear()
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            return make_llm_response(content="Default fake answer.")
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


class _Message:
    def __init__(self, content=None, tool_calls=None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls


class _ToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _ToolFunction(name, arguments)


class _ToolFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _Choice:
    def __init__(self, message):
        self.message = message


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _Response:
    def __init__(self, message, usage=None):
        self.choices = [_Choice(message)]
        self.usage = usage


def make_llm_response(
    content=None,
    tool_calls=None,
    prompt_tokens=100,
    completion_tokens=20,
):
    """
    Builds a Groq-shaped response.

    tool_calls is a list of (name, args_dict) tuples, or a raw string for
    the malformed-arguments case. Token counts default to non-zero values so
    usage accounting is exercised by default; pass None for a provider that
    reports no usage at all.
    """
    calls = None
    if tool_calls:
        import json

        calls = []
        for i, (name, args) in enumerate(tool_calls):
            raw = args if isinstance(args, str) else json.dumps(args)
            calls.append(_ToolCall(f"call_{i}", name, raw))

    reported = None
    if prompt_tokens is not None or completion_tokens is not None:
        reported = _Usage(prompt_tokens or 0, completion_tokens or 0)

    return _Response(
        _Message(content=content, tool_calls=calls), usage=reported
    )


# --- Autouse isolation ---


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
    """
    Replaces every outbound dependency with a fake, for every test.

    Caches are cleared around each test so a lazily-loaded real model can
    never leak from one test into the next.
    """
    from app.core import llm as llm_module
    from app.core.rag import reranker as reranker_module
    from app.core.rag import utils as rag_utils

    # Held by reference: monkeypatch replaces the module attributes below,
    # so teardown could no longer reach the real cached functions.
    cached = (
        llm_module._client_for,
        rag_utils.get_embedder,
        rag_utils.get_chroma_client,
        reranker_module.get_reranker,
    )

    def clear_caches():
        for fn in cached:
            fn.cache_clear()

    clear_caches()

    fake_groq = FakeGroqClient()
    embedder = FakeEmbedder()
    reranker = FakeReranker()

    monkeypatch.setattr(rag_utils, "get_embedder", lambda: embedder)
    monkeypatch.setattr(reranker_module, "get_reranker", lambda: reranker)

    # Each test gets a throwaway vector store. Patched on the consuming
    # module, which imported the value by name at import time.
    monkeypatch.setattr(rag_utils, "CHROMA_PATH", str(tmp_path / "chroma"))

    def _fake_client():
        return fake_groq

    # Every LLM call in the app goes through core.llm, so faking the
    # provider client there covers the execution loop and the RAG pipeline
    # at once. The adapter's own routing still runs for real.
    monkeypatch.setattr(
        llm_module, "_client_for", lambda provider_name: fake_groq
    )

    yield fake_groq

    clear_caches()


@pytest.fixture
def fake_llm(no_network):
    """The FakeGroqClient backing the current test."""
    return no_network


@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            db.close()

    def override_get_tenant():
        return "test_tenant"

    app.dependency_overrides[get_db] = override_get_db
    # Streamed runs record themselves from a worker thread with their own
    # session; it must land in the test database too.
    app.dependency_overrides[get_session_factory] = lambda: TestingSessionLocal
    app.dependency_overrides[get_tenant] = override_get_tenant

    yield TestClient(app)

    app.dependency_overrides.clear()


@pytest.fixture
def raw_client():
    """Client with real auth in place — for testing the middleware itself."""

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    # Streamed runs record themselves from a worker thread with their own
    # session; it must land in the test database too.
    app.dependency_overrides[get_session_factory] = lambda: TestingSessionLocal
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


@pytest.fixture
def tenant_id():
    return "test_tenant"


@pytest.fixture
def auth_headers():
    return {"x-api-key": TENANT_A_KEY}


@pytest.fixture
def supported_model():
    """The model name the API currently accepts."""
    return GROQ_MODEL


@pytest.fixture
def configured_api_keys():
    return API_KEYS
