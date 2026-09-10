# Mini Agent Platform

A multi-tenant AI agent platform. Tenants configure agents, give them
tools, upload documents to a private knowledge base, and run tasks
against them. Every run is executed by a LangGraph state machine that
calls a real LLM, executes tools, and records a full audit trail.

Built with FastAPI, SQLAlchemy, LangGraph, Groq and ChromaDB.

![CI](https://github.com/rgorodscy/mini_agent_platform/actions/workflows/ci.yml/badge.svg)

```bash
cp .env.example .env   # add your GROQ_API_KEY
docker compose up
```

---

## Table of Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running tests](#running-tests)
- [API reference](#api-reference)
- [Design decisions](#design-decisions)
- [Security model](#security-model)
- [Known limitations](#known-limitations)

---

## What it does

1. A tenant authenticates with an API key that resolves to a `tenant_id`.
2. They register **tools** and compose **agents** out of them.
3. They upload documents (text, PDF, DOCX) to `/knowledge`. Documents are
   semantically chunked, embedded and stored in a per-tenant ChromaDB
   collection.
4. They run an agent against a task. The agent loops — reasoning, calling
   tools, searching its knowledge base — until it produces an answer or
   hits its step budget.
5. Every step of every run is persisted and queryable.

No data, no vector, and no execution is ever visible across tenants.

---

## Architecture

The API follows a strict three-layer split, and the agent core sits
behind it with no knowledge of HTTP or the database:

```
Router (HTTP)  →  Service (business logic)  →  Repository (database)
                        │
                        └→  core/  (agent execution + RAG)
```

### The execution loop

`app/core/execution_loop.py` compiles a LangGraph state machine:

```
        ┌──────────────┐
   ┌───▶│  call_model  │────────┐
   │    └──────────────┘        │
   │            │               │  no tool calls
   │            │ tool calls    │  · step budget exhausted
   │            ▼               │  · 3 consecutive errors
   │    ┌──────────────┐        ▼
   └────│execute_tools │      END
        └──────────────┘
```

Each cycle appends to two accumulating channels: `messages` (what the
model sees) and `steps` (the audit trail the API returns). The loop is
bounded on three axes — step count, consecutive errors, and the model
choosing to answer in plain text.

### The RAG pipeline

`search_knowledge` is a tool like any other, but it runs a full pipeline:

```
query
  → expand_query    LLM rewrites it into several phrasings (widen recall)
  → retrieve        vector search per variation, deduplicated
  → rerank          cross-encoder scores query↔chunk pairs (raise precision)
  → answer          top chunks become grounding context for the LLM
```

Retrieval widens, reranking narrows. The cross-encoder is far more
accurate than embedding similarity because it sees the query and the
chunk together — and far too slow to run over the whole collection, which
is exactly why it only ever sees what retrieval already shortlisted.

Documents are chunked semantically (`core/rag/chunker.py`): sentences are
grouped while they stay above a similarity threshold, so a chunk breaks
where the topic changes rather than at an arbitrary character count, with
a recursive character-split fallback for oversized chunks.

---

## Project structure

```
app/
├── main.py                   # App entry point, error handlers, lifespan
├── config.py                 # Environment-driven settings
├── database.py               # SQLAlchemy engine and session
├── logger.py                 # Logging setup
├── middleware/auth.py        # API key → tenant_id
├── models/                   # SQLAlchemy ORM models
├── schemas/                  # Pydantic request/response models
├── repositories/             # Database queries (always tenant-scoped)
├── services/                 # Business logic
├── routers/                  # HTTP endpoints
└── core/                     # Agent execution — no HTTP, no DB
    ├── execution_loop.py     # LangGraph state machine
    ├── llm.py                # Provider registry and routing
    ├── guardrail.py          # Injection detection
    ├── prompt_builder.py     # Structured prompt assembly
    ├── safe_math.py          # AST-based arithmetic evaluator
    ├── usage.py              # Token/cost accounting (ContextVar)
    ├── tool_implementations.py  # Tool registry
    ├── utils.py              # PDF/DOCX/text extraction
    └── rag/
        ├── chunker.py        # Semantic + recursive chunking
        ├── indexer.py        # Ingestion
        ├── retriever.py      # Vector search
        ├── query_expander.py # Query rewriting
        ├── reranker.py       # Cross-encoder reranking
        ├── pipeline.py       # Orchestration (answer_from_knowledge)
        └── utils.py          # Embedder & ChromaDB clients

tests/                        # 290 tests, no network access
alembic/                      # Database migrations
.github/workflows/ci.yml      # Tests, migrations, image build
Dockerfile                    # Multi-stage, CPU-only torch, non-root
docker-compose.yml            # API + Postgres + persistent volumes
```

---

## Setup

### With Docker (recommended)

Brings up Postgres and the API, applies migrations on boot, and bakes the
embedding and reranking weights into the image so the first knowledge
search does not block on a download:

```bash
cp .env.example .env     # fill in GROQ_API_KEY and the tenant keys
docker compose up --build
```

The API is on `http://localhost:8000`, docs on `/docs`. The vector store
lives in a named volume, so it survives `docker compose down`.

### Without Docker

### Requirements

- Python 3.12+

### 1. Install

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`sentence-transformers` pulls in PyTorch. For a CPU-only install (much
smaller), install torch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Configure

```bash
cp .env.example .env
```

Fill in the values. The app refuses to start if a required variable is
missing. You need a free API key from [console.groq.com](https://console.groq.com).

### 3. Migrate

```bash
alembic upgrade head
```

### 4. Run

```bash
uvicorn app.main:app --reload
```

Interactive docs: `http://localhost:8000/docs`.

The embedding and reranking models (~120MB combined) are downloaded from
HuggingFace on first use and cached locally, so the first knowledge
search is slower than the rest.

---

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

290 tests, ~30 seconds, **91% coverage**.

The suite is hermetic: an autouse fixture in `tests/conftest.py` replaces
the LLM client, the embedding model and the cross-encoder with fakes for
every test. No test can reach the network or spend money — a test that
forgets to stub the LLM gets a canned response rather than a real call.
Each test also gets its own database and its own vector store.

ChromaDB, pypdf and python-docx are **not** faked. Retrieval runs against
a real vector store in a temporary directory, and the PDF and DOCX
fixtures are real files built in memory, so parsing, collection naming,
upsert semantics and tenant scoping are exercised rather than mocked away.

CI additionally applies the migrations against a real Postgres service and
builds the Docker image, then checks the container answers `/health`.

---

## API reference

Every request needs an API key header:

```
x-api-key: <your-api-key>
```

Keys come from `.env` and map to tenants:

| Environment variable | Tenant ID  |
| -------------------- | ---------- |
| `API_KEY_TENANT_A`   | `tenant_a` |
| `API_KEY_TENANT_B`   | `tenant_b` |
| `API_KEY_TENANT_C`   | `tenant_c` |

### Tools

Tools are references to implementations that ship with the platform. A
tool only becomes callable if its `name` matches an entry in
`TOOL_REGISTRY` — currently:

`calculator`, `get_weather`, `search_knowledge`, `summarize`, `think`,
`web_search`

```bash
# Create
curl -X POST http://localhost:8000/tools \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"name": "web_search", "description": "Searches the web"}'

# List (optionally filtered by agent)
curl "http://localhost:8000/tools?agent_name=Research" -H "x-api-key: $API_KEY"

# Get / update / delete
curl http://localhost:8000/tools/{tool_id} -H "x-api-key: $API_KEY"
curl -X PUT http://localhost:8000/tools/{tool_id} \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"description": "Updated description"}'
curl -X DELETE http://localhost:8000/tools/{tool_id} -H "x-api-key: $API_KEY"
```

### Agents

```bash
curl -X POST http://localhost:8000/agents \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{
    "name": "Research Agent",
    "role": "a researcher",
    "description": "Researches topics and answers from internal documents",
    "tools": ["<tool_id>"]
  }'
```

`GET /agents`, `GET /agents/{id}`, `PUT /agents/{id}` and
`DELETE /agents/{id}` behave as expected. `GET /agents?tool_name=` filters
by tool.

### Knowledge base

Accepts either raw text or a file upload (PDF, DOCX, TXT), as
`multipart/form-data`:

```bash
# Raw text
curl -X POST http://localhost:8000/knowledge \
  -H "x-api-key: $API_KEY" \
  -F "doc_id=policy-2026" \
  -F "text=Our refund policy allows returns within 30 days." \
  -F 'metadata={"source":"handbook"}'

# File upload
curl -X POST http://localhost:8000/knowledge \
  -H "x-api-key: $API_KEY" \
  -F "doc_id=contract-42" \
  -F "file=@./contract.pdf"
```

Re-posting the same `doc_id` replaces that document's chunks.

### Run an agent

```bash
curl -X POST http://localhost:8000/agents/{agent_id}/run \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"task": "What is our refund policy?", "model": "openai/gpt-oss-120b"}'
```

**Example response:**

```json
{
  "id": "abc123",
  "agent_id": "xyz789",
  "tenant_id": "tenant_a",
  "model": "openai/gpt-oss-120b",
  "task": "What is our refund policy?",
  "structured_prompt": { "system": "...", "tools": [...], "user": "..." },
  "steps": [
    {
      "step": 1,
      "type": "tool_result",
      "tool": "search_knowledge",
      "input": { "query": "refund policy" },
      "result": "Relevant knowledge for 'refund policy': ..."
    },
    {
      "step": 2,
      "type": "final_response",
      "content": "Returns are accepted within 30 days."
    }
  ],
  "final_response": "Returns are accepted within 30 days.",
  "status": "completed",
  "created_at": "2026-03-30T10:00:00"
}
```

**Step types:**

| Type             | Meaning                                        |
| ---------------- | ---------------------------------------------- |
| `tool_result`    | Tool ran and returned a result                 |
| `tool_error`     | Tool failed, or is not available to this agent |
| `tool_blocked`   | Arguments failed the injection screen          |
| `final_response` | The agent's answer                             |

**Statuses:**

| Status              | Meaning                                       |
| ------------------- | --------------------------------------------- |
| `completed`         | The agent produced an answer                  |
| `max_steps_reached` | Step budget spent without a final answer      |

### Usage and cost

```bash
curl "http://localhost:8000/usage" -H "x-api-key: $API_KEY"

# Optionally windowed
curl "http://localhost:8000/usage?since=2026-09-01T00:00:00" \
  -H "x-api-key: $API_KEY"
```

Returns tokens, LLM call count, cost and average latency for the calling
tenant, broken down by model and by agent. The tenant comes from the API
key, so there is no way to ask for another tenant's usage.

```json
{
  "tenant_id": "tenant_a",
  "executions": 1,
  "prompt_tokens": 875,
  "completion_tokens": 363,
  "total_tokens": 1238,
  "llm_calls": 4,
  "cost_usd": null,
  "avg_latency_ms": 4683,
  "by_model": [{ "model": "openai/gpt-oss-120b", "total_tokens": 1238, "...": "..." }],
  "by_agent": [{ "agent_id": "acf3647b-...", "total_tokens": 1238, "...": "..." }]
}
```

Note `llm_calls: 4` for a run with two visible steps. Two calls came from
the execution loop; the other two are the query expansion and the grounded
generation that the RAG pipeline makes inside `search_knowledge`. See
[Design decisions](#design-decisions) for how those are captured.

`cost_usd` is `null` until `MODEL_PRICING` is configured — see
[Known limitations](#known-limitations).

### Execution history

```bash
curl "http://localhost:8000/agents/{agent_id}/history?page=1&page_size=10" \
  -H "x-api-key: $API_KEY"
```

`page` ≥ 1, `page_size` between 1 and 100.

---

## Design decisions

**Three-layer architecture.** Routers never touch the database;
repositories never contain business logic. Each layer is independently
testable and replaceable.

**`tenant_id` on every query.** Every repository method takes `tenant_id`
as an explicit required parameter, and each tenant gets its own ChromaDB
collection. There is no code path that can accidentally read across
tenants.

**`search_knowledge` closes over the tenant.** The tenant is bound when
the tool is constructed, not passed as an argument the model fills in — so
no prompt can talk the model into naming another tenant's collection.

**No heavy work at import time.** Model loading, the ChromaDB client and
the LLM client are all behind `lru_cache`d accessors, so importing the
package needs no network, no API key and no GPU. Configuration that would
otherwise fail on first use is validated at startup in the app lifespan
instead. This is what makes the test suite fast and hermetic.

**The agent core knows nothing about HTTP or the database.** Everything
under `core/` takes plain arguments and returns plain data. `guardrail`
exposes `detect_injection` (pure) separately from `check_prompt_injection`
(raises HTTP 400), so the same detection serves the API layer and the
execution loop, which needs to block one tool call without failing the
whole request.

**Usage is collected in a ContextVar, not threaded through signatures.**
One run makes several LLM calls, and not all of them are visible from the
execution loop: `search_knowledge` triggers a query expansion and a grounded
generation several layers down inside the RAG pipeline. Passing a usage
object through every call would couple the RAG code to billing. Instead
`core/usage.py` keeps the accumulator in a ContextVar and `chat_completion` —
the single choke point for every LLM call — records into whichever scope is
active. ContextVars are per-task and per-thread, so concurrent requests
never share one, and the token is reset in a `finally` so an exception
cannot leak an accumulator into the next request.

**Cost is null, never zero, when a price is unknown.** A zero would read as
"this run was free" on a billing page. Same rule in the aggregate: a tenant
whose models are all unpriced gets `null`, not a confident `$0.00`.

**The advertised model list is derived, not declared.** `supported_models()`
is computed from the providers that have credentials configured, so the API
can never advertise a model it would fail to call — and the `model` on an
execution row is the model that actually ran.

**Execution history as structured JSON.** Each run stores the structured
prompt, every step with tool inputs and outputs, and the final response —
full auditability of what the agent actually did.

**Bounded loops.** Three independent stop conditions: `MAX_EXECUTION_STEPS`,
three consecutive tool errors, and the model answering in plain text.

**Validation at the schema layer.** Pydantic validators enforce non-empty
strings, length limits and pagination bounds before requests reach the
service layer.

---

## Security model

Agent platforms have an unusual threat model: the *model* decides what to
call, and the model can be steered by text the operator does not control —
a user's task, or a document that retrieval pulled into context (indirect
prompt injection).

**Inbound tasks are screened.** `check_prompt_injection` rejects known
injection and code-execution patterns with a 400 before anything runs.

**Tool arguments are screened too.** The model's tool arguments pass
through the same detection before reaching a tool. A blocked call is
recorded as a `tool_blocked` step and reported back to the model as an
error, so the run continues without executing anything.

**Retrieved text is framed as data.** The system prompt states explicitly
that tool output is data and carries no authority to change instructions.

**No `eval`.** The calculator parses expressions to an AST and interprets
an allowlist of arithmetic nodes (`core/safe_math.py`). `eval` with a
stripped `__builtins__` is *not* a sandbox —
`(1).__class__.__mro__[-1].__subclasses__()` walks straight out of it — and
these arguments are model-written, so it was a remote code execution path.
Expression length and exponent size are bounded to stop a single call from
hanging a worker.

These are defence in depth, not a guarantee. Pattern matching catches
known phrasings, not novel ones.

---

## Known limitations

**The model list has to match your provider account.** Groq retires models,
so a name that worked last month can 404 today — `llama-3.3-70b-versatile`
did, during this project's own testing. `GROQ_MODELS` is configuration for
exactly this reason, but nothing validates it against the live catalogue at
boot: a stale entry is only caught when a run against it returns 502. To
see what your account serves:

```bash
python -c "from app.core.llm import _groq_client; \
    print(sorted(m.id for m in _groq_client().models.list().data))"
```

**Only one provider is implemented.** `core/llm.py` defines the provider
registry and routing, but Groq is the only adapter. Adding OpenAI or
Anthropic means writing an adapter and a registry entry — and translating
to and from the OpenAI chat-completions shape this codebase assumes, which
not every provider SDK speaks natively.

**Tools with no implementation are silently skipped.** `POST /tools`
accepts any `name`, but only names in `TOOL_REGISTRY` become callable —
anything else is logged and dropped at execution time rather than
rejected at creation. Creating a tool named `web-search` (with a hyphen)
produces an agent whose tool never runs, with no error at creation time.

**RAG has no evaluation harness.** Query expansion and reranking are
implemented but their contribution to retrieval quality is not measured.
There are no recall@k numbers to justify the added latency and cost.

**No price table is bundled.** Token counts, latency and per-model usage
are recorded on every execution and aggregated by `GET /usage`, but
`cost_usd` stays null until `MODEL_PRICING` is configured. Shipping a price
table would mean shipping numbers that go stale silently and bill tenants
wrongly, so prices are a deployment's own responsibility.

**Synchronous endpoints.** Handlers are `def`, not `async def`, and an
agent run can hold a worker thread for tens of seconds. The RAG pipeline
makes one LLM call for expansion, N retrievals, a rerank and a final call,
all blocking, with no caching.

**SQLite in local development.** Limited concurrent writes, and no
`ALTER COLUMN` support in migrations. The Docker setup runs Postgres, and
CI applies the migrations against Postgres on every push, so the two stay
in step — but local development still defaults to SQLite.

**API keys in environment variables.** Fine for three fixed tenants,
wrong for real tenancy: no rotation, no revocation, no per-tenant limits.
Production would put these in a secrets manager with a lookup.
