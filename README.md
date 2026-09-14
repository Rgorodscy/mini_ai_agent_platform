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

Then open **http://localhost:8000** and connect with a tenant key from `.env`.

---

## Table of Contents

- [What it does](#what-it-does)
- [The console](#the-console)
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

## The console

The API ships with a web console at `/ui/` — no build step, no separate
frontend, served by the same container. Connect with a tenant key and you
can create an agent, add documents to its knowledge base, and run it:
each step appears the moment it completes, then the run's latency, LLM
calls, tokens and cost.

It is a thin client over the public API — everything it does goes through
the same endpoints documented below, with the same auth and tenant scoping.

Three details that are easy to get wrong in a page like this:

- **Model output is rendered as text, never as HTML.** An answer or a tool
  result can carry markup — a retrieved document is untrusted input — so
  every server value reaches the page through `textContent`. A test fails
  the build if an HTML-parsing sink appears in the script. A restrictive
  Content-Security-Policy (no inline script, `connect-src 'self'`, no
  framing) is the second layer, scoped to the console so it never wraps the
  API.
- **Streaming uses `fetch`, not `EventSource`.** `EventSource` can only send
  GET, and a run is a POST with a body. The page reads the response stream
  with an incremental SSE parser that tolerates frames — and `\r\n` pairs —
  split across network chunks.
- **Stop is honest about what it does.** Stopping closes the stream, but the
  run continues on the server and is still recorded and billed; the console
  says exactly that rather than implying the run was cancelled.

The API key is kept in `sessionStorage` — it survives a reload, is gone when
the tab closes, and never appears in a URL.

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

### Does the pipeline earn its cost?

Expansion adds an LLM call per search; reranking runs a cross-encoder over
every candidate. `evals/retrieval_eval.py` measures whether they pay for
themselves, against 20 documents and 22 hand-labelled queries phrased the
way a user would ask rather than by quoting the source.

```bash
docker compose exec api python -m evals.retrieval_eval
```

| configuration | recall@1 | recall@3 | MRR | seconds | LLM calls |
| ------------- | -------- | -------- | ----- | ------- | --------- |
| baseline      | 0.82     | 1.00     | 0.902 | 0.4     | 0         |
| +expansion    | **0.91** | 0.95     | **0.943** | 19.7 | 22     |
| +rerank       | 0.82     | 1.00     | 0.894 | 4.5     | 0         |
| +both         | 0.82     | 1.00     | 0.894 | 48.4    | 22        |

Two results worth reading carefully:

**Expansion helps top-1** — 0.82 to 0.91 recall@1, MRR 0.902 to 0.943 — at
roughly 50x the latency. Queries here use different vocabulary from the
documents, which is exactly the gap expansion closes.

**Reranking does not help, and cancels expansion's gain.** On its own it
leaves recall unchanged and nudges MRR down. Combined with expansion it
pulls MRR back to the un-expanded 0.894 while costing the most latency of
any configuration. The shipped default (`+both`) is the worst trade in this
table on this corpus.

**The expansion rows move between runs.** Expansion calls an LLM, which
is not deterministic. A second run gave `+expansion` recall@1 0.95 and MRR
0.966 (the table shows the first); the baseline and `+rerank` rows, which
make no LLM call, reproduced exactly. The conclusion held in both runs, but
a single run of an LLM-dependent configuration is a sample, not a
measurement — averaging several is the honest way to report it. That second
run also hit the case the expansion fallback exists for: the model refused
to rewrite one query (*"I'm sorry, but I can't provide that information"*),
and the pipeline searched with the original query instead of failing.

**These numbers do not generalise.** 22 queries is far too few for a 0.09
recall difference — that is two queries — to be significant, and baseline
recall@3 is already 1.00, so a 20-document corpus barely discriminates
between configurations at all. The cross-encoder is also trained on web
search passages, not handbook prose. The honest conclusion is narrower than
the table looks: on *this* corpus reranking does not earn its latency, and
the question is now answerable at all, which it was not before.

Both stages are therefore switches, not assumptions — `RAG_USE_EXPANSION`
and `RAG_USE_RERANK`. Run the harness against your own documents before
trusting either default.

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
├── static/                   # Web console (HTML/CSS/JS, no build step)
└── core/                     # Agent execution — no HTTP, no DB
    ├── execution_loop.py     # LangGraph state machine
    ├── llm.py                # Provider registry and routing
    ├── guardrail.py          # Injection detection
    ├── prompt_builder.py     # Structured prompt assembly
    ├── safe_math.py          # AST-based arithmetic evaluator
    ├── usage.py              # Token/cost accounting (ContextVar)
    ├── stream.py             # SSE framing + producer thread
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

tests/                        # 346 tests, no network access
evals/                        # Retrieval quality harness
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

346 tests, ~35 seconds, **92% coverage**.

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
`TOOL_REGISTRY`. Ask the API which names those are rather than hardcoding
them:

```bash
# Names with an implementation, and what each does
curl http://localhost:8000/tools/available -H "x-api-key: $API_KEY"

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

### Models

```bash
curl http://localhost:8000/models -H "x-api-key: $API_KEY"
# {"default": "openai/gpt-oss-120b", "models": ["openai/gpt-oss-120b"]}
```

Every name listed is one a run will accept: the list is derived from the
providers that have credentials configured, not hardcoded.

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

### Run an agent with streaming

Same validation and the same persisted audit trail as `/run`, but each step
arrives as it happens over Server-Sent Events:

```bash
curl -N -X POST http://localhost:8000/agents/{agent_id}/run/stream \
  -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"task": "What is the refund window?", "model": "openai/gpt-oss-120b"}'
```

```
[ 4.80s] event: step
         data: {"step":1,"type":"tool_result","tool":"search_knowledge", ...}
[ 5.17s] event: step
         data: {"step":2,"type":"final_response","content":"...30 days..."}
[ 5.23s] event: done
         data: {"execution_id":"06d37b57-...","status":"completed",
                "total_tokens":1216,"llm_calls":4,"latency_ms":5231, ...}
```

| event   | meaning                                                      |
| ------- | ------------------------------------------------------------ |
| `step`  | a tool result, tool error, blocked call, or the final response |
| `done`  | execution id, status and the run's token/cost/latency totals  |
| `error` | the run failed after the stream had already started           |

Validation runs before the response opens, so an unknown model or a rejected
task still returns `400 application/json` rather than a `200` whose body
begins with an error event.

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

**Streaming runs in a producer thread, not a bare generator.** Two reasons,
and the first is a trap. Starlette iterates a synchronous generator by
handing each `next()` to a worker thread with a *copy* of the caller's
context, so a `ContextVar.set()` made before one yield is invisible after
it — wrapping a generator body in `track_usage()` would silently measure
nothing. Running the whole execution in one thread gives it one context for
its entire life. It also decouples the agent from how fast the client reads:
the queue is bounded, so a slow reader applies backpressure instead of
letting events accumulate without limit.

**A streamed run records itself, and withholds its answer until it has.**
When a client disconnects, the server closes the response generator at
whichever `yield` it is paused on, so nothing after that point runs. The
first version wrote the execution row there — an abandoned stream left no
audit trail and no usage, even though the model had already been called and
paid for. Worse, the final answer went out *before* the row was written, so
a client could read the answer and disconnect before `done` and the tokens
never reached `/usage`. The producer thread now writes the row with a session
of its own, and holds the final answer back until that row exists. Verified
by cutting a real connection after the first event: the run finished in the
background and was recorded with all four LLM calls.

**The streaming entry point is not itself a generator.** A generator body
does not run until the first `next()`, which for a `StreamingResponse` is
after the status line and headers have gone out. Validation inside one turns
a 400 into a 200 whose body opens with an error. `stream_agent` therefore
validates eagerly and *returns* a generator, and both entry points share
`_prepare_run` so they cannot drift apart on model validation, tenant
scoping or the guardrail.

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

**Tools with no implementation are still accepted.** `GET /tools/available`
now lists the names that have an implementation, and the console only
offers those — but `POST /tools` itself still accepts any `name`, and
anything outside the registry is logged and dropped at execution time
rather than rejected at creation. An API client that creates `web-search`
(with a hyphen) still gets an agent whose tool never runs. Rejecting
unknown names with a 422 is the remaining fix; it was left out because it
would break any tenant that already has such rows.

**The eval set is too small to generalise.** `evals/` measures retrieval
quality, but 20 documents and 22 queries cannot separate configurations
whose recall@3 is already 1.00. The harness is real; the numbers are
indicative, not a benchmark. A larger labelled set is the obvious next step.

**No price table is bundled.** Token counts, latency and per-model usage
are recorded on every execution and aggregated by `GET /usage`, but
`cost_usd` stays null until `MODEL_PRICING` is configured. Shipping a price
table would mean shipping numbers that go stale silently and bill tenants
wrongly, so prices are a deployment's own responsibility.

**The work is still blocking, even when streamed.** `/run/stream` gives the
client incremental results, but the provider calls underneath are
synchronous: a run occupies a worker thread for its whole duration, and the
RAG pipeline makes an expansion call, N retrievals, a rerank and a final
call one after another with no caching. Streaming improves perceived latency
and frees the client from waiting; it does not improve throughput. An async
provider client is the change that would.

**Disconnecting does not cancel a run.** A client that closes the stream
stops receiving events, but the agent keeps going to completion, and the run
is recorded and counted towards usage. That is deliberate for accounting —
the tokens are spent either way — but it means an abandoned run still costs
its full price. Cancellation would need a stop signal checked between graph
steps, recorded as a distinct `cancelled` status.

**Token-level streaming is not implemented.** Events arrive per step, not
per token — the final response appears in one frame once the model finishes
composing it. Passing `stream=True` to the provider and relaying deltas is
the natural next step.

**SQLite in local development.** Limited concurrent writes, and no
`ALTER COLUMN` support in migrations. The Docker setup runs Postgres, and
CI applies the migrations against Postgres on every push, so the two stay
in step — but local development still defaults to SQLite.

**API keys in environment variables.** Fine for three fixed tenants,
wrong for real tenancy: no rotation, no revocation, no per-tenant limits.
Production would put these in a secrets manager with a lookup.
