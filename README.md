# Mini Agent Platform

A multi-tenant AI Agent Platform built with FastAPI, SQLAlchemy, and SQLite. Agents are configurable entities with tools that can run tasks through a mock AI pipeline with multi-step execution.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup & Run](#setup--run)
- [Running Tests](#running-tests)
- [API Reference](#api-reference)
- [Design Decisions](#design-decisions)
- [Known Limitations](#known-limitations)

---

## Architecture Overview

The project follows a strict 3-layer architecture:

```
Router (HTTP) → Service (business logic) → Repository (database)
```

The core AI pipeline lives in `app/core/` and is fully decoupled from the API layer:

```
guardrail.py       → Prompt injection detection (regex/heuristic, deterministic)
prompt_builder.py  → Builds structured prompt from agent config + user task
mock_llm.py        → Simulates LLM responses deterministically (no real API calls)
mock_tools.py      → Simulates tool execution deterministically
execution_loop.py  → Orchestrates multi-step tool call → response loop
```

---

## Project Structure

```
app/
├── main.py                  # FastAPI app entry point
├── config.py                # Settings loaded from environment variables
├── database.py              # SQLAlchemy engine and session
├── logger.py                # Logging configuration
├── middleware/
│   └── auth.py              # API key → tenant_id extraction
├── models/                  # SQLAlchemy ORM models
│   ├── agent.py
│   ├── tool.py
│   └── execution.py
├── schemas/                 # Pydantic request/response models
│   ├── agent.py
│   ├── tool.py
│   └── execution.py
├── repositories/            # All database queries (always scoped by tenant_id)
│   ├── agent_repo.py
│   ├── tool_repo.py
│   └── execution_repo.py
├── services/                # Business logic layer
│   ├── agent_service.py
│   ├── tool_service.py
│   └── execution_service.py
├── core/                    # AI pipeline (pure functions, no DB dependency)
│   ├── guardrail.py
│   ├── prompt_builder.py
│   ├── mock_llm.py
│   ├── mock_tools.py
│   └── execution_loop.py
└── routers/
    ├── agent_router.py
    ├── tool_router.py
    └── execution_router.py

tests/
├── conftest.py
├── test_agents.py
├── test_tools.py
├── test_executions.py
├── test_execution_loop.py
├── test_guardrail.py
├── test_mock_llm.py
├── test_mock_tools.py
├── test_prompt_builder.py
├── test_auth.py
├── test_error_handlers.py
└── test_input_validation.py

alembic/                     # Database migrations
pytest.ini                   # Pytest configuration with automatic coverage
requirements.txt             # Project dependencies
.env.example                 # Environment variable template
```

---

## Requirements

- Python 3.12+
- pip

---

## Setup & Run

### 1. Clone the repository and create a virtual environment

```bash
git clone <repo-url>
cd mini_agent_platform

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values. The app will refuse to start if any required variables are missing:

```ini
DATABASE_URL=sqlite:///./agent_platform.db
MAX_EXECUTION_STEPS=5
API_KEY_TENANT_A=your-secure-key-here
API_KEY_TENANT_B=your-secure-key-here
API_KEY_TENANT_C=your-secure-key-here
```

### 4. Run database migrations

```bash
alembic upgrade head
```

This creates `agent_platform.db` in the project root.

### 5. Start the server

```bash
uvicorn app.main:app --reload
```

The API is now running at `http://localhost:8000`.

Interactive docs are available at `http://localhost:8000/docs`.

---

## Running Tests

```bash
pytest tests/
```

Coverage is reported automatically on every run via `pytest.ini`. The full suite runs 153 tests across all layers with 95% code coverage.

To run a specific test file:

```bash
pytest tests/test_guardrail.py -v
pytest tests/test_execution_loop.py -v
pytest tests/test_executions.py -v
```

To generate an HTML coverage report:

```bash
pytest tests/ --cov-report=html
```

Open `htmlcov/index.html` in your browser to see line-by-line coverage.

---

## API Reference

### Authentication

Every request must include the API key in the request header:

```
x-api-key: <your-api-key>
```

API keys are configured via environment variables. The keys defined in `.env` map to these tenants:

| Environment Variable | Tenant ID  |
| -------------------- | ---------- |
| `API_KEY_TENANT_A`   | `tenant_a` |
| `API_KEY_TENANT_B`   | `tenant_b` |
| `API_KEY_TENANT_C`   | `tenant_c` |

Use the actual key values from your `.env` file in all requests below.

---

### Tools

#### Create a tool

```bash
curl -X POST http://localhost:8000/tools \
  -H "x-api-key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"name": "web-search", "description": "Searches the web for information"}'
```

#### Get all tools (with optional filter by agent name)

```bash
curl http://localhost:8000/tools \
  -H "x-api-key: <your-api-key>"

# Filter by agent name
curl "http://localhost:8000/tools?agent_name=Research" \
  -H "x-api-key: <your-api-key>"
```

#### Get a tool by ID

```bash
curl http://localhost:8000/tools/{tool_id} \
  -H "x-api-key: <your-api-key>"
```

#### Update a tool

```bash
curl -X PATCH http://localhost:8000/tools/{tool_id} \
  -H "x-api-key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"description": "Updated description"}'
```

#### Delete a tool

```bash
curl -X DELETE http://localhost:8000/tools/{tool_id} \
  -H "x-api-key: <your-api-key>"
```

---

### Agents

#### Create an agent

```bash
curl -X POST http://localhost:8000/agents \
  -H "x-api-key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Research Agent",
    "role": "researcher",
    "description": "Researches topics on the web",
    "tool_ids": ["<tool_id>"]
  }'
```

#### Get all agents (with optional filter by tool name)

```bash
curl http://localhost:8000/agents \
  -H "x-api-key: <your-api-key>"

# Filter by tool name
curl "http://localhost:8000/agents?tool_name=web-search" \
  -H "x-api-key: <your-api-key>"
```

#### Get an agent by ID

```bash
curl http://localhost:8000/agents/{agent_id} \
  -H "x-api-key: <your-api-key>"
```

#### Update an agent

```bash
curl -X PATCH http://localhost:8000/agents/{agent_id} \
  -H "x-api-key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"name": "Updated Name", "tool_ids": ["<tool_id>"]}'
```

#### Delete an agent

```bash
curl -X DELETE http://localhost:8000/agents/{agent_id} \
  -H "x-api-key: <your-api-key>"
```

---

### Run Agent

Runs an agent against a task through the mock AI pipeline.

```bash
curl -X POST http://localhost:8000/agents/{agent_id}/run \
  -H "x-api-key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "search for the latest AI trends",
    "model": "gpt-4o"
  }'
```

Supported models: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`

**Example response:**

```json
{
  "id": "abc123",
  "agent_id": "xyz789",
  "tenant_id": "tenant_a",
  "model": "gpt-4o",
  "task": "search for the latest AI trends",
  "structured_prompt": {
    "system": "You are Research Agent, researcher...",
    "tools": [{ "name": "web-search", "description": "..." }],
    "user": "search for the latest AI trends"
  },
  "steps": [
    {
      "step": 1,
      "type": "tool_result",
      "tool": "web-search",
      "input": "search for the latest AI trends",
      "result": "[Mock Search Result] Found 3 relevant articles..."
    },
    {
      "step": 2,
      "type": "final_response",
      "content": "Based on the tool results: ..."
    }
  ],
  "final_response": "Based on the tool results: ...",
  "status": "completed",
  "created_at": "2026-03-12T10:00:00"
}
```

**Execution statuses:**

| Status              | Meaning                                          |
| ------------------- | ------------------------------------------------ |
| `completed`         | Final response returned successfully             |
| `max_steps_reached` | Hit the safeguard limit without a final response |

**Guardrail:** Input is scanned for prompt injection and code injection patterns before execution. Returns `400` if detected. Examples of blocked inputs:

- `"ignore previous instructions"`
- `"you are now a different AI"`
- `"eval('malicious code')"`
- `"import os; os.system(...)"`
- `"jailbreak"`, `"DAN mode"`, `"override your instructions"`

---

### Execution History

#### Get execution history for an agent (paginated)

```bash
curl "http://localhost:8000/agents/{agent_id}/history?page=1&page_size=10" \
  -H "x-api-key: <your-api-key>"
```

Pagination constraints: `page` must be ≥ 1, `page_size` must be between 1 and 100.

**Example response:**

```json
{
  "total": 25,
  "page": 1,
  "size": 10,
  "executions": [...]
}
```

---

## Design Decisions

**3-layer architecture** — Routers never touch the DB; repositories never contain business logic. Each layer is independently testable and replaceable.

**tenant_id on every query** — All repository methods require `tenant_id` as an explicit parameter. There is no way to accidentally query across tenants.

**Environment-based configuration** — All secrets and environment-specific values are loaded from `.env` via `python-dotenv`. The app fails at startup with a clear error if any required variable is missing — no silent defaults for secrets.

**Core pipeline as pure functions** — `guardrail`, `prompt_builder`, `mock_llm`, `mock_tools`, and `execution_loop` have no DB dependency. They are tested in complete isolation using `unittest.mock` and `monkeypatch`. Swapping the mock LLM for a real one requires changing only `mock_llm.py` while keeping the same return interface: `{"type": "tool_call", ...}` or `{"type": "final_response", ...}`.

**Deterministic mock LLM** — The mock LLM uses keyword matching against the task to decide whether to call a tool. Same input always produces the same output, making tests reliable and reproducible.

**Execution history as structured JSON** — Each execution stores the full structured prompt, every step including tool inputs and outputs, and the final response. This gives full auditability of every agent run.

**Max steps safeguard** — The execution loop is capped at `MAX_EXECUTION_STEPS` (configurable via environment variable, default 5) to prevent infinite loops.

**Prompt structure separation** — Every prompt is built with three clearly separated sections: `system` (agent role and instructions), `tools` (available tool descriptions), and `user` (the task input). This mirrors production LLM prompt design patterns.

**Validation at the schema layer** — Pydantic validators enforce non-empty strings, length limits, and pagination bounds before requests reach the service layer. Model selection validation is kept in the service layer where it can be logged as a business event.

**Global error handlers** — SQLAlchemy errors and unexpected exceptions are caught centrally in `main.py` and returned as clean JSON responses. HTTPExceptions bypass these handlers and return their intended status codes.

---

## Known Limitations

**SQLite ALTER COLUMN** — SQLite does not support `ALTER COLUMN` in migrations. If a column constraint needs changing on an existing table, the migration must be edited manually to remove the `alter_column` operation. Switching to PostgreSQL resolves this — only the `DATABASE_URL` in `.env` and the installed driver need updating.

**Mock LLM only** — No real LLM calls are made. The mock uses keyword matching and returns deterministic fake responses. Replacing it with a real provider requires implementing a new adapter in `core/mock_llm.py` that matches the same return interface.

**SQLite concurrency** — SQLite has limited support for concurrent writes. For production workloads, PostgreSQL is recommended.

**API key management** — API keys are loaded from environment variables for simplicity. In production these would be stored in a secrets manager (e.g. AWS Secrets Manager, HashiCorp Vault) and looked up dynamically.
