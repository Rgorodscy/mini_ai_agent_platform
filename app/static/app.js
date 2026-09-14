"use strict";

/*
 * Agent Console.
 *
 * Two rules hold throughout:
 *
 * 1. Every piece of server-provided text reaches the page through
 *    textContent, never an HTML-parsing API. Model answers and tool results are
 *    untrusted — a document pulled in by retrieval can contain markup — so
 *    they are displayed as text, not interpreted. The Content-Security-Policy
 *    the server sends is the second layer, not the first.
 *
 * 2. The API key lives in sessionStorage only: it survives a reload, is gone
 *    when the tab closes, and never appears in a URL.
 */

const KEY_STORAGE = "agent-console.api-key";
const LONG_TEXT = 700;

const STEP_LABELS = {
  tool_result: "Tool result",
  tool_error: "Tool error",
  tool_blocked: "Blocked",
  final_response: "Answer",
};

const state = {
  apiKey: null,
  tenantId: null,
  agents: [],
  availableTools: [],
  selectedAgentId: null,
  run: null, // { controller, startedAt, timer }
};

// --- DOM helpers ---------------------------------------------------------

const $ = (id) => document.getElementById(id);

/**
 * Builds an element. String children become text nodes, so nothing passed
 * here is ever parsed as HTML.
 */
function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (name === "className") node.className = value;
    else if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else node.setAttribute(name, value === true ? "" : String(value));
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function storageGet() {
  try {
    return sessionStorage.getItem(KEY_STORAGE);
  } catch {
    return null;
  }
}

function storageSet(value) {
  try {
    if (value) sessionStorage.setItem(KEY_STORAGE, value);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch {
    /* storage unavailable: the key simply does not survive a reload */
  }
}

function showBanner(message, kind = "error") {
  const banner = $("banner");
  banner.textContent = message;
  banner.classList.toggle("is-info", kind === "info");
  banner.hidden = false;
}

function clearBanner() {
  $("banner").hidden = true;
}

// --- Formatting ----------------------------------------------------------

function formatSeconds(ms) {
  return `${(ms / 1000).toFixed(2)}s`;
}

function formatInt(value) {
  return typeof value === "number" ? value.toLocaleString() : "—";
}

function formatCost(value) {
  // Null means "no price configured", which is not the same as free.
  return typeof value === "number" ? `$${value.toFixed(6)}` : null;
}

function formatInput(input) {
  if (input === undefined || input === null) return null;
  if (typeof input === "string") return input;
  return JSON.stringify(input, null, 2);
}

// --- API -----------------------------------------------------------------

class ApiError extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
  }
}

async function errorDetail(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    }
  } catch {
    /* not JSON */
  }
  return `Request failed with status ${response.status}.`;
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("x-api-key", state.apiKey || "");
  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(path, {
    method: options.method || "GET",
    headers,
    body: options.json !== undefined ? JSON.stringify(options.json) : options.body,
    signal: options.signal,
  });

  if (!response.ok) {
    throw new ApiError(response.status, await errorDetail(response));
  }
  if (response.status === 204) return null;
  return options.raw ? response : response.json();
}

// --- Server-Sent Events --------------------------------------------------

/**
 * Incremental SSE parser.
 *
 * Network chunks do not respect frame boundaries: one chunk can hold half a
 * frame, or three and a half. Text is buffered until a blank line closes a
 * frame, and only complete frames are emitted. Handles CRLF, multi-line data
 * fields and comment lines.
 */
function createSSEParser(onEvent) {
  let buffer = "";

  function dispatch(frame) {
    let event = "message";
    const data = [];
    for (const line of frame.split("\n")) {
      if (!line || line.startsWith(":")) continue;
      const colon = line.indexOf(":");
      const field = colon === -1 ? line : line.slice(0, colon);
      let value = colon === -1 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") event = value;
      else if (field === "data") data.push(value);
    }
    if (!data.length) return;

    const raw = data.join("\n");
    let payload;
    try {
      payload = JSON.parse(raw);
    } catch {
      payload = raw;
    }
    onEvent(event, payload);
  }

  return {
    push(chunk) {
      let text = buffer + chunk;

      // A chunk ending in \r may be the first half of a \r\n split across
      // the network boundary. Normalising it now would turn \r\n into two
      // newlines — a spurious blank line that cuts the frame in half — so
      // it is held back until the next chunk shows what follows it.
      let held = "";
      if (text.endsWith("\r")) {
        held = "\r";
        text = text.slice(0, -1);
      }

      text = text.replace(/\r\n?/g, "\n");

      let boundary;
      while ((boundary = text.indexOf("\n\n")) !== -1) {
        dispatch(text.slice(0, boundary));
        text = text.slice(boundary + 2);
      }
      buffer = text + held;
    },
    flush() {
      const rest = buffer.replace(/\r\n?/g, "\n");
      if (rest.trim()) dispatch(rest);
      buffer = "";
    },
  };
}

async function readEventStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = createSSEParser(onEvent);

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    parser.push(decoder.decode(value, { stream: true }));
  }
  parser.push(decoder.decode());
  parser.flush();
}

// --- Rendering: agents ---------------------------------------------------

function renderAgents() {
  const list = $("agent-list");
  list.replaceChildren();

  $("agents-empty").hidden = state.agents.length > 0;

  for (const agent of state.agents) {
    const selected = agent.id === state.selectedAgentId;
    const chips = agent.tools.length
      ? el("div", { className: "chips" }, agent.tools.map((t) => el("span", { className: "chip" }, t.name)))
      : el("span", { className: "agent-role" }, "No tools");

    list.append(
      el("li", {}, [
        el(
          "button",
          {
            type: "button",
            className: "agent",
            "aria-pressed": selected ? "true" : "false",
            onclick: () => selectAgent(agent.id),
          },
          [
            el("span", { className: "agent-name" }, agent.name),
            el("span", { className: "agent-role" }, agent.role),
            chips,
          ],
        ),
      ]),
    );
  }
}

function selectAgent(agentId) {
  state.selectedAgentId = agentId;
  const agent = state.agents.find((a) => a.id === agentId);
  $("run-agent").textContent = agent ? agent.name : "Select an agent";
  $("run").disabled = !agent || Boolean(state.run);
  renderAgents();
}

function renderToolOptions() {
  const container = $("tool-options");
  container.replaceChildren();

  for (const tool of state.availableTools) {
    const id = `tool-${tool.name}`;
    container.append(
      el("label", { className: "tool-option", for: id }, [
        el("input", {
          id,
          type: "checkbox",
          name: "tools",
          value: tool.name,
          checked: tool.name === "search_knowledge",
        }),
        el("code", {}, tool.name),
        el("span", {}, tool.description),
      ]),
    );
  }
}

// --- Rendering: steps ----------------------------------------------------

function textField(label, text, { mono = true } = {}) {
  if (text === null || text === undefined || text === "") return null;

  const full = String(text);
  const long = full.length > LONG_TEXT;
  const body = el(mono ? "pre" : "div", { className: mono ? "step-pre" : "step-answer" });
  body.textContent = long ? `${full.slice(0, LONG_TEXT)}…` : full;

  const children = [label ? el("span", { className: "step-field-label" }, label) : null, body];

  if (long) {
    let expanded = false;
    const toggle = el("button", { type: "button", className: "link-button" }, "Show all");
    toggle.addEventListener("click", () => {
      expanded = !expanded;
      body.textContent = expanded ? full : `${full.slice(0, LONG_TEXT)}…`;
      toggle.textContent = expanded ? "Show less" : "Show all";
    });
    children.push(toggle);
  }

  return el("div", { className: "step-field" }, children);
}

/** Builds a timeline entry for one step. All step content is set as text. */
function renderStep(step, elapsedMs) {
  const type = STEP_LABELS[step.type] ? step.type : "tool_result";

  const head = el("div", { className: "step-head" }, [
    el("span", { className: "step-at" }, elapsedMs === undefined ? "" : formatSeconds(elapsedMs)),
    el("span", { className: `badge badge-${type}` }, STEP_LABELS[type]),
    step.tool ? el("span", { className: "step-tool" }, step.tool) : null,
  ]);

  const fields =
    type === "final_response"
      ? [textField("", step.content, { mono: false })]
      : [
          textField("Input", formatInput(step.input)),
          textField("Result", step.result),
          textField("Error", step.error),
        ];

  return el("li", { className: `step step-${type}` }, [head, ...fields]);
}

function metric(label, value, { muted = false } = {}) {
  return el("div", { className: "metric" }, [
    el("dt", {}, label),
    el("dd", { className: muted ? "is-muted" : null }, value),
  ]);
}

function renderDone(done) {
  const summary = $("run-summary");
  summary.className = "summary";
  const cost = formatCost(done.cost_usd);

  summary.replaceChildren(
    el(
      "p",
      { className: "summary-title" },
      done.status === "completed"
        ? "Run completed and recorded"
        : "Step budget exhausted before an answer — run recorded",
    ),
    el("dl", { className: "metrics" }, [
      metric("Latency", formatSeconds(done.latency_ms || 0)),
      metric("LLM calls", formatInt(done.llm_calls)),
      metric("Prompt tokens", formatInt(done.prompt_tokens)),
      metric("Completion tokens", formatInt(done.completion_tokens)),
      cost === null ? metric("Cost", "not priced", { muted: true }) : metric("Cost", cost),
    ]),
  );
  summary.hidden = false;
}

function renderRunMessage(message, kind) {
  const summary = $("run-summary");
  summary.className = `summary ${kind === "stopped" ? "is-stopped" : "is-error"}`;
  summary.replaceChildren(el("p", { className: "summary-title" }, message));
  summary.hidden = false;
}

// --- Rendering: usage ----------------------------------------------------

function renderUsage(report) {
  const cost = formatCost(report.cost_usd);
  $("usage").replaceChildren(
    metric("Runs", formatInt(report.executions)),
    metric("LLM calls", formatInt(report.llm_calls)),
    metric("Total tokens", formatInt(report.total_tokens)),
    metric(
      "Avg latency",
      typeof report.avg_latency_ms === "number" ? formatSeconds(report.avg_latency_ms) : "—",
    ),
    cost === null ? metric("Cost", "not priced", { muted: true }) : metric("Cost", cost),
  );
}

// --- Data loading --------------------------------------------------------

async function loadAgents() {
  state.agents = await api("/agents");
  if (!state.agents.some((a) => a.id === state.selectedAgentId)) {
    state.selectedAgentId = null;
  }
  selectAgent(state.selectedAgentId || (state.agents[0] && state.agents[0].id) || null);
}

async function loadUsage() {
  renderUsage(await api("/usage"));
}

async function loadModels() {
  const { models, default: fallback } = await api("/models");
  const select = $("model");
  select.replaceChildren(
    ...models.map((name) => el("option", { value: name, selected: name === fallback }, name)),
  );
}

async function loadAvailableTools() {
  state.availableTools = await api("/tools/available");
  renderToolOptions();
}

// --- Session -------------------------------------------------------------

async function connect(apiKey) {
  state.apiKey = apiKey;
  clearBanner();

  let report;
  try {
    // /usage doubles as the key check: it needs auth and names the tenant.
    report = await api("/usage");
  } catch (error) {
    state.apiKey = null;
    storageSet(null);
    showBanner(
      error.status === 401 ? "That API key is not valid for any tenant." : `Could not connect: ${error.message}`,
    );
    return;
  }

  state.tenantId = report.tenant_id;
  storageSet(apiKey);

  $("tenant-id").textContent = report.tenant_id;
  $("connect-form").hidden = true;
  $("session").hidden = false;
  $("welcome").hidden = true;
  $("workspace").hidden = false;
  $("api-key").value = "";

  renderUsage(report);

  try {
    await Promise.all([loadAgents(), loadModels(), loadAvailableTools()]);
  } catch (error) {
    showBanner(`Connected, but loading failed: ${error.message}`);
  }
}

function disconnect() {
  if (state.run) state.run.controller.abort();
  Object.assign(state, {
    apiKey: null,
    tenantId: null,
    agents: [],
    selectedAgentId: null,
  });
  storageSet(null);

  $("session").hidden = true;
  $("connect-form").hidden = false;
  $("workspace").hidden = true;
  $("welcome").hidden = false;
  resetTimeline();
  clearBanner();
}

// --- Running an agent ----------------------------------------------------

function resetTimeline() {
  $("timeline").replaceChildren();
  $("timeline-empty").hidden = false;
  $("run-summary").hidden = true;
  $("run-clock").textContent = "";
}

function setRunning(running) {
  $("run").hidden = running;
  $("stop").hidden = !running;
  $("run").disabled = running || !state.selectedAgentId;
  $("task").disabled = running;
  $("model").disabled = running;
}

async function runAgent(task, model) {
  resetTimeline();

  const controller = new AbortController();
  const startedAt = performance.now();
  const timer = setInterval(() => {
    $("run-clock").textContent = formatSeconds(performance.now() - startedAt);
  }, 100);

  state.run = { controller, startedAt, timer };
  setRunning(true);

  let finished = false;

  try {
    const response = await api(`/agents/${encodeURIComponent(state.selectedAgentId)}/run/stream`, {
      method: "POST",
      json: { task, model },
      signal: controller.signal,
      raw: true,
    });

    await readEventStream(response, (event, data) => {
      const elapsed = performance.now() - startedAt;
      if (event === "step") {
        $("timeline-empty").hidden = true;
        $("timeline").append(renderStep(data, elapsed));
      } else if (event === "done") {
        finished = true;
        renderDone(data);
      } else if (event === "error") {
        finished = true;
        renderRunMessage(data && data.detail ? data.detail : "The run failed.", "error");
      }
    });

    if (!finished) {
      renderRunMessage("The stream ended without a result.", "error");
    }
  } catch (error) {
    if (error.name === "AbortError") {
      // Honest about what Stop does: the server keeps going.
      renderRunMessage(
        "Stopped listening. The run continues on the server and will still be recorded and billed.",
        "stopped",
      );
    } else {
      // Rejected before the stream opened: a real status code, shown as-is.
      renderRunMessage(error.message, "error");
    }
  } finally {
    clearInterval(timer);
    $("run-clock").textContent = formatSeconds(performance.now() - startedAt);
    state.run = null;
    setRunning(false);
    loadUsage().catch(() => {});
  }
}

// --- Forms ---------------------------------------------------------------

async function createAgent(form) {
  const data = new FormData(form);
  const selected = data.getAll("tools");

  // Reuse this tenant's existing tool rows by name; create only what is missing.
  const existing = await api("/tools");
  const byName = new Map(existing.map((t) => [t.name, t.id]));

  const toolIds = [];
  for (const name of selected) {
    if (!byName.has(name)) {
      const spec = state.availableTools.find((t) => t.name === name);
      const created = await api("/tools", {
        method: "POST",
        json: { name, description: (spec && spec.description) || name },
      });
      byName.set(name, created.id);
    }
    toolIds.push(byName.get(name));
  }

  const agent = await api("/agents", {
    method: "POST",
    json: {
      name: data.get("name"),
      role: data.get("role"),
      description: data.get("description"),
      tools: toolIds,
    },
  });

  form.reset();
  renderToolOptions();
  $("new-agent").open = false;
  state.selectedAgentId = agent.id;
  await loadAgents();
}

async function addKnowledge(form) {
  const data = new FormData(form);
  const file = data.get("file");
  const text = String(data.get("text") || "").trim();
  const hasFile = file instanceof File && file.size > 0;

  if (!hasFile && !text) {
    throw new Error("Paste some text or choose a file.");
  }

  const body = new FormData();
  body.append("doc_id", data.get("doc_id"));
  if (hasFile) body.append("file", file);
  else body.append("text", text);

  const result = await api("/knowledge", { method: "POST", body });
  form.reset();
  return result;
}

function wireForm(id, handler, { busyLabel } = {}) {
  const form = $(id);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector('button[type="submit"]');
    const label = button.textContent;
    button.disabled = true;
    if (busyLabel) button.textContent = busyLabel;
    try {
      await handler(form);
    } catch (error) {
      showBanner(error.message);
    } finally {
      button.disabled = false;
      button.textContent = label;
    }
  });
}

// --- Boot ----------------------------------------------------------------

function init() {
  $("connect-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const key = $("api-key").value.trim();
    if (key) connect(key);
  });

  $("disconnect").addEventListener("click", disconnect);
  $("refresh-agents").addEventListener("click", () => loadAgents().catch((e) => showBanner(e.message)));
  $("refresh-usage").addEventListener("click", () => loadUsage().catch((e) => showBanner(e.message)));
  $("stop").addEventListener("click", () => state.run && state.run.controller.abort());

  $("run-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const task = $("task").value.trim();
    if (!task || !state.selectedAgentId || state.run) return;
    clearBanner();
    runAgent(task, $("model").value);
  });

  wireForm("agent-form", createAgent, { busyLabel: "Creating…" });

  wireForm(
    "knowledge-form",
    async (form) => {
      const status = $("knowledge-status");
      status.textContent = "Chunking and embedding…";
      try {
        const result = await addKnowledge(form);
        status.textContent = `Added “${result.doc_id}”.`;
      } catch (error) {
        status.textContent = "";
        throw error;
      }
    },
    { busyLabel: "Adding…" },
  );

  const saved = storageGet();
  if (saved) connect(saved);
}

document.addEventListener("DOMContentLoaded", init);

// Exposed for inspection and debugging only; the page does not use it.
window.agentConsole = { createSSEParser, renderStep };
