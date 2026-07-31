const configuredBase = document.querySelector('meta[name="novo-chat-base-path"]')?.content || ".";
const basePath = configuredBase === "/" ? "" : configuredBase.replace(/\/$/, "");

const state = {
  context: null,
  csrfToken: "",
  corpora: [],
  corpus: "",
  models: [],
  model: "",
  modelStatus: {},
  busy: false,
  questionHistory: [],
};

const el = Object.fromEntries([
  "corpus", "indexStatus", "indexButton", "maxSources", "maxSourcesValue", "model", "modelStatus",
  "startModel", "stopModel", "clearButton", "userName", "novoLink", "computeBanner", "computeDetail",
  "retryButton", "activeCorpus", "activeModel", "messages", "askForm", "question", "askButton", "sources",
].map((id) => [id, document.getElementById(id)]));

function apiPath(segment) {
  const clean = String(segment).replace(/^\/+/, "");
  return `${basePath}/${clean}`;
}

async function initialize() {
  bindEvents();
  await refreshContext();
}

function bindEvents() {
  el.corpus.addEventListener("change", () => { state.corpus = el.corpus.value; render(); });
  el.model.addEventListener("change", () => { state.model = el.model.value; render(); });
  el.maxSources.addEventListener("input", () => { el.maxSourcesValue.value = el.maxSources.value; });
  el.retryButton.addEventListener("click", refreshContext);
  el.clearButton.addEventListener("click", clearConversation);
  el.indexButton.addEventListener("click", () => submitAndWait({ operation: "index_rebuild", corpus: state.corpus, force: true }, "index"));
  el.startModel.addEventListener("click", () => submitAndWait({ operation: "model_start", model: state.model }, "model"));
  el.stopModel.addEventListener("click", () => submitAndWait({ operation: "model_stop", model: state.model }, "model"));
  el.askForm.addEventListener("submit", (event) => { event.preventDefault(); void ask(); });
}

async function refreshContext() {
  try {
    const data = await getJson("api/context");
    state.context = data;
    state.csrfToken = data.csrfToken || "";
    state.corpora = data.corpora || [];
    state.corpus = state.corpora.some((item) => item.corpus_key === state.corpus)
      ? state.corpus
      : state.corpora[0]?.corpus_key || "";
    applyWorkerDetails(data.worker || {});
    render();
  } catch (error) {
    addMessage("assistant", `Unable to load Novo Chat: ${error.message}`, true);
  }
}

function applyWorkerDetails(worker) {
  const capabilities = worker.capabilities || {};
  const modelRows = capabilities.models || worker.models || [];
  state.models = modelRows.map((item) => typeof item === "string" ? item : item.id || item.model).filter(Boolean);
  state.modelStatus = worker.modelStatus || capabilities.modelStatus || {};
  state.model = state.models.includes(state.model) ? state.model : state.models[0] || "";
}

function render() {
  renderIdentity();
  renderCorpora();
  renderModels();
  const worker = state.context?.worker || {};
  el.computeBanner.hidden = worker.available === true;
  el.computeDetail.textContent = worker.detail || "The compute service may be offline or reconnecting. Novo is still available.";
  const corpus = state.corpora.find((item) => item.corpus_key === state.corpus);
  el.activeCorpus.textContent = corpus?.name || "No notebook selected";
  el.activeModel.textContent = state.model || "No model selected";
  el.indexStatus.textContent = corpus?.content_revision
    ? `Live Novo revision ${corpus.content_revision}`
    : "Index freshness is checked when a job starts.";
  setBusy(state.busy);
}

function renderIdentity() {
  const user = state.context?.user;
  el.userName.textContent = user?.displayName || [user?.firstName, user?.lastName].filter(Boolean).join(" ") || user?.email || "";
  el.novoLink.href = state.context?.novoHomePath || "#";
}

function renderCorpora() {
  el.corpus.replaceChildren(...state.corpora.map((corpus) => {
    const option = document.createElement("option");
    option.value = corpus.corpus_key;
    option.textContent = corpus.name;
    return option;
  }));
  el.corpus.value = state.corpus;
}

function renderModels() {
  el.model.replaceChildren(...state.models.map((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    return option;
  }));
  el.model.value = state.model;
  const status = state.modelStatus[state.model];
  const modelState = String(status?.state || "").toLowerCase();
  const labels = {
    ready: "Model running",
    running: "Model running",
    starting: "Model starting",
    draining: "Model draining",
    stopping: "Model stopping",
    stopped: "Model stopped",
    failed: "Model failed",
    unavailable: "Model unavailable",
  };
  el.modelStatus.textContent = !state.model
    ? "No approved models reported"
    : labels[modelState] || (status?.healthy === true ? "Model running" : "Model state unknown");
}

async function ask() {
  const question = el.question.value.trim();
  if (!question || state.busy) return;
  addMessage("user", question);
  el.question.value = "";
  const pending = addMessage("assistant", "Checking notebook indexes and preparing compute…");
  const payload = {
    operation: "ask",
    corpus: state.corpus,
    question,
    retrieval_question: [...state.questionHistory.slice(-2), question].join("\n\n"),
    model: state.model,
    strategy: "hybrid",
    max_sources: Number(el.maxSources.value),
  };
  try {
    const result = await submitAndWait(payload, "ask", pending);
    state.questionHistory.push(question);
    state.questionHistory = state.questionHistory.slice(-3);
    replaceMessage(pending, result.answer || "The worker returned no answer.");
    renderSources(result.hits || result.sources || result.citations || []);
  } catch (error) {
    replaceMessage(pending, `Compute error: ${error.message}`, true);
  }
}

async function submitAndWait(payload, kind, messageNode = null) {
  if (state.busy || state.context?.worker?.available !== true) {
    throw new Error("Compute is unavailable.");
  }
  state.busy = true;
  render();
  let progressNode = messageNode;
  if (!progressNode && kind !== "ask") {
    progressNode = addMessage("assistant", kind === "index" ? "Preparing notebook data…" : "Submitting model job…");
  }
  try {
    const submission = await postJson("api/jobs", payload);
    const jobId = submission.jobId || submission.job?.jobId;
    if (!jobId) throw new Error("The gateway returned an invalid job identifier.");
    while (true) {
      const status = await getJson(`api/jobs/${encodeURIComponent(jobId)}`);
      const stateName = String(status.state || status.job?.state || "").toLowerCase();
      const progress = status.progress || status.job?.progress;
      if (progressNode) replaceMessage(progressNode, jobProgressText(kind, stateName, progress));
      if (["completed", "succeeded", "success"].includes(stateName)) break;
      if (["failed", "cancelled", "canceled"].includes(stateName)) {
        const failure = status.error || status.job?.error;
        const retryHint = failure?.retryable ? " You can retry this operation." : "";
        throw new Error(`${failure?.message || failure || `Job ${stateName}.`}${retryHint}`);
      }
      await delay(1000);
    }
    const result = await getJson(`api/jobs/${encodeURIComponent(jobId)}/result`);
    if (kind === "index") replaceMessage(progressNode, "Index rebuild completed.");
    if (kind === "model") {
      replaceMessage(progressNode, "Model operation completed.");
      await refreshContext();
    }
    return result.result || result;
  } finally {
    state.busy = false;
    render();
  }
}

function jobProgressText(kind, stateName, progress) {
  const label = kind === "index" ? "Index rebuild" : kind === "model" ? "Model operation" : "Answer";
  const rawPercent = progress?.percent ?? progress;
  const numericPercent = Number(rawPercent);
  const percent = numericPercent >= 0 && numericPercent <= 1 ? numericPercent * 100 : numericPercent;
  return `${label}: ${stateName || "queued"}${Number.isFinite(percent) ? ` (${percent.toFixed(0)}%)` : ""}`;
}

function renderSources(hits) {
  if (!hits.length) {
    el.sources.innerHTML = '<p class="muted">No retrieval yet.</p>';
    return;
  }
  el.sources.innerHTML = hits.map((hit, index) => {
    const sourceHref = safeSourceHref(hit.sourceUrl || hit.source_url || hit.novoUrl || hit.novo_url);
    return `
      <details class="source" id="source-${Number(hit.source_idx || index + 1)}">
        <summary><span class="citation">#${Number(hit.source_idx || index + 1)}</span><span>${escapeHtml(hit.title || hit.file || "Novo source")}</span></summary>
        <div class="source-body">
          <div class="source-meta">${escapeHtml(hit.notebook || hit.notebookName || "")}</div>
          ${sourceHref ? `<p><a href="${escapeAttribute(sourceHref)}" target="_blank" rel="noreferrer">Open in Novo</a></p>` : ""}
          <div>${escapeHtml(hit.text || hit.excerpt || "")}</div>
        </div>
      </details>
    `;
  }).join("");
}

function safeSourceHref(value) {
  if (typeof value !== "string" || !value.startsWith("/") || value.startsWith("//") || value.includes("\\") || /%(?:2f|5c)/i.test(value)) return "";
  try {
    const parsed = new URL(value, window.location.origin);
    if (parsed.origin !== window.location.origin) return "";
    return `${parsed.pathname}${parsed.search}${parsed.hash}`;
  } catch {
    return "";
  }
}

function addMessage(role, content, isError = false) {
  document.querySelector(".welcome")?.remove();
  const node = document.createElement("article");
  node.className = "message";
  node.innerHTML = `<div class="message-role">${role}</div><div class="message-body${isError ? " error" : ""}">${renderText(content)}</div>`;
  el.messages.appendChild(node);
  el.messages.scrollTop = el.messages.scrollHeight;
  return node;
}

function replaceMessage(node, content, isError = false) {
  const body = node?.querySelector(".message-body");
  if (!body) return;
  body.classList.toggle("error", isError);
  body.innerHTML = renderText(content);
}

function clearConversation() {
  state.questionHistory = [];
  el.messages.innerHTML = '<div class="welcome"><h2>Ask your Novo notebooks</h2><p>Answers are grounded in notebooks your current Novo account can read.</p></div>';
  renderSources([]);
}

function setBusy(busy) {
  const computeAvailable = state.context?.worker?.available === true;
  for (const button of [el.askButton, el.indexButton, el.startModel, el.stopModel]) button.disabled = busy || !computeAvailable;
  el.question.disabled = busy || !computeAvailable;
}

async function getJson(segment) {
  const response = await fetch(apiPath(segment), { credentials: "same-origin", cache: "no-store" });
  return decodeResponse(response);
}

async function postJson(segment, body) {
  const response = await fetch(apiPath(segment), {
    method: "POST",
    credentials: "same-origin",
    headers: {
      "content-type": "application/json",
      "x-csrf-token": state.csrfToken,
      "idempotency-key": globalThis.crypto?.randomUUID?.() || `browser-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    },
    body: JSON.stringify(body),
  });
  return decodeResponse(response);
}

async function decodeResponse(response) {
  const text = await response.text();
  const data = text ? tryJson(text) : {};
  if (response.status === 401) {
    window.location.assign(data.loginUrl || apiPath(""));
    throw new Error("Novo sign-in is required.");
  }
  if (!response.ok) {
    const retryHint = data.error?.retryable ? " You can retry this operation." : "";
    throw new Error(`${data.detail || data.message || text || `HTTP ${response.status}`}${retryHint}`);
  }
  return data;
}

function tryJson(text) { try { return JSON.parse(text); } catch { return { detail: text }; } }
function delay(ms) { return new Promise((resolve) => window.setTimeout(resolve, ms)); }
function renderText(value) {
  return escapeHtml(value)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[((?:\d+\s*,\s*)*\d+)\]/g, (_match, numbers) => numbers.split(",").map((n) => `<a href="#source-${n.trim()}">[${n.trim()}]</a>`).join(" "));
}
function escapeHtml(value) { return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
function escapeAttribute(value) { return escapeHtml(value).replace(/'/g, "&#39;"); }

void initialize();
