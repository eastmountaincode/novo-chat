const configuredBase = document.querySelector('meta[name="novo-chat-base-path"]')?.content || ".";
const basePath = configuredBase === "/" ? "" : configuredBase.replace(/\/$/, "");

const state = {
  context: null,
  csrfToken: "",
  corpora: [],
  corpus: "",
  indexStatus: null,
  indexStatusGeneration: 0,
  models: [],
  model: "",
  modelStatus: {},
  modelDetails: {},
  busy: false,
  rebuildingIndex: false,
  rebuildingCorpus: "",
  indexProgress: null,
  runtimeAction: null,
  runtimeModel: "",
  runtimeProgress: null,
  maxSources: 16,
  maxSourcesMax: 16,
  retrievalTopK: 16,
  questionHistory: [],
};

const MODEL_MAY_BE_RUNNING = new Set(["ready", "running", "starting", "draining", "failed"]);

const el = Object.fromEntries([
  "corpus", "indexPanel", "indexBtn", "maxSources", "maxSourcesValue", "model", "runtime",
  "modelSpecs", "clearBtn", "userName", "novoLink", "computeBanner",
  "computeDetail", "retryButton", "activeCorpus", "activeModel", "messages", "askForm",
  "question", "askBtn", "contextMeter", "sources",
].map((id) => [id, document.getElementById(id)]));

function apiPath(segment) {
  const clean = String(segment).replace(/^\/+/, "");
  return `${basePath}/${clean}`;
}

async function initialize() {
  renderStaticControls();
  bindEvents();
  await refreshContext();
  window.setInterval(() => void refreshWorker(), 15000);
}

function renderStaticControls() {
  el.clearBtn.innerHTML = `${trashIcon()}<span>Clear conversation</span>`;
  el.askBtn.innerHTML = sendIcon();
}

function bindEvents() {
  el.corpus.addEventListener("change", () => {
    state.corpus = el.corpus.value;
    state.indexStatus = null;
    state.indexStatusGeneration += 1;
    clearConversation();
    render();
    void refreshIndexStatus();
  });
  el.model.addEventListener("change", () => { state.model = el.model.value; render(); });
  el.maxSources.addEventListener("input", () => {
    state.maxSources = clampNumber(Number(el.maxSources.value), 1, state.maxSourcesMax);
    renderContextControl();
  });
  el.retryButton.addEventListener("click", () => void refreshContext());
  el.clearBtn.addEventListener("click", clearConversation);
  el.indexBtn.addEventListener("click", () => void rebuildIndex());
  el.askForm.addEventListener("submit", (event) => { event.preventDefault(); void ask(); });
  el.question.addEventListener("input", () => setBusy(state.busy));
  el.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void ask();
    }
  });
  el.messages.addEventListener("click", openCitationSource);
}

async function refreshContext() {
  try {
    const previousCorpus = state.corpus;
    const data = await getJson("api/context");
    state.context = data;
    state.csrfToken = data.csrfToken || "";
    state.corpora = sortCorpora(data.corpora || []);
    state.corpus = state.corpora.some((item) => item.corpus_key === state.corpus)
      ? state.corpus
      : state.corpora[0]?.corpus_key || "";
    if (
      data.worker?.available !== true
      || state.corpus !== previousCorpus
      || state.indexStatus?.corpus !== state.corpus
    ) {
      state.indexStatus = null;
      state.indexStatusGeneration += 1;
    }
    applyWorkerDetails(data.worker || {});
    render();
    if (data.worker?.available === true) await refreshIndexStatus();
  } catch (error) {
    addMessage("assistant", `Unable to load Novo Chat: ${error.message}`, true);
  }
}

async function refreshWorker() {
  try {
    const wasAvailable = state.context?.worker?.available === true;
    const worker = await getJson("api/worker");
    if (!state.context) return;
    state.context.worker = worker;
    applyWorkerDetails(worker);
    if (worker.available !== true) {
      state.indexStatus = null;
      state.indexStatusGeneration += 1;
    }
    const shouldRefreshIndex = worker.available === true && (
      !wasAvailable || !state.indexStatus || state.indexStatus.corpus !== state.corpus
    );
    render();
    if (shouldRefreshIndex) await refreshIndexStatus();
  } catch (_error) {
    if (!state.context) return;
    state.context.worker = {
      available: false,
      detail: "The compute service may be offline or reconnecting. Novo is still available.",
    };
    state.indexStatus = null;
    state.indexStatusGeneration += 1;
    render();
  }
}

async function refreshIndexStatus() {
  const corpus = state.corpus;
  const generation = ++state.indexStatusGeneration;
  if (!corpus || state.context?.worker?.available !== true) {
    state.indexStatus = null;
    renderIndexPanel();
    return;
  }
  try {
    const status = await getJson(`api/index-status?corpus=${encodeURIComponent(corpus)}`);
    if (state.corpus === corpus && state.indexStatusGeneration === generation) {
      state.indexStatus = status;
      renderIndexPanel();
    }
  } catch (_error) {
    if (state.corpus === corpus && state.indexStatusGeneration === generation) {
      state.indexStatus = null;
      renderIndexPanel();
    }
  }
}

function applyWorkerDetails(worker) {
  const capabilities = worker.capabilities || {};
  const modelRows = capabilities.models || worker.models || [];
  state.models = modelRows.map((item) => typeof item === "string" ? item : item.id || item.model).filter(Boolean);
  state.modelStatus = worker.modelStatus || capabilities.modelStatus || {};
  state.modelDetails = capabilities.modelDetails || capabilities.model_details || {};
  const healthy = state.models.find((model) => modelIsReady(state.modelStatus[model]));
  state.model = state.models.includes(state.model) ? state.model : healthy || state.models[0] || "";
}

function render() {
  renderIdentity();
  renderCorpora();
  renderModels();
  renderIndexPanel();
  renderContextControl();
  renderRuntime();
  renderModelSpecs();
  const worker = state.context?.worker || {};
  el.computeBanner.hidden = worker.available === true;
  el.computeDetail.textContent = worker.detail || "The compute service may be offline or reconnecting. Novo is still available.";
  const corpus = state.corpora.find((item) => item.corpus_key === state.corpus);
  el.activeCorpus.textContent = corpus?.name || state.corpus || "-";
  el.activeModel.textContent = state.model || "-";
  setBusy(state.busy);
}

function renderIdentity() {
  const user = state.context?.user;
  const label = user?.displayName || [user?.firstName, user?.lastName].filter(Boolean).join(" ") || user?.email || "";
  el.userName.textContent = label;
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

function sortCorpora(corpora) {
  const all = corpora.filter((corpus) => corpus.corpus_key === "novo:all");
  const notebooks = corpora
    .filter((corpus) => corpus.corpus_key !== "novo:all")
    .sort((a, b) => String(a.name || a.corpus_key).localeCompare(
      String(b.name || b.corpus_key), undefined, { numeric: true, sensitivity: "base" },
    ));
  return [...all, ...notebooks];
}

function renderModels() {
  el.model.replaceChildren(...state.models.map((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = `${model} (${modelOptionState(state.modelStatus[model])})`;
    return option;
  }));
  el.model.value = state.model;
}

function modelState(status) {
  return String(status?.state || (status?.healthy === true ? "ready" : "unknown")).toLowerCase();
}

function modelIsReady(status) {
  return ["ready", "running"].includes(modelState(status));
}

function modelMayBeRunning(status) {
  return MODEL_MAY_BE_RUNNING.has(modelState(status));
}

function modelOptionState(status) {
  const current = modelState(status);
  if (["ready", "running"].includes(current)) return "running";
  if (["starting", "draining", "stopping", "failed", "unavailable"].includes(current)) return current;
  return "stopped";
}

function selectedCorpus() {
  return state.corpora.find((corpus) => corpus.corpus_key === state.corpus);
}

function renderIndexPanel() {
  const corpus = selectedCorpus();
  if (!corpus) {
    el.indexPanel.innerHTML = "";
    el.indexBtn.classList.add("hidden-ui");
    return;
  }
  const rows = state.indexStatus?.indexes || [];
  const activated = rows.map((row) => row.activatedAt).filter(Boolean).sort();
  const knownChunks = rows.filter((row) => Number.isFinite(Number(row.chunkCount)));
  const chunkCount = knownChunks.reduce((total, row) => total + Number(row.chunkCount), 0);
  const readyCount = rows.filter((row) => row.exactReady).length;
  const needsRebuild = rows.length - readyCount;
  const aggregate = corpus.corpus_key === "novo:all";
  const rebuildingSelected = state.rebuildingIndex && state.rebuildingCorpus === state.corpus;
  const statusText = rebuildingSelected
    ? "Rebuilding"
    : !state.indexStatus
      ? "Checking"
      : state.indexStatus.exactReady
        ? "Ready"
        : aggregate ? `${needsRebuild} of ${rows.length} need rebuild` : "Needs rebuild";
  el.indexPanel.innerHTML = `
    <div class="specs-title">Index</div>
    <div class="specs-rows">
      ${specRow("status", statusText)}
      ${activated.length ? specRow(aggregate ? "latest rebuild" : "last rebuilt", formatBuiltAt(activated[activated.length - 1])) : ""}
      ${corpus.updated_at ? specRow("Notebook updated", formatBuiltAt(corpus.updated_at)) : ""}
      ${specRow("chunks", knownChunks.length ? formatInt(chunkCount) : "-")}
    </div>
    ${rebuildingSelected ? renderProgress("Rebuilding index", state.indexProgress) : ""}
  `;
  applyProgressWidths(el.indexPanel);
  el.indexBtn.classList.remove("hidden-ui");
  el.indexBtn.innerHTML = `${refreshIcon()}<span>${rebuildingSelected ? "Rebuilding..." : "Rebuild index"}</span>`;
}

function renderContextControl() {
  const max = Math.max(1, Number(state.maxSourcesMax || 16));
  state.maxSources = clampNumber(state.maxSources, 1, max);
  el.maxSources.min = "1";
  el.maxSources.max = String(max);
  el.maxSources.value = String(state.maxSources);
  el.maxSourcesValue.textContent = String(state.maxSources);
}

function renderRuntime() {
  const status = state.modelStatus[state.model];
  const current = modelState(status);
  const labels = {
    ready: "Running",
    running: "Running",
    starting: "Starting...",
    draining: "Draining...",
    stopping: "Stopping...",
    stopped: "Stopped",
    failed: "Model failed",
    unavailable: "Model unavailable",
    unknown: "Checking runtime status...",
  };
  const computeAvailable = state.context?.worker?.available === true;
  const controlsLocked = state.busy || !computeAvailable || state.runtimeAction !== null;
  const canStart = Boolean(state.model) && !controlsLocked && current !== "stopping" && !modelMayBeRunning(status);
  const canStop = Boolean(state.model) && !controlsLocked && current !== "stopping" && modelMayBeRunning(status);
  const visibleRuntimeAction = state.runtimeModel === state.model ? state.runtimeAction : null;
  el.runtime.innerHTML = `
    <div class="runtime-status">${escapeHtml(state.model ? labels[current] || "Checking runtime status..." : "No approved models reported")}</div>
    <div class="runtime-actions">
      <button id="startModelBtn" type="button" ${canStart ? "" : "disabled"}>
        ${playIcon()}<span>${visibleRuntimeAction === "start" ? "Starting..." : "Start"}</span>
      </button>
      <button id="stopModelBtn" type="button" ${canStop ? "" : "disabled"}>
        ${squareIcon()}<span>${visibleRuntimeAction === "stop" ? "Stopping..." : "Stop"}</span>
      </button>
    </div>
    ${visibleRuntimeAction ? renderProgress(visibleRuntimeAction === "start" ? "Starting model" : "Stopping model", state.runtimeProgress) : ""}
  `;
  applyProgressWidths(el.runtime);
  document.getElementById("startModelBtn")?.addEventListener("click", () => void controlRuntime("start"));
  document.getElementById("stopModelBtn")?.addEventListener("click", () => void controlRuntime("stop"));
}

function renderModelSpecs() {
  const spec = state.modelDetails[state.model];
  if (!spec) {
    el.modelSpecs.innerHTML = "";
    return;
  }
  const maxTokens = spec.maxTokens ?? spec.max_tokens;
  const maxModelLen = spec.maxModelLen ?? spec.max_model_len;
  const totalVramGb = spec.totalVramGb ?? spec.total_vram_gb;
  const rows = [
    ["model size", spec.modelSize || spec.model_size],
    ["total VRAM", totalVramGb != null && Number.isFinite(Number(totalVramGb))
      ? `${Number(totalVramGb).toLocaleString(undefined, { maximumFractionDigits: 2 })} GB`
      : null],
    ["max output", maxTokens != null ? `${formatInt(maxTokens)} tokens` : null],
    ["context", maxModelLen != null ? `${formatInt(maxModelLen)} tokens` : null],
    ["thinking", spec.thinking],
  ].filter((row) => row[1] !== null && row[1] !== undefined && row[1] !== "");
  el.modelSpecs.innerHTML = rows.length
    ? `<div class="specs-title">Model specs</div><div class="specs-rows">${rows.map(([label, value]) => specRow(label, value)).join("")}</div>`
    : "";
}

function specRow(label, value) {
  return `<div class="spec-row"><span>${escapeHtml(label)}</span><span class="mono">${escapeHtml(value)}</span></div>`;
}

function renderProgress(label, progress) {
  const percent = progressPercent(progress);
  return `
    <div class="progress-block">
      <div class="progress-line"><span>${escapeHtml(progress?.message || label)}</span><span class="mono">${percent.toFixed(0)}%</span></div>
      <div class="progress-bar" role="progressbar" aria-label="${escapeHtml(label)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent.toFixed(0)}">
        <div data-progress-fill="${percent.toFixed(0)}"></div>
      </div>
    </div>
  `;
}

function applyProgressWidths(root) {
  root.querySelectorAll("[data-progress-fill]").forEach((fill) => {
    fill.style.width = `${fill.dataset.progressFill}%`;
  });
}

function progressPercent(progress) {
  const raw = progress?.percent ?? progress ?? 0;
  const numeric = Number(raw);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(0, Math.min(100, numeric >= 0 && numeric <= 1 ? numeric * 100 : numeric));
}

async function ask() {
  const question = el.question.value.trim();
  if (!question || !state.corpus || state.busy) return;
  if (!modelIsReady(state.modelStatus[state.model])) {
    addMessage("assistant", "Start a model before asking.");
    return;
  }
  addMessage("user", question);
  el.question.value = "";
  const pending = addMessage("assistant", "thinking...");
  const selection = { corpus: state.corpus, model: state.model };
  const payload = {
    operation: "ask",
    corpus: selection.corpus,
    question,
    retrieval_question: [...state.questionHistory.slice(-2), question].join("\n\n"),
    model: selection.model,
    strategy: "hybrid",
    max_sources: state.maxSources,
    retrieval_top_k: state.retrievalTopK,
  };
  try {
    const result = await submitAndWait(payload, "ask", pending);
    if (!selectionMatches(selection)) {
      replaceMessage(pending, "The corpus or model changed while this job was running. Its result was not displayed.", true);
      return;
    }
    rememberQuestion(question);
    replaceMessage(pending, result.answer || "The worker returned no answer.");
    renderSources(result.hits || result.sources || result.citations || []);
    renderContextMeter(result);
  } catch (error) {
    if (selectionMatches(selection)) replaceMessage(pending, `Error: ${error.message}`, true);
  } finally {
    await refreshIndexStatus();
  }
}

async function submitAndWait(payload, kind, messageNode = null) {
  if (state.busy || state.context?.worker?.available !== true) {
    throw new Error("Compute is unavailable.");
  }
  state.busy = true;
  const targetCorpus = String(payload.corpus || "");
  const targetModel = String(payload.model || "");
  if (kind === "index") {
    state.rebuildingIndex = true;
    state.rebuildingCorpus = targetCorpus;
    state.indexProgress = { percent: 0, message: "Rebuilding index" };
  }
  if (kind === "model") {
    state.runtimeAction = payload.operation === "model_start" ? "start" : "stop";
    state.runtimeModel = targetModel;
    state.runtimeProgress = { percent: 0, message: state.runtimeAction === "start" ? "Starting model" : "Stopping model" };
  }
  render();
  try {
    const submission = await postJson("api/jobs", payload);
    const jobId = submission.jobId || submission.job?.jobId;
    if (!jobId) throw new Error("The gateway returned an invalid job identifier.");
    while (true) {
      const status = await getJson(`api/jobs/${encodeURIComponent(jobId)}`);
      const stateName = String(status.state || status.job?.state || "").toLowerCase();
      const progress = status.progress ?? status.job?.progress ?? 0;
      if (kind === "index") {
        state.indexProgress = { percent: progress, message: "Rebuilding index" };
        if (state.corpus === targetCorpus) renderIndexPanel();
      } else if (kind === "model") {
        state.runtimeProgress = {
          percent: progress,
          message: state.runtimeAction === "start" ? "Starting model" : "Stopping model",
        };
        if (state.model === targetModel) renderRuntime();
      } else if (messageNode && stateName && stateName !== "queued") {
        replaceMessage(messageNode, "thinking...");
      }
      if (["completed", "succeeded", "success"].includes(stateName)) break;
      if (["failed", "cancelled", "canceled"].includes(stateName)) {
        const failure = status.error || status.job?.error;
        const retryHint = failure?.retryable ? " You can retry this operation." : "";
        throw new Error(`${failure?.message || failure || `Job ${stateName}.`}${retryHint}`);
      }
      await delay(1000);
    }
    const result = await getJson(`api/jobs/${encodeURIComponent(jobId)}/result`);
    if (kind === "model") await refreshWorker();
    return result.result || result;
  } finally {
    state.busy = false;
    if (kind === "index") {
      state.rebuildingIndex = false;
      state.rebuildingCorpus = "";
      state.indexProgress = null;
    }
    if (kind === "model") {
      state.runtimeAction = null;
      state.runtimeModel = "";
      state.runtimeProgress = null;
    }
    render();
  }
}

async function rebuildIndex() {
  if (!state.corpus || state.busy) return;
  const corpus = state.corpus;
  try {
    await submitAndWait({ operation: "index_rebuild", corpus, force: true }, "index");
    if (state.corpus === corpus) await refreshIndexStatus();
  } catch (error) {
    addMessage("assistant", `Index error: ${error.message}`, true);
  }
}

async function controlRuntime(action) {
  const model = state.model;
  if (!model || state.busy || state.context?.worker?.available !== true) return;
  try {
    await submitAndWait({ operation: `model_${action}`, model }, "model");
  } catch (error) {
    addMessage("assistant", `Runtime error: ${error.message}`, true);
  } finally {
    await refreshWorker();
  }
}

function renderSources(hits) {
  if (!hits.length) {
    el.sources.innerHTML = '<div class="empty">No retrieval yet.</div>';
    return;
  }
  el.sources.innerHTML = hits.map((hit, index) => {
    const sourceHref = safeSourceHref(hit.sourceUrl || hit.source_url || hit.novoUrl || hit.novo_url);
    const sourceIndex = Number(hit.sourceIdx || hit.source_idx || index + 1);
    const notebookId = hit.notebookId || hit.notebook_id || "";
    const notebook = state.corpora.find((item) => item.id === notebookId)?.name || hit.notebook || hit.notebookName || notebookId;
    const used = hit.usedInContext ?? hit.used_in_context;
    return `
      <details class="source ${used === false ? "source-ranked-only" : ""}" id="source-${sourceIndex}">
        <summary>
          <span class="mono">#${sourceIndex}</span>
          <span class="mono">${Number(hit.score || 0).toFixed(3)}</span>
          <code>${escapeHtml(hit.file || hit.title || "Novo source")}</code>
        </summary>
        <div class="source-body">
          <div class="source-title">${escapeHtml(hit.title || "")}</div>
          <div class="source-meta">${escapeHtml(notebook)}</div>
          <div class="mono source-meta">${escapeHtml(sourceMetrics(hit))}</div>
          ${sourceHref ? `<div><a href="${escapeAttribute(sourceHref)}" target="_blank" rel="noreferrer">Open Novo page</a></div>` : ""}
          <div class="source-text">${escapeHtml(hit.text || hit.excerpt || "")}</div>
        </div>
      </details>
    `;
  }).join("");
}

function openCitationSource(event) {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const link = target.closest('a[href^="#source-"]');
  if (!link || !el.messages.contains(link)) return;
  const sourceId = link.getAttribute("href")?.slice(1);
  if (!sourceId) return;
  const source = document.getElementById(sourceId);
  if (!(source instanceof HTMLDetailsElement) || !el.sources.contains(source)) return;

  event.preventDefault();
  source.open = true;
  source.scrollIntoView({ behavior: "smooth", block: "nearest" });
  source.querySelector("summary")?.focus({ preventScroll: true });
}

function sourceMetrics(hit) {
  const parts = [];
  const chunk = hit.chunkIdx ?? hit.chunk_idx;
  if (chunk !== null && chunk !== undefined) parts.push(`chunk ${chunk}`);
  if (hit.bm25 !== null && hit.bm25 !== undefined) parts.push(`bm25=${Number(hit.bm25).toFixed(2)}`);
  if (hit.dense !== null && hit.dense !== undefined) parts.push(`dense=${Number(hit.dense).toFixed(3)}`);
  return parts.join(" - ");
}

function renderContextMeter(result) {
  const timings = result?.timings;
  if (!timings?.prompt_eval_count || !timings?.num_ctx) {
    el.contextMeter.innerHTML = "";
    return;
  }
  const percent = Math.min(100, (Number(timings.prompt_eval_count) / Number(timings.num_ctx)) * 100);
  el.contextMeter.innerHTML = `
    <div class="meter-label"><span>Context</span><span>${percent.toFixed(0)}% full</span></div>
    <div class="meter" role="progressbar" aria-label="Context used" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent.toFixed(0)}"><div></div></div>
    <div class="mono meter-text">${formatInt(timings.prompt_eval_count)} / ${formatInt(timings.num_ctx)} tokens</div>
  `;
  el.contextMeter.querySelector(".meter > div").style.width = `${percent}%`;
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
  const row = document.createElement("div");
  row.className = "message";
  row.innerHTML = `
    <div class="role">${role === "user" ? "user" : "assistant"}</div>
    <div class="message-body"><div class="markdown-body${isError ? " error" : ""}">${renderText(content)}</div></div>
  `;
  el.messages.appendChild(row);
  el.messages.scrollTop = el.messages.scrollHeight;
  return row;
}

function replaceMessage(node, content, isError = false) {
  const body = node?.querySelector(".markdown-body");
  if (!body) return;
  body.classList.toggle("error", isError);
  body.innerHTML = renderText(content);
  el.messages.scrollTop = el.messages.scrollHeight;
}

function clearConversation() {
  state.questionHistory = [];
  el.messages.innerHTML = "";
  renderSources([]);
  renderContextMeter(null);
}

function setBusy(busy) {
  const computeAvailable = state.context?.worker?.available === true;
  const selectionLocked = busy;
  el.corpus.disabled = selectionLocked || state.corpora.length === 0;
  el.model.disabled = selectionLocked || state.models.length === 0;
  el.maxSources.disabled = selectionLocked;
  el.clearBtn.disabled = selectionLocked;
  el.retryButton.disabled = selectionLocked;
  el.askBtn.disabled = busy || !computeAvailable || !modelIsReady(state.modelStatus[state.model]) || !el.question.value.trim();
  el.indexBtn.disabled = busy || !computeAvailable || !state.corpus;
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
    const detail = typeof data.detail === "string"
      ? data.detail
      : typeof data.message === "string"
        ? data.message
        : text || `HTTP ${response.status}`;
    throw new Error(`${detail}${retryHint}`);
  }
  return data;
}

function tryJson(text) { try { return JSON.parse(text); } catch { return { detail: text }; } }
function delay(ms) { return new Promise((resolve) => window.setTimeout(resolve, ms)); }
function rememberQuestion(question) {
  state.questionHistory = [...state.questionHistory.filter((item) => item !== question), question].slice(-3);
}
function selectionMatches(selection) {
  return state.corpus === selection.corpus && state.model === selection.model;
}
function renderText(value) {
  return normalizeCitationBrackets(escapeHtml(value))
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[((?:\d+\s*,\s*)*\d+)\]/g, (_match, numbers) => numbers.split(",").map((n) => `<a class="citation-link" href="#source-${n.trim()}">${n.trim()}</a>`).join(" "));
}
function normalizeCitationBrackets(value) {
  return value
    .replace(/[【\[](\d+)(?:†[^】\]]*)?[】\]]/g, "[$1]")
    .replace(/[【\[]((?:\d+\s*,\s*)+\d+)[】\]]/g, "[$1]");
}
function formatInt(value) { return Number(value || 0).toLocaleString(); }
function clampNumber(value, min, max) {
  const numeric = Number.isFinite(value) ? value : min;
  return Math.max(min, Math.min(max, Math.round(numeric)));
}
function formatBuiltAt(value) {
  if (!value) return "not built";
  const normalized = String(value).replace(/([+-]\d{2})(\d{2})$/, "$1:$2");
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
  }).format(date);
}
function escapeHtml(value) { return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;"); }
function escapeAttribute(value) { return escapeHtml(value).replace(/'/g, "&#39;"); }

function playIcon() {
  return `<svg class="button-icon" xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="6 3 20 12 6 21 6 3"></polygon></svg>`;
}

function squareIcon() {
  return `<svg class="button-icon" xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="6" y="6" width="12" height="12"></rect></svg>`;
}

function trashIcon() {
  return `<svg class="button-icon clear-icon" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 11v6"></path><path d="M14 11v6"></path><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"></path><path d="M3 6h18"></path><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>`;
}

function sendIcon() {
  return `<svg class="button-icon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z"></path><path d="m21.854 2.147-10.94 10.939"></path></svg>`;
}

function refreshIcon() {
  return `<svg class="button-icon" xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 0 1 15.4-6.4L21 8"></path><path d="M21 3v5h-5"></path><path d="M21 12a9 9 0 0 1-15.4 6.4L3 16"></path><path d="M3 21v-5h5"></path></svg>`;
}

void initialize();
