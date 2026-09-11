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
  maxSources: 12,
  maxSourcesMax: 16,
  retrievalTopK: 16,
  questionHistory: [],
  selectedAnswer: null,
};
let messageData = new WeakMap();

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
  const detailsOpen = el.indexPanel.querySelector("details")?.open === true;
  el.indexPanel.innerHTML = `
    <details class="settings-details"${detailsOpen ? " open" : ""}>
      <summary>Index <span class="index-state">${escapeHtml(statusText)}</span></summary>
      <div class="specs-rows">
      ${activated.length ? specRow(aggregate ? "latest rebuild" : "last rebuilt", formatBuiltAt(activated[activated.length - 1])) : ""}
      ${corpus.updated_at ? specRow("Notebook updated", formatBuiltAt(corpus.updated_at)) : ""}
      ${specRow("chunks", knownChunks.length ? formatInt(chunkCount) : "-")}
      </div>
    </details>
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
  const computeAvailable = state.context?.worker?.available === true;
  const controlsLocked = state.busy || !computeAvailable || state.runtimeAction !== null;
  const canStart = Boolean(state.model) && !controlsLocked && current !== "stopping" && !modelMayBeRunning(status);
  const canStop = Boolean(state.model) && !controlsLocked && current !== "stopping" && modelMayBeRunning(status);
  const visibleRuntimeAction = state.runtimeModel === state.model ? state.runtimeAction : null;
  el.runtime.innerHTML = `
    <div class="runtime-actions">
      <button id="startModelBtn" type="button" ${canStart ? "" : "disabled"}>
        ${playIcon()}<span>${visibleRuntimeAction === "start" ? "Starting..." : "Start"}</span>
      </button>
      <button id="stopModelBtn" type="button" ${canStop ? "" : "disabled"}>
        ${squareIcon()}<span>${visibleRuntimeAction === "stop" ? "Stopping..." : "Stop"}</span>
      </button>
    </div>
    ${visibleRuntimeAction ? renderProgress(
      visibleRuntimeAction === "start" ? "Starting model" : "Stopping model",
      state.runtimeProgress,
      { indeterminate: visibleRuntimeAction === "start" && progressPercent(state.runtimeProgress) === 0 },
    ) : ""}
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
  const detailsOpen = el.modelSpecs.querySelector("details")?.open === true;
  el.modelSpecs.innerHTML = rows.length
    ? `<details class="settings-details"${detailsOpen ? " open" : ""}><summary>Model specs</summary><div class="specs-rows">${rows.map(([label, value]) => specRow(label, value)).join("")}</div></details>`
    : "";
}

function specRow(label, value) {
  return `<div class="spec-row"><span>${escapeHtml(label)}</span><span class="mono">${escapeHtml(value)}</span></div>`;
}

function renderProgress(label, progress, { indeterminate = false } = {}) {
  const percent = progressPercent(progress);
  const progressValue = indeterminate ? "" : ` aria-valuenow="${percent.toFixed(0)}"`;
  const progressClass = indeterminate ? "progress-bar indeterminate" : "progress-bar";
  return `
    <div class="progress-block">
      <div class="progress-line"><span>${escapeHtml(progress?.message || label)}</span><span class="mono">${indeterminate ? "" : `${percent.toFixed(0)}%`}</span></div>
      <div class="${progressClass}" role="progressbar" aria-label="${escapeHtml(label)}" aria-valuemin="0" aria-valuemax="100"${progressValue}>
        <div${indeterminate ? "" : ` data-progress-fill="${percent.toFixed(0)}"`}></div>
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

function modelProgressMessage(action, progress) {
  if (action !== "start") return "Stopping model";
  const percent = progressPercent(progress);
  if (percent >= 100) return "Finalizing model";
  if (percent > 0) return "Loading model weights";
  return "Starting model";
}

function renderSearchDetails(plan, { open = false } = {}) {
  if (!plan || typeof plan !== "object") return "";
  const original = String(plan.originalQuestion || plan.original_question || "");
  const semantic = String(plan.semanticQuery || plan.semantic_query || "");
  const rawTerms = plan.bm25Terms || plan.bm25_terms || [];
  const terms = Array.isArray(rawTerms) ? rawTerms.map(String).filter(Boolean).slice(0, 12) : [];
  if (!original && !semantic && !terms.length) return "";
  const mode = String(plan.mode || "");
  const modeLabel = mode === "fallback" ? '<span class="search-mode">fallback</span>' : "";
  if (mode === "fallback") {
    const normalizedOriginal = original.trim().toLocaleLowerCase().replace(/\s+/g, " ");
    const normalizedSemantic = semantic.trim().toLocaleLowerCase().replace(/\s+/g, " ");
    const contextualQuery = semantic && normalizedSemantic !== normalizedOriginal
      ? `<dt>Contextual query</dt><dd>${escapeHtml(semantic)}</dd>`
      : "";
    return `
      <details class="search-details"${open ? " open" : ""}>
        <summary>Search details ${modeLabel}</summary>
        <dl>
          <dt>Original question</dt>
          <dd>${escapeHtml(original)}</dd>
          ${contextualQuery}
          <dt>Search expansion</dt>
          <dd class="search-empty">No additional expansion was generated.</dd>
        </dl>
      </details>
    `;
  }
  const termMarkup = terms.length
    ? terms.map((term) => `<span class="search-term">${escapeHtml(term)}</span>`).join("")
    : '<span class="search-empty">None</span>';
  return `
    <details class="search-details"${open ? " open" : ""}>
      <summary>Search details ${modeLabel}</summary>
      <dl>
        <dt>Original question</dt>
        <dd>${escapeHtml(original)}</dd>
        <dt>Semantic expansion</dt>
        <dd>${escapeHtml(semantic)}</dd>
        <dt>BM25 expansion</dt>
        <dd class="search-terms">${termMarkup}</dd>
      </dl>
    </details>
  `;
}

function queryStageLabel(status) {
  const detail = status?.progressDetail || status?.progress_detail;
  const stage = String(detail?.stage || "");
  if (stage === "planning") return "Planning search…";
  if (stage === "searching") return "Searching indexed notes…";
  if (stage === "answering") {
    const rawCount = detail?.retrievedCount ?? detail?.retrieved_count;
    const count = Number(rawCount);
    return Number.isFinite(count)
      ? `Answering from ${count} selected context chunk${count === 1 ? "" : "s"}…`
      : "Answering from retrieved notes…";
  }
  const stateName = String(status?.state || status?.job?.state || "").toLowerCase();
  return stateName === "queued" ? "Queued…" : "Preparing search…";
}

function renderQueryProgress(node, status) {
  const body = node?.querySelector(".markdown-body");
  if (!body) return;
  const detail = status?.progressDetail || status?.progress_detail;
  const plan = detail?.retrievalPlan || detail?.retrieval_plan;
  body.classList.remove("error");
  let stage = body.querySelector(".query-stage");
  if (!stage) {
    body.innerHTML = '<div class="query-stage" role="status"></div>';
    stage = body.querySelector(".query-stage");
  }
  const label = queryStageLabel(status);
  if (stage.textContent !== label) stage.textContent = label;
  const record = messageData.get(node);
  const planKey = JSON.stringify(plan || null);
  if (record && plan && record.planKey !== planKey) {
    const previous = body.querySelector(".search-details");
    const open = previous?.open === true;
    previous?.remove();
    body.insertAdjacentHTML("beforeend", renderSearchDetails(plan, { open }));
    record.planKey = planKey;
  }
}

function replaceQueryResult(node, result) {
  const body = node?.querySelector(".markdown-body");
  if (!body) return;
  body.classList.remove("error");
  const answer = result?.answer || "The worker returned no answer.";
  const plan = result?.retrievalPlan || result?.retrieval_plan;
  const record = messageData.get(node);
  if (!record) return;
  const disclosure = body.querySelector(".search-details");
  record.result = result;
  record.hits = result.hits || result.sources || result.citations || [];
  body.innerHTML = renderText(answer);
  if (disclosure && record.planKey === JSON.stringify(plan || null)) body.append(disclosure);
  else body.insertAdjacentHTML("beforeend", renderSearchDetails(plan, { open: disclosure?.open === true }));
  selectAnswer(node);
  el.messages.scrollTop = el.messages.scrollHeight;
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
  messageData.set(pending, {});
  const selection = { corpus: state.corpus, model: state.model };
  const payload = {
    operation: "ask",
    corpus: selection.corpus,
    question,
    retrieval_question: buildRetrievalQuestion(question),
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
    replaceQueryResult(pending, result);
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
    state.runtimeProgress = {
      percent: 0,
      message: modelProgressMessage(state.runtimeAction, 0),
    };
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
          message: modelProgressMessage(state.runtimeAction, progress),
        };
        if (state.model === targetModel) renderRuntime();
      } else if (messageNode) {
        renderQueryProgress(messageNode, status);
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

function selectAnswer(node) {
  const record = messageData.get(node);
  if (!record?.result) return;
  state.selectedAnswer = node;
  renderSources(record.hits);
  renderContextMeter(record.result);
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
  const node = target.closest(".message");
  if (!node || !el.messages.contains(node)) return;
  const link = target.closest('a[href^="#source-"]');
  if (!link || !messageData.get(node)?.result) return;
  event.preventDefault();
  if (state.selectedAnswer !== node) selectAnswer(node);
  const sourceId = link.getAttribute("href")?.slice(1);
  if (!sourceId) return;
  const source = document.getElementById(sourceId);
  if (!(source instanceof HTMLDetailsElement) || !el.sources.contains(source)) return;

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
  if (!timings?.prompt_eval_count || !timings?.num_ctx || !Number.isInteger(timings.eval_count) || timings.eval_count < 0) {
    el.contextMeter.innerHTML = "";
    return;
  }
  const inputTokens = Number(timings.prompt_eval_count);
  const outputTokens = timings.eval_count;
  const usedTokens = inputTokens + outputTokens;
  const percent = Math.min(100, (usedTokens / Number(timings.num_ctx)) * 100);
  const breakdown = `${formatInt(inputTokens)} input + ${formatInt(outputTokens)} output (including thinking)`;
  el.contextMeter.innerHTML = `
    <div class="meter-label"><span>Context</span><span>${percent.toFixed(0)}% full</span></div>
    <div class="meter" role="progressbar" aria-label="Context used" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent.toFixed(0)}" aria-valuetext="${breakdown}"><div></div></div>
    <div class="mono meter-text" title="${breakdown}">${formatInt(usedTokens)} / ${formatInt(timings.num_ctx)} tokens</div>
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
  state.selectedAnswer = null;
  messageData = new WeakMap();
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
function buildRetrievalQuestion(question) {
  const maximumLength = 30_000;
  const separator = "\n\n";
  const current = String(question).slice(0, maximumLength);
  const history = state.questionHistory.slice(-2).join(separator);
  const historyRoom = maximumLength - current.length - separator.length;
  if (!history || historyRoom <= 0) return current;
  const retainedHistory = history.slice(-historyRoom).trimStart();
  return retainedHistory ? `${retainedHistory}${separator}${current}` : current;
}
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
