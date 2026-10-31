const state = {
  bids: [],
  applicationRuns: [],
  runs: {application: [], historical: []},
  runScope: "application",
  inspections: new Map(),
  uploadFiles: [],
  uploadState: "selected",
  uploadedBid: null,
  comparison: null,
  lastUpload: null,
  extraction: null
};
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}
function displayValue(value) {
  if (value === null || value === undefined || value === "") return "Not found in documents";
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}
function notice(message, error = false) {
  const element = $("#notice"); element.textContent = message; element.hidden = false;
  element.classList.toggle("error", error); clearTimeout(notice.timer); notice.timer = setTimeout(() => element.hidden = true, 5000);
}
async function api(url, options = {}) {
  const response = await fetch(url, options);
  const type = response.headers.get("content-type") || "";
  const body = type.includes("json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof body === "object" ? body.detail : body;
    const message = typeof detail === "string" ? detail : (detail?.message || JSON.stringify(detail));
    throw new Error(message || `Request failed with HTTP ${response.status}`);
  }
  return body;
}
function navigate(name) {
  $$(".screen").forEach(screen => screen.classList.toggle("active", screen.id === `screen-${name}`));
  $$(".nav-item").forEach(item => item.classList.toggle("active", item.dataset.screen === name));
  const label = $(`.nav-item[data-screen='${name}']`)?.textContent.trim() || "Emplay";
  $("#page-title").textContent = label; location.hash = name; $(".sidebar").classList.remove("open");
}
function bidOptions(selected = "") {
  return state.bids.map(bid => `<option value="${escapeHtml(bid.bid_id)}" ${bid.bid_id === selected ? "selected" : ""}>${escapeHtml(bid.bid_id)}</option>`).join("");
}
function populateBidSelects() {
  $$(".bid-select").forEach(select => { const prior = select.value; select.innerHTML = bidOptions(prior); });
  const indexSelect = $("#index-bid");
  if (indexSelect) {
    const prior = state.uploadedBid || indexSelect.value;
    const eligible = state.bids.filter(bid => bid.document_count > 0);
    indexSelect.innerHTML = '<option value="">Select an uploaded bid</option>' + eligible.map(bid => `<option value="${escapeHtml(bid.bid_id)}" ${bid.bid_id === prior ? 'selected' : ''}>${escapeHtml(bid.bid_id)} · ${bid.document_count} document${bid.document_count === 1 ? '' : 's'}</option>`).join('');
    updateIndexEligibility();
  }
}
async function refreshBids() {
  const data = await api("/api/bids"); state.bids = data.bids; state.applicationRuns = data.application_runs || []; state.runs.application = state.applicationRuns; populateBidSelects(); renderDashboard();
}
function renderDashboard() {
  const documents = state.bids.reduce((sum, bid) => sum + bid.document_count, 0);
  const complete = state.bids.filter(bid => bid.extraction_status === "complete").length;
  $("#dashboard-stats").innerHTML = [
    [state.bids.length, "Bid workspaces"], [documents, "Source documents"], [complete, "Validated extractions"], ["20", "Official fields per bid"]
  ].map(([value,label]) => `<div class="stat"><small>${label}</small><strong>${value}</strong></div>`).join("");
  $("#bid-grid").innerHTML = state.bids.length ? state.bids.map(bid => `<article class="bid-card"><span class="badge ${bid.extraction_status === "complete" ? "" : "muted"}">${escapeHtml(bid.extraction_status.replace("_", " "))}</span><p class="eyebrow">${bid.document_count} DOCUMENT${bid.document_count === 1 ? "" : "S"}</p><h3>${escapeHtml(bid.bid_id)}</h3><p>Version ${escapeHtml(bid.index_version.slice(0, 10))} · evidence isolated to this workspace.</p><div class="card-footer"><span>${bid.document_count} indexed/local files</span><button class="text-button" data-open-bid="${escapeHtml(bid.bid_id)}">View fields →</button></div></article>`).join("") : `<div class="empty">No bid folders are available yet.</div>`;
  $("#recent-runs").innerHTML = state.applicationRuns.length ? state.applicationRuns.slice(0, 8).map(runCard).join('') : '<div class="empty">No application runs yet.</div>';
  $$('[data-open-bid]').forEach(button => button.onclick = () => { navigate("extract"); $("#extract-bid").value = button.dataset.openBid; loadFields(); });
  bindRunButtons($("#recent-runs"));
}

function runCard(run) {
  const identifier = run.inspection_id || run.activity_id || run.run_id;
  const bids = (run.bid_ids || []).filter(Boolean).join(", ") || "No bid";
  const status = run.status || "unknown";
  const warning = ["failed", "partial", "incomplete", "unknown"].includes(status);
  return `<article class="result-card"><div class="result-head"><div><span class="badge ${warning ? 'warn' : ''}">${escapeHtml(status)}</span> <strong>${escapeHtml(run.operation || 'workflow')}</strong></div><button class="text-button" data-inspect-run="${escapeHtml(identifier)}">Inspect →</button></div><p class="meta">${escapeHtml(bids)} · ${escapeHtml(run.source || 'application')}${run.cache_hit ? ' · cache hit' : ''}</p></article>`;
}

function bindRunButtons(root = document) {
  $$('[data-inspect-run]', root).forEach(button => button.onclick = () => {
    navigate("runs");
    $('#run-form [name=run_id]').value = button.dataset.inspectRun;
    inspectRun(button.dataset.inspectRun);
  });
}

async function refreshRuns(scope = state.runScope) {
  state.runScope = scope;
  $$('.run-tab').forEach(tab => {
    const active = tab.dataset.runScope === scope;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
  });
  $('#run-list').innerHTML = `<div class="empty">Loading ${scope} runs…</div>`;
  try {
    const payload = await api(`/api/runs?scope=${encodeURIComponent(scope)}`);
    state.runs[scope] = payload.runs || [];
    $('#run-list').innerHTML = state.runs[scope].length
      ? state.runs[scope].map(runCard).join('')
      : `<div class="empty">${scope === 'application' ? 'No application runs yet.' : 'No historical runs are available.'}</div>`;
    bindRunButtons($('#run-list'));
  } catch (error) {
    $('#run-list').innerHTML = `<div class="empty">Unable to load runs: ${escapeHtml(error.message)}</div>`;
  }
}

function renderMetrics(response) {
  if (!response.available) {
    return `<div class="empty">${escapeHtml(response.reason || 'Metrics are unavailable for this run.')}</div>`;
  }
  const metrics = response.metrics;
  const gemini = metrics.usage?.gemini || {};
  const cohere = metrics.usage?.cohere || {};
  const amount = metrics.estimated_cost?.amount;
  return `<div class="panel result-card"><p class="eyebrow">RUN METRICS</p><h3>${escapeHtml(metrics.mode)} · ${Math.round(metrics.request_latency_ms)} ms</h3><p>Retries: ${metrics.retries ?? 'unavailable'} · Provider errors: ${metrics.provider_errors ?? 'unavailable'} · Cache hit: ${metrics.cache?.hit === true ? 'yes' : 'no'}</p><p class="meta">Gemini tokens: input ${gemini.input_tokens ?? 'unavailable'}, output ${gemini.output_tokens ?? 'unavailable'}, thinking ${gemini.thinking_tokens ?? 'unavailable'} · Cohere calls ${cohere.calls ?? 'unavailable'}</p><p class="meta">Estimated cost: ${amount == null ? 'unavailable' : amount} ${escapeHtml(metrics.estimated_cost?.currency || '')}${metrics.estimated_cost?.pricing_date ? ` · pricing ${escapeHtml(metrics.estimated_cost.pricing_date)}` : ''}</p></div>`;
}

function renderTrace(trace) {
  if (!trace.available || !trace.steps?.length) {
    return `<div class="empty">${escapeHtml(trace.reason || 'No execution trace is available.')}</div>`;
  }
  const steps = trace.steps.map(step => {
    const timestamp = step.timestamp ? new Date(step.timestamp).toLocaleString() : 'timestamp unavailable';
    const duration = step.duration_ms == null ? 'duration unavailable' : `${Math.round(step.duration_ms)} ms`;
    return `<details class="trace-step"><summary><span class="trace-order">${step.order}</span><span class="trace-title"><strong>${escapeHtml(step.agent)} · ${escapeHtml(step.operation)}</strong><small>${escapeHtml(timestamp)} · ${escapeHtml(duration)}</small></span><span class="trace-status ${escapeHtml(step.status)}">${escapeHtml(step.status)}</span></summary><div class="trace-details"><pre>${escapeHtml(JSON.stringify(step.details || {}, null, 2))}</pre></div></details>`;
  }).join('');
  const context = Object.values(trace.context || {}).some(value => Array.isArray(value) ? value.length : value && typeof value === 'object' && Object.keys(value).length);
  const contextPanel = context
    ? `<details class="panel result-card context-card"><summary>Run validation, retrieval, retries and errors</summary><div class="trace-details"><pre>${escapeHtml(JSON.stringify(trace.context, null, 2))}</pre></div></details>`
    : '';
  const primarySource = trace.steps[0]?.source;
  const additional = primarySource === 'agent_trace' ? trace.workflow_events : trace.agent_traces;
  const additionalPanel = additional?.length
    ? `<details class="panel result-card context-card"><summary>${primarySource === 'agent_trace' ? 'Workflow routing events' : 'Detailed agent records'} (${additional.length})</summary><div class="trace-details"><pre>${escapeHtml(JSON.stringify(additional, null, 2))}</pre></div></details>`
    : '';
  return `<div class="trace-list">${steps}</div>${additionalPanel}${contextPanel}`;
}

async function inspectRun(identifier) {
  $('#run-form [name=run_id]').value = identifier;
  $('#inspected-run').textContent = `Inspecting ${identifier}`;
  $('#metrics-result').innerHTML = '<div class="empty">Loading recorded metrics…</div>';
  $('#trace-result').innerHTML = '<div class="empty">Loading complete trace…</div>';
  try {
    let inspection = state.inspections.get(identifier);
    if (!inspection) {
      const [metrics, trace] = await Promise.all([
        api(`/api/runs/${identifier}/metrics`),
        api(`/api/runs/${identifier}/trace`)
      ]);
      inspection = {metrics, trace};
      state.inspections.set(identifier, inspection);
    }
    $('#metrics-result').innerHTML = renderMetrics(inspection.metrics);
    $('#trace-result').innerHTML = renderTrace(inspection.trace);
    $('#inspected-run').textContent = `Inspecting ${identifier} · ${inspection.trace.step_count || 0} recorded step(s)`;
  } catch (error) {
    $('#metrics-result').innerHTML = '<div class="empty">Metrics could not be loaded.</div>';
    $('#trace-result').innerHTML = `<div class="empty">Unable to inspect this run: ${escapeHtml(error.message)}</div>`;
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function updateIndexEligibility() {
  const selected = $('#index-bid')?.value;
  const eligible = state.bids.some(bid => bid.bid_id === selected && bid.document_count > 0);
  if ($('#index-uploaded')) $('#index-uploaded').disabled = !eligible;
}

function renderUploadFiles() {
  const total = state.uploadFiles.reduce((sum, file) => sum + file.size, 0);
  const locked = ['uploading', 'uploaded', 'indexed'].includes(state.uploadState);
  const label = state.uploadFiles.length
    ? `${state.uploadFiles.length} document${state.uploadFiles.length === 1 ? '' : 's'} selected · ${formatBytes(total)}`
    : 'No files selected.';
  $('#selected-file-summary').textContent = label;
  $('#selected-files').innerHTML = state.uploadFiles.map(file => {
    const warning = ['failed', 'uploading'].includes(state.uploadState);
    const status = state.uploadState.charAt(0).toUpperCase() + state.uploadState.slice(1);
    return `<div class="selected-file"><span><strong>${escapeHtml(file.name)}</strong><small>${escapeHtml(formatBytes(file.size))} · ${escapeHtml(file.type || 'type inferred from extension')}</small></span><span class="badge upload-state ${warning ? 'warn' : ''}">${escapeHtml(status)}</span><button type="button" class="remove-file" data-remove-file="${escapeHtml(file.name)}" ${locked ? 'disabled' : ''} aria-label="Remove ${escapeHtml(file.name)}">Remove</button></div>`;
  }).join('');
  $$('[data-remove-file]', $('#selected-files')).forEach(button => button.onclick = () => {
    state.uploadFiles = state.uploadFiles.filter(file => file.name.toLocaleLowerCase() !== button.dataset.removeFile.toLocaleLowerCase());
    state.uploadState = 'selected';
    renderUploadFiles();
  });
  $('#upload-documents').disabled = !state.uploadFiles.length || locked;
}

function addUploadFiles(fileList) {
  const accepted = new Set(['.pdf', '.html', '.htm']);
  const existing = new Set(state.uploadFiles.map(file => file.name.toLocaleLowerCase()));
  const rejected = [];
  for (const file of Array.from(fileList || [])) {
    const suffix = file.name.includes('.') ? `.${file.name.split('.').pop().toLocaleLowerCase()}` : '';
    if (!accepted.has(suffix)) {
      rejected.push(`${file.name}: unsupported extension`);
      continue;
    }
    if (file.size === 0) {
      rejected.push(`${file.name}: empty file`);
      continue;
    }
    if (existing.has(file.name.toLocaleLowerCase())) {
      rejected.push(`${file.name}: duplicate filename`);
      continue;
    }
    existing.add(file.name.toLocaleLowerCase());
    state.uploadFiles.push(file);
  }
  if (state.uploadFiles.length) state.uploadState = 'selected';
  renderUploadFiles();
  if (rejected.length) {
    $('#upload-result').innerHTML = `<div class="result-card"><span class="badge warn">Not selected</span><p>${rejected.map(escapeHtml).join('<br>')}</p></div>`;
    notice(`${rejected.length} file${rejected.length === 1 ? '' : 's'} could not be selected.`, true);
  }
}
function citationButton(bidId, source, quote = "") {
  return `<button class="citation" data-citation data-bid="${escapeHtml(bidId)}" data-file="${escapeHtml(source.file || source.file_name)}" data-page="${Number(source.page || source.page_number)}" data-chunk="${escapeHtml(source.chunk_id || '')}" data-quote="${escapeHtml(quote || source.quote || "")}">${escapeHtml(source.file || source.file_name)} · p.${Number(source.page || source.page_number)}</button>`;
}
function bindCitations(root = document) { $$('[data-citation]', root).forEach(button => button.onclick = () => openEvidence(button.dataset)); }
function openEvidence(data) {
  const bid = state.bids.find(item => item.bid_id === data.bid); const doc = bid?.documents.find(item => item.name === data.file);
  if (!doc) return notice("The cited source document is not available in the local catalog.", true);
  $("#evidence-title").textContent = data.file; $("#evidence-meta").textContent = `${data.bid} · physical page ${data.page}${data.chunk ? ` · evidence ${data.chunk}` : ''}`;
  $("#evidence-quote").textContent = data.quote || "No trusted text coordinates are available. The preview is positioned at the cited physical page.";
  $("#document-frame").src = `/api/bids/${encodeURIComponent(data.bid)}/documents/${doc.document_id}#page=${encodeURIComponent(data.page)}`;
  $("#evidence-dialog").showModal();
}
async function loadFields() {
  const bidId = $("#extract-bid").value; if (!bidId) return;
  try {
    const data = await api(`/api/bids/${encodeURIComponent(bidId)}/fields`); state.extraction = data; $('#download-fields').disabled = false;
    const validation = data.validation || {};
    $("#extraction-summary").innerHTML = `<div class="summary-strip"><span><strong>${validation.passed ?? 0}</strong><small> passed</small></span><span><strong>${validation.not_found ?? 0}</strong><small> justified not found</small></span><span><strong>${validation.failed ?? 0}</strong><small> failed</small></span><span><strong>${data.addendum_changes?.length ?? 0}</strong><small> addendum changes</small></span></div>`;
    const rows = Object.entries(data.fields).map(([name, record]) => `<tr><td><strong>${escapeHtml(name)}</strong></td><td><span class="badge ${record.value == null ? 'muted' : ''}">${record.value == null ? 'Justified not found' : 'Passed'}</span></td><td class="value">${escapeHtml(displayValue(record.value))}</td><td>${Math.round((record.confidence || 0) * 100)}%</td><td>${record.sources.map(source => { const evidence = (data._evidence?.[name] || []).find(item => item.file === source.file && item.page === source.page); return citationButton(bidId, {...source, chunk_id:evidence?.chunk_id}, evidence?.quote || ''); }).join(" ") || '<span class="badge muted">No citation</span>'}</td><td>${escapeHtml(record.notes || "")}</td></tr>`).join("");
    $("#fields-table").innerHTML = `<table><thead><tr><th>Official field</th><th>Validation</th><th>Validated value</th><th>Confidence</th><th>Sources</th><th>Notes</th></tr></thead><tbody>${rows}</tbody></table>`;
    bindCitations($("#fields-table"));
  } catch (error) { $("#fields-table").innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`; }
}
async function pollJob(jobId, onUpdate = null) {
  for (let attempt = 0; attempt < 240; attempt++) {
    const job = await api(`/api/jobs/${jobId}`);
    if (onUpdate) onUpdate(job);
    notice(`${job.kind} job: ${job.status}`, job.status === "failed");
    if (["succeeded", "failed"].includes(job.status)) {
      await refreshBids();
      await refreshRuns("application");
      if (job.activity_id) await inspectRun(job.activity_id);
      return job;
    }
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  throw new Error("Job monitoring timed out; the job may still be running.");
}

document.addEventListener("DOMContentLoaded", async () => {
  $$(".nav-item").forEach(item => item.onclick = () => navigate(item.dataset.screen));
  $$('[data-go]').forEach(item => item.onclick = () => navigate(item.dataset.go));
  $("#menu-toggle").onclick = () => $(".sidebar").classList.toggle("open");
  $("#close-dialog").onclick = () => { $("#document-frame").src = "about:blank"; $("#evidence-dialog").close(); };
  $("#refresh-button").onclick = async () => { await refreshBids(); await refreshRuns(state.runScope); notice("Workspace refreshed."); };
  try { const health = await api("/api/health"); $("#health-label").textContent = health.status === "ready" ? "API ready" : health.status; $(".system-status").classList.add("ready"); } catch { $("#health-label").textContent = "API unavailable"; }
  try { await refreshBids(); } catch (error) { notice(error.message, true); }
  try { await refreshRuns("application"); } catch (error) { notice(error.message, true); }
  navigate((location.hash || "#dashboard").slice(1));
  $$('.run-tab').forEach(tab => tab.onclick = () => refreshRuns(tab.dataset.runScope));

  const sourceInput = $('#source-files');
  const dropzone = $('#upload-dropzone');
  sourceInput.addEventListener('change', event => {
    addUploadFiles(event.target.files);
    event.target.value = '';
  });
  ['dragenter', 'dragover'].forEach(name => dropzone.addEventListener(name, event => {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach(name => dropzone.addEventListener(name, event => {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.remove('dragging');
  }));
  dropzone.addEventListener('drop', event => addUploadFiles(event.dataTransfer.files));
  $('#index-bid').addEventListener('change', updateIndexEligibility);
  renderUploadFiles();

  $("#upload-form").onsubmit = async event => {
    event.preventDefault();
    const form = event.currentTarget;
    if (!form.reportValidity()) return;
    if (!state.uploadFiles.length) {
      notice('Select at least one PDF or HTML document.', true);
      return;
    }
    const bidId = form.elements.bid_id.value.trim();
    const body = new FormData();
    body.append('bid_id', bidId);
    state.uploadFiles.forEach(file => body.append('files', file, file.name));
    form.classList.add("loading");
    state.uploadState = 'uploading';
    renderUploadFiles();
    $('#upload-result').innerHTML = `<div class="result-card"><span class="badge warn">Uploading</span><p>Sending ${state.uploadFiles.length} document${state.uploadFiles.length === 1 ? '' : 's'} for ${escapeHtml(bidId)}…</p></div>`;
    try {
      const result = await api("/api/bids/upload", {method:"POST", body});
      state.lastUpload = result.bid_id;
      state.uploadedBid = result.bid_id;
      state.uploadState = 'uploaded';
      renderUploadFiles();
      await refreshBids();
      $('#index-bid').value = result.bid_id;
      updateIndexEligibility();
      $('#upload-result').innerHTML = `<div class="result-card"><span class="badge">Uploaded</span><p><strong>${escapeHtml(result.bid_id)}</strong> received all ${result.uploaded_count} document${result.uploaded_count === 1 ? '' : 's'}. Filenames and content were preserved. Ready to index.</p></div>`;
      await refreshRuns("application");
      notice("All selected documents uploaded safely.");
    } catch(error) {
      state.uploadState = 'failed';
      renderUploadFiles();
      $('#upload-result').innerHTML = `<div class="result-card"><span class="badge warn">Failed</span><p>${escapeHtml(error.message)}</p><p class="meta">The upload was rejected atomically; retry after correcting the issue.</p></div>`;
      notice(error.message, true);
    } finally {
      form.classList.remove("loading");
    }
  };
  $("#index-uploaded").onclick = async () => {
    const bidId = $('#index-bid').value;
    if (!bidId) return notice('Select an eligible uploaded bid first.', true);
    $('#index-status').innerHTML = '<div class="result-card"><span class="badge warn">Indexing</span><p>Submitting the existing ingestion and RAG indexing pipeline…</p></div>';
    try {
      const job = await api(`/api/bids/${encodeURIComponent(bidId)}/index`, {method:"POST"});
      const completed = await pollJob(job.job_id, current => {
        const progressText = current.status === 'succeeded'
          ? 'Indexed — Ready for retrieval and extraction.'
          : current.status === 'failed'
            ? `Indexing failed: ${current.error || 'Check server logs.'}`
            : 'Processing documents and updating the search index...';
        const statusLabel = current.status === 'succeeded' ? 'Indexed' : current.status;
        $('#index-status').innerHTML = `<div class="result-card"><span class="badge ${current.status === 'failed' ? 'warn' : ''}">${escapeHtml(statusLabel)}</span><p>${escapeHtml(progressText)}</p></div>`;
      });
      if (completed.status === 'failed') throw new Error(completed.error || 'Indexing failed');
      const indexOutcome = completed.result?.run_status;
      if (indexOutcome === 'partial') {
        $('#index-status').innerHTML = `<div class="result-card"><span class="badge warn">Partial</span><p>${escapeHtml(bidId)} was indexed with document-level warnings. Inspect the application run before extraction.</p></div>`;
        notice('Indexing completed partially; review the run details.', true);
        return;
      }
      if (bidId === state.uploadedBid) {
        state.uploadState = 'indexed';
        renderUploadFiles();
      }
      $('#index-status').innerHTML = '<div class="result-card"><span class="badge">Indexed</span><p>Indexed — Ready for retrieval and extraction.</p></div>';
      notice("Indexing finished.");
    } catch(error) {
      $('#index-status').innerHTML = `<div class="result-card"><span class="badge warn">Failed</span><p>${escapeHtml(error.message)}</p></div>`;
      notice(error.message, true);
    }
  };

  $("#search-form").onsubmit = async event => {
    event.preventDefault(); const values = Object.fromEntries(new FormData(event.currentTarget)); const query = new URLSearchParams(Object.entries(values).filter(([,value]) => value));
    $("#search-results").innerHTML = '<div class="empty">Searching hybrid evidence…</div>';
    try { const result = await api(`/api/search?${query}`); $("#search-results").innerHTML = result.passages.length ? result.passages.map(item => `<article class="result-card"><div class="result-head"><div><span class="badge">Rank ${item.rank}</span> <span class="meta">Score ${item.score == null ? "unavailable" : item.score.toFixed(4)}</span></div>${citationButton(values.bid_id, item.evidence, item.evidence.quote)}</div><blockquote>${escapeHtml(item.evidence.quote)}</blockquote><p class="meta">Evidence ID ${escapeHtml(item.evidence.chunk_id)}${item.evidence.section ? ` · ${escapeHtml(item.evidence.section)}` : ""}</p></article>`).join("") : '<div class="empty">No evidence matched this query.</div>'; bindCitations($("#search-results")); } catch(error) { $("#search-results").innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`; }
  };
  $("#ask-form").onsubmit = async event => {
    event.preventDefault(); const values = Object.fromEntries(new FormData(event.currentTarget)); $("#answer-result").innerHTML = '<div class="empty">Running grounded Q&amp;A…</div>';
    try { const result = await api("/api/ask", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(values)}); const answer = result.answer; $("#answer-result").innerHTML = `<article class="panel answer-card"><span class="badge">${result.cache?.hit ? "Verified cache hit" : "Fresh answer"}</span><p class="answer-text">${escapeHtml(answer.answer)}</p><p class="meta">Confidence ${Math.round(answer.confidence * 100)}% · ${escapeHtml(answer.notes || "")}</p><div class="citations">${answer.sources.map(source => { const evidence=(answer.evidence||[]).find(item=>item.bid_id===source.bid_id&&item.chunk_id===source.chunk_id); return citationButton(values.bid_id,source,evidence?.quote||''); }).join("")}</div></article>`; bindCitations($("#answer-result")); await refreshBids(); await refreshRuns("application"); } catch(error) { $("#answer-result").innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`; }
  };
  $("#load-fields").onclick = loadFields;
  $("#download-fields").onclick = () => { if (!state.extraction) return; const {_evidence, ...official} = state.extraction; const blob = new Blob([JSON.stringify(official, null, 2)], {type:'application/json'}); const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = `${$('#extract-bid').value}_extraction.json`; link.click(); URL.revokeObjectURL(link.href); };
  $("#start-extract").onclick = async () => { const bid_id = $("#extract-bid").value; try { const job = await api("/api/extract", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({bid_id,resume:true})}); await pollJob(job.job_id); await loadFields(); } catch(error) { notice(error.message,true); } };
  $("#compare-form").onsubmit = async event => {
    event.preventDefault(); const values = Object.fromEntries(new FormData(event.currentTarget)); if(values.bid_a === values.bid_b) return notice("Choose two different bids.",true);
    try { state.comparison = {bid_ids:[values.bid_a,values.bid_b]}; const result = await api("/api/compare",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(state.comparison)}); $("#export-comparison").disabled=false; $("#comparison-results").innerHTML=`<div class="table-wrap">${result.fields.map(row=>`<div class="compare-row ${row.different?'different':''}"><div class="field-label">${escapeHtml(row.field_name)}${row.different?'<p><span class="badge warn">Different</span></p>':''}</div>${result.bid_ids.map(bid=>{const record=row.bids[bid];return `<div><p class="eyebrow">${escapeHtml(bid)}</p><div class="compare-value">${escapeHtml(displayValue(record.value))}</div><p class="meta">Confidence ${Math.round((record.confidence||0)*100)}%</p><div class="citations">${record.sources.map(source=>citationButton(bid,source)).join('')}</div></div>`}).join('')}</div>`).join('')}</div>`; bindCitations($("#comparison-results")); } catch(error){notice(error.message,true);}
  };
  $("#export-comparison").onclick=async()=>{try{const text=await api('/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...state.comparison,format:'markdown'})});const blob=new Blob([text],{type:'text/markdown'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='bid-comparison.md';link.click();URL.revokeObjectURL(link.href);}catch(error){notice(error.message,true);}};
  $("#run-form").onsubmit = async event => {
    event.preventDefault();
    const runId = new FormData(event.currentTarget).get('run_id');
    await inspectRun(runId);
  };
});
