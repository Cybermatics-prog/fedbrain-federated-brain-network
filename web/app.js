// SPDX-License-Identifier: MIT
/* FedBrain Research Workspace — dependency-free, real API data only. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const icon = (name, cls = '') => `<svg${cls ? ` class="${cls}"` : ''} aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const num = (v) => (v !== null && v !== undefined && v !== '' && typeof v !== 'boolean' && Number.isFinite(Number(v))) ? Number(v) : null;
  const fmt = (v, digits = 0) => num(v) === null ? '—' : Number(v).toLocaleString('en-US', {maximumFractionDigits:digits, minimumFractionDigits:digits});
  const percentage = (v, digits = 1) => num(v) === null ? '—' : `${((Math.abs(Number(v)) <= 1 ? Number(v) * 100 : Number(v))).toFixed(digits)}%`;
  const clamp = (v, low, high) => Math.max(low, Math.min(high, v));
  const statusNames = {running:'Training',training:'Training',queued:'Queued',preparing:'Preparing',starting:'Starting',pending:'Pending',completed:'Completed',complete:'Completed',finished:'Completed',failed:'Failed',error:'Failed',stopped:'Stopped',cancelled:'Cancelled',stopping:'Stopping',ready:'Ready'};
  const activeStatuses = new Set(['running','training','queued','preparing','starting','stopping']);
  const state = {status:null, optimization:null, optimizationError:null, runs:[], run:null, selectedRun:null, selectionInitialized:false, manualSelection:false, view:'overview', subjects:[], subjectsLoaded:false, subjectsLoading:false, subject:null, selectedSubject:null, subjectLoading:false, busy:false, trainBusy:false, stopBusy:false, polling:null, subjectRequest:0, listRequest:0, lastUpdate:null};
  const siteMeta = [
    {id:'NYU',name:'New York University',code:'NY',color:'#749c80',light:'#f0f6ec',border:'#dfeada'},
    {id:'UM',name:'University of Michigan',code:'UM',color:'#9c8cb5',light:'#f6f2fa',border:'#e8e1f0'},
    {id:'USM',name:'University of Utah School of Medicine',code:'US',color:'#ab996d',light:'#f9f6ec',border:'#eee6d2'},
    {id:'UCLA',name:'University of California, Los Angeles',code:'UC',color:'#819bad',light:'#eff5f8',border:'#dfe9ee'},
    {id:'KKI',name:'Kennedy Krieger Institute',code:'KK',color:'#b38e88',light:'#faf2ef',border:'#efe1db'},
    {id:'YALE',name:'Yale University',code:'YA',color:'#8fa477',light:'#f3f6eb',border:'#e3e9d6'}
  ];
  const titles = {
    overview:['Brain Health Data','Brain Health Data','Explore ABIDE I connectivity, BOLD time series, group differences, and federated data flow.'],
    train:['Federated Training','Federated Training','Set training parameters and inspect progress and run records.'],
    inference:['Model Inference','Individual Predictions','Inspect subject connectivity and model predictions.'],
    protocol:['Study Protocol','Study Protocol','Data processing, model configuration, and evaluation methods.']
  };
  const canonicalSite = (s) => String(s || '').toUpperCase().replace(/[_\s-]/g, '').replace(/^(UM|UCLA)1$/, '$1');
  const isActive = (run) => !!run && activeStatuses.has(String(run.status || '').toLowerCase());
  const methodName = (method) => String(method || '').toLowerCase() === 'fedavg' ? 'FedAvg' : String(method || '').toLowerCase() === 'fedbrain' ? 'FedBrain' : method || 'Not recorded';
  const statusText = (status) => statusNames[String(status || '').toLowerCase()] || status || 'Not started';
  const statusClass = (status) => isActive({status}) ? 'running' : ['failed','error'].includes(String(status).toLowerCase()) ? 'failed' : ['completed','complete','finished'].includes(String(status).toLowerCase()) ? 'completed' : 'neutral';
  const hasTestMetrics = (run) => run?.test_evaluated === true || (run?.test_evaluated !== false && num(run?.metrics?.accuracy) !== null);
  const validationOnly = (run) => !!run && !hasTestMetrics(run) && (run.config?.defer_test === true || run.config?.deferred === true || run.config?.defer_test_evaluation === true);
  const runStatusText = (run) => validationOnly(run) && ['completed','complete','finished'].includes(String(run?.status).toLowerCase()) ? 'Validation selection complete · not tested' : validationOnly(run) && isActive(run) ? 'Validation selection' : statusText(run?.status);
  const selectedRound = (run = state.run) => num(run?.selected_round);
  const percentagePoints = (value) => num(value) === null ? '—' : `${Number(value) > 0 ? '+' : ''}${(Number(value)*100).toFixed(2)} pp`;
  function optimizationRunId() {
    if (state.optimization?.status !== 'completed' || !state.optimization.selected_run_id) return null;
    const run = state.runs.find((r) => String(r.id) === String(state.optimization.selected_run_id));
    return run && ['completed','complete','finished'].includes(String(run.status).toLowerCase()) && hasTestMetrics(run) && run.checkpoint_exists !== false ? run.id : null;
  }
  function duration(seconds) {
    if (num(seconds) === null) return '—';
    const s = Math.max(0, Math.round(Number(seconds)));
    return s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.floor(s % 3600 / 60)} min` : s >= 60 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${s} s`;
  }
  function bytes(value) {
    if (num(value) === null) return '—';
    const n = Number(value);
    if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
    if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(2)} MB`;
    if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${fmt(n)} B`;
  }
  function dateText(value) {
    if (!value) return 'Not recorded';
    const d = new Date(value);
    return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString('en-US', {month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});
  }
  function setNotice(id, message, isError = false, retry = false) {
    const node = $(id);
    message = message ? (window.EnglishUI?.message(message, 'Request failed. Please retry.') ?? message) : '';
    node.classList.toggle('hidden', !message);
    node.classList.toggle('error', isError);
    node.innerHTML = message ? `${icon('info')}<span>${esc(message)}</span>${retry ? '<button type="button" data-retry>Retry connection</button>' : ''}` : '';
  }
  let toastTimer;
  function toast(message) {
    $('toast').textContent = message;
    $('toast').classList.add('visible');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $('toast').classList.remove('visible'), 4000);
  }
  async function api(path, options = {}, timeoutMs = 60000) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(path, {...options, headers:{'Accept':'application/json', ...(options.body ? {'Content-Type':'application/json'} : {}), ...options.headers}, signal:controller.signal, cache:'no-store'});
      const contentType = response.headers.get('content-type') || '';
      if (!contentType.includes('json')) { const error = new Error(`The service returned a non-JSON response (HTTP ${response.status}). Check that the API is running.`); error.status = response.status; throw error; }
      const payload = await response.json();
      const data = window.EnglishUI?.translate(payload) ?? payload;
      if (!response.ok) {
        const detail = data.error || data.detail || data.message || `HTTP ${response.status}`;
        const error = new Error(typeof detail === 'object' ? JSON.stringify(detail) : detail);
        error.status = response.status;
        throw error;
      }
      return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('Request timed out. Check the data service and retry.');
      if (error instanceof TypeError) throw new Error('Cannot connect to the experiment service. Check that it is running.');
      throw error;
    } finally { clearTimeout(timeout); }
  }
  function datasetCount(dataset = {}) {
    const v = Array.isArray(dataset.subjects) ? dataset.subjects.length : dataset.subjects ?? dataset.n_subjects ?? dataset.loaded_subjects ?? dataset.count;
    return num(v);
  }
  function getSites() {
    const raw = state.status?.dataset?.sites;
    const sites = Array.isArray(raw) ? raw : raw && typeof raw === 'object' ? Object.entries(raw).map(([id, v]) => typeof v === 'object' ? {id,...v} : {id,count:v}) : [];
    return siteMeta.map((meta) => ({meta, data:sites.find((s) => canonicalSite(s.id || s.site || s.name) === meta.id) || null}));
  }
  function history() { return Array.isArray(state.run?.history) ? state.run.history.filter((h) => h && typeof h === 'object') : []; }
  function metrics() { return !validationOnly(state.run) && state.run?.metrics && typeof state.run.metrics === 'object' ? state.run.metrics : {}; }
  function getRunRound() {
    const h = history();
    return state.run?.round ?? h[h.length - 1]?.round ?? (h.length ? h.length : null);
  }
  function getTotalRounds() { return state.run?.total_rounds ?? state.run?.config?.rounds ?? null; }
  function selectedSummary() { return state.runs.find((r) => String(r.id) === String(state.selectedRun)); }
  function renderRunSelect() {
    const select = $('run-select');
    if (!state.runs.length) {
      select.innerHTML = '<option value="">No runs · start a new experiment</option>';
      select.disabled = true;
    } else {
      select.innerHTML = state.runs.map((run) => `<option value="${esc(run.id)}">${esc(run.id)} · ${esc(methodName(run.config?.method))} · ${esc(runStatusText(run))}</option>`).join('');
      select.value = state.selectedRun || '';
      select.disabled = state.busy || state.trainBusy;
    }
    const run = state.run || selectedSummary();
    $('run-status').className = `badge ${run && validationOnly(run) ? 'violet-outline' : run ? statusClass(run.status) : 'neutral'}`;
    $('run-status').textContent = run ? runStatusText(run) : state.status ? 'Not running' : 'Disconnected';
    $('run-method').textContent = run ? `${methodName(run.config?.method)} · Seed ${run.config?.seed ?? '—'} · ${dateText(run.created_at)}` : 'Real data · traceable results';
  }
  function renderMetrics() {
    const m = metrics();
    const scoreRun = state.run || selectedSummary();
    $('aux-run-context').textContent = scoreRun?.config?.experiment_tag === 'federation-process-demo'
      ? `Process demo · ${fmt(scoreRun.config.rounds)} rounds · Not the 50-round main experiment`
      : validationOnly(scoreRun) ? 'Validation-selection run · test set remains locked'
      : scoreRun ? `${methodName(scoreRun.config?.method)} · ${fmt(scoreRun.config?.rounds)} rounds · seed ${scoreRun.config?.seed ?? '—'} Single-run measurement` : 'Awaiting a run · full records linked separately';
    $('aux-pooled').textContent=percentage(m.accuracy,2);
    $('aux-macro').textContent=percentage(m.site_mean_accuracy,2);
    $('aux-auc').textContent=fmt(m.auc,3);
    $('aux-n').textContent=fmt(m.n_test_subjects);
    $('aux-evaluation-link').href=state.run?.id?`/api/runs/${encodeURIComponent(state.run.id)}`:'/api/status';
    const dataset = state.status?.dataset || {};
    $('metric-accuracy').innerHTML = `${num(m.accuracy) === null ? '—' : (Number(m.accuracy) * (Number(m.accuracy) <= 1 ? 100 : 1)).toFixed(1)}<span>%</span>`;
    $('metric-accuracy-note').textContent = num(m.accuracy) !== null ? `Subject-level pooled · ${fmt(m.n_test_subjects)} test subjects` : validationOnly(state.run) ? 'Validation selection · test set remains sealed' : isActive(state.run) ? 'Training in progress · not tested' : 'Awaiting measured evaluation results';
    const count = datasetCount(dataset);
    $('metric-subjects').innerHTML = `${fmt(count)}<span> subjects</span>`;
    $('metric-subjects-note').textContent = count !== null ? `ABIDE I · ${dataset.pipeline || 'CPAC / filt_noglobal / AAL116'}` : state.status ? 'No cached data; awaiting data preparation' : 'Awaiting data-service connection';
    const loadedSites = getSites().filter(({data}) => num(data?.count ?? data?.subjects ?? data?.n_subjects) > 0).length;
    $('metric-sites').innerHTML = `${state.status ? loadedSites : '—'}<span>/ 6</span>`;
    $('metric-sites-note').textContent = loadedSites === 6 ? 'Real subjects are available at all six sites' : 'Based on actually cached subjects';
    const selection = state.run?.selection || {};
    $('metric-features').innerHTML = `${fmt(selection.n_selected ?? selection.selected_features)}<span> connections</span>`;
    $('metric-features-note').textContent = num(selection.n_selected ?? selection.selected_features) !== null ? `Original ${fmt(selection.total_features ?? 6670)} features · selection fitted on training data` : methodName(state.run?.config?.method) === 'FedAvg' ? 'FedAvg baseline · no sparse selection' : 'L₁ sparse selection · 6,670 input features';
    $('pipeline-data').textContent = count > 0 ? 'Ready' : 'Pending';
    const round = getRunRound();
    $('round-badge').textContent = state.run ? `Trained ${fmt(round ?? 0)} / ${fmt(getTotalRounds())}  rounds` : 'Not started';
    const checkpoint = selectedRound();
    $('checkpoint-badge').classList.toggle('hidden',checkpoint === null);
    $('checkpoint-badge').textContent = checkpoint === null ? '' : `Validation-selected R${fmt(checkpoint)}`;
    const h = history(), last = h[h.length - 1];
    $('elapsed-label').textContent = `Training time ${duration(last?.elapsed_seconds ?? state.run?.elapsed_seconds)}`;
    const cumulativeBytes = [...h].reverse().find((r) => num(r.communication_bytes) !== null)?.communication_bytes ?? state.run?.communication_bytes;
    $('communication-label').textContent = `Communication ${bytes(cumulativeBytes)}`;
  }
  function renderSites() {
    $('sites-grid').innerHTML = getSites().map(({meta,data}) => {
      const count = data ? num(data.count ?? data.subjects ?? data.n_subjects) : null;
      const asd = num(data?.asd ?? data?.n_asd), hc = num(data?.hc ?? data?.n_hc);
      const total = asd !== null && hc !== null ? asd + hc : null;
      return `<article class="site-card" style="--site-color:${meta.color};--site-light:${meta.light};--site-border:${meta.border}"><div class="site-heading"><span class="site-monogram">${meta.code}</span><div><h3 class="site-title">${meta.id === 'YALE' ? 'Yale' : meta.id}</h3><div class="site-location" title="${meta.name}">${meta.name}</div></div>${count > 0 ? '<span class="status-dot" title="Local data available"></span>' : ''}</div><div class="site-number">${fmt(count)}<small>subjects</small></div><div class="site-bar" aria-label="${total !== null ? `ASD ${asd}  subjects, HC ${hc}  subjects` : 'Class counts are not available'}">${total > 0 ? `<span style="width:${asd / total * 100}%"></span><span style="width:${hc / total * 100}%"></span>` : ''}</div><div class="site-counts"><span>ASD ${fmt(asd)}</span><span>HC ${fmt(hc)}</span></div>${count === null ? '<p class="site-pending">Awaiting the local data manifest</p>' : ''}</article>`;
    }).join('');
  }
  function renderChart(id, type) {
    const h = history();
    const series = type === 'accuracy' ? [
      {name:'Validation accuracy · site macro average',color:'#2d897a',values:h.map((p) => ({x:num(p.round),y:num(p.val_accuracy)}))}
    ] : [
      {name:'Training loss',color:'#2d897a',values:h.map((p) => ({x:num(p.round),y:num(p.train_loss)}))},
      {name:'Alignment loss',color:'#a18dbd',values:h.map((p) => ({x:num(p.round),y:num(p.alignment_loss)}))}
    ];
    series.forEach((s) => { s.values = s.values.filter((p) => p.x !== null && p.y !== null).sort((a,b) => a.x-b.x); });
    const all = series.flatMap((s) => s.values);
    const width = Math.max(300, Math.min(680, $(id).clientWidth || 620)), height = 215, margin = {left:37,right:19,top:12,bottom:32};
    const plotW = width-margin.left-margin.right, plotH = height-margin.top-margin.bottom;
    const xMax = Math.max(5, ...all.map((p) => p.x), num(getTotalRounds()) || 0);
    const yMax = type === 'accuracy' ? 1 : all.length ? Math.max(.1, ...all.map((p) => p.y)) * 1.12 : 1;
    const yMin = type === 'accuracy' ? 0 : Math.min(0,...all.map((p) => p.y));
    const x = (v) => margin.left + v / xMax * plotW;
    const y = (v) => margin.top + plotH - (v-yMin) / (yMax-yMin || 1) * plotH;
    let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${type === 'accuracy' ? 'Measured round-by-round validation accuracy' : 'Measured round-by-round training loss'}${!all.length ? ', No data' : `, Includes ${h.length} round records`}"><defs><linearGradient id="fill-${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#73b499" stop-opacity=".14"/><stop offset="100%" stop-color="#73b499" stop-opacity="0"/></linearGradient></defs>`;
    for (let i=0;i<=4;i++) {
      const v = yMin + (yMax-yMin)*i/4, yy = y(v);
      svg += `<line x1="${margin.left}" y1="${yy}" x2="${width-margin.right}" y2="${yy}" stroke="#edf0e7" stroke-width="1" stroke-dasharray="3 5"/><text x="${margin.left-10}" y="${yy+3}" text-anchor="end" font-size="8" fill="#a1ac95">${type === 'accuracy' ? `${Math.round(v*100)}%` : v.toFixed(yMax < 1 ? 3 : 2)}</text>`;
    }
    for (let i=0;i<=5;i++) {
      const v = Math.round(xMax*i/5);
      svg += `<text x="${x(v)}" y="${height-12}" text-anchor="middle" font-size="8" fill="#a1ac95">${v}</text>`;
    }
    svg += `<text x="${width-margin.right}" y="${height-1}" text-anchor="end" font-size="7" fill="#b1b9a7">Communication rounds / ROUND</text>`;
    series.forEach((s, i) => {
      if (!s.values.length) return;
      const path = s.values.map((p,j) => `${j ? 'L' : 'M'}${x(p.x).toFixed(2)},${y(p.y).toFixed(2)}`).join(' ');
      if (!i && s.values.length > 1) svg += `<path d="${path} L${x(s.values[s.values.length-1].x)},${y(yMin)} L${x(s.values[0].x)},${y(yMin)}Z" fill="url(#fill-${id})"/>`;
      svg += `<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2.2" stroke-linejoin="round"${s.dash ? ' stroke-dasharray="4 4"' : ''}/>`;
      s.values.forEach((p) => { svg += `<circle cx="${x(p.x)}" cy="${y(p.y)}" r="${s.values.length < 8 ? 3 : 1.8}" fill="${s.color}"><title>${s.name} · Round ${p.x}  rounds: ${type === 'accuracy' ? percentage(p.y) : p.y.toFixed(5)}</title></circle>`; });
    });
    const checkpoint = selectedRound();
    if (type === 'accuracy' && checkpoint !== null) {
      const point = series[0].values.find((p)=>p.x===checkpoint);
      if (point) svg += `<line x1="${x(checkpoint)}" y1="${margin.top}" x2="${x(checkpoint)}" y2="${y(yMin)}" stroke="#a18dbd" stroke-width="1" stroke-dasharray="3 4"/><circle cx="${x(checkpoint)}" cy="${y(point.y)}" r="4" fill="#fff" stroke="#a18dbd" stroke-width="2"><title>Validation selected round ${checkpoint}  rounds, Validation accuracy ${percentage(point.y)}</title></circle><text x="${clamp(x(checkpoint)+7,margin.left+5,width-margin.right-72)}" y="${margin.top+8}" fill="#9381ad" font-size="8">Validation-selected R${checkpoint}</text>`;
    }
    svg += '</svg>';
    if (!all.length) svg += `<div class="chart-empty">${icon(isActive(state.run) ? 'clock' : 'pulse')}<strong>${isActive(state.run) ? 'Awaiting the first recorded round' : state.run?.error ? 'No curve records for this run' : 'Curves appear after training'}</strong><span>${isActive(state.run) ? 'Updates every 3 seconds while active' : 'No generated, imputed, or simulated points'}</span></div>`;
    $(id).innerHTML = svg;
  }
  function renderEvaluation() {
    const m = metrics(), perSite = m.per_site || {};
    const entries = Object.entries(perSite);
    let rows = '';
    getSites().forEach(({meta,data}) => {
      const entry = entries.find(([id]) => canonicalSite(id) === meta.id);
      const v = entry?.[1] || {};
      const cm = v.confusion_matrix;
      const cmText = Array.isArray(cm) && Array.isArray(cm[0]) && Array.isArray(cm[1]) ? [cm[0][0],cm[0][1],cm[1][0],cm[1][1]].map((x)=>fmt(x)).join(' / ') : '—';
      rows += `<tr><td class="table-site">${meta.id === 'YALE' ? 'Yale' : meta.id}</td><td>${fmt(v.n_test_subjects ?? v.n_subjects ?? v.count ?? v.n)}</td><td>${percentage(v.accuracy)}</td><td>${percentage(v.sensitivity ?? v.recall)}</td><td>${percentage(v.specificity)}</td><td>${cmText}</td></tr>`;
    });
    $('site-results').innerHTML = `<table><caption class="sr-only">Held-out metrics for six sites; dashes indicate unavailable results</caption><thead><tr><th scope="col">Site</th><th scope="col">Test subjects</th><th scope="col">Accuracy</th><th scope="col">Sensitivity</th><th scope="col">Specificity</th><th scope="col">TN / FP / FN / TP</th></tr></thead><tbody>${rows}</tbody></table>${!entries.length ? `<div class="table-placeholder">${validationOnly(state.run) ? 'Validation-selection run: test set is not evaluated; test metrics are not shown' : 'Measured site results appear after training and testing'}</div>` : ''}`;
    const matrix = m.confusion_matrix;
    const valid = Array.isArray(matrix) && matrix.length >= 2 && Array.isArray(matrix[0]) && Array.isArray(matrix[1]);
    const cells = [{v:valid ? matrix[0][0] : null,label:'HC → HC · TN',correct:true},{v:valid ? matrix[0][1] : null,label:'HC → ASD · FP',correct:false},{v:valid ? matrix[1][0] : null,label:'ASD → HC · FN',correct:false},{v:valid ? matrix[1][1] : null,label:'ASD → ASD · TP',correct:true}];
    $('confusion-matrix').innerHTML = `<div class="cm-layout"><div class="cm-axis-top">Predicted label: HC / ASD</div><div class="cm-axis-side">True label: HC / ASD</div><div class="cm-grid">${cells.map((c) => `<div class="cm-cell ${c.correct ? 'correct' : 'incorrect'}"><strong>${fmt(c.v)}</strong><small>${c.label}</small></div>`).join('')}</div></div>`;
    $('evaluation-minis').innerHTML = [{label:'Site macro average ACC',v:m.site_mean_accuracy},{label:'Balanced accuracy',v:m.balanced_accuracy},{label:'AUC',v:m.auc}].map((p) => `<div class="evaluation-mini"><strong>${percentage(p.v)}</strong><span>${p.label}</span></div>`).join('');
  }
  function manifestItem(label, value) { return `<div><dt>${esc(label)}</dt><dd>${esc(value ?? '—')}</dd></div>`; }
  function privacyConfig() { return state.run?.privacy || state.run?.config?.privacy || state.status?.protocol?.privacy || {}; }
  function renderTraining() {
    const run = state.run;
    const config = run?.config || {};
    const round = num(getRunRound()) ?? 0, total = num(getTotalRounds());
    const progress = total > 0 ? clamp(round/total*100,0,100) : 0;
    const error = run?.error;
    $('training-status').innerHTML = `<div class="training-status-top"><strong>${run ? esc(runStatusText(run)) : 'Start an experiment'}</strong><span>${run ? `Round ${round} / ${fmt(total)}` : 'Configure → Train → Validate → Test'}</span></div><p>${run ? `${esc(run.id)} · ${isActive(run) ? 'Measured progress updates every 3 seconds' : 'Run records saved'}${selectedRound(run)!==null ? ` · Validation-selected round ${fmt(selectedRound(run))} / trained ${round} rounds` : ''}${validationOnly(run) ? ' · Test set remains locked' : ''}` : 'Charts populate from actual API responses after training starts.'}</p><div class="progress-track" role="progressbar" aria-label="Communication round progress" aria-valuenow="${round}" aria-valuemin="0"${total ? ` aria-valuemax="${total}"` : ''}><span style="width:${progress}%"></span></div>${error ? `<p class="training-error">${esc(typeof error === 'object' ? JSON.stringify(error) : error)}</p>` : ''}`;
    const privacy = privacyConfig();
    const homogeneous = privacy.homomorphic ?? config.homomorphic;
    $('run-manifest').innerHTML = [
      manifestItem('Run ID',run?.id),manifestItem('Method',run ? methodName(config.method) : null),
      manifestItem('Random seed',config.seed),manifestItem('Started at',run?.created_at ? dateText(run.created_at) : null),
      manifestItem('Noise scale',privacy.noise_scale ?? config.noise_scale),manifestItem('Alignment weight',config.alignment_weight),
      manifestItem('Subjects per site',config.max_per_site === 0 ? 'All cached subjects' : config.max_per_site),
      manifestItem('Homomorphic encryption',homogeneous === true ? 'Paillier enabled' : homogeneous === false ? 'Disabled' : 'Not reported'),
      manifestItem('Formal differential privacy guarantee',privacy.formal_dp === true ? 'Reported enabled by the service' : 'No formal DP claim'),
      manifestItem('Evaluated subjects',metrics().n_test_subjects),
      manifestItem('Validation-selected checkpoint',selectedRound(run)!==null ? `Round ${fmt(selectedRound(run))}  rounds (Not selected by test results)` : 'Not reported'),
      manifestItem('Completed rounds',num(getRunRound())!==null ? `${fmt(getRunRound())} / ${fmt(getTotalRounds())}` : null),
      manifestItem('Final test status',validationOnly(run) ? 'Validation selection; test set locked' : hasTestMetrics(run) ? 'Final testing completed' : 'No final test results'),
      manifestItem('Training windows per subject',config.max_windows === 0 ? 'Unlimited' : config.max_windows),
      manifestItem('Maximum selected features',config.max_features),
      manifestItem('Sparse optimization steps',config.sparse_max_iter),
      manifestItem('Train / validation / test',run?.split_counts ? `${run.split_counts.train ?? '—'} / ${run.split_counts.val ?? '—'} / ${run.split_counts.test ?? '—'}` : null),
      ...[['Checkpoint selection rule',config.checkpoint_selection],['Alignment normalization',config.alignment_normalization],['Reference-statistics update',config.alignment_reference],['L₂ weight decay',config.weight_decay]].filter(([,value])=>value!==undefined).map(([label,value])=>manifestItem(label,value))
    ].join('');
    const h = history().slice(-50).reverse();
    $('round-history').innerHTML = h.length ? `<table><caption class="sr-only">Latest 50 training rounds</caption><thead><tr><th>Round</th><th>Training loss</th><th>Alignment loss</th><th>Validation accuracy</th><th>Elapsed time</th><th>Communication</th></tr></thead><tbody>${h.map((p) => `<tr><td class="table-site">${fmt(p.round)}</td><td>${fmt(p.train_loss,5)}</td><td>${fmt(p.alignment_loss,5)}</td><td>${percentage(p.val_accuracy)}</td><td>${duration(p.elapsed_seconds)}</td><td>${bytes(p.communication_bytes)}</td></tr>`).join('')}</tbody></table>` : '<div class="empty-block">'+icon('book')+'<strong>No recorded training rounds</strong><p>Measured loss, validation metrics, and communication are recorded per round.</p></div>';
    updateControls();
  }
  function updateControls() {
    const globalActive = !!state.status?.active_run || state.runs.some(isActive);
    $('start-train').disabled = state.trainBusy || globalActive || !state.status;
    $('train-fields').disabled = state.trainBusy;
    $('start-train').innerHTML = state.trainBusy ? `${icon('refresh','spin')}Submitting training…` : globalActive ? `${icon('clock')}An experiment is running` : `${icon('play')}Start Federated Training`;
    $('stop-train').disabled = state.stopBusy || !isActive(state.run) || String(state.run?.status).toLowerCase() === 'stopping';
    $('stop-train').innerHTML = state.stopBusy ? `${icon('refresh','spin')}Stopping…` : `${icon('stop')}Stop Training`;
    $('refresh-button').disabled = state.busy;
    $('run-select').disabled = state.busy || state.trainBusy || !state.runs.length;
    $('export-run').disabled = !state.run;
    $('export-protocol').disabled = !state.status;
    document.querySelectorAll('[data-open-optimized]').forEach((button)=>{button.disabled=state.busy || !optimizationRunId();});
  }
  function displayValue(value) {
    if (value === null || value === undefined) return 'Not reported';
    if (typeof value === 'boolean') return value ? 'Yes' : 'No';
    if (Array.isArray(value)) return value.map((v) => typeof v === 'object' ? JSON.stringify(v) : String(v)).join(' / ');
    if (typeof value === 'object') return Object.entries(value).map(([k,v]) => `${k}: ${displayValue(v)}`).join('; ');
    return String(value);
  }
  function protocolItem(label, value) { return `<div class="protocol-item"><span class="protocol-item-label">${esc(label)}</span><span class="protocol-item-value">${esc(displayValue(value))}</span></div>`; }
  function renderProtocol() {
    const paper = state.status?.paper || {};
    if (num(paper.reported_accuracy) !== null) $('paper-accuracy').innerHTML = `${Number(paper.reported_accuracy) * (Number(paper.reported_accuracy) <= 1 ? 100 : 1)}<span>%</span>`;
    if (paper.doi) $('paper-link').href = `https://doi.org/${String(paper.doi).replace(/^https?:\/\/(?:dx\.)?doi\.org\//,'')}`;
    const p = state.status?.protocol || {}, privacy = privacyConfig();
    const homomorphic = privacy.homomorphic ?? state.run?.config?.homomorphic;
    const items = [
      protocolItem('Deployment mode','Six-client single-process simulation, not a deployed hospital federation'),
      protocolItem('Selected model',state.run ? methodName(state.run.config?.method) : 'Select a run'),
      protocolItem('Data source','ABIDE I · real locally cached data'),
      protocolItem('Cached subjects',datasetCount(state.status?.dataset) !== null ? `${fmt(datasetCount(state.status?.dataset))}  subjects (Actually cached)` : 'Awaiting the data service'),
      protocolItem('Local versus paper cohort','The downloaded cohort may differ from paper Table 1'),
      protocolItem('Subject-level split',p.split ?? p.split_strategy ?? p.splitting ?? 'See the saved run protocol and service records'),
      protocolItem('Homomorphic encryption',homomorphic === true ? 'Paillier enabled for this run' : homomorphic === false ? 'Disabled for this run (smoke-test configuration)' : 'Awaiting run configuration; not assumed enabled'),
      protocolItem('Privacy statement',privacy.formal_dp === true ? 'The service reports formal DP enabled' : 'Parameter noise is not a formal DP guarantee'),
      protocolItem('Metric aggregation','Pooled subject accuracy and site macro accuracy are separate; the published reference is a site average'),
      protocolItem('Measured results',validationOnly(state.run) ? 'Validation-selection run; no test metrics yet' : state.run ? `pooled ${percentage(metrics().accuracy)}; macro-site ${percentage(metrics().site_mean_accuracy)}` : 'Awaiting measured evaluation'),
      protocolItem('Checkpoint selection',selectedRound()!==null ? `Validation selected round ${fmt(selectedRound())}  rounds; Trained ${fmt(getRunRound())}  rounds` : 'No validation checkpoint reported; no round is assumed'),
      protocolItem('Intended use','Research only; not for clinical diagnosis')
    ];
    const names = {implementation:'Implementation notes',implementation_note:'Implementation choices',limitations:'Implementation limitations',notes:'Protocol notes',deviations:'Differences from the paper',cohort_note:'Cohort notes',reference_evaluation:'Published evaluation protocol',evaluation:'Evaluation definition',feature_selection:'Feature selection',normalization:'Feature scaling',window:'Window length',window_size:'Window length',stride:'Window stride',encryption:'Encryption notes',privacy_note:'Privacy scope'};
    Object.entries(names).forEach(([key,label]) => { if (p[key] !== undefined) items.push(protocolItem(label,p[key])); });
    $('protocol-details').innerHTML = items.join('');
    const run = state.run, m = metrics(), h = history();
    $('protocol-evidence').innerHTML = [
      {label:'01 / Data evidence',value:`${fmt(datasetCount(state.status?.dataset))} subjects`,note:state.status?.dataset?.pipeline || 'Local cache manifest is not ready'},
      {label:'02 / Run configuration',value:run?.id || 'No runs',note:run ? `${methodName(run.config?.method)} · seed ${run.config?.seed ?? '—'} · ${fmt(getTotalRounds())} rounds` : 'Parameters and seed are saved when training starts'},
      {label:'03 / Training trace',value:`${h.length} round records`,note:selectedRound()!==null ? `Validation selection R${fmt(selectedRound())}, Trained ${fmt(getRunRound())}  rounds; Rounds are not selected by test performance` : 'Curves come only from history; validation is not testing'},
      {label:'04 / Final testing',value:validationOnly(run) ? 'Validation selection · not tested' : num(m.accuracy) !== null ? `ACC ${percentage(m.accuracy)}` : 'No measured results',note:num(m.n_test_subjects) !== null ? `Held-out test set ${fmt(m.n_test_subjects)} subjects` : 'Lock the variant and checkpoint on validation first; published references do not populate local metrics'}
    ].map((e) => `<div class="evidence-item"><span>${e.label}</span><strong>${esc(e.value)}</strong><p>${esc(e.note)}</p></div>`).join('');
  }
  function renderOptimization() {
    const o = state.optimization || {status:'pending'}, complete = o.status === 'completed', confirming = o.status === 'confirming', searching = ['searching','confirming'].includes(o.status);
    const variants = Array.isArray(o.variants) ? o.variants.filter((v)=>v && typeof v === 'object') : [];
    const selectedId = typeof o.selected_variant === 'object' ? o.selected_variant?.id : o.selected_variant;
    const selected = variants.find((v)=>String(v.id)===String(selectedId));
    const selectedLabel = selected?.label || (typeof o.selected_variant === 'object' ? o.selected_variant?.label : selectedId);
    const passGate = o.comparison?.pass_gate;
    $('optimization-panel').classList.toggle('gate-failed',complete && passGate===false);
    $('optimization-status').textContent = complete ? passGate===true ? 'Confirmation passed · gate met' : passGate===false ? 'Confirmation complete · no automatic deployment' : 'Test results summarized · awaiting verdict' : confirming ? 'Confirming the locked variant' : searching ? 'Validation selection in progress' : 'Awaiting optimization evidence';
    $('optimization-status').className = `badge ${complete && passGate===true ? 'teal-outline' : searching ? 'violet-outline' : 'neutral'}`;
    const warning = state.optimizationError ? `<div class="optimization-warning">${icon('info')}<span>${esc(state.optimizationError)}</span></div>` : '';
    const summary = `<div class="optimization-summary"><p>${complete ? `Validation-locked${passGate===true ? 'Variant' : 'Candidate'}: <strong>${esc(selectedLabel || 'Variant name not provided')}</strong>${passGate===false ? '.The preregistered stability gate was not met; the original reproduction remains the default.' : ''}` : confirming ? 'The variant is locked by validation; fixed-seed held-out confirmation is in progress and comparisons are not final.' : searching ? 'Select the variant and checkpoint on validation, then unlock final testing.' : 'Optimization evidence is pending; retain the original run without prefilled scores or improvements.'}</p>${optimizationRunId() ? `<button class="text-button" data-open-optimized>${String(state.selectedRun)===String(optimizationRunId()) ? passGate===true ? 'Viewing the locked run' : 'Viewing the validation-selected candidate' : passGate===true ? 'View locked run' : 'View validation-selected candidate'} ${icon('arrow')}</button>` : ''}</div>`;
    let comparison = '';
    if (complete) {
      const c = o.comparison && typeof o.comparison === 'object' ? o.comparison : {}, seeds = Array.isArray(c.seeds) ? c.seeds : [];
      const originalDelta = num(c.delta_original_macro) ?? (num(c.original_macro_mean)!==null && num(c.optimized_macro_mean)!==null ? Number(c.optimized_macro_mean)-Number(c.original_macro_mean) : null);
      comparison = `<div class="optimization-comparison"><div><span>Validation-selected control · site macro</span><strong>${percentage(c.baseline_macro_mean,2)}</strong><small>Pooled mean ${percentage(c.baseline_pooled_mean,2)}</small></div><div><span>Locked candidate · site macro</span><strong>${percentage(c.optimized_macro_mean,2)}</strong><small>Pooled mean ${percentage(c.optimized_pooled_mean,2)}</small></div><div class="optimization-delta${c.pass_gate===true && num(c.delta_macro)!==null && Number(c.delta_macro)>0 ? ' positive' : ''}"><span>Difference from validation control · pp</span><strong>${percentagePoints(c.delta_macro)}</strong><small>${c.pass_gate===true ? 'Preregistered stability gate met' : c.pass_gate===false ? 'Preregistered stability gate not met' : 'Gate verdict not reported'}</small></div></div><div class="optimization-original"><div><span>Original final-round variant · matched-seed mean</span><strong>Macro average ${percentage(c.original_macro_mean,2)}</strong><small>pooled ${percentage(c.original_pooled_mean,2)}</small></div><div><span>Candidate versus original final-round variant</span><strong>${percentagePoints(originalDelta)}</strong><small>Macro-average change · pooled ${percentagePoints(c.delta_original_pooled)}</small></div></div><div class="optimization-matching"><span>${seeds.length ? `Matched seeds ${seeds.map(esc).join(' / ')} · Not a single best run` : 'Seed list unavailable; repetition count is not assumed'}</span>${num(c.success_count)!==null ? `<span>Improved seeds ${fmt(c.success_count)}${seeds.length ? ` / ${seeds.length}` : ''}</span>` : ''}</div>`;
      if (num(c.fedavg_best_val_macro_mean)!==null) {
        const controlDelta = num(c.optimized_macro_mean)!==null ? Number(c.optimized_macro_mean)-Number(c.fedavg_best_val_macro_mean) : null;
        comparison += `<div class="optimization-control"><span>Matched sparse-mask FedAvg · validation-selected</span><strong>Macro average ${percentage(c.fedavg_best_val_macro_mean,2)}</strong><small>Candidate versus this control ${percentagePoints(controlDelta)}</small></div>`;
      }
    } else if (!variants.length) comparison = `<div class="optimization-pending">${icon(searching?'pulse':'clock')}<div><strong>${searching?'Awaiting measured validation records':'Only measured API optimization evidence is shown'}</strong><p>${searching?'Validation search does not populate test metrics.':'Comparison values remain blank while the API is unavailable or pending.'}</p></div></div>`;
    const changes = Array.isArray(o.changes) && o.changes.length ? `<div class="optimization-change-label">Implementation changes under test · saved locked configuration is authoritative</div><div class="optimization-changes">${o.changes.map((v)=>`<span>${icon('check')}${esc(displayValue(v))}</span>`).join('')}</div>` : '';
    const candidates = variants.length ? `<details class="optimization-variants"${searching?' open':''}><summary>Validation search records <span>${variants.length} API records · test results do not select variants</span></summary><div class="table-scroll" tabindex="0" aria-label="Scrollable variants and search-time test-isolation status"><table><caption class="sr-only">Select variants and rounds by validation accuracy only; the test column describes search-time isolation, not the current final-test state</caption><thead><tr><th scope="col">Candidate variant</th><th scope="col">Validation accuracy</th><th scope="col">Selected validation round</th><th scope="col">Search-time test status</th></tr></thead><tbody>${variants.map((v)=>`<tr${String(v.id)===String(selectedId)?' class="locked-variant"':''}><td>${esc(v.label || v.id || 'Unnamed')}${String(v.id)===String(selectedId)?'<span class="badge teal-outline">Validation-locked</span>':''}</td><td>${percentage(v.validation_accuracy,2)}</td><td>${num(v.selected_round)===null?'—':`R${fmt(v.selected_round)}`}</td><td>${v.test_evaluated===true?'Evaluated':v.test_evaluated===false?'Not evaluated':'Not reported'}</td></tr>`).join('')}</tbody></table></div></details>` : '';
    const notes = `<div class="optimization-notes"><p>Rescaling classification versus alignment is a local implementation change, not an exact recovery of the original objective or source code.</p>${Array.isArray(o.notes) ? o.notes.map((v)=>`<p>${esc(displayValue(v))}</p>`).join('') : ''}</div>`;
    $('optimization-content').innerHTML = warning + summary + comparison + changes + candidates + notes;
  }
  async function loadOptimization() {
    try {
      const result = await api('/api/optimization',{},5000);
      state.optimization = result && typeof result === 'object' ? result : {status:'pending'};
      state.optimizationError = null;
    } catch(error) {
      if (!state.optimization) state.optimization = {status:'pending'};
      state.optimizationError = error.status===404 ? null : `Could not refresh optimization evidence: ${error.message}`;
    }
  }
  function renderAll() {
    renderRunSelect();renderMetrics();renderSites();renderChart('overview-chart','accuracy');renderChart('loss-chart','loss');renderEvaluation();renderTraining();renderProtocol();renderOptimization();
    window.Federation?.update({run:state.run,status:state.status});
    if (state.status?.device) {
      const d = state.status.device;
      $('device-chip').innerHTML = `<span class="status-dot${d.cuda ? '' : ' gray'}"></span><span>${esc(d.name || (d.cuda ? 'CUDA GPU' : 'CPU'))}${d.cuda ? ' · CUDA' : ' · CPU'}</span>`;
      $('device-chip').title = `Compute device: ${d.name || 'Not reported'}; CUDA ${d.cuda ? 'Available' : 'Unavailable'}`;
    }
    $('updated-at').textContent = state.lastUpdate ? `${state.lastUpdate.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false})} Updated` : 'Not synchronized';
  }
  function schedulePoll() {
    clearTimeout(state.polling);
    state.polling = null;
    const active = !!state.status?.active_run || state.runs.some(isActive) || isActive(state.run) || ['searching','confirming'].includes(state.optimization?.status);
    if (active && !document.hidden) state.polling = setTimeout(() => refresh({quiet:true}),3000);
  }
  async function refresh({quiet = false} = {}) {
    if (state.busy) return;
    state.busy = true;
    document.body.classList.add('busy');
    updateControls();
    const previousId = state.selectedRun;
    try {
      await loadOptimization();
      const status = await api('/api/status');
      state.status = status;
      const rawRuns = Array.isArray(status.runs) ? status.runs : (await api('/api/runs')).runs;
      state.runs = (Array.isArray(rawRuns) ? rawRuns : []).filter((r) => r && r.id).sort((a,b) => String(b.created_at || b.id).localeCompare(String(a.created_at || a.id)));
      const requested = new URLSearchParams(location.search).get('run');
      const activeId = typeof status.active_run === 'object' ? status.active_run?.id : status.active_run;
      const processDemo = state.runs.find((r)=>r.id==='federation-demo-s42') || state.runs.filter((r)=>r.federation_available===true).sort((a,b)=>(Date.parse(b.created_at)||0)-(Date.parse(a.created_at)||0))[0];
      const processId = processDemo && await window.Federation?.available(processDemo.id) ? processDemo.id : null;
      const optimizedId = processId || (state.optimization?.comparison?.pass_gate === true ? optimizationRunId() : null);
      if (!state.selectionInitialized || !state.runs.some((r) => String(r.id) === String(state.selectedRun)) || (!state.manualSelection && optimizedId && String(state.selectedRun)!==String(optimizedId))) {
        const completedFull = (r) => r.config?.max_per_site === 0 && ['completed','complete','finished'].includes(String(r.status).toLowerCase()) && !validationOnly(r);
        const fullCompleted = state.runs.find((r) => r.id === 'full-fedbrain-s42' && completedFull(r)) || state.runs.find(completedFull);
        const explicitRequest = !state.selectionInitialized && requested && state.runs.some((r)=>String(r.id)===requested);
        if (explicitRequest) state.manualSelection = true;
        state.selectedRun = (explicitRequest ? requested : optimizedId || fullCompleted?.id || activeId || state.runs[0]?.id) || null;
        state.selectionInitialized = true;
      }
      if (state.selectedRun) {
        try { state.run = await api(`/api/runs/${encodeURIComponent(state.selectedRun)}`); }
        catch(error) { state.run = selectedSummary() || null; throw new Error(`Connected to status, but run details could not be read: ${error.message}`); }
      } else state.run = null;
      state.lastUpdate = new Date();
      setNotice('global-notice','');
      if (previousId !== state.selectedRun) resetSubjectData();
      renderAll();
      if (state.view === 'inference' && !state.subjectsLoaded && !state.subjectsLoading) loadSubjects();
      if (!quiet && previousId === state.selectedRun && state.lastUpdate) { /* manual refresh is reflected by the timestamp */ }
    } catch(error) {
      setNotice('global-notice',error.message,true,true);
      renderAll();
      $('device-chip').innerHTML = '<span class="status-dot gray"></span><span>Disconnected</span>';
    } finally {
      state.busy = false;
      document.body.classList.remove('busy');
      updateControls();
      schedulePoll();
    }
  }
  async function selectRun(id) {
    if (state.busy || String(id) === String(state.selectedRun)) return;
    state.selectedRun = id || null;
    state.manualSelection = true;
    state.run = selectedSummary() || null;
    resetSubjectData();
    const url = new URL(location.href);
    id ? url.searchParams.set('run',id) : url.searchParams.delete('run');
    window.history.replaceState(null,'',url);
    renderAll();
    state.busy = true;updateControls();
    try {
      state.run = id ? await api(`/api/runs/${encodeURIComponent(id)}`) : null;
      setNotice('global-notice','');
    } catch(error) { setNotice('global-notice',`Could not read run: ${error.message}`,true,true); }
    finally { state.busy = false;renderAll();schedulePoll();if (state.view === 'inference') loadSubjects(); }
  }
  function navigate(view, updateHash = true) {
    if (!titles[view]) view = 'overview';
    state.view = view;
    document.body.classList.toggle('health-overview',view==='overview');
    if(view!=='overview') window.Federation?.pause();
    document.querySelectorAll('.view').forEach((node) => {const active = node.id === `view-${view}`;node.classList.toggle('active',active);node.hidden = !active;});
    document.querySelectorAll('.nav-item').forEach((node) => {const active=node.dataset.view===view;node.classList.toggle('active',active);active ? node.setAttribute('aria-current','page') : node.removeAttribute('aria-current');});
    $('breadcrumb-current').textContent = titles[view][0];
    $('page-title').innerHTML = `${titles[view][1]}<span class="title-dot">.</span>`;
    $('page-description').textContent = titles[view][2];
    document.title = `${titles[view][0]} · FedBrain Research Workspace`;
    const menuWasOpen = $('sidebar').classList.contains('open');
    $('sidebar').classList.remove('open');$('menu-toggle').setAttribute('aria-expanded','false');$('menu-toggle').setAttribute('aria-label','Open navigation');
    if (menuWasOpen) $('main-content').focus({preventScroll:true});
    if (updateHash && location.hash !== `#${view}`) window.history.replaceState(null,'',`#${view}`);
    if (view === 'inference' && !state.subjectsLoaded && !state.subjectsLoading) loadSubjects();
    if (view === 'overview') renderChart('overview-chart','accuracy');
    if (view === 'train') renderChart('loss-chart','loss');
  }
  function resetSubjectData() {
    state.listRequest++;state.subjectRequest++;
    state.subjects=[];state.subjectsLoaded=false;state.subjectsLoading=false;state.subject=null;state.selectedSubject=null;state.subjectLoading=false;
    $('subject-search').value='';$('site-filter').value='';
    setNotice('subject-notice','');renderSubjectList();renderSubject();
  }
  async function loadSubjects() {
    if (state.subjectsLoading) return;
    const request=++state.listRequest, runId=state.selectedRun;
    state.subjectsLoading=true;renderSubjectList();
    try {
      const data=await api(`/api/subjects${runId ? `?run_id=${encodeURIComponent(runId)}` : ''}`);
      if(request!==state.listRequest)return;
      state.subjects=Array.isArray(data.subjects)?data.subjects:[];state.subjectsLoaded=true;
      const sites=[...new Set(state.subjects.map((s)=>s.site).filter(Boolean))].sort();
      const selected=$('site-filter').value;
      $('site-filter').innerHTML='<option value="">All sites</option>'+sites.map((s)=>`<option value="${esc(s)}">${esc(s)}</option>`).join('');
      $('site-filter').value=sites.includes(selected)?selected:'';
      setNotice('subject-notice','');
      if (!state.selectedSubject && state.subjects.length) selectSubject(state.subjects.find((s)=>s.split==='test')?.id || state.subjects[0].id);
    } catch(error) {
      if(request!==state.listRequest)return;
      setNotice('subject-notice',`Could not load subjects: ${error.message}`,true);
      state.subjectsLoaded=false;
    } finally {if(request===state.listRequest){state.subjectsLoading=false;renderSubjectList();}}
  }
  function labelOf(s) { return s?.label_name || (num(s?.label) === 1 ? 'ASD' : num(s?.label) === 0 ? 'HC' : 'Unlabeled'); }
  function splitText(s) { return ({train:'Training set',val:'Validation set',validation:'Validation set',test:'Test set'})[String(s || '').toLowerCase()] || s || 'Split not assigned'; }
  function renderSubjectList() {
    $('subject-list').setAttribute('aria-busy',String(state.subjectsLoading));
    $('subjects-count').textContent=fmt(state.subjects.length);
    if (state.subjectsLoading && !state.subjects.length) {$('subject-list').innerHTML='<div class="list-empty">Loading real subjects…<span class="loading-line"></span></div>';return;}
    const search=$('subject-search').value.trim().toLowerCase(), site=$('site-filter').value;
    const filtered=state.subjects.filter((s)=>(!site||s.site===site)&&(!search||`${s.id} ${s.site}`.toLowerCase().includes(search)));
    if(!filtered.length){$('subject-list').innerHTML=`<div class="list-empty">${state.subjects.length?'No subjects match the filters':state.subjectsLoaded?'No real subjects; prepare data first':'Subject data not loaded'}${!state.subjectsLoaded && !state.subjectsLoading?'<br><button class="text-button" data-load-subjects>Load / Retry</button>':''}</div>`;return;}
    $('subject-list').innerHTML=filtered.slice(0,600).map((s)=>`<button class="subject-button${String(s.id)===String(state.selectedSubject)?' selected':''}" data-subject-id="${esc(s.id)}" aria-label="View subject ${esc(s.id)}, ${esc(s.site)}, ${esc(labelOf(s))}" aria-pressed="${String(s.id)===String(state.selectedSubject)}"><span class="subject-avatar">${icon('people')}</span><span><span class="subject-id">${esc(s.id)}</span><span class="subject-site">${esc(s.site)} · ${esc(splitText(s.split))}</span></span><span class="badge ${String(labelOf(s)).toUpperCase()==='ASD'?'label-asd':'label-hc'}">${esc(labelOf(s))}</span></button>`).join('')+(filtered.length>600?'<div class="list-more">Showing the first 600 subjects; narrow the search</div>':'');
  }
  async function selectSubject(id) {
    if (!id) return;
    const request=++state.subjectRequest;
    state.selectedSubject=id;state.subject=null;state.subjectLoading=true;
    renderSubjectList();renderSubject();setNotice('subject-notice','');
    if(validationOnly(state.run)) {
      state.subjectLoading=false;
      setNotice('subject-notice','This run is for validation selection and has no final test. Select a locked, evaluated run for individual inference.');
      renderSubject();return;
    }
    try {
      const data=await api(`/api/subjects/${encodeURIComponent(id)}${state.selectedRun ? `?run_id=${encodeURIComponent(state.selectedRun)}` : ''}`);
      if(request!==state.subjectRequest)return;
      state.subject=data;renderSubject();
    } catch(error){if(request===state.subjectRequest){setNotice('subject-notice',`Could not load subject: ${error.message}`,true);renderSubject();}}
    finally{if(request===state.subjectRequest){state.subjectLoading=false;renderSubject();}}
  }
  function roiLabel(index) {
    const value=state.subject?.roi_labels?.[index];
    return value && typeof value==='object' ? value.name || value.label || value.id || `AAL-${String(index+1).padStart(3,'0')}` : value || `AAL-${String(index+1).padStart(3,'0')}`;
  }
  function emptyBlock(title, note, ico='brain') {return `<div class="empty-block">${icon(ico)}<strong>${esc(title)}</strong><p>${esc(note)}</p></div>`;}
  function renderSubject() {
    const s=state.subject, pred=s?.prediction;
    document.querySelector('.subject-detail').setAttribute('aria-busy',String(state.subjectLoading));
    const isLoading=state.subjectLoading && !s;
    const summary=state.subjects.find((v)=>String(v.id)===String(state.selectedSubject));
    $('subject-heading').textContent=s?.id || (isLoading ? `Loading ${state.selectedSubject}…` : state.selectedSubject ? String(state.selectedSubject) : 'Select a real subject');
    $('subject-meta').innerHTML=s?`<span>${esc(s.site || 'Site not reported')}</span><span class="badge ${String(labelOf(s)).toUpperCase()==='ASD'?'label-asd':'label-hc'}">True label ${esc(labelOf(s))}</span><span class="badge neutral">${esc(splitText(s.split))}</span>${selectedRound()!==null?`<span class="badge violet-outline">Validation checkpoint R${fmt(selectedRound())}</span>`:''}${num(s.age)!==null?`<span>${fmt(s.age,1)}  years</span>`:''}`:isLoading?'Reading measured connectivity and model outputs…':'Select a subject on the left to inspect connectivity and model outputs.';
    if (pred && num(pred.asd_probability)!==null) {
      const asd=Number(pred.asd_probability), label=String(pred.predicted_label ?? (asd>=.5?'ASD':'HC')).toUpperCase();
      const labelText=label==='1'?'ASD':label==='0'?'HC':label;
      $('prediction-result').innerHTML=`<strong class="${labelText==='HC'?'hc':''}">${esc(labelText)}</strong><small>ASD probability ${percentage(asd)}</small>`;
    } else $('prediction-result').innerHTML=`<span class="badge neutral">${validationOnly(state.run)?'Validation selection · inference not released':isLoading?'Loading inference':s?'No model output':'Select a subject'}</span>`;
    renderHeatmap();renderNetwork();renderExperts();
    const note=s?.note;
    $('inference-note').textContent=note ? `${typeof note==='object'?JSON.stringify(note):note} · Research only; not for clinical diagnosis.` : 'Model outputs are for research; ASD probability is not clinical risk. Training-subject predictions are not generalization estimates.';
  }
  function correlationColor(value) {
    const v=clamp(Number(value),-1,1), center=[247,248,243], end=v>=0?[42,139,124]:[121,99,162], a=Math.abs(v);
    return center.map((c,i)=>Math.round(c+(end[i]-c)*a));
  }
  function renderHeatmap() {
    const fc=state.subject?.fc, canvas=$('fc-canvas'), ctx=canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width,canvas.height);
    const valid=Array.isArray(fc)&&fc.length>0&&fc.every((row)=>Array.isArray(row)&&row.length===fc.length);
    $('heatmap-empty').classList.toggle('hidden',valid);
    if(!valid){$('heatmap-empty').innerHTML=`${icon('layers')}<strong>${state.subjectLoading?'Reading the measured matrix':'Awaiting connectivity data'}</strong><span>${state.subject?'The service has not returned a valid matrix':'Select a subject to load the measured matrix'}</span>`;return;}
    const n=fc.length, buffer=document.createElement('canvas');buffer.width=n;buffer.height=n;
    const bctx=buffer.getContext('2d'), img=bctx.createImageData(n,n);
    for(let row=0;row<n;row++)for(let col=0;col<n;col++){
      const value=num(fc[row][col]), p=(row*n+col)*4, rgb=value===null?[233,236,226]:correlationColor(value);
      img.data[p]=rgb[0];img.data[p+1]=rgb[1];img.data[p+2]=rgb[2];img.data[p+3]=255;
    }
    bctx.putImageData(img,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(buffer,0,0,canvas.width,canvas.height);
    canvas.setAttribute('aria-label',`Subject ${state.subject.id}: ${n} × ${n} Pearson correlation matrix, color range -1 to +1`);
    $('heatmap-readout').textContent=`${n} × ${n} · Pearson r ∈ [−1, +1] · Hover to inspect a connection`;
  }
  function renderNetwork() {
    const s=state.subject, raw=Array.isArray(s?.top_connections)?s.top_connections:[];
    const n=Array.isArray(s?.fc)?s.fc.length:116;
    const edges=raw.map((edge)=>({...edge,source:num(edge.source),target:num(edge.target),value:num(edge.value)})).filter((e)=>e.source!==null&&e.target!==null&&e.value!==null&&e.source>=0&&e.target>=0&&e.source<n&&e.target<n).slice(0,60);
    if(!edges.length){$('roi-network').innerHTML=emptyBlock(state.subjectLoading?'Reading connections':'Awaiting measured connections','Only API top_connections are drawn; no simulated network is generated.','brain');$('network-count').textContent='Awaiting connections';return;}
    const size=360,cx=180,cy=180,r=133, pos=(i)=>({x:cx+r*Math.cos(-Math.PI/2+i/n*Math.PI*2),y:cy+r*Math.sin(-Math.PI/2+i/n*Math.PI*2)});
    const used=new Set(edges.flatMap((e)=>[e.source,e.target]));
    const labelIds=[];
    [...used].sort((a,b)=>a-b).forEach((i)=>{if(labelIds.length<12&&labelIds.every((j)=>Math.min(Math.abs(i-j),n-Math.abs(i-j))>=8))labelIds.push(i);});
    let svg=`<svg viewBox="0 0 ${size} ${size}" role="img" aria-label="${edges.length} measured strong connections in a circular ROI schematic; not anatomical coordinates"><circle cx="180" cy="180" r="133" fill="none" stroke="#edf1e6" stroke-width="1"/><circle cx="180" cy="180" r="110" fill="none" stroke="#f4f6ef" stroke-width="1" stroke-dasharray="2 5"/>`;
    edges.forEach((e)=>{const a=pos(e.source),b=pos(e.target);svg+=`<path d="M${a.x.toFixed(2)} ${a.y.toFixed(2)} Q180 180 ${b.x.toFixed(2)} ${b.y.toFixed(2)}" fill="none" stroke="${e.value>=0?'#5b9e87':'#ad94c3'}" stroke-width="${(.5+Math.min(1,Math.abs(e.value))*1.4).toFixed(2)}" opacity=".46"><title>${esc(roiLabel(e.source))} ↔ ${esc(roiLabel(e.target))}: r = ${e.value.toFixed(4)}</title></path>`;});
    for(let i=0;i<n;i++){const p=pos(i);svg+=`<circle cx="${p.x.toFixed(2)}" cy="${p.y.toFixed(2)}" r="${used.has(i)?2.5:1.35}" fill="${used.has(i)?'#82a98b':'#dce5d3'}"><title>${esc(roiLabel(i))}</title></circle>`;}
    labelIds.forEach((i)=>{const a=-Math.PI/2+i/n*Math.PI*2,rr=147,xx=cx+rr*Math.cos(a),yy=cy+rr*Math.sin(a);let label=String(roiLabel(i));if(label.length>14)label=label.slice(0,13)+'…';svg+=`<text x="${xx.toFixed(2)}" y="${(yy+2.5).toFixed(2)}" text-anchor="${Math.cos(a)>.15?'start':Math.cos(a)<-.15?'end':'middle'}" font-size="6.5" fill="#a0ad90">${esc(label)}</text>`;});
    svg+='</svg>';$('roi-network').innerHTML=svg;$('network-count').textContent=`${edges.length}  connectionsConnections`;
  }
  function globalGate(pred) {
    const gate=pred?.gate;
    if(num(gate)!==null)return clamp(Number(gate),0,1);
    if(gate&&typeof gate==='object'&&!Array.isArray(gate)){const v=gate.global??gate.global_weight;if(num(v)!==null)return clamp(Number(v),0,1);}
    return null;
  }
  function renderExperts() {
    const p=state.subject?.prediction;
    if(!p||[p.global_probability,p.private_probability,p.asd_probability].every((v)=>num(v)===null)){$('expert-content').innerHTML=emptyBlock(state.subjectLoading?'Reading expert outputs':'No model inference available','Complete training and select that run to inspect measured global/private expert and gate outputs.','layers');return;}
    const g=globalGate(p), windows=Array.isArray(p.window_probabilities)?p.window_probabilities.filter((v)=>num(v)!==null):[];
    $('expert-content').innerHTML=`<div class="expert-flow"><div class="expert-pair"><div class="expert-box"><div><h3>Global expert</h3><p>Global · Shared cross-site knowledge</p></div><strong>${percentage(p.global_probability)}</strong></div><div class="expert-box private"><div><h3>Private expert</h3><p>Private · Site-specific information</p></div><strong>${percentage(p.private_probability)}</strong></div></div><div class="expert-arrow">${icon('arrow')}</div><div class="gate-box"><h3>Gated fusion · mean weight</h3><div class="gate-bar" aria-label="${g===null?'Gate weights unavailable':`Global weight ${percentage(g)}, Private weight ${percentage(1-g)}`}">${g!==null?`<span style="width:${g*100}%"></span><span style="width:${(1-g)*100}%"></span>`:''}</div><div class="gate-values"><span>Global ${percentage(g)}</span><span>Private ${g===null?'—':percentage(1-g)}</span></div><div class="gate-output"><span>Final ASD probability</span><strong>${percentage(p.asd_probability)}</strong></div></div></div>${windows.length?`<p class="window-summary">This subject ${fmt(windows.length)} measured window outputs · window ASD probability range ${percentage(Math.min(...windows))} — ${percentage(Math.max(...windows))}; The service aggregates final probabilities by subject.</p>`:''}`;
  }
  async function startTraining(event) {
    event.preventDefault();
    if(state.trainBusy||$('start-train').disabled)return;
    const form=$('train-form');if(!form.reportValidity())return;
    const payload={rounds:Number($('rounds').value),seed:Number($('seed').value),method:$('method').value,max_per_site:Number($('max-per-site').value),noise_scale:Number($('noise-scale').value),alignment_weight:Number($('alignment-weight').value)};
    if(!Number.isInteger(payload.rounds)||!Number.isInteger(payload.seed)||!Number.isInteger(payload.max_per_site)){setNotice('train-notice','Rounds, seed, and subject limit must be integers.',true);return;}
    state.trainBusy=true;updateControls();setNotice('train-notice','Submitting a real training job…');
    try {
      const response=await api('/api/train',{method:'POST',body:JSON.stringify(payload)});
      if(!response.run_id)throw new Error('The service returned no run_id; the run cannot be linked.');
      state.selectedRun=response.run_id;state.selectionInitialized=true;state.manualSelection=true;state.run={id:response.run_id,status:response.status||'queued',config:payload};
      resetSubjectData();
      setNotice('train-notice',`Run ${response.run_id} submitted. Progress updates on the right.`);toast('Training submitted; awaiting measured records');
      await refresh({quiet:true});
    } catch(error){setNotice('train-notice',error.message,true);}
    finally{state.trainBusy=false;updateControls();schedulePoll();}
  }
  async function stopTraining() {
    if(state.stopBusy||!state.selectedRun||!isActive(state.run))return;
    state.stopBusy=true;updateControls();
    try{const result=await api(`/api/runs/${encodeURIComponent(state.selectedRun)}/stop`,{method:'POST',body:'{}'});if(state.run)state.run.status=result.status||'stopping';toast('Stop requested; saving run state');await refresh({quiet:true});}
    catch(error){setNotice('train-notice',`Could not stop training: ${error.message}`,true);}
    finally{state.stopBusy=false;updateControls();schedulePoll();}
  }
  function exportJson(data,filename) {
    if(!data)return;
    const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');
    a.href=url;a.download=filename;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('Exported actual API records');
  }
  document.addEventListener('click',(event)=>{
    const nav=event.target.closest('[data-view],[data-navigate]');if(nav){navigate(nav.dataset.view||nav.dataset.navigate);return;}
    if(event.target.closest('[data-open-optimized]')){const id=optimizationRunId();if(id)selectRun(id);return;}
    const subject=event.target.closest('[data-subject-id]');if(subject&&!subject.closest('#health-root')){selectSubject(subject.dataset.subjectId);return;}
    if(event.target.closest('[data-retry]'))refresh();
    if(event.target.closest('[data-load-subjects]'))loadSubjects();
    if($('sidebar').classList.contains('open')&&!event.target.closest('#sidebar,#menu-toggle')){$('sidebar').classList.remove('open');$('menu-toggle').setAttribute('aria-expanded','false');}
  });
  $('run-select').addEventListener('change',(event)=>selectRun(event.target.value));
  $('refresh-button').addEventListener('click',()=>{if(state.view==='inference'){state.subjectsLoaded=false;if(state.selectedSubject)selectSubject(state.selectedSubject);}refresh();});
  $('train-form').addEventListener('submit',startTraining);
  $('stop-train').addEventListener('click',stopTraining);
  $('subject-search').addEventListener('input',renderSubjectList);
  $('site-filter').addEventListener('change',renderSubjectList);
  $('menu-toggle').addEventListener('click',()=>{const open=$('sidebar').classList.toggle('open');$('menu-toggle').setAttribute('aria-expanded',String(open));$('menu-toggle').setAttribute('aria-label',open?'Close navigation':'Open navigation');});
  document.addEventListener('keydown',(event)=>{if(event.key==='Escape'){$('sidebar').classList.remove('open');$('menu-toggle').setAttribute('aria-expanded','false');}});
  $('export-run').addEventListener('click',()=>exportJson(state.run,`fedbrain-${state.selectedRun || 'run'}.json`));
  $('export-protocol').addEventListener('click',()=>exportJson({exported_at:new Date().toISOString(),status:state.status,optimization:state.optimization,selected_run:state.run},'fedbrain-reproduction-snapshot.json'));
  $('fc-canvas').addEventListener('pointermove',(event)=>{
    const fc=state.subject?.fc;if(!Array.isArray(fc)||!fc.length)return;
    const rect=event.currentTarget.getBoundingClientRect(),col=clamp(Math.floor((event.clientX-rect.left)/rect.width*fc.length),0,fc.length-1),row=clamp(Math.floor((event.clientY-rect.top)/rect.height*fc.length),0,fc.length-1);
    $('heatmap-readout').textContent=`${roiLabel(row)} ↔ ${roiLabel(col)} · r = ${fmt(fc[row]?.[col],4)}`;
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden){clearTimeout(state.polling);state.polling=null;}else if(!!state.status?.active_run||state.runs.some(isActive)||isActive(state.run)||['searching','confirming'].includes(state.optimization?.status))refresh({quiet:true});});
  window.addEventListener('hashchange',()=>navigate(location.hash.replace('#',''),false));
  let resizeFrame;
  window.addEventListener('resize',()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{if(state.view==='overview')renderChart('overview-chart','accuracy');if(state.view==='train')renderChart('loss-chart','loss');});});
  renderAll();renderSubjectList();renderSubject();navigate(location.hash.replace('#','')||'overview',false);refresh({quiet:true});
})();
