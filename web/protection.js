/* Actual public site previews and recorded shared vectors; packet motion is schematic. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
  const number = (v) => v !== null && v !== undefined && Number.isFinite(Number(v)) ? Number(v) : null;
  const fmt = (v) => number(v) === null ? 'Not recorded' : Number(v).toLocaleString('en-US');
  const errorText = (error, fallback) => window.EnglishUI?.message(error?.message ?? error, fallback) || fallback;
  const colors = ['#578d80', '#8da466', '#8d87ad', '#a59066', '#699aa4', '#b08b9d'];
  const state = { facts: null, previews: null, profileError: null, previewError: null,
    profileLoading: false, mode: 'local', runId: '', request: 0, abort: null,
    context: null, playing: false, timers: [] };

  function scaffold() {
    return `<article class="card protection-card dataflow-card">
      <header class="df-heading"><div><span class="section-kicker">FEDERATED LEARNING</span><h2>Federated Data Flow</h2></div>
        <span class="df-simulation">Single-process demo · Illustrative data flow</span>
        <div class="df-modes" role="group" aria-label="Local data or upload payloads"><button data-df-mode="local" aria-pressed="true">Local Data</button><button data-df-mode="upload" aria-pressed="false">Upload Payloads</button></div>
        <button id="df-play" class="button primary" disabled>Play Transfer</button>
      </header>
      <div class="df-toolbar"><span id="df-preview-scope">Loading actual site examples…</span><label for="protection-run">Mechanism Record</label><select id="protection-run" disabled><option value="">Loading…</option></select><span id="protection-run-context"></span></div>
      <div class="df-map"><section class="df-local-domain" aria-label="Local data boundaries for six sites"><div class="df-domain-label"><strong>Local Data Boundary</strong><span>FC, BOLD, individual labels, private heads and gates stay local</span></div><div id="df-sites" class="df-sites"></div></section>
        <div id="df-transmission" class="df-transmission" aria-label="Transfer paths for parameters and aggregate statistics only"></div>
        <section id="df-server" class="df-server" aria-label="Aggregator displays shared parameters, not individual data"></section>
      </div>
      <div class="df-footer"><span class="df-key"><i class="df-key-local"></i>Local Health Data</span><span class="df-key"><i class="df-key-theta"></i>θ Shared Parameters</span><span class="df-key"><i class="df-key-stats"></i>Σ Alignment Statistics</span><span id="df-phase" role="status" aria-live="polite">Local data do not move with the illustrative packets</span></div>
      <div id="protection-mechanisms" class="df-mechanisms"></div>
      <details class="df-details"><summary>Records and Scope</summary><div id="df-record-notes"></div></details>
      <p class="df-scope">Single-process public-data demo. The diagram shows training-message boundaries, not access control or captured network traffic.</p>
    </article>`;
  }

  function init() {
    const root = $('data-protection-root');
    if (!root) return;
    document.body.classList.add('protection-integrated');
    document.body.classList.remove('protection-protocol');
    root.innerHTML = scaffold();
    root.addEventListener('click', (event) => {
      const mode = event.target.closest('[data-df-mode]');
      const site = event.target.closest('[data-df-site]');
      if (mode) { state.mode = mode.dataset.dfMode; renderMode(); }
      if (site && $('health-site')) {
        $('health-site').value = site.dataset.dfSite;
        $('health-site').dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
    $('df-play').addEventListener('click', play);
    $('protection-run').addEventListener('change', (event) => loadProfile(event.target.value));
    window.addEventListener('health-data-context', (event) => {
      state.context = event.detail;
      markSelectedSite();
    });
    state.context = window.HealthData?.context();
    window.matchMedia('(max-width:820px)').addEventListener('change', () => {
      stop();
      renderPaths();
      $('df-play').disabled = !state.facts?.transfer?.available || !state.previews;
    });
    render();
    loadSites();
    loadProfile();
  }

  async function get(url, signal) {
    const response = await fetch(url, { signal, cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    return window.EnglishUI?.translate(payload) ?? payload;
  }

  async function loadSites() {
    try { state.previews = await get('/api/health-data/sites'); }
    catch (error) { state.previewError = `Site examples are unavailable: ${errorText(error, 'Request failed')}`; }
    render();
  }

  async function loadProfile(runId = '') {
    stop();
    const generation = ++state.request;
    state.abort?.abort();
    state.abort = new AbortController();
    state.facts = null;
    state.profileError = null;
    state.profileLoading = true;
    state.runId = runId;
    render();
    try {
      const facts = await get(`/api/data-protection${runId ? `?run_id=${encodeURIComponent(runId)}` : ''}`, state.abort.signal);
      if (generation !== state.request) return;
      state.facts = facts;
      state.runId = facts.run?.id || runId;
    } catch (error) {
      if (generation === state.request && error.name !== 'AbortError') state.profileError = `Parameter records are unavailable: ${errorText(error, 'Request failed')}`;
    } finally {
      if (generation === state.request) { state.profileLoading = false; render(); }
    }
  }

  function transferSite(id) {
    return (state.facts?.transfer?.sites || []).find((site) => site.id === id);
  }

  function validVector(values) {
    return Array.isArray(values) && values.length === 64 && values.every((v) => number(v) !== null);
  }

  function paramGrid(values, kind) {
    if (!validVector(values)) return '<div class="df-unrecorded">No parameter snapshot</div>';
    const bound = Math.max(0.000001, ...values.map(Math.abs));
    return `<div class="df-param-grid ${kind}" aria-label="64 actual recorded shared-parameter coordinates">${values.map((value, i) => {
      const opacity = 0.12 + Math.abs(value) / bound * 0.76;
      return `<i class="df-param-cell" data-coordinate="${i}" data-value="${value}" style="height:${4 + Math.abs(value) / bound * 42}px;background:${value >= 0 ? `rgba(72,127,179,${opacity})` : `rgba(139,112,177,${opacity})`}" title="Recorded coordinate ${i + 1}: ${value.toFixed(6)}"></i>`;
    }).join('')}</div>`;
  }

  function boldSpark(example, color) {
    const bold = example?.bold;
    if (!bold?.values?.length || !bold.frame_indices?.length) return '<span class="df-unrecorded">BOLD unavailable</span>';
    const extent = Math.max(1, ...bold.values.map(Math.abs));
    const first = bold.frame_indices[0], span = Math.max(1, bold.frame_indices[bold.frame_indices.length - 1] - first);
    const points = bold.values.map((value, i) => `${((bold.frame_indices[i] - first) / span * 108).toFixed(2)},${(24 - value / extent * 19).toFixed(2)}`).join(' ');
    return `<svg class="df-bold" viewBox="0 0 108 48" role="img" aria-label="${esc(bold.label)} recorded BOLD frame series"><path d="M0 24H108" stroke="#e3ebda"/><polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.2"/><title>${esc(bold.label)} · ${bold.values.length} recorded frames${bold.downsampled ? ' (evenly sampled from the full series)' : ''}</title></svg>`;
  }

  function renderSites() {
    const sites = state.previews?.sites || [];
    $('df-sites').innerHTML = sites.length ? sites.map((site, index) => {
      const example = site.example, transfer = transferSite(site.id);
      return `<button class="df-site" data-df-site="${esc(site.id)}" style="--site-color:${colors[index % 6]}">
        <span class="df-site-heading"><strong>${esc(site.id)}</strong><small>${fmt(site.total)} participants</small><span class="df-local-tag">Local Example</span></span>
        <span class="df-local-preview">${example?.fc ? `<canvas class="df-fc" id="df-fc-${esc(site.id)}" width="116" height="116" role="img" aria-label="${esc(site.id)} actual 116 × 116 individual FC"></canvas>` : '<span class="df-unrecorded">FC unavailable</span>'}<span class="df-spark-wrap">${boldSpark(example, colors[index % 6])}<small>BOLD · frames</small></span></span>
        <span class="df-upload-preview">${paramGrid(transfer?.uploaded_shared_sketch, 'df-upload-grid')}<span class="df-coordinate-label">θ · 64 / ${fmt(state.facts?.channels?.shared_model?.parameter_count)}<small>Parameter coordinates, not FC</small></span></span>
        <span class="df-example-id">${example ? `${esc(example.subject_id)} · ${esc(example.label_name)} · ${fmt(example.n_frames)} frames` : 'Actual example pending'}<span class="df-stays">Private blocks stay local</span></span>
      </button>`;
    }).join('') : `<div class="df-empty">${esc(state.previewError || 'Loading recorded FC and BOLD from six sites…')}</div>`;
    sites.forEach((site) => drawFC($(`df-fc-${site.id}`), site.example?.fc));
    markSelectedSite();
  }

  function drawFC(canvas, matrix) {
    if (!canvas || !Array.isArray(matrix) || matrix.length !== 116) return;
    const context = canvas.getContext('2d'), image = context.createImageData(116, 116);
    for (let row = 0; row < 116; row++) {
      for (let col = 0; col < 116; col++) {
        const value = matrix[row][col], mix = Math.min(1, Math.abs(value));
        const end = value >= 0 ? [44,143,118] : [145,111,178];
        image.data.set([249,249,243].map((v, i) => Math.round(v + (end[i] - v) * mix)).concat(255), (row * 116 + col) * 4);
      }
    }
    context.putImageData(image, 0, 0);
  }

  function renderServer() {
    const transfer = state.facts?.transfer;
    const vector = transfer?.aggregate_shared_sketch;
    const siteRows = (transfer?.sites || []).map((site, index) => {
      const values = site.uploaded_shared_sketch;
      const bound = validVector(values) ? Math.max(.000001, ...values.map(Math.abs)) : 1;
      return `<div class="df-contribution"><span>${esc(site.id)}</span><div>${validVector(values) ? values.slice(0, 12).map((value) => `<i style="height:${3 + Math.abs(value) / bound * 13}px;background:${colors[index % 6]}" title="Shared parameter ${value.toFixed(6)}"></i>`).join('') : '<small>Not recorded</small>'}</div></div>`;
    }).join('');
    $('df-server').innerHTML = `<div class="df-server-icon" aria-hidden="true">∑</div><h3>Aggregator</h3><p class="df-server-subtitle">Shared-Parameter Aggregation</p>${paramGrid(vector, 'df-aggregate-grid')}<div class="df-contributions">${siteRows}</div>
      <div class="df-no-raw"><svg viewBox="0 0 30 25" aria-hidden="true"><rect x="4" y="3" width="22" height="18" rx="2" fill="none"/><path d="M4 9H26M11 3V21M19 3V21M2 24L28 1"/></svg><span>No individual FC / BOLD in these messages</span></div><span class="df-global-label">θG Global model returns to sites</span>`;
  }

  function renderPaths() {
    const sites = state.previews?.sites || [];
    const align = state.facts?.channels?.alignment_statistics?.uploaded === true;
    const mobile = window.matchMedia('(max-width:820px)').matches;
    let svg = `<svg viewBox="${mobile ? '0 0 340 260' : '0 0 220 300'}"` + ' role="img" aria-label="Protocol diagram: six sites upload shared parameters and alignment statistics, then receive the global model"><defs><marker id="df-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#8ca77b"/></marker><marker id="df-back" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#a490b5"/></marker></defs>';
    sites.forEach((site, index) => {
      const yy = 24 + index * 42, xx = 35 + index * 53;
      const d = mobile ? `M${xx} 47 C${xx} 130 170 135 170 240` : `M40 ${yy} C100 ${yy} 125 138 205 138`;
      const values = transferSite(site.id)?.uploaded_shared_sketch;
      const value = validVector(values) ? values[0].toFixed(3) : 'Not recorded';
      svg += `<text x="${mobile ? xx - 15 : 2}" y="${mobile ? 16 : yy - 7}" class="df-route-site">${esc(site.id)}</text><path class="df-upload-path" d="${d}" fill="none" stroke="${colors[index % 6]}" stroke-opacity=".38" marker-end="url(#df-arrow)"/><text x="${mobile ? xx - 20 : 42}" y="${mobile ? 36 : yy - 7}" class="df-route-value">θ ${value}</text>`;
      svg += packet(d, 'θ', '#5a8fbd', 0, 'parameter');
      if (align) svg += packet(d, 'Σ', '#beab76', .3, 'statistics');
    });
    const back = mobile ? 'M215 238 C332 238 332 62 310 48' : 'M205 174 C160 174 160 282 40 282';
    svg += `<path d="${back}" fill="none" stroke="#ae99bf" stroke-dasharray="4 4" marker-end="url(#df-back)"/><text x="${mobile ? 228 : 72}" y="${mobile ? 210 : 298}" class="df-route-return">${mobile ? 'θG Return ↑' : 'θG Global Model Return'}</text>${packet(back, 'θG', '#a48bb8', 1.9, 'global')}`;
    svg += `<text x="${mobile ? 97 : 70}" y="${mobile ? 94 : 12}" class="df-route-label">Parameters / Statistics Only</text></svg>`;
    $('df-transmission').innerHTML = svg;
  }

  function packet(path, label, color, delay, kind) {
    return `<g class="df-packet" opacity="0" data-packet-kind="${kind}" data-delay="${delay}"><rect x="-10" y="-8" width="23" height="17" rx="4" fill="${color}"/><text x="1" y="4" text-anchor="middle" fill="white" font-size="10">${label}</text><animateMotion begin="indefinite" dur="1.6s" path="${path}" fill="remove"/><animate attributeName="opacity" begin="indefinite" dur="1.6s" values="0;1;1;0" keyTimes="0;.08;.92;1" fill="remove"/></g>`;
  }

  function markSelectedSite() {
    document.querySelectorAll('[data-df-site]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.dfSite === state.context?.site)));
  }

  function renderMode() {
    $('data-protection-root').dataset.flowMode = state.mode;
    document.querySelectorAll('[data-df-mode]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.dfMode === state.mode)));
  }

  function stop() {
    state.timers.forEach(clearTimeout);
    state.timers = [];
    state.playing = false;
    $('data-protection-root')?.classList.remove('df-playing');
  }

  function play() {
    if (state.playing || !state.facts?.transfer?.available || !state.previews) return;
    state.playing = true;
    $('df-play').disabled = true;
    $('data-protection-root').classList.add('df-playing');
    $('df-phase').textContent = 'Upload θ parameters and Σ statistics; raw data stay local';
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!reduced) document.querySelectorAll('.df-packet').forEach((packetNode) => {
      state.timers.push(setTimeout(() => packetNode.querySelectorAll('animateMotion,animate').forEach((animation) => animation.beginElement()), Number(packetNode.dataset.delay) * 1000));
    });
    state.timers.push(setTimeout(() => { $('df-phase').textContent = 'After aggregation, the shared global model returns; private heads and gates stay local'; }, reduced ? 0 : 1800));
    state.timers.push(setTimeout(() => {
      stop();
      $('df-play').disabled = !state.facts?.transfer?.available;
      $('df-phase').textContent = reduced ? 'Transfer paths highlighted statically (reduced motion)' : 'Illustration complete · Only parameters and statistics moved; local health data stayed local';
    }, reduced ? 500 : 3700));
  }

  function render() {
    if (!$('df-sites')) return;
    const facts = state.facts, runs = facts?.available_runs || [], transfer = facts?.transfer;
    $('data-protection-root').dataset.ready = facts ? 'true' : 'false';
    $('data-protection-root').dataset.previewsReady = state.previews ? 'true' : 'false';
    $('data-protection-root').dataset.run = state.runId;
    $('df-preview-scope').textContent = state.previews ? `${fmt(state.previews.total_subjects)} participants in the public cohort · Six actual local examples` : state.previewError || 'Loading actual site data…';
    $('protection-run').innerHTML = runs.length ? runs.map((run) => `<option value="${esc(run.id)}">${esc(run.id)}${run.homomorphic ? ' · HE' : ''}</option>`).join('') : '<option value="">Parameter records pending</option>';
    $('protection-run').value = state.runId;
    $('protection-run').disabled = state.profileLoading || !runs.length;
    $('protection-run-context').textContent = facts ? `${fmt(facts.run?.data_subjects)} participants / ${fmt(facts.run?.rounds_completed)} recorded rounds${facts.mechanisms?.paillier?.configured ? ' · Historical encryption example' : ''}` : '';
    const noise = facts?.mechanisms?.perturbation, he = facts?.mechanisms?.paillier, align = facts?.channels?.alignment_statistics;
    $('protection-mechanisms').innerHTML = facts ? `<span>Parameter Noise: ${noise?.configured ? `b=${esc(noise.noise_scale)}${noise.observed ? ' Applied' : ' Configured Only'}` : 'Disabled'} · No certified ε-DP guarantee</span><span class="df-he-off">Homomorphic Encryption: ${he?.configured ? `Paillier ${esc(he.key_bits)} · Keys in the same process` : 'Disabled for this run'}</span><span>Σ ${align?.uploaded ? 'Means / class means / window counts · Plaintext statistics' : 'No alignment statistics for this method'}</span>` : `<span>${esc(state.profileError || 'Loading recorded protection settings…')}</span>`;
    $('df-record-notes').textContent = `${transfer?.coordinate_note || 'No values are generated without parameter snapshots.'} ${transfer?.note || ''} Site examples are public subjects, not necessarily participants in the selected run. Noise and Paillier apply only to shared parameters, not alignment statistics; the returned model is plaintext.`;
    $('df-play').disabled = !transfer?.available || !state.previews || state.playing;
    $('df-phase').textContent = transfer?.available ? 'Local data do not move with the illustrative packets' : state.profileLoading ? 'Loading parameter records…' : 'No transfer snapshots for this run; parameters and ciphertext are not fabricated';
    renderSites();
    renderServer();
    renderPaths();
    renderMode();
  }

  init();
})();
