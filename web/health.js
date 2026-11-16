/* Brain-health data atlas. All matrices, cohorts and signals come from health-data APIs. */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
  const number = (value) => value !== null && value !== undefined && value !== '' &&
    Number.isFinite(Number(value)) ? Number(value) : null;
  const format = (value, digits = 0) => number(value) === null ? '—' :
    Number(value).toLocaleString('en-US', { maximumFractionDigits: digits });
  const errorText = (error, fallback) => window.EnglishUI?.message(error?.message ?? error, fallback) || fallback;
  const palette = ['#5a9985', '#9b84b4', '#bd9c63'];
  const state = {
    cohort: null, subject: null, subjectId: null, group: 'HC', mode: 'cohort',
    site: '', ageMin: '', ageMax: '', query: '', edges: 40, selectedROI: null,
    cohortLoading: false, subjectLoading: false, cohortError: null, subjectError: null,
    cohortRequest: 0, subjectRequest: 0, cohortAbort: null, subjectAbort: null
  };

  function endpoint() {
    const params = new URLSearchParams();
    if (state.site) params.set('site', state.site);
    if (state.ageMin !== '') params.set('age_min', state.ageMin);
    if (state.ageMax !== '') params.set('age_max', state.ageMax);
    return `/api/health-data${params.size ? `?${params}` : ''}`;
  }

  async function request(url, signal) {
    const response = await fetch(url, { signal, cache: 'no-store' });
    if (!response.ok) {
      const error = new Error(`HTTP ${response.status}`);
      error.status = response.status;
      throw error;
    }
    const payload = await response.json();
    return window.EnglishUI?.translate(payload) ?? payload;
  }

  function scaffold() {
    return `
      <article class="card health-controls">
        <div class="health-filter-top">
          <div class="health-group" role="group" aria-label="Select cohort connectivity view">
            <button data-health-group="HC" aria-pressed="true">Healthy Controls (HC)</button>
            <button data-health-group="ASD" aria-pressed="false">ASD Group</button>
            <button data-health-group="difference" aria-pressed="false">ASD − HC Difference</button>
          </div>
          <div class="health-filters">
            <label>Site <select id="health-site"><option value="">All Sites</option></select></label>
            <label>Age <input id="health-age-min" type="number" min="0" max="120" step="0.1" placeholder="Any" aria-label="Minimum age"></label>
            <span class="health-age-dash">—</span>
            <input id="health-age-max" type="number" min="0" max="120" step="0.1" placeholder="Any" aria-label="Maximum age">
            <button id="health-apply" class="text-button">Apply Filters</button>
          </div>
        </div>
        <div class="health-count-strip" id="health-counts"></div>
        <div id="health-notice" role="status" aria-live="polite"></div>
        <p class="health-source" id="health-source">Loading data.</p>
      </article>
      <div id="data-protection-root" aria-label="Health data boundaries in the federated training protocol"></div>
      <div class="health-data-grid">
        <article class="card health-matrix-card">
          <div class="card-heading"><div><div class="section-kicker">FUNCTIONAL CONNECTIVITY</div><h2 id="health-matrix-title">HC Brain Connectivity</h2></div><span class="badge teal-outline" id="health-matrix-count">Loading</span></div>
          <p class="health-provenance" data-health-provenance>Research data browser · Not a training upload</p>
          <div class="health-modes" role="group" aria-label="Select connectivity matrix source">
            <button data-health-mode="cohort" aria-pressed="true">Cohort Mean</button>
            <button data-health-mode="subject" aria-pressed="false">Individual Data</button>
            <button data-health-mode="reference" aria-pressed="false">Individual − HC Reference</button>
          </div>
          <div class="health-matrix-wrap"><canvas id="health-fc" width="464" height="464" role="img" aria-label="Actual 116 × 116 connectivity matrix"></canvas><div id="health-matrix-empty" class="health-chart-empty">Loading connectivity matrix.</div></div>
          <div class="health-colorbar"><span id="health-scale-min">−1</span><i></i><span id="health-scale-max">+1</span><small id="health-scale-label">Pearson r</small></div>
          <div id="health-matrix-readout" class="health-readout" aria-live="polite">Hover for ROI pairs and connectivity values; click to highlight a node.</div>
          <p id="health-matrix-note" class="health-caption">Equal-subject Fisher-z cohort mean; file column order is not anatomical position.</p>
        </article>
        <article class="card health-network-card">
          <div class="card-heading"><div><div class="section-kicker">AAL116 · CONNECTIVITY</div><h2>Brain Connectivity</h2></div><span class="badge violet-outline">Schematic circle · Not anatomical</span></div>
          <div class="health-edge-controls"><label for="health-edges">Show strongest <strong id="health-edge-count">40</strong> connections</label><input id="health-edges" type="range" min="10" max="80" step="10" value="40"><button id="health-roi-reset" class="text-button">Clear Selection</button></div>
          <div id="health-network" class="health-network"></div>
          <div class="health-network-legend"><span><i class="health-positive"></i>Positive</span><span><i class="health-negative"></i>Negative</span><small>Ranked by absolute connectivity · Nodes are AAL file columns</small></div>
          <div id="health-network-readout" class="health-readout">Select an ROI to inspect connections. Circular placement is not anatomical.</div>
        </article>
      </div>
      <div class="health-distributions">
        <article class="card"><div class="card-heading"><div><div class="section-kicker">COHORT COMPOSITION</div><h2>Participants by Site</h2></div><div class="health-mini-legend"><span>● HC</span><span>● ASD</span></div></div><div id="health-sites"></div><p class="health-caption">Select a site to filter. Counts reflect the current filters.</p></article>
        <article class="card"><div class="card-heading"><div><div class="section-kicker">AGE DISTRIBUTION · YEARS</div><h2>Age Distribution</h2></div><span id="health-age-summary" class="badge neutral">Pending</span></div><div id="health-ages"></div><p id="health-age-note" class="health-caption"></p></article>
      </div>
      <article class="card health-cases">
        <div class="card-heading"><div><div class="section-kicker">SUBJECT TIME SERIES</div><h2>Individual BOLD Time Series</h2></div><span class="badge neutral" id="health-case-count">HC First</span></div>
        <div class="health-case-grid">
          <aside class="health-case-browser"><label class="sr-only" for="health-search">Search subject ID or site</label><input id="health-search" type="search" placeholder="Search subject ID / site"><div id="health-subjects" class="health-subjects" aria-label="Actual subject list"></div></aside>
          <div class="health-case-detail"><div id="health-subject-meta"></div><div id="health-signal-chart"></div><div id="health-subject-notice" class="health-caption" role="status"></div><div class="health-case-actions"><button data-health-mode="subject" class="text-button">View Individual FC / Network ↑</button><button data-health-mode="reference" class="text-button">Compare with HC Reference ↑</button></div></div>
        </div>
      </article>
      <p class="health-research-note">HC is a research control label, not a clinical health certification. ASD–HC differences are not adjusted for age, site or other confounders and are not biomarkers, individual risks or diagnoses.</p>`;
  }

  function init() {
    const root = $('health-root');
    if (!root) return;
    root.innerHTML = scaffold();
    root.addEventListener('click', onClick);
    root.addEventListener('keydown', (event) => {
      const roi = event.target.closest('[data-health-roi]');
      if (roi && (event.key === 'Enter' || event.key === ' ')) {
        event.preventDefault();
        state.selectedROI = Number(roi.dataset.healthRoi);
        drawNetwork();
      }
    });
    $('refresh-button')?.addEventListener('click', loadCohort);
    $('health-apply').addEventListener('click', applyFilters);
    $('health-site').addEventListener('change', applyFilters);
    $('health-search').addEventListener('input', (event) => {
      state.query = event.target.value;
      renderSubjects();
    });
    $('health-edges').addEventListener('input', (event) => {
      state.edges = Number(event.target.value);
      $('health-edge-count').textContent = state.edges;
      drawNetwork();
    });
    $('health-roi-reset').addEventListener('click', () => {
      state.selectedROI = null;
      drawNetwork();
    });
    $('health-fc').addEventListener('mousemove', inspectMatrix);
    $('health-fc').addEventListener('click', (event) => {
      const pair = matrixPair(event);
      if (pair) { state.selectedROI = pair.row; drawNetwork(); }
    });
    render();
    loadCohort();
  }

  function onClick(event) {
    const group = event.target.closest('[data-health-group]');
    const mode = event.target.closest('[data-health-mode]');
    const subject = event.target.closest('[data-health-subject]');
    const site = event.target.closest('[data-health-site]');
    const roi = event.target.closest('[data-health-roi]');
    if (group) {
      state.group = group.dataset.healthGroup;
      state.mode = 'cohort';
      state.selectedROI = null;
      ensureSubject();
      render();
    }
    if (mode) { state.mode = mode.dataset.healthMode; state.selectedROI = null; render(); }
    if (subject) { state.mode = 'subject'; loadSubject(subject.dataset.healthSubject); }
    if (site) { $('health-site').value = site.dataset.healthSite; applyFilters(); }
    if (roi) { state.selectedROI = Number(roi.dataset.healthRoi); drawNetwork(); }
  }

  function applyFilters() {
    const min = $('health-age-min').value;
    const max = $('health-age-max').value;
    if (min !== '' && max !== '' && Number(min) > Number(max)) {
      $('health-notice').innerHTML = '<div class="notice error">Minimum age must not exceed maximum age.</div>';
      return;
    }
    state.site = $('health-site').value;
    state.ageMin = min;
    state.ageMax = max;
    state.mode = 'cohort';
    loadCohort();
  }

  async function loadCohort() {
    const generation = ++state.cohortRequest;
    state.cohortAbort?.abort();
    state.cohortAbort = new AbortController();
    state.subjectAbort?.abort();
    state.subjectRequest++;
    state.cohort = null;
    state.subject = null;
    state.subjectId = null;
    state.cohortError = null;
    state.subjectError = null;
    state.cohortLoading = true;
    state.subjectLoading = false;
    render();
    try {
      const data = await request(endpoint(), state.cohortAbort.signal);
      if (generation !== state.cohortRequest) return;
      state.cohort = data;
      ensureSubject();
    } catch (error) {
      if (error.name !== 'AbortError' && generation === state.cohortRequest) {
        state.cohortError = error.status === 404 ?
          'The health-data API is not ready. Model outputs and simulated data are not used as substitutes.' : `Unable to load health data: ${errorText(error, 'Request failed')}`;
      }
    } finally {
      if (generation === state.cohortRequest) { state.cohortLoading = false; render(); }
    }
  }

  function eligibleSubjects() {
    const subjects = Array.isArray(state.cohort?.subjects) ? state.cohort.subjects : [];
    return subjects.filter((subject) => state.group === 'difference' || subject.label_name === state.group);
  }

  function ensureSubject() {
    const subjects = eligibleSubjects();
    if (subjects.some((subject) => subject.id === state.subjectId)) return;
    const first = subjects.find((subject) => subject.label_name === 'HC') || subjects[0];
    if (first) loadSubject(first.id);
    else {
      state.subjectAbort?.abort();
      state.subjectRequest++;
      state.subject = null;
      state.subjectId = null;
      state.subjectLoading = false;
    }
  }

  async function loadSubject(id) {
    const generation = ++state.subjectRequest;
    state.subjectAbort?.abort();
    state.subjectAbort = new AbortController();
    state.subjectId = id;
    state.subject = null;
    state.subjectError = null;
    state.subjectLoading = true;
    render();
    try {
      const subject = await request(`/api/health-data/subjects/${encodeURIComponent(id)}`, state.subjectAbort.signal);
      if (generation !== state.subjectRequest) return;
      state.subject = subject;
    } catch (error) {
      if (error.name !== 'AbortError' && generation === state.subjectRequest) {
        state.subjectError = `Unable to load subject data: ${errorText(error, 'Request failed')}`;
      }
    } finally {
      if (generation === state.subjectRequest) { state.subjectLoading = false; render(); }
    }
  }

  function currentMatrix() {
    if (state.mode === 'subject') return state.subject?.fc || null;
    if (state.mode === 'reference') return state.subject?.reference?.difference_fc || null;
    return state.group === 'difference' ? state.cohort?.difference?.fc || null :
      state.cohort?.groups?.[state.group]?.mean_fc || null;
  }

  function roiLabel(index) {
    return state.cohort?.roi_labels?.[index] || `AAL-${String(index + 1).padStart(3, '0')}`;
  }

  function render() {
    if (!$('health-root')) return;
    const data = state.cohort;
    $('health-root').dataset.group = state.group;
    $('health-root').dataset.mode = state.mode;
    $('health-root').dataset.subjectId = state.subject?.id || '';
    $('health-root').dataset.ready = data ? 'true' : 'false';
    document.querySelectorAll('[data-health-group]').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.healthGroup === state.group));
    });
    document.querySelectorAll('.health-modes [data-health-mode]').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.healthMode === state.mode));
    });
    $('health-apply').disabled = state.cohortLoading;
    $('health-counts').innerHTML = [
      ['Full Cohort', data?.total_subjects, 'Recorded Participants'], ['Current Selection', data?.selected_subjects, 'participants'],
      ['Healthy Controls (HC)', data?.n_hc, 'controls'], ['ASD Group', data?.n_asd, 'participants']
    ].map(([label, value, unit]) => `<div><span>${label}</span><strong>${format(value)}</strong><small>${unit}</small></div>`).join('');
    $('health-notice').innerHTML = state.cohortError ? `<div class="notice error">${escape(state.cohortError)}</div>` :
      state.cohortLoading ? '<div class="health-loading">Loading cohort and connectivity cache…</div>' : '';
    if (data) {
      const source = typeof data.source === 'object' ? data.source : {};
      $('health-source').innerHTML = `${escape(source.name || 'ABIDE I')} · ${escape(source.pipeline || '')} / ${escape(source.strategy || '')} / ${escape(source.atlas || 'AAL116')} <span>Independent of trained models</span> <a href="${escape(endpoint())}" target="_blank" rel="noopener">View Source Data Summary ↗</a>`;
      const previous = $('health-site').value;
      $('health-site').innerHTML = '<option value="">All Sites</option>' +
        (data.site_counts || []).map((site) => `<option value="${escape(site.id)}">${escape(site.id)}</option>`).join('');
      $('health-site').value = previous || state.site;
    }
    drawMatrix();
    drawNetwork();
    drawDistributions();
    renderSubjects();
    drawSignals();
    document.querySelectorAll('[data-health-provenance]').forEach((node) => {
      node.textContent = `Research data browser · Not a training upload · ${state.site || 'Multi-site public cohort'} / ${format(data?.selected_subjects)} participants`;
    });
    if (data) window.dispatchEvent(new CustomEvent('health-data-context', { detail: context() }));
  }

  function matrixRange(matrix) {
    if (state.mode === 'reference' || state.group === 'difference' && state.mode === 'cohort') {
      return Math.max(0.000001, ...matrix.flat().filter((v) => number(v) !== null).map(Math.abs));
    }
    return 1;
  }

  function rgb(value, range) {
    const amount = Math.min(1, Math.abs(value) / range);
    const start = [249, 249, 243], end = value >= 0 ? [44, 143, 118] : [145, 111, 178];
    return start.map((v, i) => Math.round(v + (end[i] - v) * amount));
  }

  function drawMatrix() {
    const matrix = currentMatrix(), canvas = $('health-fc'), context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    const valid = Array.isArray(matrix) && matrix.length === 116;
    $('health-matrix-empty').classList.toggle('hidden', valid);
    $('health-matrix-empty').textContent = state.cohortLoading || state.subjectLoading && state.mode !== 'cohort' ?
      'Loading recorded connectivity matrix…' : 'No matrix for the current selection; empty cohorts are not filled with zero connectivity.';
    const cohortMode = state.mode === 'cohort';
    const difference = state.group === 'difference' && cohortMode;
    $('health-matrix-title').textContent = cohortMode ? difference ? 'ASD − HC Descriptive Connectivity Difference' : `${state.group} Cohort Brain Connectivity` :
      state.mode === 'reference' ? `${state.subjectId || 'Subject'} − HC Reference` : `${state.subjectId || 'Subject'} · Recorded Brain Connectivity`;
    $('health-matrix-count').textContent = cohortMode ? difference ? `HC ${format(state.cohort?.n_hc)} / ASD ${format(state.cohort?.n_asd)}` :
      `n = ${format(state.cohort?.groups?.[state.group]?.count)}` : `Subject ${state.subjectId || '—'}`;
    $('health-matrix-note').textContent = cohortMode ? difference ? state.cohort?.difference?.note || 'Cohort differences are descriptive, not biomarkers.' :
      'Equal-subject Fisher-z mean, not duration-weighted; AAL IDs follow file column order.' : state.mode === 'reference' ?
        state.subject?.reference?.note || 'Reference comparisons are descriptive, not individual risk scores.' : 'Pearson connectivity from the full resting-state series; independent of training or predictions.';
    if (!valid) return;
    const range = matrixRange(matrix), buffer = document.createElement('canvas');
    buffer.width = buffer.height = 116;
    const pixels = buffer.getContext('2d').createImageData(116, 116);
    for (let row = 0; row < 116; row++) {
      for (let col = 0; col < 116; col++) {
        const [r, g, b] = rgb(Number(matrix[row][col]), range), offset = (row * 116 + col) * 4;
        pixels.data.set([r, g, b, 255], offset);
      }
    }
    buffer.getContext('2d').putImageData(pixels, 0, 0);
    context.imageSmoothingEnabled = false;
    context.drawImage(buffer, 0, 0, canvas.width, canvas.height);
    $('health-scale-min').textContent = `−${format(range, 3)}`;
    $('health-scale-max').textContent = `+${format(range, 3)}`;
    $('health-scale-label').textContent = state.mode === 'reference' || difference ? 'Connectivity difference Δr · Plot-specific scale' : 'Pearson r';
  }

  function matrixPair(event) {
    const matrix = currentMatrix();
    if (!Array.isArray(matrix)) return null;
    const box = $('health-fc').getBoundingClientRect();
    const row = Math.min(115, Math.max(0, Math.floor((event.clientY - box.top) / box.height * 116)));
    const col = Math.min(115, Math.max(0, Math.floor((event.clientX - box.left) / box.width * 116)));
    return { row, col, value: matrix[row][col] };
  }

  function inspectMatrix(event) {
    const pair = matrixPair(event);
    if (pair) $('health-matrix-readout').textContent = `${roiLabel(pair.row)} ↔ ${roiLabel(pair.col)} · ${Number(pair.value).toFixed(5)}`;
  }

  function strongest(matrix) {
    const links = [];
    for (let row = 0; row < matrix.length; row++) {
      for (let col = row + 1; col < matrix.length; col++) {
        if (number(matrix[row]?.[col]) !== null) links.push({ source: row, target: col, value: Number(matrix[row][col]) });
      }
    }
    return links.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }

  function drawNetwork() {
    const matrix = currentMatrix();
    if (!Array.isArray(matrix) || matrix.length !== 116) {
      $('health-network').innerHTML = '<div class="health-chart-empty static">No connectivity data.</div>';
      return;
    }
    const links = strongest(matrix).slice(0, state.edges), range = matrixRange(matrix);
    const point = (i, radius = 168) => ({ x: 225 + radius * Math.cos(i / 116 * Math.PI * 2 - Math.PI / 2), y: 225 + radius * Math.sin(i / 116 * Math.PI * 2 - Math.PI / 2) });
    let svg = '<svg viewBox="0 0 450 450" role="img" aria-label="Schematic circular network of 116 actual ROIs and strongest connections; not anatomical coordinates"><circle cx="225" cy="225" r="168" fill="#fcfdf9" stroke="#e3ebda"/>';
    links.forEach((link) => {
      const a = point(link.source), b = point(link.target);
      const active = state.selectedROI === null || link.source === state.selectedROI || link.target === state.selectedROI;
      svg += `<path d="M${a.x} ${a.y} Q225 225 ${b.x} ${b.y}" fill="none" stroke="${link.value >= 0 ? '#5a9d88' : '#9e83b5'}" stroke-opacity="${active ? 0.45 : 0.045}" stroke-width="${0.5 + Math.abs(link.value) / range * 1.5}"><title>${roiLabel(link.source)} ↔ ${roiLabel(link.target)}: ${link.value.toFixed(5)}</title></path>`;
    });
    for (let i = 0; i < 116; i++) {
      const p = point(i), label = point(i, 188), selected = i === state.selectedROI;
      svg += `<g data-health-roi="${i}" role="button" tabindex="0" aria-label="Highlight ${roiLabel(i)}"><circle cx="${p.x}" cy="${p.y}" r="${selected ? 4 : 2.4}" fill="${selected ? '#a88bbb' : '#8daf79'}"/><text x="${label.x}" y="${label.y}" text-anchor="middle" dominant-baseline="central" font-size="${selected ? 9 : 6.2}" fill="${selected ? '#9472ae' : '#a3b48f'}">${String(i + 1).padStart(3, '0')}</text><title>${roiLabel(i)} · File column ${i + 1}</title></g>`;
    }
    $('health-network').innerHTML = svg + '</svg>';
    const selectedLinks = state.selectedROI === null ? [] : strongest(matrix).filter((link) => link.source === state.selectedROI || link.target === state.selectedROI).slice(0, 3);
    $('health-network-readout').textContent = state.selectedROI === null ? `116 ROIs · Showing ${links.length} strongest recorded connections · Select a node to highlight` :
      `${roiLabel(state.selectedROI)}: ` + selectedLinks.map((link) => `${roiLabel(link.source === state.selectedROI ? link.target : link.source)} ${link.value.toFixed(3)}`).join(' / ');
  }

  function drawDistributions() {
    const data = state.cohort;
    if (!data) {
      $('health-sites').innerHTML = '<div class="health-empty-small">Awaiting cohort counts.</div>';
      $('health-ages').innerHTML = '<div class="health-empty-small">Awaiting age distribution.</div>';
      return;
    }
    const sites = data.site_counts || [], maximum = Math.max(1, ...sites.map((site) => site.total));
    $('health-sites').innerHTML = sites.map((site) => `<button class="health-site-row" data-health-site="${escape(site.id)}"><strong>${escape(site.id)}</strong><span class="health-site-track"><i class="hc" style="width:${site.hc / maximum * 100}%"></i><i class="asd" style="width:${site.asd / maximum * 100}%"></i></span><small>HC ${format(site.hc)} / ASD ${format(site.asd)}</small><b>${format(site.total)}</b></button>`).join('');
    const hist = data.age_histogram || {}, labels = hist.labels || [], hc = hist.hc || [], asd = hist.asd || [];
    const max = Math.max(1, ...labels.map((_, i) => Number(hc[i] || 0) + Number(asd[i] || 0)));
    const width = 600, height = 183, plot = 126, bw = 520 / Math.max(1, labels.length);
    let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Recorded age histogram grouped by HC and ASD research labels"><path d="M42 15V148H575" fill="none" stroke="#e1e9d6"/><text x="32" y="22" text-anchor="end" font-size="8" fill="#a4b392">${max}</text><text x="32" y="149" text-anchor="end" font-size="8" fill="#a4b392">0</text>`;
    labels.forEach((label, i) => {
      const x = 46 + i * bw, hh = Number(hc[i] || 0) / max * plot, ah = Number(asd[i] || 0) / max * plot;
      svg += `<rect x="${x}" y="${147 - hh}" width="${Math.max(2, bw - 7)}" height="${hh}" fill="#86ae91"><title>${escape(label)} years HC: ${hc[i] || 0}</title></rect><rect x="${x}" y="${147 - hh - ah}" width="${Math.max(2, bw - 7)}" height="${ah}" fill="#b5a1c7"><title>${escape(label)} years ASD: ${asd[i] || 0}</title></rect><text x="${x + (bw - 7) / 2}" y="164" text-anchor="middle" font-size="7" fill="#9eae8d">${escape(label)}</text>`;
    });
    $('health-ages').innerHTML = svg + '</svg>';
    const ages = data.age_summary || {};
    $('health-age-summary').textContent = `Median ${format(ages.median, 1)} years`;
    $('health-age-note').textContent = `Observed range ${format(ages.min, 1)}–${format(ages.max, 1)} years · Missing age: HC ${format(hist.missing_hc)} / ASD ${format(hist.missing_asd)}; Subjects with missing age are excluded when age filters are active.`;
  }

  function renderSubjects() {
    let subjects = eligibleSubjects();
    const query = state.query.toLowerCase().trim();
    if (query) subjects = subjects.filter((subject) => `${subject.id} ${subject.site}`.toLowerCase().includes(query));
    $('health-case-count').textContent = state.cohort ? `${state.group === 'difference' ? 'HC / ASD' : state.group} · ${subjects.length} available subjects` : 'Awaiting subjects';
    $('health-subjects').innerHTML = subjects.length ? subjects.map((subject) => `
      <button class="health-subject ${subject.id === state.subjectId ? 'selected' : ''}" data-health-subject="${escape(subject.id)}" aria-pressed="${subject.id === state.subjectId}"><span><strong>${escape(subject.id)}</strong><small>${escape(subject.site)} · ${format(subject.age, 1)} years</small></span><i class="${subject.label_name === 'ASD' ? 'asd' : 'hc'}">${escape(subject.label_name)}</i></button>`).join('') :
      `<div class="health-empty-small">${state.cohortLoading ? 'Loading actual subjects…' : 'No subjects match the current filters.'}</div>`;
  }

  function drawSignals() {
    const subject = state.subject;
    if (!subject) {
      $('health-subject-meta').textContent = state.subjectLoading ? `Loading recorded time series for ${state.subjectId}…` : 'Select a subject to inspect recorded signals.';
      $('health-signal-chart').innerHTML = '<div class="health-empty-small">Select a subject to view BOLD time series.</div>';
      $('health-subject-notice').textContent = state.subjectError || '';
      return;
    }
    $('health-subject-meta').innerHTML = `<span class="health-provenance">Research data browser · Not a training upload · ${escape(subject.site)}</span><h3>${escape(subject.id)} <span class="badge ${subject.label_name === 'HC' ? 'teal-outline' : 'violet-outline'}">${escape(subject.label_name)} Research Label</span></h3><p>${escape(subject.site)} · ${format(subject.age, 1)} years · ${format(subject.n_frames)} frames · Mean Absolute Connectivity ${format(subject.mean_abs_connectivity, 3)}</p>`;
    const signals = subject.signals || {}, series = signals.series || [], frames = signals.frames || [];
    if (!series.length || !frames.length) {
      $('health-signal-chart').innerHTML = '<div class="health-empty-small">No readable recorded time series for this subject.</div>';
      return;
    }
    const extent = Math.max(1, ...series.flatMap((item) => item.values || []).filter((v) => number(v) !== null).map(Math.abs));
    const xMax = Math.max(1, Number(frames[frames.length - 1])), x = (v) => 37 + v / xMax * 684, y = (v) => 88 - v / extent * 58;
    let svg = '<svg viewBox="0 0 750 190" role="img" aria-label="Standardized BOLD signals from three actual ROIs; the x-axis is frame index, not seconds">';
    [-1, 0, 1].forEach((v) => {
      const yy = y(v * extent);
      svg += `<path d="M37 ${yy}H722" fill="none" stroke="#eaf0e1"/><text x="28" y="${yy + 3}" text-anchor="end" font-size="8" fill="#a6b495">${(v * extent).toFixed(1)}</text>`;
    });
    series.forEach((item, index) => {
      const points = (item.values || []).map((value, i) => `${x(Number(frames[i] ?? i)).toFixed(2)},${y(Number(value)).toFixed(2)}`).join(' ');
      svg += `<polyline points="${points}" fill="none" stroke="${palette[index % 3]}" stroke-width="1.05" stroke-opacity=".83"><title>${escape(item.label)} · ${item.values?.length || 0} recorded time points</title></polyline>`;
    });
    svg += `<text x="37" y="166" font-size="8" fill="#9fb08e">0</text><text x="722" y="166" text-anchor="end" font-size="8" fill="#9fb08e">${xMax}</text><text x="380" y="181" text-anchor="middle" font-size="9" fill="#94a980">Frame index (not seconds; TR unavailable) · Within-ROI BOLD z-score</text></svg>`;
    $('health-signal-chart').innerHTML = `<div class="health-signal-legend">${series.map((item, i) => `<span style="color:${palette[i % 3]}">● ${escape(item.label)}</span>`).join('')}</div>${svg}`;
    $('health-subject-notice').textContent = `Full-cohort HC reference: ${format(subject.reference?.hc_count)} participants${subject.reference?.leave_one_out ? ' (current subject excluded)' : ''}.`;
  }

  function context() {
    return { site: state.site, ageMin: state.ageMin, ageMax: state.ageMax,
      siteCounts: state.cohort?.site_counts || [], selectedSubjects: state.cohort?.selected_subjects,
      totalSubjects: state.cohort?.total_subjects, group: state.group };
  }

  window.HealthData = { reload: loadCohort, context };
  init();
})();
