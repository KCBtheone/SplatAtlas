const state = {
  benchmark: null,
  vol: null,
  dtu: null,
  view: 'dataset',
  activeDataset: 'tandt',
  activeMetric: 'psnr',
  sortDirection: 'desc',
  query: '',
  selectedMethodId: null,
  sceneDataset: 'tandt',
  volDataset: 'tnt',
  volMetrics: { tnt: 'f_score', dtu: 'cd' },
};

const elements = {
  datasetTabs: document.querySelector('#dataset-tabs'),
  metricTabs: document.querySelector('#metric-tabs'),
  methodRows: document.querySelector('#method-rows'),
  methodCount: document.querySelector('#method-count'),
  search: document.querySelector('#method-search'),
  empty: document.querySelector('#empty-state'),
  datasetView: document.querySelector('#dataset-view'),
  methodView: document.querySelector('#method-view'),
  methodSelect: document.querySelector('#method-select'),
  selectedMethodSummary: document.querySelector('#selected-method-summary'),
  sceneDatasetTabs: document.querySelector('#scene-dataset-tabs'),
  sceneRows: document.querySelector('#scene-rows'),
  sceneCount: document.querySelector('#scene-count'),
  exportAll: document.querySelector('#export-all'),
  volDescription: document.querySelector('#vol-description'),
  volPrimaryMetric: document.querySelector('#vol-primary-metric'),
  volPrimaryUnit: document.querySelector('#vol-primary-unit'),
  volDatasetTabs: document.querySelector('#vol-dataset-tabs'),
  volMetricTabs: document.querySelector('#vol-metric-tabs'),
  volNotice: document.querySelector('#vol-notice'),
  volHead: document.querySelector('#vol-head'),
  volRows: document.querySelector('#vol-rows'),
  volNotes: document.querySelector('#vol-notes'),
  volCount: document.querySelector('#vol-count'),
  volSnapshot: document.querySelector('#vol-snapshot'),
  toast: document.querySelector('#toast'),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function formatScore(metric, value) {
  if (!Number.isFinite(value)) return '—';
  return metric === 'psnr' ? value.toFixed(2) : value.toFixed(4);
}

function datasetById(id) {
  return state.benchmark.datasets.find((dataset) => dataset.id === id);
}

function methodById(id) {
  return state.benchmark.methods.find((method) => method.id === id);
}

function saveJson(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  showToast(`Downloaded ${filename}`);
}

let toastTimer;
function showToast(message) {
  window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add('is-visible');
  toastTimer = window.setTimeout(() => elements.toast.classList.remove('is-visible'), 2200);
}

function setView(view) {
  state.view = view;
  const isDataset = view === 'dataset';
  elements.datasetView.hidden = !isDataset;
  elements.methodView.hidden = isDataset;
  document.querySelectorAll('[data-view]').forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-selected', String(active));
  });
  if (!isDataset && !state.selectedMethodId) state.selectedMethodId = state.benchmark.methods[0]?.id ?? null;
  renderMethodView();
}

function setDataset(datasetId) {
  if (!datasetById(datasetId)) return;
  state.activeDataset = datasetId;
  state.sceneDataset = datasetId;
  renderDatasetView();
  renderMethodView();
}

function renderDatasetTabs() {
  elements.datasetTabs.innerHTML = state.benchmark.datasets.map((dataset) => `
    <button class="dataset-tab${dataset.id === state.activeDataset ? ' is-active' : ''}" type="button" role="tab" data-select-dataset="${escapeHtml(dataset.id)}" aria-selected="${dataset.id === state.activeDataset}" aria-label="Rank by ${escapeHtml(dataset.name)}">
      ${escapeHtml(dataset.short)}
    </button>
  `).join('');

  elements.sceneDatasetTabs.innerHTML = state.benchmark.datasets.map((dataset) => `
    <button class="dataset-tab${dataset.id === state.sceneDataset ? ' is-active' : ''}" type="button" role="tab" data-scene-dataset="${escapeHtml(dataset.id)}" aria-selected="${dataset.id === state.sceneDataset}">
      ${escapeHtml(dataset.name)} <span>(${dataset.scenes.length})</span>
    </button>
  `).join('');

  document.querySelectorAll('[data-dataset-header]').forEach((header) => {
    const dataset = datasetById(header.dataset.datasetHeader);
    if (!dataset) return;
    header.classList.toggle('is-active', dataset.id === state.activeDataset);
    const button = header.querySelector('[data-select-dataset]');
    button.innerHTML = `${escapeHtml(dataset.name)} <small>${dataset.scenes.length} scenes</small>`;
  });
}

function renderMetricTabs() {
  elements.metricTabs.querySelectorAll('[data-metric]').forEach((button) => {
    const metric = button.dataset.metric;
    const active = metric === state.activeMetric;
    button.classList.toggle('is-active', active);
    const arrow = active ? (state.sortDirection === 'desc' ? ' ↓' : ' ↑') : '';
    button.textContent = `${metric.toUpperCase()}${arrow}`;
    button.setAttribute('aria-pressed', String(active));
    button.setAttribute('aria-label', `Sort by ${metric.toUpperCase()} ${active ? (state.sortDirection === 'desc' ? 'descending' : 'ascending') : ''}`.trim());
  });
  document.querySelectorAll('[data-sort-metric]').forEach((header, index) => {
    const dataset = state.benchmark.datasets[Math.floor(index / 3)];
    const active = dataset?.id === state.activeDataset && header.dataset.sortMetric === state.activeMetric;
    header.classList.toggle('is-sorted', active);
    header.setAttribute('aria-sort', active ? (state.sortDirection === 'desc' ? 'descending' : 'ascending') : 'none');
  });
}

function renderMethodOptions() {
  const currentId = state.selectedMethodId;
  elements.methodSelect.innerHTML = state.benchmark.methods.map((method) => `
    <option value="${escapeHtml(method.id)}">${escapeHtml(method.name)}</option>
  `).join('');
  if (currentId && methodById(currentId)) elements.methodSelect.value = currentId;
  else state.selectedMethodId = elements.methodSelect.value || null;
}

function visibleMethods() {
  const query = state.query.trim().toLocaleLowerCase();
  return state.benchmark.methods
    .filter((method) => method.name.toLocaleLowerCase().includes(query))
    .sort((a, b) => {
      const aScore = a.scores[state.activeDataset]?.[state.activeMetric];
      const bScore = b.scores[state.activeDataset]?.[state.activeMetric];
      if (!Number.isFinite(aScore)) return Number.isFinite(bScore) ? 1 : a.name.localeCompare(b.name);
      if (!Number.isFinite(bScore)) return -1;
      const difference = state.sortDirection === 'desc' ? bScore - aScore : aScore - bScore;
      return difference || a.name.localeCompare(b.name);
    });
}

function renderScoreCells(method, dataset) {
  const score = method.scores[dataset.id];
  const coverage = score ? `${score.availableScenes}/${score.totalScenes}` : `0/${dataset.scenes.length}`;
  return ['psnr', 'ssim', 'lpips'].map((metric) => {
    const value = score?.[metric];
    const missing = !Number.isFinite(value);
    return `<td class="metric-cell${dataset.id === state.activeDataset ? ' is-active' : ''}" data-metric="${metric}" data-dataset="${escapeHtml(dataset.id)}">
      <button class="score-cell-button" type="button" data-method-id="${escapeHtml(method.id)}" data-method-dataset="${escapeHtml(dataset.id)}" title="${coverage} scenes with recorded values" aria-label="Inspect ${escapeHtml(method.name)} ${metric.toUpperCase()} values for ${escapeHtml(dataset.name)}">
        <span class="${missing ? 'score-missing' : 'score-value'}">${missing ? '—' : formatScore(metric, value)}</span>
      </button>
    </td>`;
  }).join('');
}

function renderDatasetView() {
  renderDatasetTabs();
  renderMetricTabs();
  const methods = visibleMethods();
  elements.methodRows.innerHTML = methods.map((method, index) => `
    <tr>
      <td class="method-column"><button class="method-name-button" type="button" data-open-method="${escapeHtml(method.id)}" aria-label="Open per-scene results for ${escapeHtml(method.name)}"><span class="rank-number">${String(index + 1).padStart(2, '0')}</span><span class="method-name">${escapeHtml(method.name)}</span></button></td>
      ${state.benchmark.datasets.map((dataset) => renderScoreCells(method, dataset)).join('')}
    </tr>
  `).join('');
  elements.empty.hidden = methods.length > 0;
  elements.methodCount.textContent = `${methods.length} of ${state.benchmark.methods.length} methods · ranked by ${datasetById(state.activeDataset).name} ${state.activeMetric.toUpperCase()}`;
  document.querySelector('#dataset-hint').textContent = `Sort by ${datasetById(state.activeDataset).name}; click a method or score cell to inspect its scene values.`;
}

function renderMethodView() {
  if (!state.benchmark) return;
  renderMethodOptions();
  renderDatasetTabs();
  const method = methodById(state.selectedMethodId);
  if (!method) {
    elements.selectedMethodSummary.textContent = 'No method selected.';
    elements.sceneRows.innerHTML = '';
    elements.sceneCount.textContent = '';
    return;
  }
  const dataset = datasetById(state.sceneDataset);
  const score = method.scores[dataset.id];
  const complete = score?.availableScenes ?? 0;
  elements.selectedMethodSummary.innerHTML = `
    <strong>${escapeHtml(method.name)}</strong>
    <span>${complete}/${dataset.scenes.length} ${escapeHtml(dataset.short)} scenes</span>
    <span>PSNR ${formatScore('psnr', score?.psnr)}</span>
    <span>SSIM ${formatScore('ssim', score?.ssim)}</span>
    <span>LPIPS ${formatScore('lpips', score?.lpips)}</span>
    ${method.note ? `<p class="method-note">${escapeHtml(method.note)}</p>` : ''}
  `;
  const scenes = method.scenes[dataset.id] ?? [];
  elements.sceneRows.innerHTML = scenes.map((scene) => {
    const available = scene.status === 'complete';
    return `
      <tr>
        <td>${escapeHtml(scene.name)}</td>
        <td class="numeric">${formatScore('psnr', scene.metrics.psnr)}</td>
        <td class="numeric">${formatScore('ssim', scene.metrics.ssim)}</td>
        <td class="numeric">${formatScore('lpips', scene.metrics.lpips)}</td>
        <td><span class="cell-state${available ? '' : ' is-missing'}">${available ? 'Available' : '—'}</span></td>
      </tr>
    `;
  }).join('');
  elements.sceneCount.textContent = `${scenes.length} listed scenes · ${complete} complete metric records`;
}

function renderVol() {
  const isDtu = state.volDataset === 'dtu';
  const data = isDtu ? state.dtu : state.vol;
  const metric = data.metrics.find((item) => item.id === state.volMetrics[state.volDataset]) ?? data.metrics[0];
  const scenes = data.scenes;
  const summary = data.summary;
  elements.volDatasetTabs.innerHTML = [
    { id: 'tnt', label: 'Tanks & Temples', meta: `${state.vol.scenes.length} scenes` },
    { id: 'dtu', label: 'DTU', meta: `${state.dtu.scenes.length} scans` },
  ].map((tab) => `
    <button class="dataset-tab${tab.id === state.volDataset ? ' is-active' : ''}" type="button" role="tab" data-vol-dataset="${tab.id}" aria-selected="${tab.id === state.volDataset}">
      ${tab.label} <span>(${tab.meta})</span>
    </button>
  `).join('');
  elements.volMetricTabs.hidden = false;
  elements.volMetricTabs.innerHTML = data.metrics.map((item) => `
    <button class="metric-tab${item.id === metric.id ? ' is-active' : ''}" type="button" data-vol-metric="${item.id}" aria-pressed="${item.id === metric.id}">${item.label}</button>
  `).join('');

  const valueFor = (method, scene) => method.scores[scene]?.[metric.id];
  const validCells = data.methods.reduce((sum, method) => sum + scenes.filter((scene) => Number.isFinite(valueFor(method, scene))).length, 0);
  const totalCells = summary.methods * summary.scenes;
  elements.volPrimaryMetric.textContent = metric.label;
  const direction = metric.direction ?? (metric.unit === '%' ? 'Higher is better' : 'Lower is better');
  elements.volPrimaryUnit.textContent = `${direction} · ${metric.unit}`;
  elements.volDescription.textContent = `${data.dataset} geometry benchmark · ${summary.methods} methods × ${summary.scenes} ${isDtu ? 'scans' : 'scenes'} · ${metric.label} (${metric.unit}) · ${validCells}/${totalCells} selected values available.`;
  elements.volNotice.textContent = data.notice;
  elements.volSnapshot.textContent = data.snapshot;
  elements.volCount.textContent = `${summary.methods} methods · ${validCells}/${totalCells} selected values · ${data.status === 'complete' ? 'project-evaluated geometry results' : 'missing and unverified cells stay blank'}`;
  document.querySelector('.vol-table').classList.toggle('is-dtu', isDtu);
  elements.volHead.innerHTML = `<tr><th>Method</th>${scenes.map((scene) => `<th class="${!isDtu && scene === 'Church' ? 'extension-column' : ''}">${isDtu ? `Scan ${escapeHtml(scene)}` : escapeHtml(scene)}${!isDtu && scene === 'Church' ? '<small class="extension-label">Extension</small>' : ''}</th>`).join('')}<th>Coverage</th></tr>`;
  elements.volRows.innerHTML = data.methods.map((method) => {
    const values = scenes.map((scene) => valueFor(method, scene));
    const coverage = values.filter(Number.isFinite).length;
    return `<tr><td>${escapeHtml(method.name)}</td>${scenes.map((scene) => {
      const value = valueFor(method, scene);
      const sceneLabel = isDtu ? `Scan ${scene}` : scene;
      const formatted = metric.unit === '%' ? (value * 100).toFixed(2) : value?.toFixed(3);
      const provenance = method.scores[scene]?.provenance;
      const tooltip = [
        `${method.name} · ${sceneLabel} · ${metric.label}`,
        provenance?.protocol ? `Protocol: ${provenance.protocol}` : '',
        Number.isFinite(provenance?.thresholdMeters) ? `Distance threshold: ${provenance.thresholdMeters} m` : '',
        provenance?.inputRevision ? `Input revision: ${provenance.inputRevision}` : '',
      ].filter(Boolean).join('\n');
      return `<td class="${!isDtu && scene === 'Church' ? 'extension-column' : ''}">${Number.isFinite(value) ? `<span class="vol-value" title="${escapeHtml(tooltip)}">${formatted}</span>` : '<span class="vol-empty">—</span>'}</td>`;
    }).join('')}<td><span class="vol-coverage">${coverage}/${scenes.length}</span></td></tr>`;
  }).join('');
  elements.volNotes.innerHTML = data.notes.map((note) => `<div class="vol-note">${escapeHtml(note)}</div>`).join('');
}

function openMethod(methodId, datasetId = state.activeDataset) {
  if (!methodById(methodId)) return;
  state.selectedMethodId = methodId;
  state.sceneDataset = datasetById(datasetId) ? datasetId : state.activeDataset;
  setView('method');
  document.querySelector('#view-method-button').focus({ preventScroll: true });
}

function renderAll() {
  renderDatasetView();
  renderMethodView();
  renderVol();
}

document.querySelector('.view-switch').addEventListener('click', (event) => {
  const button = event.target.closest('[data-view]');
  if (button) setView(button.dataset.view);
});

elements.datasetTabs.addEventListener('click', (event) => {
  const button = event.target.closest('[data-select-dataset]');
  if (button) setDataset(button.dataset.selectDataset);
});

document.querySelector('.matrix-table thead').addEventListener('click', (event) => {
  const button = event.target.closest('[data-select-dataset]');
  if (button) setDataset(button.dataset.selectDataset);
});

elements.metricTabs.addEventListener('click', (event) => {
  const button = event.target.closest('[data-metric]');
  if (!button) return;
  const metric = button.dataset.metric;
  if (state.activeMetric === metric) state.sortDirection = state.sortDirection === 'desc' ? 'asc' : 'desc';
  else {
    state.activeMetric = metric;
    state.sortDirection = metric === 'lpips' ? 'asc' : 'desc';
  }
  renderDatasetView();
});

elements.search.addEventListener('input', (event) => {
  state.query = event.target.value;
  renderDatasetView();
});

elements.methodRows.addEventListener('click', (event) => {
  const scoreCell = event.target.closest('[data-method-dataset]');
  const methodButton = event.target.closest('[data-open-method]');
  if (scoreCell) openMethod(scoreCell.dataset.methodId, scoreCell.dataset.methodDataset);
  else if (methodButton) openMethod(methodButton.dataset.openMethod);
});

elements.methodSelect.addEventListener('change', (event) => {
  state.selectedMethodId = event.target.value;
  renderMethodView();
});

elements.sceneDatasetTabs.addEventListener('click', (event) => {
  const button = event.target.closest('[data-scene-dataset]');
  if (!button) return;
  state.sceneDataset = button.dataset.sceneDataset;
  renderMethodView();
});

elements.volDatasetTabs.addEventListener('click', (event) => {
  const button = event.target.closest('[data-vol-dataset]');
  if (!button) return;
  state.volDataset = button.dataset.volDataset;
  renderVol();
});

elements.volMetricTabs.addEventListener('click', (event) => {
  const button = event.target.closest('[data-vol-metric]');
  if (!button) return;
  state.volMetrics[state.volDataset] = button.dataset.volMetric;
  renderVol();
});

elements.exportAll.addEventListener('click', () => {
  saveJson('splatatlas-benchmarks.json', { splatatlas: state.benchmark, splatatlasvol: state.vol, splatatlasvol_dtu: state.dtu });
});

async function initialize() {
  try {
    const [benchmarkResponse, volResponse, dtuResponse] = await Promise.all([
      fetch('./data/benchmark.json'),
      fetch('./data/vol-benchmark.json?v=20261018'),
      fetch('./data/vol-dtu-benchmark.json?v=20261018'),
    ]);
    if (!benchmarkResponse.ok) throw new Error(`SplatAtlas data unavailable (HTTP ${benchmarkResponse.status}).`);
    if (!volResponse.ok) throw new Error(`SplatAtlasVol data unavailable (HTTP ${volResponse.status}).`);
    if (!dtuResponse.ok) throw new Error(`SplatAtlasVol DTU data unavailable (HTTP ${dtuResponse.status}).`);
    [state.benchmark, state.vol, state.dtu] = await Promise.all([benchmarkResponse.json(), volResponse.json(), dtuResponse.json()]);
    state.selectedMethodId = state.benchmark.methods[0]?.id ?? null;
    renderAll();
  } catch (error) {
    elements.methodRows.innerHTML = '';
    elements.empty.hidden = false;
    elements.empty.innerHTML = `<strong>Benchmark data could not be loaded.</strong><span>${escapeHtml(error.message ?? error)}</span>`;
    elements.methodCount.textContent = 'Data unavailable';
  }
}

initialize();
