import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const inputs = parseArgs(process.argv.slice(2));
const nvsRows = parseCsv(await readFile(inputs.nvs, 'utf8'));
const sceneRows = parseCsv(await readFile(inputs.scenes, 'utf8'));
const volRows = parseCsv(await readFile(inputs.vol, 'utf8'));

const datasetConfig = [
  { id: 'tandt', sourceName: 'Tanks&Temples', name: 'Tanks & Temples', short: 'T&T' },
  { id: 'mip360', sourceName: 'Mip-NeRF 360', name: 'Mip-NeRF 360', short: 'Mip-360' },
  { id: 'synthetic', sourceName: 'NeRF-Synthetic', name: 'NeRF Synthetic', short: 'Synthetic' },
  { id: 'deepblending', sourceName: 'Deep Blending', name: 'Deep Blending', short: 'Deep Blending' },
];
const datasetBySource = new Map(datasetConfig.map((item) => [item.sourceName, item]));
const datasetById = new Map(datasetConfig.map((item) => [item.id, item]));
const sceneDisplay = new Map(sceneRows.map((row) => [row.SceneKey, row.DisplayName]));
const expectedScenes = new Map(datasetConfig.map((dataset) => [dataset.id, []]));

for (const row of sceneRows) {
  const dataset = datasetBySource.get(row.Dataset);
  if (!dataset) throw new Error(`Unknown dataset in scene inventory: ${row.Dataset}`);
  expectedScenes.get(dataset.id).push(row.SceneKey);
}

const methods = new Map();
const seenCells = new Set();
for (const row of nvsRows) {
  const dataset = datasetBySource.get(row.dataset);
  if (!dataset) throw new Error(`Unknown dataset in NVS table: ${row.dataset}`);
  const key = `${row.method}|${dataset.id}|${row.scene}`;
  if (seenCells.has(key)) throw new Error(`Duplicate NVS cell: ${key}`);
  seenCells.add(key);

  let method = methods.get(row.method);
  if (!method) {
    method = {
      id: row.method,
      name: row.method_display,
      category: row.category,
      tier: row.tier,
      note: row.method === '3dgsmcmc'
        ? 'The source benchmark marks this method with a paper-level failure note; interpret its reported values with care.'
        : '',
      scores: {},
      scenes: Object.fromEntries(datasetConfig.map((item) => [item.id, []])),
    };
    methods.set(row.method, method);
  }

  const metrics = {
    psnr: numberOrNull(row.PSNR_dB),
    ssim: numberOrNull(row.SSIM),
    lpips: numberOrNull(row.LPIPS),
  };
  const valid = row.metric_status === 'complete' && Object.values(metrics).every(Number.isFinite);
  method.scenes[dataset.id].push({
    id: row.scene,
    name: sceneDisplay.get(row.scene) ?? titleCase(row.scene),
    metrics: valid ? metrics : { psnr: null, ssim: null, lpips: null },
    status: valid ? 'complete' : (row.metric_status || 'missing'),
  });
}

if (nvsRows.length !== 31 * 31 || methods.size !== 31 || sceneRows.length !== 31) {
  throw new Error(`Unexpected SplatAtlas coverage: ${nvsRows.length} cells, ${methods.size} methods, ${sceneRows.length} scenes.`);
}

for (const method of methods.values()) {
  for (const dataset of datasetConfig) {
    const scenes = method.scenes[dataset.id];
    const totals = { psnr: 0, ssim: 0, lpips: 0 };
    let count = 0;
    for (const scene of scenes) {
      if (scene.status !== 'complete') continue;
      count += 1;
      for (const metric of Object.keys(totals)) totals[metric] += scene.metrics[metric];
    }
    method.scores[dataset.id] = {
      psnr: count ? totals.psnr / count : null,
      ssim: count ? totals.ssim / count : null,
      lpips: count ? totals.lpips / count : null,
      availableScenes: count,
      totalScenes: expectedScenes.get(dataset.id).length,
    };
  }
}

const benchmark = {
  title: 'SplatAtlas',
  updated: '2026-10-13',
  source: 'SplatAtlas core per-scene NVS table (30 displayed methods × 31 scenes; lod_gs omitted by request; incomplete cells retained).',
  aggregation: 'Arithmetic mean over available complete scenes within each dataset. Missing cells are not imputed.',
  datasets: datasetConfig.map((dataset) => ({
    id: dataset.id,
    name: dataset.name,
    short: dataset.short,
    scenes: expectedScenes.get(dataset.id).map((id) => ({ id, name: sceneDisplay.get(id) ?? titleCase(id) })),
  })),
  methods: [...methods.values()].filter((method) => method.id !== 'lod_gs').sort((a, b) => a.name.localeCompare(b.name)),
};

const volMethods = new Map();
const volScenes = ['Barn', 'Caterpillar', 'Church', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck'];
const volMetrics = [
  { id: 'f_score', label: 'F-score', unit: '%', direction: 'Higher is better' },
  { id: 'precision', label: 'Precision (P)', unit: '%', direction: 'Higher is better' },
  { id: 'recall', label: 'Recall (R)', unit: '%', direction: 'Higher is better' },
];
for (const row of volRows) {
  let method = volMethods.get(row.method);
  if (!method) {
    method = { id: row.method, name: row.method_display, scores: {} };
    volMethods.set(row.method, method);
  }
  const selected = ['DONE_VALID', 'DONE_USER_PRIMARY'].includes(row.geometry_status);
  method.scores[row.scene] = {
    f_score: selected ? numberOrNull(row.F_score) : null,
    precision: selected ? numberOrNull(row.Precision) : null,
    recall: selected ? numberOrNull(row.Recall) : null,
  };
}

const vol = {
  title: 'SplatAtlasVol',
  dataset: 'Tanks & Temples',
  metric: 'F-score',
  unit: '%',
  metrics: volMetrics,
  snapshot: '2026-10-11',
  status: 'in_progress',
  statusLabel: 'Work in progress',
  notice: 'Provisional snapshot only. Some values are under review and are being rerun; do not treat this table as final or cite it as a finalized benchmark.',
  notes: [
    'Use the metric selector to switch between F-score, precision (P), and recall (R); P and R come from the same selected result record. Unavailable or unverified cells remain blank.',
    'GeoSVR and AmbiSuR have rerun entries in the project queue. The Church column is an extension scene, not one of the six core Tanks & Temples scenes.',
    'The SplatAtlasVol NVS LPIPS values are not shown here because the historical table used an incorrect input range and standard LPIPS is being recomputed.',
  ],
  scenes: volScenes,
  methods: [...volMethods.values()].sort((a, b) => a.name.localeCompare(b.name)),
};

const volValidCells = vol.methods.reduce((sum, method) => sum + Object.values(method.scores).filter((score) => Number.isFinite(score.f_score)).length, 0);
const volValidCellsByMetric = Object.fromEntries(volMetrics.map(({ id }) => [
  id,
  vol.methods.reduce((sum, method) => sum + Object.values(method.scores).filter((score) => Number.isFinite(score[id])).length, 0),
]));
vol.summary = {
  methods: vol.methods.length,
  scenes: volScenes.length,
  validCells: volValidCellsByMetric.f_score,
  totalCells: vol.methods.length * volScenes.length,
  validCellsByMetric: volValidCellsByMetric,
};

await writeFile(path.join(root, 'data', 'benchmark.json'), `${JSON.stringify(benchmark, null, 2)}\n`);
await writeFile(path.join(root, 'data', 'vol-benchmark.json'), `${JSON.stringify(vol, null, 2)}\n`);
console.log(`Prepared ${benchmark.methods.length} methods, ${benchmark.datasets.length} datasets, ${seenCells.size} scene cells and ${volValidCells}/${vol.summary.totalCells} provisional SplatAtlasVol values.`);

function parseArgs(args) {
  const values = {};
  for (let index = 0; index < args.length; index += 2) {
    const key = args[index];
    const value = args[index + 1];
    if (!['--nvs', '--scenes', '--vol'].includes(key) || !value) throw new Error('Usage: node scripts/build-site-data.mjs --nvs <csv> --scenes <csv> --vol <csv>');
    values[key.slice(2)] = path.resolve(value);
  }
  for (const key of ['nvs', 'scenes', 'vol']) if (!values[key]) throw new Error('Usage: node scripts/build-site-data.mjs --nvs <csv> --scenes <csv> --vol <csv>');
  return values;
}

function parseCsv(source) {
  const text = source.replace(/^\uFEFF/, '');
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') { field += '"'; index += 1; }
      else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"' && field === '') quoted = true;
    else if (char === ',') { row.push(field); field = ''; }
    else if (char === '\n') {
      row.push(field.replace(/\r$/, '')); field = '';
      if (row.some((value) => value !== '')) rows.push(row);
      row = [];
    } else field += char;
  }
  if (field !== '' || row.length) { row.push(field.replace(/\r$/, '')); if (row.some((value) => value !== '')) rows.push(row); }
  const headers = rows.shift();
  return rows.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ''])));
}

function numberOrNull(value) {
  if (value === '' || value == null) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function titleCase(value) {
  return value.replace(/([a-z])([A-Z])/g, '$1 $2').replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}
