import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const inputs = parseArgs(process.argv.slice(2));
const rows = parseCsv(await readFile(inputs.nvs, 'utf8'));
const register = parseCsv(await readFile(inputs.methods, 'utf8'));
const benchmark = JSON.parse(await readFile(path.join(root, 'data/benchmark.json'), 'utf8'));
const previous = new Map(benchmark.methods.map((method) => [method.id, method]));
const datasetIds = {
  nerf_synthetic: 'synthetic',
  mipnerf360: 'mip360',
  tanks_and_temples: 'tandt',
  deep_blending: 'deepblending',
};
const expected = new Map(benchmark.datasets.map((dataset) => [
  dataset.id, new Map(dataset.scenes.map((scene) => [scene.id.toLowerCase(), scene])),
]));
const notes = {
  '3dgsmcmc': 'Four T&T scenes use the completed SfM initialization repairs: auditorium, lighthouse, palace and temple.',
  lightgaussian: '10k checkpoints are used for stump, playground and truck; the other checkpoints are 30k.',
  mipsplatting: 'A 20k checkpoint is used for ballroom; the other checkpoints are 30k.',
  radegs: 'Bicycle, stump and treehill use the completed cap-5M configuration repairs.',
  octree_gs: '40k checkpoints are used for all scenes.',
  ghap: 'The evaluated GHAP configuration retains GMR at 15k with ratio 0.1.',
  gslpm: 'Image scores use retained training-time RGB on 30 scenes and an independently reloaded retrained 30k Room model. Native and intervention readouts retain their recorded model versions.',
};
const approved = new Set(register.map((row) => row.method));
if (approved.size !== register.length || register.some((row) => row.nvs_scenes !== '31')) {
  throw new Error('The approved paper method register must contain unique full31 methods.');
}
if (rows.some((row) => !approved.has(row.method))) throw new Error('Scene data includes a method outside the approved paper table.');

const methods = register.map((entry) => {
  const selected = rows.filter((row) => row.method === entry.method);
  if (selected.length !== 31 || new Set(selected.map((row) => row.scene)).size !== 31) {
    throw new Error('Incomplete or duplicate scene coverage for ' + entry.method);
  }
  const old = previous.get(entry.method);
  const method = {
    id: entry.method,
    name: entry.method_display,
    ...(old?.category ? { category: old.category } : {}),
    ...(old?.tier ? { tier: old.tier } : {}),
    note: notes[entry.method] || '',
    scores: {},
    scenes: Object.fromEntries(benchmark.datasets.map((dataset) => [dataset.id, []])),
  };
  for (const row of selected) {
    const dataset = datasetIds[row.dataset];
    const scene = expected.get(dataset)?.get(row.scene.toLowerCase());
    if (!scene) throw new Error('Unexpected scene in the paper source: ' + row.scene);
    const value = (text) => text?.trim() ? Number(text) : NaN;
    const metrics = { psnr: value(row.PSNR), ssim: value(row.SSIM), lpips: value(row.LPIPS_VGG) };
    if (!Object.values(metrics).every(Number.isFinite) || !Number.isInteger(Number(row.n_scored_views)) || Number(row.n_scored_views) < 1) {
      throw new Error('Incomplete full-view metrics: ' + row.method + '/' + row.scene);
    }
    method.scenes[dataset].push({
      id: scene.id,
      name: scene.name,
      metrics,
      status: 'complete',
      iteration: Number(row.iteration),
      views: Number(row.n_scored_views),
      configuration: row.configuration,
      qualification: row.qualification,
    });
  }
  for (const dataset of benchmark.datasets) {
    const scenes = method.scenes[dataset.id];
    if (scenes.length !== dataset.scenes.length || scenes.some((scene) => !expected.get(dataset.id).has(scene.id.toLowerCase()))) {
      throw new Error('Dataset coverage mismatch: ' + entry.method + '/' + dataset.id);
    }
    const order = new Map(dataset.scenes.map((scene, index) => [scene.id, index]));
    scenes.sort((a, b) => order.get(a.id) - order.get(b.id));
    method.scores[dataset.id] = {
      ...Object.fromEntries(['psnr', 'ssim', 'lpips'].map((metric) => [
        metric, scenes.reduce((sum, scene) => sum + scene.metrics[metric], 0) / scenes.length,
      ])),
      availableScenes: scenes.length,
      totalScenes: dataset.scenes.length,
    };
  }
  return method;
});

benchmark.updated = '2026-11-04';
benchmark.source = 'Paper Appendix A: 30 approved methods on all 31 shared scenes; complete held-out RGB evaluation from the recorded image sources.';
benchmark.aggregation = 'Scene-equal dataset means. Each scene uses its complete held-out RGB image set. LPIPS uses VGG. Checkpoint stages and configuration repairs are retained in the scene records and method notes.';
benchmark.methods = methods;
await writeFile(path.join(root, 'data/benchmark.json'), JSON.stringify(benchmark, null, 2) + '\n');
console.log('Updated image quality: ' + methods.length + ' methods, ' + rows.length + ' scene records, ' + methods.length * benchmark.datasets.length + ' dataset summaries. Geometry data left unchanged.');

function parseArgs(args) {
  const values = {};
  for (let index = 0; index < args.length; index += 2) {
    const key = args[index];
    const value = args[index + 1];
    if (!['--nvs', '--methods'].includes(key) || !value) {
      throw new Error('Usage: node scripts/build-site-data.mjs --nvs <current-scene-csv> --methods <paper-method-register-csv>');
    }
    values[key.slice(2)] = path.resolve(value);
  }
  if (!values.nvs || !values.methods) throw new Error('Both current scene data and the approved paper method register are required.');
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

