import { cp, mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const staticRoot = new URL('../static/', import.meta.url);
const dataFile = new URL('../data/benchmark.json', import.meta.url);
const volDataFile = new URL('../data/vol-benchmark.json', import.meta.url);
const volDtuDataFile = new URL('../data/vol-dtu-benchmark.json', import.meta.url);
const outputRoot = new URL('../dist/', import.meta.url);
const outputData = new URL('../dist/data/', import.meta.url);

await rm(outputRoot, { recursive: true, force: true });
await mkdir(outputRoot, { recursive: true });
await mkdir(outputData, { recursive: true });
await cp(staticRoot, outputRoot, { recursive: true });
await cp(dataFile, join(fileURLToPath(outputData), 'benchmark.json'));
await cp(volDataFile, join(fileURLToPath(outputData), 'vol-benchmark.json'));
await cp(volDtuDataFile, join(fileURLToPath(outputData), 'vol-dtu-benchmark.json'));

console.log(`Built static site at ${fileURLToPath(outputRoot)}`);
