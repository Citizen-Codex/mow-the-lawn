import { readFileSync } from "fs";

interface Model {
  vocabulary: Record<string, number>;
  ngram_range: [number, number];
  coef: number[][];
  intercept: number[];
  classes: string[];
}

function charNgrams(path: string, min: number, max: number): string[] {
  const ngrams: string[] = [];
  for (let n = min; n <= max; n++) {
    for (let i = 0; i <= path.length - n; i++) {
      ngrams.push(path.slice(i, i + n));
    }
  }
  return ngrams;
}

function vectorize(
  ngrams: string[],
  vocabulary: Record<string, number>
): Map<number, number> {
  const vec = new Map<number, number>();
  for (const ng of ngrams) {
    const idx = vocabulary[ng];
    if (idx !== undefined) {
      vec.set(idx, (vec.get(idx) ?? 0) + 1);
    }
  }
  return vec;
}

function predict(
  vec: Map<number, number>,
  coef: number[][],
  intercept: number[],
  classes: string[]
): { label: string; probabilities: Record<string, number> } {
  const nClasses = classes.length;
  const scores = new Float64Array(nClasses);

  for (let c = 0; c < nClasses; c++) {
    let sum = intercept[c];
    for (const [idx, count] of vec) {
      sum += coef[c][idx] * count;
    }
    scores[c] = sum;
  }

  // softmax
  const maxScore = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - maxScore));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sumExp);

  let bestIdx = 0;
  for (let i = 1; i < nClasses; i++) {
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  }

  const probabilities: Record<string, number> = {};
  for (let i = 0; i < nClasses; i++) {
    probabilities[classes[i]] = probs[i];
  }

  return { label: classes[bestIdx], probabilities };
}

// --- main ---

const model: Model = JSON.parse(readFileSync("data/model.json", "utf-8"));

const csvLines = readFileSync("data/labelled_paths.csv", "utf-8")
  .trim()
  .split("\n");
const headers = csvLines[0].split(",");
const rows = csvLines.slice(1).map((line) => {
  const cols = line.split(",");
  const row: Record<string, string> = {};
  headers.forEach((h, i) => (row[h] = cols[i]));
  return row;
});

const samples = [
  { path: rows[0].snake_path, expected: "snake" },
  { path: rows[0].spiral_path, expected: "spiral" },
  { path: rows[0].random_walk_path, expected: "random_walk" },
  { path: rows[1].snake_path, expected: "snake" },
  { path: rows[1].spiral_path, expected: "spiral" },
  { path: rows[1].random_walk_path, expected: "random_walk" },
];

console.log("POC: JS/TS inference from exported sklearn model\n");

let allMatch = true;
for (const { path, expected } of samples) {
  const ngrams = charNgrams(path, model.ngram_range[0], model.ngram_range[1]);
  const vec = vectorize(ngrams, model.vocabulary);
  const result = predict(vec, model.coef, model.intercept, model.classes);

  const match = result.label === expected;
  if (!match) allMatch = false;

  const probStr = Object.entries(result.probabilities)
    .map(([k, v]) => `${k}: ${v.toFixed(3)}`)
    .join(", ");

  console.log(`Expected: ${expected.padEnd(12)} Got: ${result.label.padEnd(12)} ${match ? "OK" : "MISMATCH"}  [${probStr}]`);
}

console.log(`\n${allMatch ? "All predictions match!" : "Some predictions did not match."}`);
