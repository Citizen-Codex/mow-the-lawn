import { readFileSync } from "fs";
import { type Model, classify } from "./web/classifier";

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
  const result = classify(model, path);

  const match = result.label === expected;
  if (!match) allMatch = false;

  const probStr = Object.entries(result.probabilities)
    .map(([k, v]) => `${k}: ${v.toFixed(3)}`)
    .join(", ");

  console.log(`Expected: ${expected.padEnd(12)} Got: ${result.label.padEnd(12)} ${match ? "OK" : "MISMATCH"}  [${probStr}]`);
}

console.log(`\n${allMatch ? "All predictions match!" : "Some predictions did not match."}`);
