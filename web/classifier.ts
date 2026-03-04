export interface Model {
  vocabulary: Record<string, number>;
  ngram_range: [number, number];
  coef: number[][];
  intercept: number[];
  classes: string[];
}

export interface PredictionResult {
  label: string;
  probabilities: Record<string, number>;
}

export function charNgrams(path: string, min: number, max: number): string[] {
  const ngrams: string[] = [];
  for (let n = min; n <= max; n++) {
    for (let i = 0; i <= path.length - n; i++) {
      ngrams.push(path.slice(i, i + n));
    }
  }
  return ngrams;
}

export function vectorize(
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

export function predict(
  vec: Map<number, number>,
  coef: number[][],
  intercept: number[],
  classes: string[]
): PredictionResult {
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

export function classify(model: Model, path: string): PredictionResult {
  const ngrams = charNgrams(path, model.ngram_range[0], model.ngram_range[1]);
  const vec = vectorize(ngrams, model.vocabulary);
  return predict(vec, model.coef, model.intercept, model.classes);
}
