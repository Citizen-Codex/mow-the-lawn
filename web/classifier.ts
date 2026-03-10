export interface Model {
  chars: string[];
  ngram_range: [number, number];
  coef: number[][];
  intercept: number[];
  classes: string[];
}

export interface PredictionResult {
  label: string;
  probabilities: Record<string, number>;
}

function countStringsWithPrefix(
  prefixLength: number,
  min: number,
  max: number,
  alphabetSize: number
): number {
  let total = 0;
  for (let length = Math.max(min, prefixLength); length <= max; length++) {
    total += alphabetSize ** (length - prefixLength);
  }
  return total;
}

function buildCharToDigit(chars: string[]): Record<string, number> {
  const charToDigit: Record<string, number> = {};
  chars.forEach((char, index) => {
    charToDigit[char] = index;
  });
  return charToDigit;
}

function ngramIndex(
  ngram: string,
  chars: string[],
  charToDigit: Record<string, number>,
  min: number,
  max: number
): number | undefined {
  let rank = 0;
  const alphabetSize = chars.length;

  for (let i = 0; i < ngram.length; i++) {
    const ch = ngram[i];
    const digit = charToDigit[ch];
    if (digit === undefined) {
      return undefined;
    }

    const prefixLength = i + 1;
    for (const candidate of chars) {
      if (charToDigit[candidate] >= digit) {
        break;
      }
      rank += countStringsWithPrefix(prefixLength, min, max, alphabetSize);
    }

    if (prefixLength >= min && i < ngram.length - 1) {
      rank += 1;
    }
  }

  return rank;
}

export function charNgrams(path: string, min: number, max: number): string[] {
  if (typeof path !== "string") {
    throw new TypeError("classify path must be a string");
  }

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
  chars: string[],
  min: number,
  max: number
): Map<number, number> {
  const vec = new Map<number, number>();
  const charToDigit = buildCharToDigit(chars);
  for (const ng of ngrams) {
    if (ng.length < min || ng.length > max) {
      continue;
    }
    const idx = ngramIndex(ng, chars, charToDigit, min, max);
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
  const [min, max] = model.ngram_range;
  const ngrams = charNgrams(path, min, max);
  const vec = vectorize(ngrams, model.chars, min, max);
  return predict(vec, model.coef, model.intercept, model.classes);
}

