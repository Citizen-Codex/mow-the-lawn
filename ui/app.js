const ARROW_MOVES = {
  ArrowUp: { code: "u", label: "up", delta: [-1, 0] },
  ArrowDown: { code: "d", label: "down", delta: [1, 0] },
  ArrowLeft: { code: "l", label: "left", delta: [0, -1] },
  ArrowRight: { code: "r", label: "right", delta: [0, 1] },
};

const state = {
  gridData: null,
  displayedSolution: null,
  optimalReference: null,
  currentStep: 0,
  playTimer: null,
  manualPath: null,
  manualActive: false,
  humanResults: null,
  humanResultsError: null,
  selectedHumanConfigId: null,
  selectedHumanAttemptId: null,
  hardLevels: null,
  hardLevelsError: null,
  selectedHardLevelRank: null,
};

const elements = {
  sizeInput: document.getElementById("sizeInput"),
  seedInput: document.getElementById("seedInput"),
  removedMinInput: document.getElementById("removedMinInput"),
  removedMaxInput: document.getElementById("removedMaxInput"),
  clusterCountMinInput: document.getElementById("clusterCountMinInput"),
  clusterCountMaxInput: document.getElementById("clusterCountMaxInput"),
  clusterSizeMinInput: document.getElementById("clusterSizeMinInput"),
  clusterSizeMaxInput: document.getElementById("clusterSizeMaxInput"),
  randomizeSeedButton: document.getElementById("randomizeSeedButton"),
  generateButton: document.getElementById("generateButton"),
  solverSelect: document.getElementById("solverSelect"),
  strategySelect: document.getElementById("strategySelect"),
  startModeSelect: document.getElementById("startModeSelect"),
  solveButton: document.getElementById("solveButton"),
  optimalButton: document.getElementById("optimalButton"),
  manualToggleButton: document.getElementById("manualToggleButton"),
  resetManualButton: document.getElementById("resetManualButton"),
  undoManualButton: document.getElementById("undoManualButton"),
  showManualButton: document.getElementById("showManualButton"),
  manualModeChip: document.getElementById("manualModeChip"),
  manualStats: document.getElementById("manualStats"),
  humanResultsChip: document.getElementById("humanResultsChip"),
  humanConfigSelect: document.getElementById("humanConfigSelect"),
  humanAttemptSelect: document.getElementById("humanAttemptSelect"),
  humanPrevButton: document.getElementById("humanPrevButton"),
  humanNextButton: document.getElementById("humanNextButton"),
  humanResultsSummary: document.getElementById("humanResultsSummary"),
  humanAttemptSummary: document.getElementById("humanAttemptSummary"),
  hardLevelsChip: document.getElementById("hardLevelsChip"),
  hardLevelSelect: document.getElementById("hardLevelSelect"),
  hardLevelSummary: document.getElementById("hardLevelSummary"),
  prevButton: document.getElementById("prevButton"),
  playButton: document.getElementById("playButton"),
  nextButton: document.getElementById("nextButton"),
  fullButton: document.getElementById("fullButton"),
  stepSlider: document.getElementById("stepSlider"),
  stepLabel: document.getElementById("stepLabel"),
  statusText: document.getElementById("statusText"),
  gridStats: document.getElementById("gridStats"),
  solutionStats: document.getElementById("solutionStats"),
  referenceStats: document.getElementById("referenceStats"),
  gridCanvas: document.getElementById("gridCanvas"),
};

function isBusy() {
  return elements.generateButton.disabled;
}

function setStatus(text, tone = "info") {
  elements.statusText.textContent = text;
  elements.statusText.className = `status ${tone}`;
}

function randomizeSeed() {
  elements.seedInput.value = String(Math.floor(Math.random() * 1_000_000));
}

function readOptionalInt(input, label) {
  const raw = input.value.trim();
  if (raw === "") {
    return null;
  }
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${label} must be a positive integer or blank.`);
  }
  return value;
}

function readGridOptions() {
  const size = Number(elements.sizeInput.value);
  const seed = Number(elements.seedInput.value);
  const removedMin = Number(elements.removedMinInput.value);
  const removedMax = Number(elements.removedMaxInput.value);
  const clusterCountMin = readOptionalInt(elements.clusterCountMinInput, "Cluster count min");
  const clusterCountMax = readOptionalInt(elements.clusterCountMaxInput, "Cluster count max");
  const clusterSizeMin = readOptionalInt(elements.clusterSizeMinInput, "Cluster size min");
  const clusterSizeMax = readOptionalInt(elements.clusterSizeMaxInput, "Cluster size max");

  if (!Number.isInteger(size) || size <= 0) {
    throw new Error("Grid size must be a positive integer.");
  }
  if (!Number.isInteger(seed) || seed < 0) {
    throw new Error("Seed must be a non-negative integer.");
  }
  if (!(removedMin >= 0 && removedMin <= removedMax && removedMax < 1)) {
    throw new Error("Removed fractions must satisfy 0 <= min <= max < 1.");
  }
  if (clusterCountMin !== null && clusterCountMax !== null && clusterCountMin > clusterCountMax) {
    throw new Error("Cluster count min must be <= max.");
  }
  if (clusterSizeMin !== null && clusterSizeMax !== null && clusterSizeMin > clusterSizeMax) {
    throw new Error("Cluster size min must be <= max.");
  }

  return {
    size,
    seed,
    removedMin,
    removedMax,
    clusterCountMin,
    clusterCountMax,
    clusterSizeMin,
    clusterSizeMax,
  };
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatDurationMs(durationMs) {
  if (!Number.isFinite(durationMs)) {
    return "-";
  }

  if (durationMs < 1000) {
    return `${Math.round(durationMs)} ms`;
  }

  return `${(durationMs / 1000).toFixed(1)} s`;
}

function formatAverage(value) {
  return Number.isFinite(value) ? value.toFixed(1) : "-";
}

function getHumanAttempts() {
  return state.humanResults?.attempts || [];
}

function getHumanConfigMap() {
  return new Map((state.humanResults?.configs || []).map((config) => [String(config.id), config]));
}

function getHumanConfigEntries() {
  return (state.humanResults?.configs || [])
    .map((config) => {
      const attempts = getHumanAttempts()
        .filter((attempt) => attempt.configId === config.id)
        .sort((left, right) => left.id - right.id);

      const durationTotal = attempts.reduce((sum, attempt) => sum + attempt.durationMs, 0);
      const moveTotal = attempts.reduce((sum, attempt) => sum + attempt.moveCount, 0);
      const overlapTotal = attempts.reduce((sum, attempt) => sum + attempt.overlapCount, 0);
      const completedCount = attempts.filter((attempt) => attempt.coverageComplete).length;

      return {
        ...config,
        attempts,
        submissionCount: attempts.length,
        averageDurationMs: attempts.length ? durationTotal / attempts.length : 0,
        averageMoveCount: attempts.length ? moveTotal / attempts.length : 0,
        averageOverlapCount: attempts.length ? overlapTotal / attempts.length : 0,
        completedCount,
      };
    })
    .sort((left, right) => left.id - right.id);
}

function getSelectedHumanConfigEntry() {
  return (
    getHumanConfigEntries().find(
      (config) => String(config.id) === String(state.selectedHumanConfigId)
    ) || null
  );
}

function getAttemptsForSelectedConfig() {
  return getSelectedHumanConfigEntry()?.attempts || [];
}

function getHumanAttemptById(attemptId) {
  return getHumanAttempts().find((attempt) => attempt.id === attemptId) || null;
}

function ensureHumanSelection() {
  const configEntries = getHumanConfigEntries();
  if (!configEntries.length) {
    state.selectedHumanConfigId = null;
    state.selectedHumanAttemptId = null;
    return;
  }

  if (
    !configEntries.some(
      (config) => String(config.id) === String(state.selectedHumanConfigId)
    )
  ) {
    state.selectedHumanConfigId = configEntries[0].id;
  }

  const attempts = getAttemptsForSelectedConfig();
  if (!attempts.some((attempt) => attempt.id === state.selectedHumanAttemptId)) {
    state.selectedHumanAttemptId = null;
  }
}

function applyHumanAttempt(attemptId, { keepStatus = false } = {}) {
  const attempt = getHumanAttemptById(attemptId);
  if (!attempt) {
    return;
  }

  const config = getHumanConfigMap().get(String(attempt.configId));
  if (!config) {
    return;
  }

  stopPlayback();
  state.selectedHumanConfigId = attempt.configId;
  state.selectedHumanAttemptId = attempt.id;
  state.gridData = config.grid;
  state.displayedSolution = attempt;
  state.currentStep = attempt.moveCount;
  state.optimalReference = null;
  state.manualPath = null;
  state.manualActive = false;

  elements.sizeInput.value = String(config.size);
  elements.seedInput.value = String(config.seed);
  const removedRange = config.grid.removedFractionRange || [0.18, 0.42];
  elements.removedMinInput.value = String(removedRange[0]);
  elements.removedMaxInput.value = String(removedRange[1]);

  if (!keepStatus) {
    setStatus(
      `Loaded human attempt ${attempt.id} for ${attempt.user} on ${config.label}.`,
      "success"
    );
  }

  render();
}

async function loadHumanResults() {
  elements.humanResultsChip.textContent = "loading";
  elements.humanResultsChip.className = "chip chip-idle";
  elements.humanResultsSummary.textContent = "Loading human results from the CSV...";

  try {
    state.humanResults = await fetchJson("/api/human-results");
    state.humanResultsError = null;
    state.selectedHumanConfigId = null;
    state.selectedHumanAttemptId = null;
    render();
  } catch (error) {
    state.humanResults = null;
    state.humanResultsError = error.message;
    render();
  }
}

async function loadHardLevels() {
  elements.hardLevelsChip.textContent = "loading";
  elements.hardLevelsChip.className = "chip chip-idle";
  elements.hardLevelSummary.textContent = "Loading hard-level candidates from the CSV...";

  try {
    state.hardLevels = await fetchJson("/api/hard-levels");
    state.hardLevelsError = null;
    render();
  } catch (error) {
    state.hardLevels = null;
    state.hardLevelsError = error.message;
    render();
  }
}

function applyHardLevelCandidate(rank) {
  const candidate = (state.hardLevels?.candidates || []).find(
    (entry) => entry.rank === rank
  );
  if (!candidate) {
    return;
  }
  state.selectedHardLevelRank = rank;

  elements.sizeInput.value = String(candidate.size);
  elements.seedInput.value = String(candidate.seed);
  elements.removedMinInput.value = String(candidate.removedFractionRange[0]);
  elements.removedMaxInput.value = String(candidate.removedFractionRange[1]);
  elements.clusterCountMinInput.value = String(candidate.clusterCountRange[0]);
  elements.clusterCountMaxInput.value = String(candidate.clusterCountRange[1]);
  elements.clusterSizeMinInput.value = String(candidate.clusterSizeRange[0]);
  elements.clusterSizeMaxInput.value = String(candidate.clusterSizeRange[1]);

  generateGrid();
}

function setBusy(busy) {
  const disabled = Boolean(busy);
  elements.generateButton.disabled = disabled;
  elements.solveButton.disabled = disabled;
  elements.optimalButton.disabled = disabled;
  syncSolverControls();
  syncManualControls();
  syncHumanResultsControls();
  syncPlaybackControls();
}

function syncSolverControls() {
  const needsPathControls = ["snake", "spiral"].includes(elements.solverSelect.value);
  elements.strategySelect.disabled = isBusy() || !needsPathControls;
  elements.startModeSelect.disabled = isBusy() || !needsPathControls;
}

function syncManualControls() {
  const hasGrid = Boolean(state.gridData);
  const hasManualPath = Boolean(state.manualPath);
  elements.manualToggleButton.disabled = isBusy() || !hasGrid;
  elements.resetManualButton.disabled = isBusy() || !hasGrid;
  elements.undoManualButton.disabled =
    isBusy() || !hasManualPath || state.manualPath.moves.length === 0;
  elements.showManualButton.disabled = isBusy() || !hasManualPath;
}

function syncPlaybackControls() {
  const hasSolution = Boolean(state.displayedSolution);
  const disabled = isBusy() || !hasSolution;
  elements.prevButton.disabled = disabled;
  elements.playButton.disabled = disabled;
  elements.nextButton.disabled = disabled;
  elements.fullButton.disabled = disabled;
  elements.stepSlider.disabled = disabled;
}

function syncHumanResultsControls() {
  const configEntries = getHumanConfigEntries();
  const attempts = getAttemptsForSelectedConfig();
  const currentIndex = attempts.findIndex(
    (attempt) => attempt.id === state.selectedHumanAttemptId
  );
  const hasResults = configEntries.length > 0;

  elements.humanConfigSelect.disabled = isBusy() || !hasResults;
  elements.humanAttemptSelect.disabled = isBusy() || attempts.length === 0;
  elements.humanPrevButton.disabled =
    isBusy() || attempts.length <= 1 || currentIndex <= 0;
  elements.humanNextButton.disabled =
    isBusy() || attempts.length <= 1 || currentIndex === -1 || currentIndex >= attempts.length - 1;
}

async function generateGrid() {
  stopPlayback();

  let options;
  try {
    options = readGridOptions();
  } catch (error) {
    setStatus(error.message, "error");
    return;
  }

  setBusy(true);
  setStatus("Generating grid...", "info");

  try {
    const query = new URLSearchParams({
      size: String(options.size),
      seed: String(options.seed),
      removedMin: String(options.removedMin),
      removedMax: String(options.removedMax),
    });
    if (options.clusterCountMin !== null) query.set("clusterCountMin", String(options.clusterCountMin));
    if (options.clusterCountMax !== null) query.set("clusterCountMax", String(options.clusterCountMax));
    if (options.clusterSizeMin !== null) query.set("clusterSizeMin", String(options.clusterSizeMin));
    if (options.clusterSizeMax !== null) query.set("clusterSizeMax", String(options.clusterSizeMax));
    state.gridData = await fetchJson(`/api/grid?${query.toString()}`);
    state.displayedSolution = null;
    state.optimalReference = null;
    state.currentStep = 0;
    state.manualPath = null;
    state.manualActive = false;
    state.selectedHumanAttemptId = null;
    setStatus("Grid generated. Choose a solver or start a manual path.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
    render();
  }
}

async function requestSolution(solver) {
  if (!state.gridData) {
    throw new Error("Generate a grid first.");
  }

  const body = {
    grid: state.gridData.grid,
    solver,
    strategy: elements.strategySelect.value,
    startMode: elements.startModeSelect.value,
    seed: Number(elements.seedInput.value),
  };

  const payload = await fetchJson("/api/solve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  return payload.solution;
}

function pauseManualInput() {
  if (!state.manualActive) {
    return;
  }
  state.manualActive = false;
  if (state.displayedSolution?.solverKey === "manual") {
    state.displayedSolution = buildManualSolution();
  }
}

async function solveDisplayed() {
  const solver = elements.solverSelect.value;

  stopPlayback();
  pauseManualInput();
  setBusy(true);
  setStatus(`Solving ${solver}...`, "info");

  try {
    const solution = await requestSolution(solver);
    state.displayedSolution = solution;
    state.currentStep = solution.moveCount;
    if (solver === "optimal") {
      state.optimalReference = solution;
    }
    setStatus(`Displayed ${solution.solverLabel.toLowerCase()} solution.`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
    render();
  }
}

async function loadOptimalReference() {
  stopPlayback();
  pauseManualInput();
  setBusy(true);
  setStatus("Loading optimal reference...", "info");

  const previousDisplayedSolution = state.displayedSolution;
  const previousStep = state.currentStep;

  try {
    state.optimalReference = await requestSolution("optimal");
    if (previousDisplayedSolution) {
      state.displayedSolution = previousDisplayedSolution;
      state.currentStep = Math.min(previousStep, previousDisplayedSolution.moveCount);
    } else {
      state.displayedSolution = state.optimalReference;
      state.currentStep = state.optimalReference.moveCount;
    }
    setStatus("Loaded optimal reference for the current grid.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
    render();
  }
}

function createEmptyManualPath() {
  if (!state.gridData?.start) {
    throw new Error("This grid does not have a valid start cell.");
  }
  return {
    moves: [],
    points: [state.gridData.start.slice()],
  };
}

function buildManualSolution() {
  if (!state.gridData || !state.manualPath) {
    return null;
  }

  const points = state.manualPath.points.map((point) => [...point]);
  const visitCounts = countVisits(points);
  let overlapCount = 0;
  let visitedOpenCells = 0;

  for (const [key, count] of visitCounts.entries()) {
    const [row, col] = key.split(",").map(Number);
    if (state.gridData.grid[row]?.[col] === 1) {
      visitedOpenCells += 1;
    }
    if (count > 1) {
      overlapCount += count - 1;
    }
  }

  return {
    solverKey: "manual",
    solverLabel: "Manual",
    detailLabel: state.manualActive ? "arrow-key input active" : "user-created path",
    pathString: state.manualPath.moves.join(""),
    points,
    moveCount: state.manualPath.moves.length,
    overlapCount,
    visitedOpenCells,
    openCellCount: state.gridData.openCells,
    coverageComplete: visitedOpenCells === state.gridData.openCells,
    start: points[0] || null,
    end: points[points.length - 1] || null,
  };
}

function showManualPath({ keepStatus = false } = {}) {
  if (!state.manualPath) {
    setStatus("Start a manual path first.", "info");
    return;
  }

  stopPlayback();
  state.displayedSolution = buildManualSolution();
  state.currentStep = state.displayedSolution.moveCount;
  if (!keepStatus) {
    setStatus("Displayed the current manual path.", "success");
  }
  render();
}

function toggleManualInput() {
  if (!state.gridData) {
    setStatus("Generate a grid first.", "error");
    return;
  }

  stopPlayback();

  if (!state.manualActive) {
    if (!state.manualPath) {
      state.manualPath = createEmptyManualPath();
    }
    state.manualActive = true;
    state.displayedSolution = buildManualSolution();
    state.currentStep = state.displayedSolution.moveCount;
    setStatus(
      "Manual input active. Use arrow keys to move and Backspace to undo.",
      "success"
    );
  } else {
    state.manualActive = false;
    if (state.displayedSolution?.solverKey === "manual") {
      state.displayedSolution = buildManualSolution();
    }
    setStatus("Manual input paused.", "info");
  }

  render();
}

function resetManualPath() {
  if (!state.gridData) {
    setStatus("Generate a grid first.", "error");
    return;
  }

  stopPlayback();
  state.manualPath = createEmptyManualPath();
  state.manualActive = true;
  state.displayedSolution = buildManualSolution();
  state.currentStep = 0;
  setStatus("Manual path reset. Use arrow keys to move.", "success");
  render();
}

function undoManualMove({ keepStatus = false } = {}) {
  if (!state.manualPath) {
    setStatus("Start a manual path first.", "info");
    return;
  }

  if (state.manualPath.moves.length === 0) {
    if (!keepStatus) {
      setStatus("Manual path is already at the start cell.", "info");
    }
    return;
  }

  stopPlayback();
  state.manualPath.moves.pop();
  state.manualPath.points.pop();
  state.displayedSolution = buildManualSolution();
  state.currentStep = state.displayedSolution.moveCount;

  if (!keepStatus) {
    setStatus("Removed the last manual move.", "success");
  }

  render();
}

function attemptManualMove(move) {
  if (!state.manualActive || !state.gridData) {
    return;
  }

  if (!state.manualPath) {
    state.manualPath = createEmptyManualPath();
  }

  const currentPoint = state.manualPath.points[state.manualPath.points.length - 1];
  const nextRow = currentPoint[0] + move.delta[0];
  const nextCol = currentPoint[1] + move.delta[1];

  if (
    nextRow < 0 ||
    nextRow >= state.gridData.rows ||
    nextCol < 0 ||
    nextCol >= state.gridData.cols ||
    state.gridData.grid[nextRow][nextCol] !== 1
  ) {
    setStatus(`Blocked move: cannot go ${move.label} from the current cell.`, "error");
    return;
  }

  stopPlayback();
  state.manualPath.moves.push(move.code);
  state.manualPath.points.push([nextRow, nextCol]);
  state.displayedSolution = buildManualSolution();
  state.currentStep = state.displayedSolution.moveCount;

  if (state.displayedSolution.coverageComplete) {
    setStatus("Manual path now covers every open cell.", "success");
  } else {
    setStatus(
      "Manual input active. Use arrow keys to move and Backspace to undo.",
      "info"
    );
  }

  render();
}

function shouldIgnoreManualKeydown(event) {
  if (event.altKey || event.ctrlKey || event.metaKey) {
    return true;
  }

  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return false;
  }

  return (
    target.isContentEditable ||
    ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)
  );
}

function handleManualKeydown(event) {
  if (!state.manualActive || isBusy() || shouldIgnoreManualKeydown(event)) {
    return;
  }

  if (event.key === "Backspace") {
    event.preventDefault();
    undoManualMove({ keepStatus: true });
    setStatus("Manual input active. Use arrow keys to move and Backspace to undo.", "info");
    return;
  }

  const move = ARROW_MOVES[event.key];
  if (!move) {
    return;
  }

  event.preventDefault();
  attemptManualMove(move);
}

function renderHumanResults() {
  if (state.humanResultsError) {
    elements.humanResultsChip.textContent = "error";
    elements.humanResultsChip.className = "chip chip-paused";
    elements.humanResultsSummary.textContent = state.humanResultsError;
    elements.humanAttemptSummary.textContent = "Attempt data unavailable.";
    elements.humanConfigSelect.innerHTML = '<option value="">Unavailable</option>';
    elements.humanAttemptSelect.innerHTML = '<option value="">Unavailable</option>';
    syncHumanResultsControls();
    return;
  }

  if (!state.humanResults) {
    elements.humanResultsChip.textContent = "loading";
    elements.humanResultsChip.className = "chip chip-idle";
    elements.humanResultsSummary.textContent = "Loading human results from the CSV...";
    elements.humanAttemptSummary.textContent = "Preparing config and submission selectors...";
    elements.humanConfigSelect.innerHTML = '<option value="">Loading...</option>';
    elements.humanAttemptSelect.innerHTML = '<option value="">Loading...</option>';
    syncHumanResultsControls();
    return;
  }

  ensureHumanSelection();
  const configEntries = getHumanConfigEntries();
  const selectedConfig = getSelectedHumanConfigEntry();
  const attempts = getAttemptsForSelectedConfig();
  const selectedAttempt = getHumanAttemptById(state.selectedHumanAttemptId);
  const selectedAttemptIndex = attempts.findIndex(
    (attempt) => attempt.id === state.selectedHumanAttemptId
  );

  elements.humanResultsChip.className = "chip chip-active";
  elements.humanConfigSelect.innerHTML = configEntries
    .map(
      (config) =>
        `<option value="${config.id}" ${String(config.id) === String(state.selectedHumanConfigId) ? "selected" : ""}>${escapeHtml(`${config.label} | ${config.size}x${config.size} | ${config.submissionCount} submissions`)}</option>`
    )
    .join("");
  elements.humanAttemptSelect.innerHTML = attempts.length
    ? [
        `<option value="" ${selectedAttempt ? "" : "selected"}>Choose a submission...</option>`,
        ...attempts.map(
          (attempt, index) =>
            `<option value="${attempt.id}" ${attempt.id === state.selectedHumanAttemptId ? "selected" : ""}>${escapeHtml(`${index + 1}. ${attempt.user} | ${formatDurationMs(attempt.durationMs)} | ${attempt.moveCount} moves | #${attempt.id}`)}</option>`
        ),
      ].join("")
    : '<option value="">No submissions</option>';

  if (!selectedConfig) {
    elements.humanResultsChip.textContent = `${state.humanResults.attemptCount} runs`;
    elements.humanResultsSummary.textContent = "No config data available.";
    elements.humanAttemptSummary.textContent = "No submissions available.";
    syncHumanResultsControls();
    return;
  }

  elements.humanResultsChip.textContent = `${selectedConfig.submissionCount} submissions`;
  elements.humanResultsSummary.textContent = `${selectedConfig.label} | size ${selectedConfig.size} | seed ${selectedConfig.seed} | submissions ${selectedConfig.submissionCount} | avg duration ${formatDurationMs(selectedConfig.averageDurationMs)} | avg moves ${formatAverage(selectedConfig.averageMoveCount)} | complete ${selectedConfig.completedCount}/${selectedConfig.submissionCount}`;

  if (!selectedAttempt || selectedAttemptIndex === -1) {
    elements.humanAttemptSummary.textContent = "Choose a submission to load it, or switch configs to auto-load the first run for that config.";
    syncHumanResultsControls();
    return;
  }

  elements.humanAttemptSummary.textContent = `Submission ${selectedAttemptIndex + 1}/${attempts.length} | user ${selectedAttempt.user} | duration ${formatDurationMs(selectedAttempt.durationMs)} | moves ${selectedAttempt.moveCount} | overlaps ${selectedAttempt.overlapCount} | coverage ${selectedAttempt.visitedOpenCells}/${selectedAttempt.openCellCount} ${selectedAttempt.coverageComplete ? "(complete)" : "(partial)"} | ${new Date(selectedAttempt.createdAt).toLocaleString()}`;

  syncHumanResultsControls();
}

function renderStats() {
  if (!state.gridData) {
    elements.gridStats.textContent = "No grid loaded.";
  } else {
    const grid = state.gridData;
    const countRange = grid.clusterCountRange;
    const sizeRange = grid.clusterSizeRange;
    const clusterText =
      countRange && sizeRange
        ? ` | clusters ${countRange[0]}-${countRange[1]} of size ${sizeRange[0]}-${sizeRange[1]}`
        : "";
    elements.gridStats.textContent = `Grid ${grid.rows}x${grid.cols} | seed ${grid.seed} | open ${grid.openCells} | removed ${grid.removedCells} | density ${grid.density.toFixed(2)}${clusterText}`;
  }

  if (!state.displayedSolution) {
    elements.solutionStats.textContent = "No displayed solution yet.";
  } else {
    const solution = state.displayedSolution;
    const durationSummary =
      solution.solverKey === "human"
        ? ` | duration ${formatDurationMs(solution.durationMs)} | avg step ${formatDurationMs(solution.averageStepMs)}`
        : "";
    elements.solutionStats.textContent = `Displayed: ${solution.solverLabel} (${solution.detailLabel}) | moves ${solution.moveCount} | overlaps ${solution.overlapCount} | coverage ${solution.visitedOpenCells}/${solution.openCellCount} ${solution.coverageComplete ? "(complete)" : "(partial)"}${durationSummary}`;
  }

  if (!state.optimalReference) {
    elements.referenceStats.textContent = "Optimal reference not loaded for this grid.";
  } else if (!state.displayedSolution) {
    const reference = state.optimalReference;
    elements.referenceStats.textContent = `Optimal reference: ${reference.moveCount} moves | overlaps ${reference.overlapCount}`;
  } else {
    const reference = state.optimalReference;
    const solution = state.displayedSolution;
    const moveDelta = solution.moveCount - reference.moveCount;
    const overlapDelta = solution.overlapCount - reference.overlapCount;
    elements.referenceStats.textContent = `Optimal reference: ${reference.moveCount} moves | overlaps ${reference.overlapCount} | delta vs displayed ${formatDelta(moveDelta)} moves, ${formatDelta(overlapDelta)} overlaps`;
  }
}

function renderManualControls() {
  if (!state.gridData) {
    elements.manualModeChip.textContent = "no grid";
    elements.manualModeChip.className = "chip chip-idle";
    elements.manualStats.textContent = "Generate a grid before starting a manual path.";
  } else if (state.manualActive) {
    elements.manualModeChip.textContent = "input active";
    elements.manualModeChip.className = "chip chip-active";
    const manualSolution = buildManualSolution();
    elements.manualStats.textContent = manualSolution
      ? `Manual path: moves ${manualSolution.moveCount} | overlaps ${manualSolution.overlapCount} | coverage ${manualSolution.visitedOpenCells}/${manualSolution.openCellCount} ${manualSolution.coverageComplete ? "(complete)" : "(partial)"}`
      : "Start a manual path to use the arrow keys.";
  } else if (state.manualPath) {
    elements.manualModeChip.textContent = "paused";
    elements.manualModeChip.className = "chip chip-paused";
    const manualSolution = buildManualSolution();
    elements.manualStats.textContent = `Manual path: moves ${manualSolution.moveCount} | overlaps ${manualSolution.overlapCount} | coverage ${manualSolution.visitedOpenCells}/${manualSolution.openCellCount} ${manualSolution.coverageComplete ? "(complete)" : "(partial)"}`;
  } else {
    elements.manualModeChip.textContent = "idle";
    elements.manualModeChip.className = "chip chip-idle";
    elements.manualStats.textContent = "Manual path not started.";
  }

  elements.manualToggleButton.textContent = state.manualActive
    ? "Pause manual input"
    : state.manualPath
      ? "Resume manual input"
      : "Start manual path";

  syncManualControls();
}

function formatDelta(value) {
  return `${value >= 0 ? "+" : ""}${value}`;
}

function renderStepControls() {
  const maxStep = state.displayedSolution ? state.displayedSolution.moveCount : 0;
  elements.stepSlider.max = String(maxStep);
  elements.stepSlider.value = String(Math.min(state.currentStep, maxStep));
  elements.stepLabel.textContent = `Step ${Math.min(state.currentStep, maxStep)} / ${maxStep}`;
  syncPlaybackControls();
}

function renderHardLevels() {
  if (state.hardLevelsError) {
    elements.hardLevelsChip.textContent = "error";
    elements.hardLevelsChip.className = "chip chip-paused";
    elements.hardLevelSummary.textContent = state.hardLevelsError;
    elements.hardLevelSelect.innerHTML = '<option value="">Unavailable</option>';
    elements.hardLevelSelect.disabled = true;
    return;
  }

  if (!state.hardLevels) {
    elements.hardLevelsChip.textContent = "loading";
    elements.hardLevelsChip.className = "chip chip-idle";
    elements.hardLevelSummary.textContent = "Loading hard-level candidates from the CSV...";
    elements.hardLevelSelect.innerHTML = '<option value="">Loading...</option>';
    elements.hardLevelSelect.disabled = true;
    return;
  }

  if (!state.hardLevels.available) {
    elements.hardLevelsChip.textContent = "missing";
    elements.hardLevelsChip.className = "chip chip-paused";
    elements.hardLevelSummary.textContent =
      state.hardLevels.message || "Hard-levels CSV not found.";
    elements.hardLevelSelect.innerHTML = '<option value="">Not generated</option>';
    elements.hardLevelSelect.disabled = true;
    return;
  }

  const candidates = state.hardLevels.candidates || [];
  elements.hardLevelsChip.textContent = `${candidates.length} candidates`;
  elements.hardLevelsChip.className = "chip chip-active";
  elements.hardLevelSelect.disabled = isBusy() || candidates.length === 0;

  const selected = candidates.find(
    (entry) => entry.rank === state.selectedHardLevelRank
  );
  elements.hardLevelSelect.innerHTML = [
    `<option value="" ${selected ? "" : "selected"}>Choose a candidate...</option>`,
    ...candidates.map(
      (entry) => `<option value="${entry.rank}" ${
        entry.rank === state.selectedHardLevelRank ? "selected" : ""
      }>${escapeHtml(
        `#${entry.rank} ${entry.regime} ${entry.size}x${entry.size} seed=${entry.seed} | score ${entry.score.toFixed(2)}`
      )}</option>`
    ),
  ].join("");

  if (!selected) {
    elements.hardLevelSummary.textContent =
      "Select a candidate to load its grid and form params.";
    return;
  }

  const m = selected.metrics;
  const altStr =
    m.altOptimaFraction === null || m.altOptimaFraction === undefined
      ? "n/a"
      : `${Math.round(m.altOptimaFraction * 100)}%`;
  elements.hardLevelSummary.textContent =
    `#${selected.rank} | ${selected.regime} | ${selected.size}x${selected.size} | seed ${selected.seed} | ` +
    `score ${selected.score.toFixed(3)} | optimal ${m.optimalMoves} moves (${m.optimalOverlap} overlaps) | ` +
    `snake ${m.snakeMoves} | gap ${(m.heuristicGap * 100).toFixed(0)}% | ` +
    `turn ${Math.round(m.turnDensity * 100)}% | bridges ${m.bridges} | choice ${m.choicePoints} | alt ${altStr}`;
}

function handleHardLevelChange(event) {
  const raw = event.target.value;
  if (raw === "") {
    state.selectedHardLevelRank = null;
    render();
    return;
  }
  const rank = Number(raw);
  if (!Number.isInteger(rank)) {
    return;
  }
  applyHardLevelCandidate(rank);
}

function render() {
  renderHumanResults();
  renderHardLevels();
  renderStats();
  renderManualControls();
  renderStepControls();
  drawCanvas();
  syncSolverControls();
}

function drawCanvas() {
  const canvas = elements.gridCanvas;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!state.gridData) {
    return;
  }

  const grid = state.gridData.grid;
  const rows = grid.length;
  const cols = grid[0].length;
  const maxGridPixels = 680;
  const cellSize = Math.max(
    24,
    Math.min(64, Math.floor(maxGridPixels / Math.max(rows, cols)))
  );
  const margin = 28;
  const width = cols * cellSize + margin * 2;
  const height = rows * cellSize + margin * 2;
  canvas.width = width;
  canvas.height = height;

  const bbox = (row, col) => {
    const x0 = margin + col * cellSize;
    const y0 = margin + row * cellSize;
    return [x0, y0, x0 + cellSize, y0 + cellSize];
  };

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const [x0, y0, x1, y1] = bbox(row, col);
      ctx.fillStyle = grid[row][col] === 1 ? "#ffffff" : "#334155";
      ctx.strokeStyle = grid[row][col] === 1 ? "#cbd5e1" : "#1e293b";
      ctx.lineWidth = 1;
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }
  }

  const start = state.gridData.start;
  let prefixPoints = [];
  let visitCounts = new Map();
  let currentPoint = null;

  if (state.displayedSolution) {
    const maxStep = state.displayedSolution.moveCount;
    state.currentStep = Math.max(0, Math.min(state.currentStep, maxStep));
    prefixPoints = state.displayedSolution.points.slice(0, state.currentStep + 1);
    visitCounts = countVisits(prefixPoints);
    currentPoint = prefixPoints[prefixPoints.length - 1] || null;

    for (const [key, count] of visitCounts.entries()) {
      const [row, col] = key.split(",").map(Number);
      if (grid[row]?.[col] !== 1) {
        continue;
      }
      const [x0, y0, x1, y1] = bbox(row, col);
      ctx.fillStyle = count === 1 ? "#dcfce7" : "#fecaca";
      ctx.fillRect(x0 + 4, y0 + 4, x1 - x0 - 8, y1 - y0 - 8);
    }

    for (let index = 1; index < prefixPoints.length; index += 1) {
      const [prevRow, prevCol] = prefixPoints[index - 1];
      const [nextRow, nextCol] = prefixPoints[index];
      const [px0, py0, px1, py1] = bbox(prevRow, prevCol);
      const [nx0, ny0, nx1, ny1] = bbox(nextRow, nextCol);
      drawArrow(
        ctx,
        (px0 + px1) / 2,
        (py0 + py1) / 2,
        (nx0 + nx1) / 2,
        (ny0 + ny1) / 2,
        "#0f766e"
      );
    }

    for (const [key, count] of visitCounts.entries()) {
      if (count <= 1) {
        continue;
      }
      const [row, col] = key.split(",").map(Number);
      if (grid[row]?.[col] !== 1) {
        continue;
      }
      const [x0, y0, x1, y1] = bbox(row, col);
      const cx = (x0 + x1) / 2;
      const cy = (y0 + y1) / 2;
      const radius = cellSize * 0.16;
      ctx.fillStyle = "#be123c";
      ctx.strokeStyle = "#881337";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 11px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(count), cx, cy);
    }
  }

  if (start) {
    const [x0, y0, x1, y1] = bbox(start[0], start[1]);
    ctx.strokeStyle = "#16a34a";
    ctx.lineWidth = 3;
    ctx.strokeRect(x0 + 5, y0 + 5, x1 - x0 - 10, y1 - y0 - 10);
    ctx.fillStyle = "#15803d";
    ctx.font = "bold 12px Inter, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("S", x0 + 11, y0 + 9);
  }

  if (currentPoint) {
    const [x0, y0, x1, y1] = bbox(currentPoint[0], currentPoint[1]);
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    const radius = cellSize * 0.18;
    ctx.fillStyle = "#2563eb";
    ctx.strokeStyle = "#1d4ed8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    if (
      state.displayedSolution &&
      state.displayedSolution.moveCount > 0 &&
      state.currentStep >= state.displayedSolution.moveCount
    ) {
      ctx.strokeStyle = "#dc2626";
      ctx.lineWidth = 3;
      ctx.strokeRect(x0 + 5, y0 + 5, x1 - x0 - 10, y1 - y0 - 10);
      ctx.fillStyle = "#b91c1c";
      ctx.font = "bold 12px Inter, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText("E", x0 + 11, y0 + 9);
    }
  }
}

function countVisits(points) {
  const counts = new Map();
  for (const point of points) {
    const key = point.join(",");
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return counts;
}

function drawArrow(ctx, startX, startY, endX, endY, color) {
  const headLength = 10;
  const angle = Math.atan2(endY - startY, endX - startX);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(startX, startY);
  ctx.lineTo(endX, endY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(endX, endY);
  ctx.lineTo(
    endX - headLength * Math.cos(angle - Math.PI / 6),
    endY - headLength * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    endX - headLength * Math.cos(angle + Math.PI / 6),
    endY - headLength * Math.sin(angle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fill();
}

function stopPlayback() {
  if (state.playTimer !== null) {
    window.clearTimeout(state.playTimer);
    state.playTimer = null;
  }
  elements.playButton.textContent = "Play";
}

function playNextFrame() {
  if (!state.displayedSolution) {
    stopPlayback();
    return;
  }

  if (state.currentStep >= state.displayedSolution.moveCount) {
    stopPlayback();
    render();
    return;
  }

  state.currentStep += 1;
  render();
  state.playTimer = window.setTimeout(playNextFrame, 140);
}

function togglePlayback() {
  if (!state.displayedSolution) {
    return;
  }
  if (state.playTimer !== null) {
    stopPlayback();
    return;
  }
  if (state.currentStep >= state.displayedSolution.moveCount) {
    state.currentStep = 0;
  }
  elements.playButton.textContent = "Pause";
  playNextFrame();
}

function stepBack() {
  if (!state.displayedSolution) {
    return;
  }
  stopPlayback();
  state.currentStep = Math.max(0, state.currentStep - 1);
  render();
}

function stepForward() {
  if (!state.displayedSolution) {
    return;
  }
  stopPlayback();
  state.currentStep = Math.min(
    state.displayedSolution.moveCount,
    state.currentStep + 1
  );
  render();
}

function stepToFull() {
  if (!state.displayedSolution) {
    return;
  }
  stopPlayback();
  state.currentStep = state.displayedSolution.moveCount;
  render();
}

function handleHumanConfigChange(event) {
  if (event.target.value === "") {
    return;
  }

  const configId = Number(event.target.value);
  if (!Number.isInteger(configId)) {
    return;
  }

  state.selectedHumanConfigId = configId;
  const attempts = getAttemptsForSelectedConfig();
  const firstAttempt = attempts[0] || null;
  if (!firstAttempt) {
    state.selectedHumanAttemptId = null;
    render();
    return;
  }

  applyHumanAttempt(firstAttempt.id);
}

function handleHumanAttemptChange(event) {
  if (event.target.value === "") {
    state.selectedHumanAttemptId = null;
    render();
    return;
  }

  const attemptId = Number(event.target.value);
  if (!Number.isInteger(attemptId)) {
    return;
  }

  applyHumanAttempt(attemptId);
}

function navigateHumanAttempt(direction) {
  ensureHumanSelection();
  const attempts = getAttemptsForSelectedConfig();
  if (attempts.length <= 1) {
    return;
  }

  const currentIndex = attempts.findIndex(
    (attempt) => attempt.id === state.selectedHumanAttemptId
  );
  if (currentIndex === -1) {
    applyHumanAttempt(attempts[0].id);
    return;
  }

  const nextIndex = currentIndex + direction;
  if (nextIndex < 0 || nextIndex >= attempts.length) {
    return;
  }

  applyHumanAttempt(attempts[nextIndex].id);
}

elements.randomizeSeedButton.addEventListener("click", randomizeSeed);
elements.generateButton.addEventListener("click", generateGrid);
elements.solveButton.addEventListener("click", solveDisplayed);
elements.optimalButton.addEventListener("click", loadOptimalReference);
elements.manualToggleButton.addEventListener("click", toggleManualInput);
elements.resetManualButton.addEventListener("click", resetManualPath);
elements.undoManualButton.addEventListener("click", () => undoManualMove());
elements.showManualButton.addEventListener("click", () => showManualPath());
elements.solverSelect.addEventListener("change", syncSolverControls);
elements.humanConfigSelect.addEventListener("change", handleHumanConfigChange);
elements.humanAttemptSelect.addEventListener("change", handleHumanAttemptChange);
elements.hardLevelSelect.addEventListener("change", handleHardLevelChange);
elements.humanPrevButton.addEventListener("click", () => navigateHumanAttempt(-1));
elements.humanNextButton.addEventListener("click", () => navigateHumanAttempt(1));
elements.prevButton.addEventListener("click", stepBack);
elements.playButton.addEventListener("click", togglePlayback);
elements.nextButton.addEventListener("click", stepForward);
elements.fullButton.addEventListener("click", stepToFull);
elements.stepSlider.addEventListener("input", (event) => {
  if (!state.displayedSolution) {
    return;
  }
  stopPlayback();
  state.currentStep = Number(event.target.value);
  render();
});
window.addEventListener("keydown", handleManualKeydown);

generateGrid();
loadHumanResults();
loadHardLevels();
