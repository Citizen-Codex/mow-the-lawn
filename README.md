# Mow The Lawn

Mow The Lawn is a small Python project for generating connected random lawn grids, solving traversal paths with snake and spiral strategies, and visualizing the result. It also includes a simulation pipeline to generate labelled path data for basic classification experiments.

## Project purpose

- Generate deterministic random grids from a size and seed.
- Compare two traversal strategies: snake and spiral.
- Visualize base grids and computed paths.
- Simulate many grids to build labelled training data (`snake_path`, `spiral_path`, `random_walk_path`).

## Codebase components

- `main.py`: Entrypoint for running a single grid demo and visualizing solver output.
- `simulate.py`: Entrypoint for generating labelled CSV data through the `simulate()` function.
- `classifier.py`: Train/evaluate/classify path strings as snake-like, spiral-like, or random-walk-like.
- `src/grid.py`: Random grid generation with connectivity-safe cell removals.
- `src/solvers.py`: Snake/spiral solvers and pathing strategies (`shortest`, `least_overlap`).
- `src/visualize.py`: Tkinter visualization helpers and path statistics.
- `src/shared_types.py`: Shared types (`Grid`, `Path`, moves, and deltas).
- `data/labelled_paths.csv`: Example generated dataset.

## Quickstart

### 1) Setup

```bash
uv sync
```

### 2) Run the main demo

`main.py` calls `main(n, seed)` to:

- Create one deterministic random grid.
- Solve it with snake, spiral, and random walk strategies.
- Print path overlap stats.
- Attempt to launch Tkinter windows for grid/path visualization.

Run it:

```bash
uv run main.py 12 7
```

### 3) Run the simulation pipeline

`simulate.py` calls `simulate()` (CLI entrypoint), which uses `create_labelled_paths(count, size, output_path)` to:

- Generate many deterministic random grids.
- Solve each grid with snake, spiral, and random walk strategies.
- Save the move sequences as CSV rows.

Run it:

```bash
uv run simulate.py 100 12 --output data/labelled_paths.csv
```

Output CSV columns:

- `snake_path`
- `spiral_path`
- `random_walk_path`

Path strings use only: `u`, `d`, `l`, `r`.

## 4) Use the classifier

Evaluate model performance on a train/test split:

```bash
uv run classifier.py eval
```

Classify a single path string:

```bash
uv run classifier.py classify --path "dddddddddruuuuuuuuurdddddddd"
```

## Notes

- Tkinter windows may not open in headless environments; CLI output still runs.
- Use the same seed to reproduce the same generated grid for a given size.
