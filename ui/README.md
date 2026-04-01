# UI App

This folder contains a small browser application for exploring generated mowing grids and solver behavior interactively.

## Run

```bash
uv run python3 ui/app.py
```

Then open `http://127.0.0.1:8765` in a browser.

## What It Does

- Generates grids with the existing `create_random_grid()` logic.
- Lets you choose `snake`, `spiral`, `random_walk`, or `optimal` solving.
- Exposes the existing heuristic path strategy controls for snake and spiral.
- Lets you create your own path interactively with the arrow keys and compare it against the optimal reference.
- Lets you scrub through the computed path step by step or play it back.
- Loads the exact optimal path as a reference so you can compare move count and overlap count against the displayed solution.

## Notes

- The optimal solver is exact and can take noticeably longer on larger or denser grids.
- The UI is served locally with the Python standard library, so it does not require a frontend build step.
