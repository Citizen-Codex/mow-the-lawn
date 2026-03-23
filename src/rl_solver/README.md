# RL Solver

This module adds a reinforcement-learning solver stack for the grid problem using `Gymnasium` and `MaskablePPO`.

## Files

- `config.py`: environment and training configuration dataclasses.
- `env.py`: custom `Gymnasium` environment for lawn coverage.
- `model.py`: training and inference helpers for `MaskablePPO`.
- `metrics.py`: path metrics with completion, overlap, and move counts.
- `evaluate.py`: evaluate the RL policy on generated grids.

## Workflow

Train a fixed-size model:

```bash
uv run train_rl.py --size 10 --timesteps 1000000 --output data/rl/maskable_ppo_10x10.zip
```

Resume training from an existing checkpoint:

```bash
uv run train_rl.py --size 10 --resume data/rl/maskable_ppo_10x10.best.zip --timesteps 500000 --output data/rl/maskable_ppo_10x10_resumed.zip
```

Training now uses a single streamlined backend, auto-tunes rollout/update sizes from `--n-envs`, trains directly on the full target distribution, and evaluates held-out seeds during training to save a `*.best.zip` checkpoint based on completion and coverage.

You can also override PPO rollout/update parameters directly:

```bash
uv run train_rl.py --size 10 --timesteps 1000000 --n-envs 16 --n-steps 256 --batch-size 1024 --n-epochs 6 --output data/rl/maskable_ppo_10x10.zip
```

Evaluate it on generated grids:

```bash
uv run eval_rl.py --model data/rl/maskable_ppo_10x10.zip --size 10 --seeds 50
```

Visualize one generated grid and the trained model path:

```bash
uv run visualize_rl.py --model data/rl/maskable_ppo_10x10.zip --seed 7
```

The policy is trained on one grid size at a time. The observation contains the open-cell mask, visited-cell mask, a log-scaled revisit-intensity map, the current and previous agent positions, scalar progress features, and per-action features that expose whether a move reaches new territory, how far it is from a frontier, and how heavily that edge has already been reused.

Rewards now prioritize completion first: new cells and completion are worth substantially more than routine revisits, repeated two-cell oscillation gets penalized, reusing the same corridor over and over gets penalized, moving closer to the nearest unvisited frontier gets a positive shaping reward, and timeout cost still scales with uncovered area.

The environment ends an episode immediately once any single cell exceeds the configured `max_overlaps`. This limit is per-cell, not global: the total overlaps across the whole grid can be larger, as long as no individual cell goes over the cap.

When resuming training, the checkpoint only needs to match the same grid size. To fine-tune behavior with different reward settings, update the defaults in `src/rl_solver/config.py` before launching the resumed run; the new checkpoint will save the updated config alongside the model.
