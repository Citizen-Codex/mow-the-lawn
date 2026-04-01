# RL Solver

This module adds a reinforcement-learning solver stack for the lawn grid using `Gymnasium` and `MaskablePPO`.

## Files

- `config.py`: environment and training configuration dataclasses plus config save/load helpers.
- `env.py`: padded `Gymnasium` environment with action masking, reward shaping, and curriculum-size sampling.
- `features.py`: custom CNN-plus-scalar feature extractor for padded grid observations.
- `expert.py`: offline optimal-path library loading and behavior-cloning pretraining.
- `model.py`: PPO training, checkpoint resume, held-out evaluation, and inference helpers.
- `rollout.py`: run a trained policy on a specific grid or seed and collect metrics.
- `metrics.py`: path, rollout, and summary metrics.
- `evaluate.py`: batch evaluation over generated seeds.

## Model Shape

Each model is trained with a maximum grid size from `--size`.

- Observations are padded to `(size, size)`.
- Training can sample smaller active grids inside that padded space.
- New models use a custom CNN-based feature extractor over the grid maps plus scalar/action features, instead of the default flat dict extractor.
- Inference and evaluation support any square grid whose side length is `<= size`.
- Resume compatibility is still based on the same maximum `size`.

## Training Workflows

Train a baseline model at one maximum size:

```bash
uv run train_rl.py --size 10 --timesteps 1000000 --output data/rl/maskable_ppo_10x10.zip
```

Resume training from an existing checkpoint with the same maximum size:

```bash
uv run train_rl.py --size 10 --resume data/rl/maskable_ppo_10x10.best.zip --timesteps 500000 --output data/rl/maskable_ppo_10x10_resumed.zip
```

Warm-start PPO with demonstrations loaded from `src/optimal/library/optimal_paths.csv` before RL fine-tuning:

```bash
uv run train_rl.py --size 12 --timesteps 1000000 --bc-epochs 5 --bc-grid-count 64 --bc-size-min 7 --bc-size-max 7 --output data/rl/maskable_ppo_12x12_bc.zip
```

Train one padded model with a size-biased curriculum, softer overlap handling, and a target-size-only finish phase:

```bash
uv run train_rl.py --size 12 --timesteps 5000000 --curriculum-min-size 6 --curriculum-warmup-episodes 2000 --curriculum-target-size-prob 0.5 --curriculum-size-bias-power 2.0 --final-target-only-timesteps 1500000 --no-terminate-on-overlap-limit --max-overlaps 2 --eval-seed-count 50 --bc-epochs 10 --bc-grid-count 128 --bc-size-min 7 --bc-size-max 7 --bc-removed-fraction-min 0.10 --bc-removed-fraction-max 0.28 --output data/rl/maskable_ppo_12x12_curriculum.zip
```

Override PPO rollout/update parameters directly:

```bash
uv run train_rl.py --size 10 --timesteps 1000000 --learning-rate 1e-4 --gamma 0.997 --gae-lambda 0.95 --ent-coef 0.001 --clip-range 0.2 --n-envs 16 --n-steps 512 --batch-size 1024 --n-epochs 8 --output data/rl/maskable_ppo_10x10.zip
```

## Curriculum And BC

Training uses one PPO backend and evaluates held-out seeds during learning to save a `*.best.zip` checkpoint based on completion and coverage.

Curriculum training:

- `--curriculum-min-size`: smallest active grid size sampled during training.
- `--curriculum-warmup-episodes`: number of generated episodes per environment used to expand from the minimum size to the maximum `--size`.
- `--curriculum-size-bias-power`: biases unlocked curriculum sizes toward larger boards instead of sampling them uniformly.
- `--curriculum-target-size-prob`: once the maximum size is unlocked, sample it directly with this probability.
- `--final-target-only-timesteps`: optional second PPO stage that runs only on the maximum `--size` after the main curriculum stage.
- Held-out evaluation still runs at the full maximum size, so best-checkpoint selection reflects large-grid performance rather than easier curriculum episodes.

Behavior cloning pretraining:

- `--bc-epochs`: supervised pretraining epochs before PPO.
- `--bc-grid-count`: number of rows to load from the optimal-path library.
- `--bc-size-min` and `--bc-size-max`: size range for expert-library grids, capped by the model maximum `--size`. By default, BC can use any library size from `--curriculum-min-size` or `1` up to `--size`.
- `--bc-removed-fraction-min` and `--bc-removed-fraction-max`: optional density filter applied to library rows.
- `--bc-learning-rate` and `--bc-batch-size`: optimizer controls.
- `--bc-seed-start`: row offset into the filtered optimal-path library before sampling demonstrations.

The expert dataset is built from `src/optimal/library/optimal_paths.csv`. Training filters the library by size and removed-cell fraction, samples the requested number of rows, replays those stored optimal paths through `LawnMowingEnv`, and collects observation, action-mask, and action targets for the policy head.

The checked-in library currently contains precomputed `7x7` optimal paths. To use additional expert sizes, regenerate the library in `src/optimal/library.py` with a wider size range before training.

If the expert-library sizes are much smaller than the target model size, training prints a warning because aggressive BC on small boards can slow down later PPO adaptation.

## Reward And Episode Controls

- `--max-overlaps`: per-cell overlap threshold used by the revisit-limit heuristic.
- `--terminate-on-overlap-limit` / `--no-terminate-on-overlap-limit`: choose whether crossing the overlap threshold ends the episode immediately or only applies the configured penalty.
- `--max-steps-factor`: total step budget as a multiple of open-cell count.
- `--removed-fraction-min` and `--removed-fraction-max`: control the density range of generated grids used for training, evaluation, rollout-by-seed, and visualization.

New `train_rl.py` runs default to `--no-terminate-on-overlap-limit`, which keeps the overlap limit as a learning signal without prematurely cutting off promising large-grid rollouts. Older saved configs still load with their saved overlap behavior.

## PPO Tuning Controls

`train_rl.py` now exposes the main PPO knobs directly:

- `--learning-rate`
- `--gamma`
- `--gae-lambda`
- `--ent-coef`
- `--clip-range`
- `--n-envs`
- `--n-steps`
- `--batch-size`
- `--n-epochs`
- `--eval-freq-timesteps`
- `--eval-seed-count`
- `--save-best-checkpoint` / `--no-save-best-checkpoint`

## Evaluation And Visualization

Evaluate a model on generated grids at any supported size:

```bash
uv run eval_rl.py --model data/rl/maskable_ppo_10x10.zip --size 10 --seeds 50
```

Visualize a rollout at the model maximum size:

```bash
uv run visualize_rl.py --model data/rl/maskable_ppo_10x10.zip --seed 7
```

Visualize a smaller grid with the same padded model:

```bash
uv run visualize_rl.py --model data/rl/maskable_ppo_12x12_curriculum.zip --size 8 --seed 7
```

Evaluation, `rollout_model_on_seed`, and visualization now reuse the saved removed-cell fraction range from the model config so diagnostics stay aligned with the training distribution.

## Observation And Reward Design

The observation contains:

- open-cell mask
- visited-cell mask
- log-scaled revisit-intensity map
- current position
- previous position
- scalar progress features
- per-action features for move validity, new-cell reachability, revisit intensity, frontier distance, and edge reuse

Rewards prioritize completion-first behavior:

- strong reward for covering a new cell
- large completion bonus
- penalties for revisits
- penalties for repeated edge reuse
- penalties for two-cell oscillation loops
- stall penalties after too many steps without new coverage
- timeout cost scaled by uncovered area
- overlap-limit penalty when a single cell exceeds `max_overlaps`, with optional immediate termination

The overlap limit is per-cell, not global. Total overlaps across the whole path may be larger as long as no single cell exceeds the configured cap.

## Checkpoints

- Model checkpoints save alongside a matching `.config.json` payload.
- Resuming requires the same maximum `size`.
- Changing reward defaults or curriculum settings and resuming is supported as long as the maximum size matches.
- New checkpoints save the CNN feature extractor configuration inside the PPO policy.
