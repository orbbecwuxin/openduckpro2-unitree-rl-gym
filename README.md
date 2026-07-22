# openduckpro3 RL Gym

OpenDuck-only reinforcement-learning training workspace built on Isaac Gym and `rsl_rl`.
The repository includes the OpenDuck Mini and openduckpro3 task definitions, robot descriptions, training scripts, evaluation tools, and parameter-search configuration. Training logs and model checkpoints are intentionally excluded.

## Included Tasks

- `openduckpro3`: 10-DoF openduckpro3 locomotion task.
- `open_duck`: OpenDuck Mini locomotion task.

Reference robot implementations and assets are not shipped in this repository.

## Layout

- `legged_gym/envs/base`: shared Isaac Gym environment code.
- `legged_gym/envs/openduckpro3`: openduckpro3 environment and PPO configuration.
- `legged_gym/envs/open_duck`: OpenDuck Mini environment and configuration.
- `openduckpro3`: openduckpro3 URDF and meshes used by training.
- `resources/robots/open_duck_mini`: OpenDuck Mini model assets.
- `configs/auto_train`: reproducible automated-training configurations.
- `scripts`: manual training and playback helpers.

## Requirements

The code targets Python 3.8, NVIDIA Isaac Gym, PyTorch, and `rsl_rl` 1.0.2. Install Isaac Gym and `rsl_rl` separately, then install this repository in the active environment:

```bash
pip install -e .
```

## Train openduckpro3

A direct 10K run can be started with:

```bash
python legged_gym/scripts/train.py \
  --task=openduckpro3 \
  --headless \
  --num_envs=4096 \
  --max_iterations=10000 \
  --sim_device=cuda:0 \
  --rl_device=cuda:0
```

Training output is written under `logs/openduckpro3/` and is not versioned.

## Play A Checkpoint

```bash
python legged_gym/scripts/play.py \
  --task=openduckpro3 \
  --num_envs=1 \
  --load_run=<run-directory> \
  --checkpoint=<iteration> \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --keyboard_commands
```

The portable launcher is `play_openduckpro3_unitree.sh`; set `LOAD_RUN` and `CHECKPOINT` as environment variables.

## Automated Training

The automated workflow is described in `docs/auto_training_framework.md`. Start from an openduckpro3 configuration under `configs/auto_train/`; generated workspaces, logs, checkpoints, and local decision artifacts remain outside source control.

## License

See `LICENSE` and `legged_gym/LICENSE`.
