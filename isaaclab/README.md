# OpenDuckPro3 Isaac Lab Migration

This directory contains an external Isaac Lab task. It does not modify the
Isaac Lab installation. The first migration stage preserves the Isaac Gym task's
10 actions, 41 policy observations, 44 critic observations, PD gains, command
range, gait phase, domain randomization, and enabled reward formulas.

## Server Setup

The default paths target the existing server installation:

- Isaac Lab: `/data2/wuxin/IsaacLab-2.3.2`
- Conda environment: `/data2/conda/envs/leggedlab-train`
- Physical GPU: `1`

Run a one-environment import and physics smoke test:

```bash
PYTHONPATH="$PWD/isaaclab/source/isaaclab_openduck" \
ISAACLAB_ROOT=/data2/wuxin/IsaacLab-2.3.2 \
/data2/conda/envs/leggedlab-train/bin/python \
  isaaclab/scripts/run_upstream.py zero \
  --task Isaac-OpenDuckPro3-Direct-v0 --device cuda:0 \
  --num_envs 1 --headless
```

Start the default 4096-environment, 10K-iteration training:

```bash
./isaaclab/train_openduckpro3.sh
```

Routine choices are grouped at the top of the shell script. They can also be
overridden without editing it:

```bash
PHYSICAL_GPU=2 NUM_ENVS=1024 MAX_ITERATIONS=10 \
  RUN_NAME=smoke ./isaaclab/train_openduckpro3.sh
```

Isaac Gym checkpoints are not binary-compatible with Isaac Lab/RSL-RL 3.2.
Training starts from a new policy while retaining the task semantics.
