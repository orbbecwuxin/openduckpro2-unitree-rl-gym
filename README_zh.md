# openduckpro3 RL Gym

这是一个只包含 OpenDuck 的强化学习训练仓库，基于 Isaac Gym 和 `rsl_rl`。仓库保留 OpenDuck Mini、openduckpro3 的任务定义、机器人模型、训练脚本、评估工具和参数搜索配置；训练日志与模型 checkpoint 不进入 Git。

## 保留的任务

- `openduckpro3`：10 自由度 openduckpro3 行走任务。
- `open_duck`：OpenDuck Mini 行走任务。

仓库不再包含其他参考机器人的实现和模型资源。

## 目录

- `legged_gym/envs/base`：通用 Isaac Gym 环境。
- `legged_gym/envs/openduckpro3`：openduckpro3 环境和 PPO 配置。
- `legged_gym/envs/open_duck`：OpenDuck Mini 环境和配置。
- `openduckpro3`：训练实际使用的 openduckpro3 URDF 和 mesh。
- `resources/robots/open_duck_mini`：OpenDuck Mini 模型资源。
- `configs/auto_train`：自动训练配置。
- `scripts`：手动训练和推理脚本。

## 环境要求

代码面向 Python 3.8、NVIDIA Isaac Gym、PyTorch 和 `rsl_rl` 1.0.2。分别安装 Isaac Gym 与 `rsl_rl` 后，在当前环境安装本仓库：

```bash
pip install -e .
```

## 训练 openduckpro3

直接启动 1 万轮训练：

```bash
python legged_gym/scripts/train.py \
  --task=openduckpro3 \
  --headless \
  --num_envs=4096 \
  --max_iterations=10000 \
  --sim_device=cuda:0 \
  --rl_device=cuda:0
```

训练结果保存在 `logs/openduckpro3/`，不会提交到 Git。

## 推理 checkpoint

```bash
python legged_gym/scripts/play.py \
  --task=openduckpro3 \
  --num_envs=1 \
  --load_run=<训练目录> \
  --checkpoint=<轮次> \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --keyboard_commands
```

可移植启动脚本为 `play_openduckpro3_unitree.sh`，通过环境变量设置 `LOAD_RUN` 和 `CHECKPOINT`。

## 自动训练

自动训练流程见 `docs/auto_training_framework.md`。从 `configs/auto_train/` 中的 openduckpro3 配置启动；生成的 workspace、日志、模型和本地决策记录均不进入源码仓库。

## 许可证

见 `LICENSE` 与 `legged_gym/LICENSE`。
