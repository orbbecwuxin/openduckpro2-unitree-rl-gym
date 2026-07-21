# 规范：OpenDuckPro2 自动训练框架

## 目标

构建一个仓库内执行器，由 Codex 主导 OpenDuckPro2 双足强化学习训练。执行器在远程训练服务器上训练 reward 变体，运行 Isaac Gym policy rollout，使用多个指标节点对轨迹评分，写入 simlog 诊断，然后等待 Codex 决定下一轮 reward 源码修改或 candidate 修改。所有仓库修改都必须保留 Git 提交记录。

## 技术栈

- `legged_gym/scripts/` 下的 Python 脚本
- 现有 Isaac Gym 任务：`openduckpro2`
- 现有训练后端：通过 `task_registry` 使用 `rsl_rl`
- Shell 入口风格沿用 `train_openduckpro2_unitree.sh`
- 自动提交使用命令级身份：`Auto Train <auto-train@local>`

## 常用命令

运行已配置的优化循环：

```bash
./auto_train_openduckpro2.sh --config configs/auto_train/openduckpro2_default.json
```

launcher 默认使用 `CONDA_ENV=openduck-unitree`，并在非交互 SSH 会话中 source `/home/orbbec/miniconda3/etc/profile.d/conda.sh`。它还会把 `${CONDA_PREFIX}/lib` 追加到 `LD_LIBRARY_PATH` 前面，使 Isaac Gym 能在该环境中找到 Python shared library。

```bash
CONDA_ENV=openduck-unitree ./auto_train_openduckpro2.sh --config configs/auto_train/openduckpro2_default.json
```

不启动 Isaac Gym 训练，仅做短 dry run：

```bash
./auto_train_openduckpro2.sh --config configs/auto_train/openduckpro2_default.json --dry-run --no-commit
```

训练单个 reward 变体：

```bash
python legged_gym/scripts/train_reward_variant.py \
  --reward-overrides auto_train_runs/example/reward_overrides.json \
  --auto-train-meta auto_train_runs/example/train_meta.json \
  --task=openduckpro2 \
  --experiment_name=openduckpro2 \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --num_envs=4096 \
  --max_iterations=5000 \
  --run_name=manual_variant \
  --headless
```

评估一个 checkpoint 并写入轨迹指标：

```bash
python legged_gym/scripts/evaluate_policy.py \
  --output auto_train_runs/example/evaluation.json \
  --task=openduckpro2 \
  --experiment_name=openduckpro2 \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --num_envs=32 \
  --load_run=<run-dir-name> \
  --checkpoint=-1 \
  --headless
```

## 项目结构

- `auto_train_openduckpro2.sh`：在仓库内启动框架。
- `configs/auto_train/openduckpro2_default.json`：定义 GPU、并行度、训练规模、评估规模和初始 reward candidates。
- `configs/auto_train/openduckpro2_multislot.json`：通过 `gpus: [0, 0, 1, 1]` 在每张 GPU 上运行两个训练 slot。
- `configs/auto_train/openduckpro2_multislot_smoke.json`：验证四个并发 workspace candidates，其中 GPU 0 两个 slot、GPU 1 两个 slot。
- `legged_gym/scripts/auto_train.py`：编排 cycle 和并行 GPU jobs，写入 Codex review request，然后等待 Codex decision。
- `legged_gym/scripts/train_reward_variant.py`：在内存中应用 reward overrides 并运行 PPO 训练。
- `legged_gym/scripts/evaluate_policy.py`：运行 headless Isaac Gym 推理，采集轨迹样本，并生成指标节点评分和 simlog 诊断。
- `legged_gym/scripts/auto_train_common.py`：共享 JSON、reward、checkpoint 和 Git helper。
- `auto_train_runs/<run_id>/`：保存已提交的 candidate 定义和评估摘要。
- `auto_train_workspaces/<run_id>/cycle_XXX/<candidate>/`：保存每个 candidate 的隔离 branch workspace、训练日志和保留的模型 checkpoint。

## 代码风格

框架优先使用 Python 标准库，用显式 JSON 文件保存实验状态，并把长时间运行的 Isaac Gym job 隔离到 subprocess 边界。

```python
reward_overrides = {
    "name": candidate["name"],
    "cycle": cycle_index,
    "reward_scales": candidate.get("reward_scales", {}),
}
write_json(candidate_dir / "reward_overrides.json", reward_overrides)
```

## Codex 主导的优化循环

默认配置以 Codex-owned streaming loop 运行，并保持已配置训练 slot 持续占满。任意一个 candidate 完成后，执行器立刻评估该 candidate，在该 candidate 目录写入 `codex_review_request.json` 和 `codex_review_request.md`，然后等待该 candidate 的 `codex_decision.json`。它不会等待同一 cycle 的所有 seed candidates 都完成后才评分或请求下一步 reward 决策。

旧的 cycle-barrier 行为仍可通过设置 `streaming_candidates.enabled=false` 使用。

### 流式执行行为

- 在配置的 `gpus` 上最多启动 `max_parallel_jobs` 个 candidates。
- 某个 candidate 一完成，就运行 Isaac Gym evaluation，并写入 `evaluation.json`、`trajectory.csv` 和 `trajectory.svg`。
- review request 写在该 candidate 目录内。
- Codex 检查该 candidate 的 simlog，并写入 `codex_decision.json`。
- 空出的 GPU slot 会立即补入新的 Codex candidate，其它 slot 上的训练不受影响并继续运行。
- 剩余 seed candidates 继续排队，除非配置让 Codex candidates 具有更高优先级。

### Codex 必须读取的证据

- `candidate_result.json`
- `evaluation.json`
- `trajectory.csv`
- `trajectory.svg`
- `train.log`
- `evaluate.log`

Codex 根据这些证据决定是否新增、删除或重写 reward 源码逻辑，是否调整 candidate scales，是否增加 iteration budget，或是否停止。如果 reward 行为存在结构性错误，例如同一只脚重复抬脚、摔倒、蹲伏或没有清晰 swing phase，Codex 应修改 reward 源码项，而不是只改 reward 权重。

### Reward 源码文件

- `legged_gym/envs/openduckpro2/openduckpro2_env.py`：reward 函数，例如 `_reward_contact`、`_reward_feet_swing_height` 和其它 gait/trajectory 项。
- `legged_gym/envs/openduckpro2/openduckpro2_config.py`：`rewards.scales` 下的 reward scale 声明。

### Reward 优化策略

- 当 simlog 暴露缺失行为约束时，新增 reward 项，例如左右脚交替抬升序列。
- 当 simlog 显示某个 reward 鼓励错误捷径时，删除或禁用该项，例如双脚都贴地 shuffle。
- 当指标方向存在但 frame、phase 或高度基准错误时，重写 reward 项。
- 只有当行为结构有效、simlog 显示只是现有目标之间的权衡不对时，才只调整权重。
- 后续任何 reward 源码修改前，都必须先运行 reward challenge subagent，并在 candidate 目录写入 `reward_challenge.json`。
- 如果 rollout 有结构性失败但 Codex 想做 scale-only，也必须先通过 challenge 说明为什么不改源码是合理的。

### Decision 文件 schema

```json
{
  "action": "continue",
  "train_max_iterations": 5000,
  "reward_source_commit": "<git-sha-or-empty>",
  "reward_source_changes": {
    "added_rewards": ["alternating_lift_sequence"],
    "removed_rewards": [],
    "modified_rewards": ["feet_swing_height"],
    "scale_only": false,
    "evidence": "simlog showed repeated same-foot lift-offs and foot reference error"
  },
  "reward_challenge": {
    "artifact": "reward_challenge.json",
    "decision": "source_change_required",
    "primary_failure": "repeated same-foot lift-offs",
    "accepted_by_main_codex": true
  },
  "notes": "Codex changed reward logic because simlog showed repeated same-foot lift-offs.",
  "next_candidates": [
    {
      "name": "codex_reward_fix_c001",
      "reward_scales": {}
    }
  ]
}
```

允许的动作：

- `continue`：Codex 提供 `next_candidates`；用于 Codex 已检查 simlog，并可选地提交 reward 源码修改之后。
- `auto_mutate`：Codex 明确允许脚本建议的 reward-scale mutation 进入下一轮。
- `stop`：Codex 或用户终止 run。

循环只在以下条件之一满足时停止：

- 用户停止：创建 `auto_train_runs/<run_id>/STOP` 或仓库根目录 `STOP_AUTO_TRAIN`。
- 分数目标：best candidate 达到 `termination.score_target`。
- 平台期：达到 `termination.min_cycles` 后，最近 best score 在 `termination.patience_cycles` 内提升小于 `termination.min_delta`。
- 显式上限：传入 `--cycles N`，或把 `cycles` 设置为一个固定数字。
- 需要 Codex decision 且没有 wait mode：`codex_control.wait_for_decision=false` 会写入 review request 后停止。

默认 `cycles` 值是 `null`，因此除非用户设置，否则没有固定 cycle 上限。启用 `codex_control.wait_for_decision=true` 时，进程会持续运行，并在等待 Codex 时刷新 `auto_train_runs/<run_id>/heartbeat.json`。触发停止条件时，框架写入 `auto_train_runs/<run_id>/stop_reason.json` 并提交。

训练 iteration budget 是动态的。默认从 `train.max_iterations=5000` 开始；如果分数低于 `dynamic_iterations.score_threshold`，或最近提升低于 `dynamic_iterations.plateau_delta`，下一轮会增加 `dynamic_iterations.increment`，直到 `dynamic_iterations.max_iterations=6000`。正在运行的 PPO job 不会被原地修改；更大的 iteration 数只应用于下一轮 candidate。

Automatic mutation 现在只是建议，除非 Codex 明确写入 `action=auto_mutate`。建议 mutation 使用各节点 evaluation score：

- `survival` 低：增加稳定性相关 reward，例如 `orientation`、`base_height`、`contact`。
- `velocity_tracking` 低：增加 command tracking reward。
- `height_stability` 或 `upright` 低：增加对应稳定性 reward。
- `foot_trajectory` 低：增加 swing-height 和 landing-posture 约束。
- `foot_alternation` 低：增加 phase contact 和 swing-height 压力，鼓励左右脚交替抬升。
- `energy` 或 `smoothness` 低：增加 torque、acceleration、action-rate 和 velocity regularization。

使用 `--cycles N` 可以给一次 run 设置循环上限。

## 测试策略

- 使用 `python -m py_compile` 对新增 Python 脚本做静态语法检查。
- 使用 `--dry-run --no-commit` 验证命令生成，不启动训练。
- 在完整运行前，把 `max_iterations` 降低后做一次短训练 smoke test。
- 完整 Isaac Gym 训练和评估依赖 GPU 和 simulator runtime，应视为集成测试。

## 评分准则

评估基于轨迹，不使用学习出来的 judge 模型。每个 candidate 运行 Isaac Gym inference，并写入 `evaluation.json`、`trajectory.csv` 和 `trajectory.svg`。`evaluation.json` 还包含 `simlog`，用于总结失败证据和 Codex reward-source action hints。

硬 gate：

- 如果任何被评估环境摔倒，`total_score` 强制为 `0.0`。
- 摔倒通过环境 `done` signal、base height 低于 `0.55 * base_height_target`，或 roll/pitch 超过 termination thresholds 检测。

加权评分节点：

- `survival`：rollout 中没有 fall/reset。
- `velocity_tracking`：base XY velocity 跟随 command。
- `yaw_tracking`：base yaw velocity 跟随 command。
- `height_stability`：base height 保持在 `base_height_target` 附近。
- `upright`：roll 和 pitch 保持较小。
- `foot_trajectory`：左右 foot link height 在与 `env.feet_pos` 相同的 z frame 中跟随交替 phase reference trajectory。
- `foot_alternation`：lift-off event 左右交替；同一只脚重复 lift-off 会被惩罚。
- `energy`：mean absolute torque 越低越好。
- `smoothness`：action delta 越低越好。

Foot reference 使用估计的 stance link height 作为 baseline，使用 `swing_height_target` 作为 swing target。SVG 图会展示 base height、左右脚高度与 reference height 对比、以及左右脚 X 轨迹。重复左-左或右-右抬脚序列会降低 `foot_alternation`；一只脚连续踩两次再换脚的 policy 不应获得高排名。

## 边界

- Always：并行训练时，把生成的 reward variants 保存在 JSON 文件中，不直接编辑 `openduckpro2_config.py`。
- Always：把生成的框架状态写入仓库内 `auto_train_runs/`。
- Always：当 `workspace.enabled=true` 时，在隔离子目录 workspace 中训练 candidates。
- Always：每个 candidate workspace 保留 model checkpoints，并在 `candidate_result.json` 记录路径。
- Always：除非显式传入 `--no-commit`，否则提交框架生成的仓库修改。
- Always：Codex 负责优化决策；脚本负责执行、日志和等待。
- Always：当行为失败属于结构性问题时，基于 simlog 证据修改 reward 源码。
- Always：把 `STOP_AUTO_TRAIN` 和 `auto_train_runs/<run_id>/STOP` 当作用户控制文件，而不是实验 artifact。
- Ask first：修改 robot asset 或 base environment dynamics；reward 源码修改遵循 reward challenge gate 和 Git 提交流程。
- Ask first：把 GPU/slot 并行度提高到用户已授权资源之外。
- Never：写出 `/data2/wuxin/Open_Duck_Mini/unitree_rl_gym`。
- Never：提交 model checkpoints 或 `logs/` 下的 TensorBoard logs。
- Never：stage 无关 dirty files。

## 成功准则

- 框架可以在配置的 GPU 和 slot 上启动多个 candidate training。
- 每个 candidate 使用自己的 reward override 文件和唯一 run name。
- Evaluation 输出 `evaluation.json`，其中包含 trajectory samples、aggregate metrics、per-node scores 和 total score。
- Orchestrator 可以从 best evaluated candidate 生成下一轮 reward candidates。
- 默认 run 持续运行，直到用户停止、达到分数目标或 plateau convergence；每个 cycle 保留 best candidate 作为 elite candidate，并从 best evaluated result mutation 下一轮 reward candidates。
- 当分数仍低或提升停滞时，框架可以提高下一轮训练 iterations。
- 框架只提交生成的框架文件，并保持已有 dirty files 不变。

## 未决事项

- 默认 conda 环境是 `openduck-unitree`；launcher 会设置 `LD_LIBRARY_PATH`，使 Isaac Gym 能加载 `libpython3.8.so.1.0`。
- 默认 iteration budget 从 5000 开始，并可在后续 cycle 动态增长到 6000。
