# OpenDuckPro2 10K 自动训练流程

本文描述 `legged_gym/scripts/auto_train.py` 的当前控制契约。该流程只负责 OpenDuckPro2，并以 reward scale 搜索为边界。

## 强制约束

- 每个 candidate 使用一个训练进程、一个 `run_name`、一个日志目录和一条 TensorBoard 序列，连续训练到 10000 iteration。
- 1000 到 9000 iteration 只保存 checkpoint，不停止、不 resume、不新建 TensorBoard run，也不等待决策。
- 到达 10000 后，依次加载同一 run 中的 `model_1000.pt` 到 `model_10000.pt` 做 post-hoc 评估。
- reward 名称、公式、mask、目标、坐标系和相位逻辑在搜索期间冻结，只允许覆盖已有非零 reward 的 scale。
- 每个新 candidate 必须先通过 `gh` 在 `orbbecwuxin/openduck-training-control` 创建独立 draft PR，并把 PR 元数据写入 candidate。
- checkpoint、TensorBoard、完整日志、临时 workspace 和决策产物不得提交到源码仓库。

## 配置

默认模板是 `configs/auto_train/openduckpro2_default.json`。模板故意将 `candidates` 留空，防止没有 PR gate 的任务被误启动。

每个 candidate 至少包含：

```json
{
  "name": "candidate-name",
  "seed": 1,
  "reward_scales": {
    "tracking_lin_vel": 1.0
  },
  "training_pr": {
    "repo": "orbbecwuxin/openduck-training-control",
    "number": 123,
    "url": "https://github.com/orbbecwuxin/openduck-training-control/pull/123",
    "head_ref": "train/candidate-name",
    "base_ref": "main",
    "created_with": "gh",
    "state": "OPEN",
    "is_draft": true
  }
}
```

允许覆盖的正 reward：

- `tracking_lin_vel`
- `tracking_ang_vel`
- `alive`
- `contact`

允许覆盖的 penalty：

- `lin_vel_z`
- `ang_vel_xy`
- `orientation`
- `base_height`
- `torques`
- `dof_acc`
- `dof_vel`
- `action_rate`
- `dof_pos_limits`
- `hip_pos`
- `contact_no_vel`
- `feet_swing_height`

scale 不能为零，不能反转正负号。`feet_air_time` 与 `collision` 保持禁用。

## 启动

在 Isaac Gym、PyTorch 和 `rsl_rl` 环境已经激活后执行：

```bash
./auto_train_openduckpro2.sh \
  --config configs/auto_train/openduckpro2_default.json \
  --run-id <unique-run-id>
```

启动前会校验 10K budget、1K milestone 列表、reward 白名单、候选是否从零开始，以及 draft PR 元数据。任一条件不满足都会阻止训练。

## 产物布局

```text
auto_train_runs/<run-id>/
  cycle_000/<candidate>/
    train.log
    evaluate.log
    candidate_result.json
    evaluation.json
    trajectory.csv
    trajectory.svg
    milestones/<iteration>/
    codex_review_request.json
    codex_decision.json
```

实际 PPO 日志位于 `logs/openduckpro2/<run-name>/`。同一 candidate 在 10K 前必须始终绑定到同一目录。

## 评估顺序

1. 先判断训练是否正常结束以及 checkpoint 是否完整。
2. 检查 episode fall rate、base height、tilt 和 NaN/Inf。
3. 检查左右脚交替、同脚重复、双脚同时抬起、拖脚与摆动高度。
4. 检查速度跟踪、动作饱和、力矩、平滑性和能耗。
5. 只有 10K 后的完整 milestone 证据可以产生下一 candidate 的 scale 决策。

若结构问题不能在既有 scale 合约内解决，应报告 blocker，而不是修改 reward 源码。

## 停止与恢复

正常停止使用 run 对应的 stop file；不要用 checkpoint resume 把一个 candidate 拆成多个 TensorBoard run。运行异常时保留原日志并创建新的 candidate、run 和 PR，不把异常退出伪装成连续训练。
