# openduckpro3 Reward Scale Challenge

该协议用于挑战 openduckpro3 的 reward scale 假设。当前训练合约冻结 reward 名称与公式，因此 challenge 不能授权新增、删除、重命名或改写 reward。

## 触发时机

在创建新的 scale candidate 前，或 10K 评估显示摔倒、静止、拖脚、不交替、M 型抬腿、落脚冲击、速度偏差或动作饱和时，执行 challenge。

1K 到 9K 是观察点，不单独产生换权重决策。除非出现 NaN、进程退出或 checkpoint 缺失，否则等待连续 10K 完成。

## 必查证据

- candidate、slot、run、checkpoint 和 TensorBoard 路径是否一一对应。
- `train.log`、`evaluate.log`、`evaluation.json`、`trajectory.csv` 和 `trajectory.svg`。
- episode fall rate、base height、roll/pitch/yaw、速度跟踪和动作标准差。
- 左右脚接触、摆动高度、同脚重复、双抬脚、拖脚和落脚速度。
- 力矩、动作变化率、关节速度/加速度与饱和比例。
- command、URDF、碰撞体、控制频率、PD 参数和 evaluator 是否可以解释现象。

## 根因分类

challenge 必须把结论归入以下一种：

- `scale_only_candidate`：步态结构存在，剩余问题可由一个 reward 权重族的小幅调整验证。
- `continue_training`：证据尚未成熟，不修改参数并继续到 10K。
- `non_reward_blocker`：URDF、碰撞、控制、观测、命令或评估器存在问题，禁止用 reward 掩盖。
- `reward_contract_blocker`：既有 scale 无法表达所需约束；报告 blocker，不修改 reward 公式。
- `insufficient_evidence`：artifact 不完整或相互冲突。

## Scale 边界

只允许以下已有非零项：

```text
tracking_lin_vel tracking_ang_vel alive contact
lin_vel_z ang_vel_xy orientation base_height torques dof_acc dof_vel
action_rate dof_pos_limits hip_pos contact_no_vel feet_swing_height
```

正 reward 必须保持正值，penalty 必须保持负值，任何项不得设为零。每轮只修改一个权重族，并使用相同定义的双 seed 复验。

稳定性优先于步态结构，步态结构优先于速度跟踪，速度跟踪优先于平滑性和能耗。

## Challenge 记录

每次 challenge 写入 candidate 本地 artifact：

```json
{
  "schema_version": "reward_scale_challenge/v1",
  "run_id": "...",
  "slot_id": "...",
  "candidate_name": "...",
  "milestone_iteration": 10000,
  "checkpoint_path": ".../model_10000.pt",
  "input_artifacts": [],
  "observed_failure": "...",
  "alternative_causes": [],
  "decision": "scale_only_candidate",
  "reward_family": "tracking",
  "scale_delta": {},
  "expected_change": "...",
  "falsification_checks": [],
  "risks": []
}
```

若结论是 `scale_only_candidate`，后续 candidate 仍必须先创建独立 draft PR，并通过配置校验后才能启动。
