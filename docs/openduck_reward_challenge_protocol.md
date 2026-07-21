# OpenDuck Reward Challenge 协议

该协议是 Codex 主导 OpenDuckPro2 reward 修改时的强制流程。

## 负责人模型

Codex 仍然是优化负责人。训练脚本只负责执行训练和评估。reward challenge subagent 是在 reward 源码修改前使用的独立批评者，用于挑战主 Codex 的修改假设。subagent 不决定最终训练动作，也不编辑 reward 文件，除非主 Codex 明确委托一个边界清晰的 patch。

## 触发条件

在修改以下内容前，必须运行一次 reward challenge：

- `legged_gym/envs/openduckpro2/openduckpro2_env.py`
- `legged_gym/envs/openduckpro2/openduckpro2_config.py`
- 任意 `_reward_*` 函数、reward 默认 scale、reward helper，或 reward 相关的 phase/contact/height 逻辑

当完成的 rollout 有结构性失败，但 Codex 想避免源码修改、只做 scale-only 决策时，也必须运行 challenge。challenge 必须说明为什么 scale-only 是允许的。

1K/2K 阶段的 gait failure 通常不足以触发源码修改 challenge，除非存在训练健康证据，例如 NaN、无 checkpoint、进程失败，或 reward 明显发散。1K/2K 阶段主要判断训练是否健康。

## Challenge 必答问题

每次 challenge 必须回答：

1. 检查了哪些 artifacts：`evaluation.json`、`trajectory.csv`、`trajectory.svg`、`train.log`、`evaluate.log`、simlog issues、checkpoint、milestone iteration。
2. 失败现象是什么：fall、base height collapse、tilt、left/right non-alternation、same-foot repeats、foot dragging、foot-reference error、tracking error、energy、smoothness、action saturation、reward shortcut。
3. 可能 root cause 属于 missing reward structure、wrong reward formula、coordinate/phase bug、scale imbalance，还是 training health。
4. 哪个文件或 symbol 会改变：target file、reward function、config field、scale name。
5. 为什么拒绝 scale-only，或为什么允许 scale-only。
6. 预期行为变化是什么，下一轮哪些 artifacts 能验证。
7. 副作用和风险是什么。
8. 哪些反证会推翻拟议修改。
9. 最终结论必须是以下之一：`source_change_required`、`scale_only_allowed`、`no_reward_change_continue`、`insufficient_evidence`。

## 何时必须改源码

出现以下证据时，不要只做 scale-only：

- 左右脚不交替、同一只脚重复抬脚、同步 shuffle，或缺失 swing/contact sequence 约束。
- base height collapse、蹲伏行走、fall gate，或 reward 行为导致的持续 tilt。
- foot trajectory/reference mismatch 可能来自 frame、phase、reference-height 或 foot-index bug。
- reward shortcut，例如原地站立、拖脚、过度接触，或 energy minimization 压制 gait。
- 现有 reward 的 sign、normalization、mask、phase gate、coordinate frame 或左右脚 indexing 可能错误。
- simlog 暴露缺失约束，例如 swing clearance、anti-drag、anti-double-contact 或 anti-tilt。
- 3K/4K rollout 即使 scalar score 非零，仍然缺少有效 gait structure。

只有 rollout 结构已经有效时，才允许 scale-only：没有 fall、存在左右交替、高度合理、foot reference 大致匹配，剩余问题只是 tracking、energy、smoothness 或 stability margin 的权重权衡。

## Artifact 契约

每次 challenge 都必须写入 candidate-local artifact：

```text
auto_train_runs/<run_id>/cycle_000/<candidate>/reward_challenge.json
```

使用以下 schema 形状：

```json
{
  "schema_version": "reward_challenge/v1",
  "timestamp": "ISO-8601",
  "run_id": "...",
  "candidate_id": "...",
  "milestone_iteration": 3000,
  "checkpoint": "model_3000.pt",
  "challenge_agent": {
    "agent_type": "subagent",
    "agent_id": "...",
    "role": "reward_critic"
  },
  "source_files_under_review": [
    "legged_gym/envs/openduckpro2/openduckpro2_env.py",
    "legged_gym/envs/openduckpro2/openduckpro2_config.py"
  ],
  "input_artifacts": {
    "evaluation_json": "...",
    "trajectory_csv": "...",
    "trajectory_svg": "...",
    "train_log": "...",
    "evaluate_log": "..."
  },
  "evidence_summary": {
    "score": 0.0,
    "fall_gate": false,
    "base_height": "...",
    "tilt": "...",
    "left_right_alternation": "...",
    "same_foot_repeats": "...",
    "foot_reference_error": "...",
    "velocity_tracking": "...",
    "energy": "...",
    "smoothness": "...",
    "action_saturation": "..."
  },
  "diagnosis": {
    "primary_failure": "...",
    "root_cause_hypothesis": "...",
    "scale_only_rejected_reason": "...",
    "alternative_explanations": []
  },
  "proposed_reward_change": {
    "decision": "source_change_required",
    "target_file": "...",
    "target_symbols": [],
    "added_rewards": [],
    "removed_rewards": [],
    "modified_rewards": [],
    "scale_changes": [],
    "expected_behavior_change": "..."
  },
  "falsification_checks": [],
  "validation_plan": [],
  "risk_notes": [],
  "challenge_result": {
    "approved_for_main_codex": true,
    "confidence": "medium",
    "required_before_editing": []
  }
}
```

## Decision 引用

任何 challenge 之后写入的 `codex_decision.json` 都必须包含：

```json
{
  "reward_challenge": {
    "artifact": "reward_challenge.json",
    "decision": "source_change_required",
    "candidate_id": "...",
    "primary_failure": "...",
    "scale_only_rejected_reason": "...",
    "accepted_by_main_codex": true
  }
}
```

如果 Codex 拒绝 challenge，必须包含 `accepted_by_main_codex=false`，并写出明确的 `override_reason`。

## Anti-Agreement 检查清单

challenge 必须主动寻找不应该修改拟议 reward 的理由：

- 这是否只是训练太早，尤其是 1K/2K？
- checkpoint、evaluator、command sequence、terrain、initial state 或 termination config 是否可以解释这个现象？
- 是否存在 scalar score 更低但 gait structure 更健康的 candidate？
- 拟议 reward 是否可能制造原地站立、拖脚、过度抬脚或牺牲速度的 shortcut？
- scale tuning 是否在掩盖 formula、frame 或 phase bug？
- 删除或削弱某个 reward 是否会移除 stability、energy 或 smoothness 保护？
- trajectory 和 simlog 是否冲突？如果冲突，标记为 `insufficient_evidence`。
- 是否有更小的源码修改可以验证该假设？
- 下一轮什么证据会证明这次 challenge 是错的？
- 如果主 Codex 已经倾向某个修改，challenge 仍必须给出至少一个可测试的反论点。
