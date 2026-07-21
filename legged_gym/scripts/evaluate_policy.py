import argparse
import math
import sys
from pathlib import Path

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

from auto_train_common import apply_reward_overrides, load_json, write_json


DEFAULT_COMMAND_PLAN = [
    {"name": "forward_slow", "vx": 0.12, "vy": 0.0, "yaw": 0.0, "steps": 400},
    {"name": "forward_turn", "vx": 0.10, "vy": 0.0, "yaw": 0.35, "steps": 300},
    {"name": "lateral", "vx": 0.0, "vy": 0.08, "yaw": 0.0, "steps": 300},
]

DEFAULT_SCORE_WEIGHTS = {
    "survival": 0.20,
    "velocity_tracking": 0.13,
    "yaw_tracking": 0.05,
    "height_stability": 0.13,
    "upright": 0.10,
    "foot_trajectory": 0.12,
    "foot_alternation": 0.07,
    "single_swing_topology": 0.15,
    "energy": 0.025,
    "smoothness": 0.025,
}

TOPOLOGY_GATE_DEFAULTS = {
    "strict_double_lift_max": 25,
    "strict_sample_same_foot_max": 0,
    "admissible_double_lift_max": 100,
    "admissible_sample_same_foot_max": 2,
    "admissible_topology_cap": 0.35,
    "strict_debounced_same_foot_repeat_rate_max": 0.02,
    "admissible_debounced_same_foot_repeat_rate_max": 0.10,
    "admissible_fall_env_rate_max": 0.05,
}

TOPOLOGY_FULL_GATE_MIN_ITERATION = 7000
CONTACT_DEBOUNCE_SECONDS = 0.06


def parse_custom_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reward-overrides")
    parser.add_argument("--output", required=True)
    parser.add_argument("--command-plan")
    parser.add_argument("--score-config")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--sample-env", type=int, default=0)
    parser.add_argument("--trajectory-stride", type=int, default=10)
    parser.add_argument("--trajectory-output")
    parser.add_argument("--plot-output")
    parser.add_argument("--foot-review-warmup-seconds", type=float, default=1.0)
    parser.add_argument("--milestone-iteration", type=int)
    custom, remaining = parser.parse_known_args(argv)
    return custom, remaining


def actor_obs(observations):
    if isinstance(observations, tuple):
        return observations[0]
    return observations


def unpack_step(step_result):
    if len(step_result) == 5:
        obs, _, rews, dones, infos = step_result
        return obs, rews, dones, infos
    return step_result


def tensor_mean(value):
    return float(torch.mean(value.detach().float()).item())


def to_list(value):
    return [float(x) for x in value.detach().cpu().tolist()]


def _values(rows, key):
    return [float(row[key]) for row in rows if key in row]


def _mean(values):
    return sum(values) / max(1, len(values))


def _minmax(values):
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": _mean(values)}


def _moving_average(values, radius=1):
    if not values:
        return []
    return [
        _mean(values[max(0, index - radius) : min(len(values), index + radius + 1)])
        for index in range(len(values))
    ]


def _complete_swing_segments(rows, foot, swing_threshold):
    phase_key = f"{foot}_phase"
    height_key = f"{foot}_relative_z"
    segments = []
    current = []
    saw_stance = False
    current_episode = None

    for row in rows:
        if row.get("topology_reset"):
            current = []
            saw_stance = False
            current_episode = None
            continue
        episode_id = row.get("sample_episode_id")
        if current_episode is not None and episode_id != current_episode:
            current = []
            saw_stance = False
        current_episode = episode_id
        raw_phase = row.get(phase_key)
        raw_height = row.get(height_key)
        if raw_phase in (None, "") or raw_height in (None, ""):
            current = []
            saw_stance = False
            continue

        phase = float(raw_phase)
        if phase < swing_threshold:
            if current and saw_stance and len(current) >= 4:
                segments.append(current)
            current = []
            saw_stance = True
            continue

        progress = (phase - swing_threshold) / max(1.0 - swing_threshold, 1e-6)
        if current and progress < current[-1][0]:
            if saw_stance and len(current) >= 4:
                segments.append(current)
            current = []
        current.append((progress, float(raw_height)))

    return segments


def summarize_single_swing_topology(
    rows,
    swing_threshold,
    min_peak_clearance,
    min_peak_separation,
    mid_drop_threshold,
):
    per_foot = {}
    all_drops = []

    for foot in ("left", "right"):
        complete_segments = _complete_swing_segments(rows, foot, swing_threshold)
        violations = 0
        foot_drops = []
        examples = []
        for segment in complete_segments:
            progress = [point[0] for point in segment]
            heights = _moving_average([point[1] for point in segment])
            peaks = [
                index
                for index in range(1, len(heights) - 1)
                if 0.10 <= progress[index] <= 0.90
                and heights[index] >= heights[index - 1]
                and heights[index] > heights[index + 1]
                and heights[index] >= min_peak_clearance
            ]
            max_drop = 0.0
            for first_peak, second_peak in zip(peaks, peaks[1:]):
                if progress[second_peak] - progress[first_peak] < min_peak_separation:
                    continue
                valley_index = min(
                    range(first_peak, second_peak + 1), key=lambda index: heights[index]
                )
                if not 0.25 <= progress[valley_index] <= 0.75:
                    continue
                mid_drop = min(heights[first_peak], heights[second_peak]) - heights[valley_index]
                max_drop = max(max_drop, mid_drop)
                if mid_drop >= mid_drop_threshold:
                    break
            if max_drop >= mid_drop_threshold:
                violations += 1
                foot_drops.append(max_drop)
                examples.append(
                    {
                        "max_mid_drop": max_drop,
                        "steps": len(segment),
                    }
                )
        per_foot[foot] = {
            "complete_count": len(complete_segments),
            "multi_peak_count": violations,
            "multi_peak_rate": violations / max(1, len(complete_segments)),
            "mid_drop_mean": _mean(foot_drops) if foot_drops else 0.0,
            "mid_drop_max": max(foot_drops, default=0.0),
            "examples": examples[:8],
        }
        all_drops.extend(foot_drops)

    complete_count = sum(result["complete_count"] for result in per_foot.values())
    multi_peak_count = sum(result["multi_peak_count"] for result in per_foot.values())
    violation_rate = multi_peak_count / max(1, complete_count)
    mean_excess_drop = _mean(
        [max(0.0, drop - mid_drop_threshold) for drop in all_drops]
    ) if all_drops else 0.0
    worst_foot = max(
        per_foot, key=lambda foot: per_foot[foot]["multi_peak_rate"]
    )
    worst_drop_foot = max(
        per_foot, key=lambda foot: per_foot[foot]["mid_drop_max"]
    )
    worst_foot_rate = per_foot[worst_foot]["multi_peak_rate"]
    worst_foot_drop = per_foot[worst_drop_foot]["mid_drop_max"]
    topology_score = max(0.0, 1.0 - worst_foot_rate) * math.exp(
        -mean_excess_drop / 0.020
    )
    mid_drop_max = max((result["mid_drop_max"] for result in per_foot.values()), default=0.0)
    return {
        "eligible": min(result["complete_count"] for result in per_foot.values()) >= 6,
        "complete_count": complete_count,
        "multi_peak_count": multi_peak_count,
        "multi_peak_rate": violation_rate,
        "worst_foot": worst_foot,
        "worst_foot_multi_peak_rate": worst_foot_rate,
        "worst_drop_foot": worst_drop_foot,
        "worst_foot_mid_drop_max": worst_foot_drop,
        "mid_drop_mean": _mean(all_drops) if all_drops else 0.0,
        "mid_drop_max": mid_drop_max,
        "score": topology_score,
        "per_foot": per_foot,
    }


def _split_review_segments(rows):
    segments = []
    current = []
    current_episode = None
    for row in rows:
        if row.get("topology_reset"):
            if current:
                segments.append(current)
            current = []
            current_episode = None
            continue
        if int(row.get("foot_review_ready", 1) or 0) != 1:
            continue
        episode_id = row.get("sample_episode_id")
        if current and episode_id != current_episode:
            segments.append(current)
            current = []
        current_episode = episode_id
        current.append(row)
    if current:
        segments.append(current)
    return segments


def summarize_continuous_m_shape(
    rows,
    swing_threshold,
    peak_min=0.015,
    drop_min=0.008,
    drop_ratio_min=0.25,
    rebound_min=0.006,
    min_peak_separation_steps=4,
):
    per_foot = {}
    total_events = 0
    all_drops = []
    review_segments = _split_review_segments(rows)

    for foot in ("left", "right"):
        key = f"{foot}_relative_z"
        phase_key = f"{foot}_phase"
        events = []
        phase_segments = []
        for review_segment in review_segments:
            current = []
            previous_phase = None
            for row in review_segment:
                phase = float(row[phase_key])
                if (
                    phase < swing_threshold
                    or (previous_phase is not None and phase < previous_phase)
                ):
                    if current:
                        phase_segments.append(current)
                    current = []
                else:
                    current.append(row)
                previous_phase = phase
            if current:
                phase_segments.append(current)
        for segment in phase_segments:
            values = _moving_average([float(row[key]) for row in segment], radius=1)
            steps = [int(float(row["step"])) for row in segment]
            if len(values) < 7:
                continue
            peaks = [
                index
                for index in range(1, len(values) - 1)
                if values[index] >= values[index - 1]
                and values[index] > values[index + 1]
                and values[index] >= peak_min
            ]
            for first_peak, second_peak in zip(peaks, peaks[1:]):
                if second_peak - first_peak < min_peak_separation_steps:
                    continue
                valley_index = min(
                    range(first_peak, second_peak + 1),
                    key=lambda index: values[index],
                )
                valley = values[valley_index]
                p1 = values[first_peak]
                p2 = values[second_peak]
                drop_abs = min(p1, p2) - valley
                drop_ratio = drop_abs / max(min(p1, p2), 1e-6)
                rebound = p2 - valley
                if (
                    p1 >= peak_min
                    and p2 >= peak_min
                    and rebound >= rebound_min
                    and (drop_abs >= drop_min or drop_ratio >= drop_ratio_min)
                ):
                    events.append(
                        {
                            "start_step": steps[first_peak],
                            "valley_step": steps[valley_index],
                            "end_step": steps[second_peak],
                            "first_peak": p1,
                            "valley": valley,
                            "second_peak": p2,
                            "drop_abs": drop_abs,
                            "drop_ratio": drop_ratio,
                            "rebound": rebound,
                        }
                    )
        drops = [event["drop_abs"] for event in events]
        per_foot[foot] = {
            "phase_segment_count": len(phase_segments),
            "event_count": len(events),
            "max_drop": max(drops, default=0.0),
            "mean_drop": _mean(drops) if drops else 0.0,
            "examples": events[:8],
        }
        total_events += len(events)
        all_drops.extend(drops)

    max_drop = max(all_drops, default=0.0)
    worst_foot = max(per_foot, key=lambda foot: per_foot[foot]["max_drop"])
    return {
        "eligible": sum(
            result["phase_segment_count"] for result in per_foot.values()
        ) >= 4,
        "review_segment_count": len(review_segments),
        "event_count": total_events,
        "max_drop": max_drop,
        "mean_drop": _mean(all_drops) if all_drops else 0.0,
        "worst_foot": worst_foot,
        "stable_m_risk": total_events >= 2 or max_drop >= drop_min,
        "thresholds": {
            "peak_min": peak_min,
            "drop_min": drop_min,
            "drop_ratio_min": drop_ratio_min,
            "rebound_min": rebound_min,
            "min_peak_separation_steps": min_peak_separation_steps,
        },
        "per_foot": per_foot,
    }


def summarize_sample_trajectory(rows, base_height_target):
    if not rows:
        return {
            "available": False,
            "lift_sequence": [],
            "same_foot_repeat_count": 0,
        }

    lift_sequence = []
    same_foot_repeat_count = 0
    previous = None
    last_lifted = None
    both_air_count = 0
    both_contact_count = 0
    low_height_count = 0
    left_ref_errors = []
    right_ref_errors = []

    for row in rows:
        left_contact = int(row["left_contact"])
        right_contact = int(row["right_contact"])
        if not left_contact and not right_contact:
            both_air_count += 1
        if left_contact and right_contact:
            both_contact_count += 1
        if float(row["base_height"]) < base_height_target * 0.55:
            low_height_count += 1

        left_ref_errors.append(abs(float(row["left_foot_z"]) - float(row["left_ref_z"])))
        right_ref_errors.append(abs(float(row["right_foot_z"]) - float(row["right_ref_z"])))

        current = (left_contact, right_contact)
        if previous is not None:
            for side, prev_contact, current_contact in (
                ("left", previous[0], current[0]),
                ("right", previous[1], current[1]),
            ):
                if prev_contact and not current_contact:
                    lift_sequence.append(side)
                    if last_lifted == side:
                        same_foot_repeat_count += 1
                    last_lifted = side
        previous = current

    return {
        "available": True,
        "base_height": _minmax(_values(rows, "base_height")),
        "base_roll_abs_max": max((abs(x) for x in _values(rows, "base_roll")), default=0.0),
        "base_pitch_abs_max": max((abs(x) for x in _values(rows, "base_pitch")), default=0.0),
        "left_foot_z": _minmax(_values(rows, "left_foot_z")),
        "right_foot_z": _minmax(_values(rows, "right_foot_z")),
        "left_ref_error_mean": _mean(left_ref_errors),
        "right_ref_error_mean": _mean(right_ref_errors),
        "both_air_rate": both_air_count / max(1, len(rows)),
        "both_contact_rate": both_contact_count / max(1, len(rows)),
        "low_height_rate": low_height_count / max(1, len(rows)),
        "lift_sequence": lift_sequence[:80],
        "left_lift_count": lift_sequence.count("left"),
        "right_lift_count": lift_sequence.count("right"),
        "same_foot_repeat_count": same_foot_repeat_count,
        "alternating_lift_sequence_ok": same_foot_repeat_count == 0 and len(lift_sequence) >= 2,
    }


def build_simlog(metrics, score, rows, base_height_target, swing_height_target):
    sample = summarize_sample_trajectory(rows, base_height_target)
    issues = []
    diagnostic_only = bool(score.get("topology_diagnostic_only"))
    topology_reward_action = (
        "Record the single-swing topology issue for trend review only; do not stop this 1K-6K maturation candidate for topology evidence alone."
        if diagnostic_only
        else "Treat persistent mature topology failure as a G1 reward-contract blocker; do not change reward names or formulas. Only a new scale-only candidate may be tried."
    )

    def add_issue(
        code,
        severity,
        evidence,
        codex_reward_action,
        severity_label=None,
        **metadata,
    ):
        issue = {
            "code": code,
            "severity": severity,
            "evidence": evidence,
            "codex_reward_action": codex_reward_action,
        }
        if severity_label:
            issue["severity_label"] = severity_label
        issue.update(metadata)
        issues.append(issue)

    if metrics["fall_detected"]:
        fall_is_gate = bool(score.get("fall_gate"))
        add_issue(
            "fall_detected",
            "blocker" if fall_is_gate else "diagnostic",
            {
                "fall_count": metrics["fall_count"],
                "fall_rate": metrics["fall_rate"],
                "fall_env_rate": metrics.get("fall_env_rate"),
                "done_rate": metrics["done_rate"],
                "diagnostic_only": not fall_is_gate,
            },
            (
                "Treat this as a mature-policy stability gate."
                if fall_is_gate
                else "Record the low-rate early fall for trend review; do not reset this maturing checkpoint from an any-fall boolean."
            ),
        )
    if metrics["mean_height_error"] > 0.06 or sample.get("low_height_rate", 0.0) > 0.0:
        add_issue(
            "base_height_unstable",
            "high",
            {
                "target": base_height_target,
                "mean_height_error": metrics["mean_height_error"],
                "sample_base_height": sample.get("base_height"),
                "sample_low_height_rate": sample.get("low_height_rate"),
            },
            "Use only the existing base_height and orientation scales in a new scale-only candidate; do not change reward formulas.",
        )
    if metrics["mean_tilt_error"] > 0.25 or sample.get("base_roll_abs_max", 0.0) > 0.8 or sample.get("base_pitch_abs_max", 0.0) > 1.0:
        add_issue(
            "upright_tilt_unstable",
            "high",
            {
                "mean_tilt_error": metrics["mean_tilt_error"],
                "sample_roll_abs_max": sample.get("base_roll_abs_max"),
                "sample_pitch_abs_max": sample.get("base_pitch_abs_max"),
            },
            "Use only the existing orientation, ang_vel_xy, and base_height scales in a new scale-only candidate; do not add reward logic.",
        )
    alternation_score = score["nodes"]["foot_alternation"]["score"]
    debounced_repeat_rate = metrics.get("debounced_same_foot_repeat_rate")
    alternation_failed = alternation_score < 0.80 or (
        not diagnostic_only
        and debounced_repeat_rate is not None
        and float(debounced_repeat_rate) > 0.10
    )
    if alternation_failed:
        add_issue(
            "foot_alternation_failed",
            "diagnostic" if diagnostic_only else "high",
            {
                "node_score": alternation_score,
                "metric_source": score.get("gait_contact_metric_source"),
                "contact_debounce_seconds": metrics.get("contact_debounce_seconds"),
                "debounced_lift_event_count": metrics.get("debounced_lift_event_count"),
                "debounced_same_foot_repeat_count": metrics.get(
                    "debounced_same_foot_repeat_count"
                ),
                "debounced_same_foot_repeat_rate": debounced_repeat_rate,
                "debounced_lift_events_per_cycle": metrics.get(
                    "debounced_lift_events_per_cycle"
                ),
                "raw_lift_event_count": metrics["lift_event_count"],
                "raw_double_lift_violation_count": metrics[
                    "double_lift_violation_count"
                ],
            },
            (
                "Record the immature alternation metric for trend review only."
                if diagnostic_only
                else "Treat persistent mature alternation failure as a G1 reward-contract blocker; only scale existing contact, contact_no_vel, and feet_swing_height rewards."
            ),
            diagnostic_only=diagnostic_only,
            blocks_candidate=not diagnostic_only,
        )
    topology = metrics.get("single_swing_topology", {})
    continuous_m_shape = metrics.get("continuous_m_shape", {})
    if topology.get("eligible") and (
        topology.get("worst_foot_multi_peak_rate", 0.0) >= 0.25
        or topology.get("worst_foot_mid_drop_max", 0.0) >= 0.015
    ):
        critical = (
            topology.get("worst_foot_multi_peak_rate", 0.0) >= 0.50
            or topology.get("worst_foot_mid_drop_max", 0.0) >= 0.030
        )
        add_issue(
            "single_swing_multi_peak_m_trajectory",
            "critical" if critical else "high",
            {
                "left": topology.get("per_foot", {}).get("left", {}),
                "right": topology.get("per_foot", {}).get("right", {}),
                "multi_peak_rate": topology.get("multi_peak_rate"),
                "worst_foot": topology.get("worst_foot"),
                "worst_foot_multi_peak_rate": topology.get("worst_foot_multi_peak_rate"),
                "worst_foot_mid_drop_max": topology.get("worst_foot_mid_drop_max"),
                "mid_drop_mean": topology.get("mid_drop_mean"),
                "mid_drop_max": topology.get("mid_drop_max"),
                "single_swing_topology_score": topology.get("score"),
                "pre_topology_gate_total_score": score.get("pre_topology_gate_total_score"),
                "total_score": score.get("total_score"),
                "score_cap": score.get("topology_score_cap"),
                "diagnostic_only": score.get("topology_diagnostic_only"),
                "foot_review_warmup_seconds": metrics.get("foot_review_warmup_seconds"),
                "foot_review_min_completed_swings": metrics.get("foot_review_min_completed_swings"),
                "foot_review_skipped_steps": metrics.get("foot_review_skipped_steps"),
            },
            topology_reward_action,
            "特别严重" if critical else None,
            diagnostic_only=diagnostic_only,
            blocks_candidate=not diagnostic_only,
        )
    if continuous_m_shape.get("eligible") and continuous_m_shape.get("stable_m_risk"):
        add_issue(
            "continuous_m_shape_trajectory",
            "critical",
            {
                "event_count": continuous_m_shape.get("event_count"),
                "max_drop": continuous_m_shape.get("max_drop"),
                "mean_drop": continuous_m_shape.get("mean_drop"),
                "worst_foot": continuous_m_shape.get("worst_foot"),
                "left": continuous_m_shape.get("per_foot", {}).get("left", {}),
                "right": continuous_m_shape.get("per_foot", {}).get("right", {}),
                "thresholds": continuous_m_shape.get("thresholds"),
                "pre_topology_gate_total_score": score.get("pre_topology_gate_total_score"),
                "total_score": score.get("total_score"),
                "score_cap": score.get("topology_score_cap"),
                "diagnostic_only": score.get("topology_diagnostic_only"),
                "foot_review_warmup_seconds": metrics.get("foot_review_warmup_seconds"),
                "foot_review_min_completed_swings": metrics.get("foot_review_min_completed_swings"),
            },
            "Treat persistent mature M-shape behavior as a G1 reward-contract blocker; do not add or rewrite rewards. A replacement may change only existing reward scales.",
            "特别严重",
            diagnostic_only=diagnostic_only,
            blocks_candidate=not diagnostic_only,
        )
    if metrics["mean_foot_ref_error"] > 0.04:
        add_issue(
            "foot_reference_tracking_error",
            "medium",
            {
                "mean_foot_ref_error": metrics["mean_foot_ref_error"],
                "left_ref_error_mean": sample.get("left_ref_error_mean"),
                "right_ref_error_mean": sample.get("right_ref_error_mean"),
                "swing_height_target": swing_height_target,
            },
            "Use only the existing feet_swing_height, contact, and contact_no_vel scales in a new candidate; do not modify reward source logic.",
        )
    if metrics["both_feet_contact_rate"] > 0.75:
        add_issue(
            "shuffling_or_no_clear_swing",
            "medium",
            {
                "both_feet_contact_rate": metrics["both_feet_contact_rate"],
                "sample_both_contact_rate": sample.get("both_contact_rate"),
            },
            "Use only existing contact, contact_no_vel, and feet_swing_height scales; do not add a clearance reward.",
        )
    if metrics["mean_lin_vel_error"] > 0.15:
        add_issue(
            "velocity_tracking_error",
            "medium",
            {
                "mean_lin_vel_error": metrics["mean_lin_vel_error"],
                "mean_yaw_error": metrics["mean_yaw_error"],
            },
            "Inspect command-specific windows and adjust only existing tracking_lin_vel or tracking_ang_vel scales in a new candidate.",
        )

    return {
        "summary": {
            "total_score": score["total_score"],
            "fall_gate": score["fall_gate"],
            "base_height_target": base_height_target,
            "swing_height_target": swing_height_target,
            "single_swing_topology": topology,
            "continuous_m_shape": continuous_m_shape,
            "pre_topology_gate_total_score": score.get("pre_topology_gate_total_score"),
            "topology_score_cap": score.get("topology_score_cap"),
            "topology_diagnostic_only": score.get("topology_diagnostic_only"),
            "foot_review_warmup_seconds": metrics.get("foot_review_warmup_seconds"),
            "foot_review_min_completed_swings": metrics.get("foot_review_min_completed_swings"),
            "foot_review_skipped_steps": metrics.get("foot_review_skipped_steps"),
        },
        "sample_trajectory": sample,
        "issues": issues,
        "codex_policy": {
            "decision_owner": "codex",
            "rule": "Use simlog evidence to tune only existing nonzero reward scales; report a blocker when structural failure cannot be expressed by the frozen reward contract.",
            "expected_next_step": "Codex reviews evaluation.json, trajectory.csv/svg, train.log, and evaluate.log, then records a scale-only decision or blocker before creating a PR-backed candidate.",
        },
    }


def reference_foot_heights(
    phases,
    swing_threshold,
    stance_heights,
    target_height,
    profile_power=1.0,
    min_clearance=0.0,
):
    swing_den = max(1.0 - swing_threshold, 1e-6)
    progress = torch.clamp((phases - swing_threshold) / swing_den, 0.0, 1.0)
    swing_amplitude = torch.clamp(
        torch.full_like(stance_heights, target_height) - stance_heights,
        min=min_clearance,
    )
    return torch.where(
        phases >= swing_threshold,
        stance_heights
        + swing_amplitude
        * torch.pow(torch.clamp(torch.sin(progress * math.pi), min=0.0), profile_power),
        stance_heights,
    )


def resolve_foot_columns(env):
    foot_handles = [int(x) for x in env.feet_indices.detach().cpu().tolist()]
    resolved = {}
    for side, body_name in (("left", "left_ankle_pitch_link"), ("right", "right_ankle_pitch_link")):
        handle = env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], body_name)
        if handle in foot_handles:
            resolved[side] = foot_handles.index(handle)
    if "left" not in resolved or "right" not in resolved:
        resolved = {"left": 0, "right": min(1, len(foot_handles) - 1)}
    return [resolved["left"], resolved["right"]]


def write_trajectory_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "step",
        "command_name",
        "base_height",
        "base_roll",
        "base_pitch",
        "base_yaw",
        "left_foot_x",
        "left_foot_y",
        "left_foot_z",
        "right_foot_x",
        "right_foot_y",
        "right_foot_z",
        "left_ref_z",
        "right_ref_z",
        "left_phase",
        "right_phase",
        "left_stance_z",
        "right_stance_z",
        "left_relative_z",
        "right_relative_z",
        "foot_review_ready",
        "sample_reset",
        "left_contact",
        "right_contact",
        "action_abs_mean",
        "action_abs_max",
        "raw_torque_abs_mean",
        "raw_torque_abs_max",
        "torque_abs_max",
        "torque_saturation_rate",
        "torque_clip_excess_mean",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(column, "")) for column in columns) + "\n")


def _series_points(rows, key):
    return [(float(row["step"]), float(row[key])) for row in rows]


def _polyline(points, x_min, x_max, y_min, y_max, left, top, width, height):
    if not points:
        return ""
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    coords = []
    for x, y in points:
        px = left + (x - x_min) / x_span * width
        py = top + height - (y - y_min) / y_span * height
        coords.append(f"{px:.1f},{py:.1f}")
    return " ".join(coords)


def write_trajectory_svg(path, rows, base_height_target, swing_height_target):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_min = float(rows[0]["step"])
    x_max = float(rows[-1]["step"])
    width = 1200
    panel_height = 190
    left = 70
    plot_width = 1060
    colors = {
        "base": "#2f6f9f",
        "target": "#7a7a7a",
        "left": "#d45500",
        "right": "#007a5a",
        "left_ref": "#f0a35e",
        "right_ref": "#66bfa0",
    }

    panels = [
        (
            "Base height",
            [
                ("base_height", colors["base"], "base z"),
            ],
            [base_height_target],
        ),
        (
            "Foot link height vs reference",
            [
                ("left_foot_z", colors["left"], "left foot z"),
                ("right_foot_z", colors["right"], "right foot z"),
                ("left_ref_z", colors["left_ref"], "left ref z"),
                ("right_ref_z", colors["right_ref"], "right ref z"),
            ],
            [swing_height_target],
        ),
        (
            "Foot x trajectory",
            [
                ("left_foot_x", colors["left"], "left foot x"),
                ("right_foot_x", colors["right"], "right foot x"),
            ],
            [],
        ),
    ]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="720" viewBox="0 0 {width} 720">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:monospace;font-size:14px}.title{font-size:18px;font-weight:bold}.axis{stroke:#999;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.line{fill:none;stroke-width:2}</style>',
        '<text x="70" y="32" class="title">OpenDuckPro2 rollout trajectory</text>',
    ]
    top = 60
    for title, series, reference_values in panels:
        values = []
        for key, _, _ in series:
            values.extend(float(row[key]) for row in rows)
        values.extend(reference_values)
        y_min = min(values)
        y_max = max(values)
        pad = max((y_max - y_min) * 0.12, 0.02)
        y_min -= pad
        y_max += pad
        svg.append(f'<text x="{left}" y="{top - 12}" class="title">{title}</text>')
        svg.append(f'<line x1="{left}" y1="{top + panel_height}" x2="{left + plot_width}" y2="{top + panel_height}" class="axis"/>')
        svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}" class="axis"/>')
        svg.append(f'<text x="12" y="{top + 12}">{y_max:.3f}</text>')
        svg.append(f'<text x="12" y="{top + panel_height}">{y_min:.3f}</text>')
        for ref in reference_values:
            y = top + panel_height - (ref - y_min) / max(y_max - y_min, 1e-6) * panel_height
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="{colors["target"]}" stroke-dasharray="6 6"/>')
        legend_x = left + 740
        for idx, (key, color, label) in enumerate(series):
            points = _series_points(rows, key)
            svg.append(
                f'<polyline class="line" stroke="{color}" points="{_polyline(points, x_min, x_max, y_min, y_max, left, top, plot_width, panel_height)}"/>'
            )
            svg.append(f'<line x1="{legend_x}" y1="{top + idx * 20}" x2="{legend_x + 24}" y2="{top + idx * 20}" stroke="{color}" stroke-width="3"/>')
            svg.append(f'<text x="{legend_x + 32}" y="{top + idx * 20 + 5}">{label}</text>')
        top += panel_height + 50
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def score_metrics(metrics, score_config, allow_topology_gate=True):
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    weights.update((score_config or {}).get("weights", {}))
    if not allow_topology_gate:
        weights["single_swing_topology"] = 0.0
    gate_cfg = dict(TOPOLOGY_GATE_DEFAULTS)
    gate_cfg.update((score_config or {}).get("topology_gate", {}))
    fall_metric_name = (
        "fall_episode_rate" if "fall_episode_rate" in metrics else "fall_env_rate"
    )
    fall_metric_value = float(
        metrics.get(
            fall_metric_name,
            metrics.get("fall_count", 0) / max(1, metrics.get("num_envs", 1)),
        )
    )
    fall_gate = allow_topology_gate and fall_metric_value > float(
        gate_cfg["admissible_fall_env_rate_max"]
    )

    nodes = {
        "survival": max(0.0, min(1.0, 1.0 - fall_metric_value)),
        "velocity_tracking": math.exp(-metrics["mean_lin_vel_error"] / 0.25),
        "yaw_tracking": math.exp(-metrics["mean_yaw_error"] / 0.50),
        "height_stability": math.exp(-metrics["mean_height_error"] / 0.08),
        "upright": math.exp(-metrics["mean_tilt_error"] / 0.40),
        "foot_trajectory": math.exp(-metrics["mean_foot_ref_error"] / 0.06),
        "foot_alternation": metrics.get(
            "debounced_foot_alternation_score",
            metrics["foot_alternation_score"],
        ),
        "single_swing_topology": metrics.get("single_swing_topology_score", 1.0),
        "energy": math.exp(-metrics["mean_abs_torque"] / 15.0),
        "smoothness": math.exp(-metrics["mean_action_rate"] / 0.30),
    }
    total_weight = sum(max(0.0, float(v)) for v in weights.values()) or 1.0
    total = 0.0
    weighted_nodes = {}
    for name, node_score in nodes.items():
        weight = max(0.0, float(weights.get(name, 0.0)))
        weighted_nodes[name] = {"score": float(node_score), "weight": weight}
        total += weight * node_score
    pre_topology_gate_total_score = total / total_weight
    topology_score_cap = None
    topology_gate_reasons = []
    if allow_topology_gate and metrics.get("single_swing_topology_eligible", False):
        if (
            metrics.get("single_swing_worst_foot_multi_peak_rate", 0.0) >= 0.50
            or metrics.get("single_swing_worst_foot_mid_drop_max", 0.0) >= 0.030
        ):
            topology_score_cap = 0.45
            topology_gate_reasons.append("critical_single_swing_topology")
        elif (
            metrics.get("single_swing_worst_foot_multi_peak_rate", 0.0) >= 0.25
            or metrics.get("single_swing_worst_foot_mid_drop_max", 0.0) >= 0.015
        ):
            topology_score_cap = 0.65
            topology_gate_reasons.append("single_swing_topology")
    if allow_topology_gate and metrics.get("continuous_m_shape_eligible", False):
        if metrics.get("continuous_m_shape_stable_m_risk", False):
            topology_score_cap = (
                min(topology_score_cap, 0.30)
                if topology_score_cap is not None
                else 0.30
            )
            topology_gate_reasons.append("continuous_m_shape")
    has_debounced_gait_metrics = "debounced_same_foot_repeat_rate" in metrics
    if allow_topology_gate and has_debounced_gait_metrics:
        debounced_repeat_rate = float(
            metrics.get("debounced_same_foot_repeat_rate", 0.0) or 0.0
        )
        if debounced_repeat_rate > float(
            gate_cfg["admissible_debounced_same_foot_repeat_rate_max"]
        ):
            topology_score_cap = (
                min(topology_score_cap, float(gate_cfg["admissible_topology_cap"]))
                if topology_score_cap is not None
                else float(gate_cfg["admissible_topology_cap"])
            )
            topology_gate_reasons.append("debounced_same_foot_repeat")
    elif allow_topology_gate:
        double_lift_count = int(metrics.get("double_lift_violation_count", 0) or 0)
        sample_same_foot_count = int(metrics.get("sample_same_foot_repeat_count", 0) or 0)
        if double_lift_count > int(gate_cfg["admissible_double_lift_max"]):
            topology_score_cap = (
                min(topology_score_cap, float(gate_cfg["admissible_topology_cap"]))
                if topology_score_cap is not None
                else float(gate_cfg["admissible_topology_cap"])
            )
            topology_gate_reasons.append("double_lift_violation")
        if sample_same_foot_count > int(gate_cfg["admissible_sample_same_foot_max"]):
            topology_score_cap = (
                min(topology_score_cap, float(gate_cfg["admissible_topology_cap"]))
                if topology_score_cap is not None
                else float(gate_cfg["admissible_topology_cap"])
            )
            topology_gate_reasons.append("same_foot_repeat")
    total_score = 0.0 if fall_gate else pre_topology_gate_total_score
    if topology_score_cap is not None:
        total_score = min(total_score, topology_score_cap)
    strict_alternation = (
        float(metrics.get("debounced_same_foot_repeat_rate", 0.0) or 0.0)
        <= float(gate_cfg["strict_debounced_same_foot_repeat_rate_max"])
        if has_debounced_gait_metrics
        else (
            int(metrics.get("double_lift_violation_count", 0) or 0)
            <= int(gate_cfg["strict_double_lift_max"])
            and int(metrics.get("sample_same_foot_repeat_count", 0) or 0)
            <= int(gate_cfg["strict_sample_same_foot_max"])
        )
    )
    strict_champion = (
        int(metrics.get("fall_count", 0) or 0) == 0
        and not topology_gate_reasons
        and strict_alternation
        and int(metrics.get("continuous_m_shape_event_count", 0) or 0) == 0
        and int(metrics.get("single_swing_multi_peak_count", 0) or 0) == 0
    )
    return {
        "total_score": total_score,
        "fall_gate": bool(fall_gate),
        "fall_metric_name": fall_metric_name,
        "fall_metric_value": fall_metric_value,
        "fall_diagnostic_only": bool(metrics["fall_detected"] and not fall_gate),
        "topology_gate": topology_score_cap is not None,
        "topology_gate_reasons": topology_gate_reasons,
        "topology_diagnostic_only": not allow_topology_gate,
        "gait_contact_metric_source": (
            "debounced_60ms" if has_debounced_gait_metrics else "legacy_raw"
        ),
        "topology_admissible": not fall_gate and topology_score_cap is None,
        "topology_strict_champion": strict_champion,
        "topology_gate_thresholds": gate_cfg,
        "topology_score_cap": topology_score_cap,
        "pre_topology_gate_total_score": pre_topology_gate_total_score,
        "nodes": weighted_nodes,
    }


def load_command_plan(path, max_steps=None):
    plan = load_json(path, default=DEFAULT_COMMAND_PLAN) or DEFAULT_COMMAND_PLAN
    if max_steps is None:
        return plan

    remaining = max_steps
    clipped = []
    for command in plan:
        if remaining <= 0:
            break
        command = dict(command)
        command["steps"] = min(int(command.get("steps", remaining)), remaining)
        clipped.append(command)
        remaining -= command["steps"]
    return clipped


def evaluate(args, custom):
    command_plan = load_command_plan(custom.command_plan, custom.steps)
    score_config = load_json(custom.score_config, default={}) or {}
    milestone_iteration = custom.milestone_iteration
    if milestone_iteration is None:
        try:
            milestone_iteration = int(getattr(args, "checkpoint", 0) or 0)
        except (TypeError, ValueError):
            milestone_iteration = 0
    topology_diagnostic_only = (
        0 < milestone_iteration < TOPOLOGY_FULL_GATE_MIN_ITERATION
    )

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.num_envs is None:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 64)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = False
    env_cfg.commands.resampling_time = 1e9
    apply_reward_overrides(env_cfg, custom.reward_overrides)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = actor_obs(env.get_observations())

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    foot_columns = resolve_foot_columns(env)
    foot_column_tensor = torch.tensor(foot_columns, dtype=torch.long, device=env.device)
    foot_handle_tensor = env.feet_indices[foot_column_tensor]
    swing_threshold = float(getattr(env.cfg.rewards, "swing_phase_threshold", 0.55))
    swing_height_target = float(getattr(env.cfg.rewards, "swing_height_target", 0.08))
    swing_profile_power = float(getattr(env.cfg.rewards, "swing_profile_power", 1.0))
    swing_min_clearance = float(getattr(env.cfg.rewards, "swing_min_clearance", 0.0))
    torque_group_names = ("leg_yaw", "leg_roll", "leg_pitch", "knee", "ankle")
    torque_group_indices = {
        group_name: torch.tensor(
            [i for i, dof_name in enumerate(env.dof_names) if group_name in dof_name],
            dtype=torch.long,
            device=env.device,
        )
        for group_name in torque_group_names
    }

    sums = {
        "lin_vel_error": 0.0,
        "yaw_error": 0.0,
        "height_error": 0.0,
        "tilt_error": 0.0,
        "foot_ref_error": 0.0,
        "abs_torque": 0.0,
        "raw_abs_torque": 0.0,
        "raw_torque_abs_max": 0.0,
        "torque_saturation_rate": 0.0,
        "torque_clip_excess": 0.0,
        "action_abs": 0.0,
        "action_abs_max": 0.0,
        "action_rate": 0.0,
        "reward": 0.0,
        "done_rate": 0.0,
        "timeout_rate": 0.0,
        "terminal_fall_rate": 0.0,
        "fall_rate": 0.0,
        "both_air_rate": 0.0,
        "both_contact_rate": 0.0,
        "debounced_both_air_rate": 0.0,
        "debounced_both_contact_rate": 0.0,
        "debounced_phase_contact_accuracy": 0.0,
    }
    for group_name in torque_group_names:
        sums[f"{group_name}_torque_saturation_rate"] = 0.0
        sums[f"{group_name}_raw_abs_torque"] = 0.0
    sample = []
    topology_rows = [{"topology_reset": True}]
    total_steps = 0
    sample_env = min(custom.sample_env, env.num_envs - 1)
    base_height_target = float(getattr(env.cfg.rewards, "base_height_target", 0.0))
    fall_detected = False
    fall_count = 0
    fall_episode_count = 0
    episode_reset_count = 0
    fall_seen_by_env = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    fall_episode_latched = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    lift_events = 0
    double_lift_violations = 0
    prev_contacts = None
    contact_debounce_steps = max(
        2,
        int(math.ceil(CONTACT_DEBOUNCE_SECONDS / max(float(env.dt), 1e-6))),
    )
    debounced_contacts = torch.zeros(
        env.num_envs, 2, dtype=torch.bool, device=env.device
    )
    debounce_initialized = torch.zeros_like(debounced_contacts)
    debounce_candidate = torch.zeros_like(debounced_contacts)
    debounce_candidate_steps = torch.zeros(
        env.num_envs, 2, dtype=torch.long, device=env.device
    )
    debounce_touchdown_peak_force = torch.zeros(
        env.num_envs, 2, dtype=torch.float, device=env.device
    )
    debounce_touchdown_impulse = torch.zeros_like(debounce_touchdown_peak_force)
    debounced_lift_events = 0
    debounced_same_foot_repeats = 0
    debounced_alternation_opportunities = 0
    debounced_simultaneous_lift_events = 0
    debounced_touchdown_events = 0
    debounced_touchdown_peak_force_sum = 0.0
    debounced_touchdown_peak_force_max = 0.0
    debounced_touchdown_impulse_sum = 0.0
    reference_swing_completion_events = 0
    debounced_last_lifted = [-1 for _ in range(env.num_envs)]
    foot_stance_z = None
    foot_stance_valid = None
    last_lifted = [-1 for _ in range(env.num_envs)]
    foot_review_warmup_seconds = max(1.0, custom.foot_review_warmup_seconds)
    foot_review_skipped_steps = 0
    foot_review_reset_count = 0
    foot_review_min_completed_swings = 2
    foot_ref_valid_steps = 0
    foot_contact_valid_steps = 0
    foot_review_episode_ids = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    foot_review_steps_since_reset = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    foot_review_completed_swings = torch.zeros(
        env.num_envs, 2, dtype=torch.long, device=env.device
    )
    foot_review_previous_phases = torch.zeros(
        env.num_envs, 2, dtype=torch.float, device=env.device
    )
    foot_review_previous_phase_valid = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )

    with torch.no_grad():
        for command_index, command in enumerate(command_plan):
            vx = float(command.get("vx", 0.0))
            vy = float(command.get("vy", 0.0))
            yaw = float(command.get("yaw", 0.0))
            steps = int(command.get("steps", 0))
            for _ in range(steps):
                env.commands[:, 0] = vx
                env.commands[:, 1] = vy
                env.commands[:, 2] = yaw
                env.compute_observations()
                obs = actor_obs(env.get_observations())
                prev_actions = env.actions.clone()
                actions = policy(obs.detach())
                clip_actions = float(env.cfg.normalization.clip_actions)
                clipped_actions = torch.clip(actions.detach(), -clip_actions, clip_actions)
                raw_torques = (
                    env.p_gains
                    * (
                        clipped_actions * env.cfg.control.action_scale
                        + env.default_dof_pos
                        - env.dof_pos
                    )
                    - env.d_gains * env.dof_vel
                )
                torque_limits = env.torque_limits.unsqueeze(0)
                torque_saturation = torch.abs(raw_torques) >= (torque_limits - 1e-6)
                torque_clip_excess = torch.clamp(torch.abs(raw_torques) - torque_limits, min=0.0)
                obs, rews, dones, infos = unpack_step(env.step(actions.detach()))

                command_xy = torch.tensor([vx, vy], dtype=torch.float, device=env.device)
                lin_error = torch.norm(env.base_lin_vel[:, :2] - command_xy, dim=1)
                yaw_error = torch.abs(env.base_ang_vel[:, 2] - yaw)
                height_error = torch.abs(env.base_pos[:, 2] - base_height_target)
                tilt_error = torch.abs(env.rpy[:, 0]) + torch.abs(env.rpy[:, 1])
                action_rate = torch.norm(env.actions - prev_actions, dim=1)
                foot_pos = env.feet_pos[:, foot_column_tensor, :]
                foot_z = foot_pos[:, :, 2]
                contacts = torch.norm(env.contact_forces[:, foot_handle_tensor, :3], dim=2) > 1.0
                reset_mask = dones.bool()
                reset_mask_2d = reset_mask.unsqueeze(1)
                contact_normal_force = torch.abs(
                    env.contact_forces[:, foot_handle_tensor, 2]
                )

                debounced_contacts = torch.where(
                    reset_mask_2d,
                    torch.zeros_like(debounced_contacts),
                    debounced_contacts,
                )
                debounce_initialized = debounce_initialized & ~reset_mask_2d
                debounce_candidate = torch.where(
                    reset_mask_2d,
                    torch.zeros_like(debounce_candidate),
                    debounce_candidate,
                )
                debounce_candidate_steps = torch.where(
                    reset_mask_2d,
                    torch.zeros_like(debounce_candidate_steps),
                    debounce_candidate_steps,
                )
                debounce_touchdown_peak_force = torch.where(
                    reset_mask_2d,
                    torch.zeros_like(debounce_touchdown_peak_force),
                    debounce_touchdown_peak_force,
                )
                debounce_touchdown_impulse = torch.where(
                    reset_mask_2d,
                    torch.zeros_like(debounce_touchdown_impulse),
                    debounce_touchdown_impulse,
                )
                initialize_contact = ~debounce_initialized & ~reset_mask_2d
                debounced_contacts = torch.where(
                    initialize_contact, contacts, debounced_contacts
                )
                debounce_candidate = torch.where(
                    initialize_contact, contacts, debounce_candidate
                )
                debounce_initialized = debounce_initialized | initialize_contact

                contact_changed = (
                    debounce_initialized
                    & ~reset_mask_2d
                    & (contacts != debounced_contacts)
                )
                candidate_continues = contact_changed & (
                    contacts == debounce_candidate
                )
                next_candidate_steps = torch.where(
                    contact_changed,
                    torch.where(
                        candidate_continues,
                        debounce_candidate_steps + 1,
                        torch.ones_like(debounce_candidate_steps),
                    ),
                    torch.zeros_like(debounce_candidate_steps),
                )
                next_candidate = torch.where(
                    contact_changed, contacts, debounced_contacts
                )
                pending_touchdown = contact_changed & contacts
                next_touchdown_peak_force = torch.where(
                    pending_touchdown,
                    torch.where(
                        candidate_continues,
                        torch.maximum(
                            debounce_touchdown_peak_force, contact_normal_force
                        ),
                        contact_normal_force,
                    ),
                    torch.zeros_like(debounce_touchdown_peak_force),
                )
                next_touchdown_impulse = torch.where(
                    pending_touchdown,
                    torch.where(
                        candidate_continues,
                        debounce_touchdown_impulse + contact_normal_force * env.dt,
                        contact_normal_force * env.dt,
                    ),
                    torch.zeros_like(debounce_touchdown_impulse),
                )
                contact_transition = contact_changed & (
                    next_candidate_steps >= contact_debounce_steps
                )
                previous_debounced_contacts = debounced_contacts.clone()
                transition_touchdown_peak_force = next_touchdown_peak_force.clone()
                transition_touchdown_impulse = next_touchdown_impulse.clone()
                debounced_contacts = torch.where(
                    contact_transition, next_candidate, debounced_contacts
                )
                debounce_candidate = torch.where(
                    contact_transition, debounced_contacts, next_candidate
                )
                debounce_candidate_steps = torch.where(
                    contact_transition,
                    torch.zeros_like(next_candidate_steps),
                    next_candidate_steps,
                )
                debounce_touchdown_peak_force = torch.where(
                    contact_transition,
                    torch.zeros_like(next_touchdown_peak_force),
                    next_touchdown_peak_force,
                )
                debounce_touchdown_impulse = torch.where(
                    contact_transition,
                    torch.zeros_like(next_touchdown_impulse),
                    next_touchdown_impulse,
                )
                if foot_stance_z is None:
                    foot_stance_z = foot_z.detach().clone()
                    foot_stance_valid = ~reset_mask.clone()
                else:
                    foot_stance_valid = foot_stance_valid & ~reset_mask
                    initialize_stance = ~foot_stance_valid & ~reset_mask
                    foot_stance_z = torch.where(
                        initialize_stance.unsqueeze(1),
                        foot_z.detach(),
                        foot_stance_z,
                    )
                    foot_stance_valid = foot_stance_valid | initialize_stance
                lower_foot_z = torch.minimum(foot_stance_z, foot_z.detach())
                foot_stance_z = torch.where(contacts | (foot_z.detach() < foot_stance_z), lower_foot_z, foot_stance_z)
                phases = getattr(env, "leg_phase", torch.zeros(env.num_envs, 2, device=env.device))[:, :2]
                ref_foot_z = reference_foot_heights(
                    phases,
                    swing_threshold,
                    foot_stance_z,
                    swing_height_target,
                    swing_profile_power,
                    swing_min_clearance,
                )
                foot_ref_error = torch.mean(torch.abs(foot_z - ref_foot_z), dim=1)
                foot_review_episode_ids += reset_mask.long()
                foot_review_steps_since_reset = torch.where(
                    reset_mask,
                    torch.zeros_like(foot_review_steps_since_reset),
                    foot_review_steps_since_reset + 1,
                )
                completed_swing = (
                    foot_review_previous_phase_valid.unsqueeze(1)
                    & ~reset_mask.unsqueeze(1)
                    & (foot_review_previous_phases >= swing_threshold)
                    & (phases < swing_threshold)
                )
                foot_review_completed_swings = torch.where(
                    reset_mask.unsqueeze(1),
                    torch.zeros_like(foot_review_completed_swings),
                    foot_review_completed_swings + completed_swing.long(),
                )
                foot_review_previous_phases = phases.detach()
                foot_review_previous_phase_valid = ~reset_mask
                foot_review_ready_mask = (
                    ~reset_mask
                    & (
                        foot_review_steps_since_reset.float() * env.dt
                        >= foot_review_warmup_seconds
                    )
                    & torch.all(
                        foot_review_completed_swings
                        >= foot_review_min_completed_swings,
                        dim=1,
                    )
                )
                valid_foot_ref_error = foot_ref_error[
                    foot_stance_valid & foot_review_ready_mask
                ]
                reference_swing_completion_events += int(
                    torch.sum(completed_swing & foot_review_ready_mask.unsqueeze(1)).item()
                )
                if prev_contacts is not None:
                    lift_mask = (
                        prev_contacts
                        & ~contacts
                        & foot_review_ready_mask.unsqueeze(1)
                    )
                    for env_index, row in enumerate(lift_mask.detach().cpu().tolist()):
                        for foot_index, lifted in enumerate(row):
                            if lifted:
                                lift_events += 1
                                if last_lifted[env_index] == foot_index:
                                    double_lift_violations += 1
                                last_lifted[env_index] = foot_index
                debounced_lift_mask = (
                    contact_transition
                    & previous_debounced_contacts
                    & ~debounced_contacts
                    & foot_review_ready_mask.unsqueeze(1)
                )
                debounced_touchdown_mask = (
                    contact_transition
                    & ~previous_debounced_contacts
                    & debounced_contacts
                    & foot_review_ready_mask.unsqueeze(1)
                )
                debounced_simultaneous_lift_events += int(
                    torch.sum(torch.all(debounced_lift_mask, dim=1)).item()
                )
                for env_index, row in enumerate(
                    debounced_lift_mask.detach().cpu().tolist()
                ):
                    for foot_index, lifted in enumerate(row):
                        if not lifted:
                            continue
                        debounced_lift_events += 1
                        if debounced_last_lifted[env_index] >= 0:
                            debounced_alternation_opportunities += 1
                            if debounced_last_lifted[env_index] == foot_index:
                                debounced_same_foot_repeats += 1
                        debounced_last_lifted[env_index] = foot_index
                touchdown_peak_values = transition_touchdown_peak_force[
                    debounced_touchdown_mask
                ]
                touchdown_impulse_values = transition_touchdown_impulse[
                    debounced_touchdown_mask
                ]
                if touchdown_peak_values.numel() > 0:
                    debounced_touchdown_events += int(
                        touchdown_peak_values.numel()
                    )
                    debounced_touchdown_peak_force_sum += float(
                        torch.sum(touchdown_peak_values).item()
                    )
                    debounced_touchdown_peak_force_max = max(
                        debounced_touchdown_peak_force_max,
                        float(torch.max(touchdown_peak_values).item()),
                    )
                    debounced_touchdown_impulse_sum += float(
                        torch.sum(touchdown_impulse_values).item()
                    )
                for env_index, reset in enumerate(reset_mask.detach().cpu().tolist()):
                    if reset:
                        last_lifted[env_index] = -1
                        debounced_last_lifted[env_index] = -1
                prev_contacts = contacts.clone()

                time_outs = infos.get("time_outs")
                if time_outs is None:
                    time_outs = getattr(env, "time_out_buf", torch.zeros_like(dones, dtype=torch.bool))
                time_outs = time_outs.bool()
                terminal_fall = dones.bool() & ~time_outs
                height_fall = env.base_pos[:, 2] < (base_height_target * 0.55)
                tilt_fall = (torch.abs(env.rpy[:, 0]) > 0.8) | (torch.abs(env.rpy[:, 1]) > 1.0)
                fall_now = terminal_fall | height_fall | tilt_fall
                new_fall_episode = fall_now & ~fall_episode_latched
                if torch.any(fall_now):
                    fall_detected = True
                    fall_count += int(torch.sum(fall_now).item())
                    fall_seen_by_env = fall_seen_by_env | fall_now
                if torch.any(new_fall_episode):
                    fall_episode_count += int(torch.sum(new_fall_episode).item())
                fall_episode_latched = fall_episode_latched | fall_now
                episode_reset_count += int(torch.sum(reset_mask).item())
                fall_episode_latched = torch.where(
                    reset_mask,
                    torch.zeros_like(fall_episode_latched),
                    fall_episode_latched,
                )

                sums["lin_vel_error"] += tensor_mean(lin_error)
                sums["yaw_error"] += tensor_mean(yaw_error)
                sums["height_error"] += tensor_mean(height_error)
                sums["tilt_error"] += tensor_mean(tilt_error)
                if valid_foot_ref_error.numel() > 0:
                    sums["foot_ref_error"] += tensor_mean(valid_foot_ref_error)
                    foot_ref_valid_steps += 1
                sums["abs_torque"] += tensor_mean(torch.abs(env.torques))
                sums["raw_abs_torque"] += tensor_mean(torch.abs(raw_torques))
                sums["raw_torque_abs_max"] += float(torch.max(torch.abs(raw_torques)).item())
                sums["torque_saturation_rate"] += tensor_mean(torque_saturation.float())
                sums["torque_clip_excess"] += tensor_mean(torque_clip_excess)
                sums["action_abs"] += tensor_mean(torch.abs(clipped_actions))
                sums["action_abs_max"] += float(torch.max(torch.abs(clipped_actions)).item())
                for group_name, group_indices in torque_group_indices.items():
                    if group_indices.numel() == 0:
                        continue
                    sums[f"{group_name}_torque_saturation_rate"] += tensor_mean(
                        torque_saturation[:, group_indices].float()
                    )
                    sums[f"{group_name}_raw_abs_torque"] += tensor_mean(
                        torch.abs(raw_torques[:, group_indices])
                    )
                sums["action_rate"] += tensor_mean(action_rate)
                sums["reward"] += tensor_mean(rews)
                sums["done_rate"] += tensor_mean(dones)
                sums["timeout_rate"] += tensor_mean(time_outs)
                sums["terminal_fall_rate"] += tensor_mean(terminal_fall)
                sums["fall_rate"] += tensor_mean(fall_now)
                valid_contact_mask = foot_review_ready_mask
                if torch.any(valid_contact_mask):
                    sums["both_air_rate"] += tensor_mean(
                        torch.all(~contacts, dim=1)[valid_contact_mask]
                    )
                    sums["both_contact_rate"] += tensor_mean(
                        torch.all(contacts, dim=1)[valid_contact_mask]
                    )
                    sums["debounced_both_air_rate"] += tensor_mean(
                        torch.all(~debounced_contacts, dim=1)[valid_contact_mask]
                    )
                    sums["debounced_both_contact_rate"] += tensor_mean(
                        torch.all(debounced_contacts, dim=1)[valid_contact_mask]
                    )
                    expected_stance = phases < swing_threshold
                    phase_contact_accuracy = torch.mean(
                        (debounced_contacts == expected_stance).float(), dim=1
                    )
                    sums["debounced_phase_contact_accuracy"] += tensor_mean(
                        phase_contact_accuracy[valid_contact_mask]
                    )
                    foot_contact_valid_steps += 1

                sample_foot_pos = foot_pos[sample_env]
                sample_ref_z = ref_foot_z[sample_env]
                sample_contacts = contacts[sample_env]
                sample_debounced_contacts = debounced_contacts[sample_env]
                sample_contact_normal_force = contact_normal_force[sample_env]
                sample_actions = clipped_actions[sample_env]
                sample_raw_torques = raw_torques[sample_env]
                sample_torques = env.torques[sample_env]
                sample_torque_saturation = torque_saturation[sample_env]
                sample_torque_clip_excess = torque_clip_excess[sample_env]
                sample_phases = phases[sample_env]
                sample_stance_z = foot_stance_z[sample_env]
                sample_relative_z = sample_foot_pos[:, 2] - sample_stance_z
                sample_reset = bool(dones[sample_env].item())
                foot_review_ready = bool(
                    foot_review_ready_mask[sample_env].item()
                )
                sample_episode_id = int(
                    foot_review_episode_ids[sample_env].item()
                )
                sample_steps_since_reset = int(
                    foot_review_steps_since_reset[sample_env].item()
                )
                sample_completed_swings = [
                    int(value)
                    for value in foot_review_completed_swings[sample_env]
                    .detach()
                    .cpu()
                    .tolist()
                ]
                trajectory_row = {
                    "step": total_steps,
                    "command_name": command.get("name", f"command_{command_index}"),
                    "base_height": float(env.base_pos[sample_env, 2].item()),
                    "base_roll": float(env.rpy[sample_env, 0].item()),
                    "base_pitch": float(env.rpy[sample_env, 1].item()),
                    "base_yaw": float(env.rpy[sample_env, 2].item()),
                    "left_foot_x": float(sample_foot_pos[0, 0].item()),
                    "left_foot_y": float(sample_foot_pos[0, 1].item()),
                    "left_foot_z": float(sample_foot_pos[0, 2].item()),
                    "right_foot_x": float(sample_foot_pos[1, 0].item()),
                    "right_foot_y": float(sample_foot_pos[1, 1].item()),
                    "right_foot_z": float(sample_foot_pos[1, 2].item()),
                    "left_ref_z": float(sample_ref_z[0].item()),
                    "right_ref_z": float(sample_ref_z[1].item()),
                    "left_phase": float(sample_phases[0].item()),
                    "right_phase": float(sample_phases[1].item()),
                    "left_stance_z": float(sample_stance_z[0].item()),
                    "right_stance_z": float(sample_stance_z[1].item()),
                    "left_relative_z": float(sample_relative_z[0].item()),
                    "right_relative_z": float(sample_relative_z[1].item()),
                    "sample_episode_id": sample_episode_id,
                    "sample_steps_since_reset": sample_steps_since_reset,
                    "left_completed_swings": sample_completed_swings[0],
                    "right_completed_swings": sample_completed_swings[1],
                    "foot_review_ready": int(foot_review_ready),
                    "sample_reset": int(sample_reset),
                    "left_contact": int(bool(sample_contacts[0].item())),
                    "right_contact": int(bool(sample_contacts[1].item())),
                    "left_contact_debounced": int(
                        bool(sample_debounced_contacts[0].item())
                    ),
                    "right_contact_debounced": int(
                        bool(sample_debounced_contacts[1].item())
                    ),
                    "left_contact_normal_force": float(sample_contact_normal_force[0].item()),
                    "right_contact_normal_force": float(sample_contact_normal_force[1].item()),
                    "action_abs_mean": float(torch.mean(torch.abs(sample_actions)).item()),
                    "action_abs_max": float(torch.max(torch.abs(sample_actions)).item()),
                    "raw_torque_abs_mean": float(torch.mean(torch.abs(sample_raw_torques)).item()),
                    "raw_torque_abs_max": float(torch.max(torch.abs(sample_raw_torques)).item()),
                    "torque_abs_max": float(torch.max(torch.abs(sample_torques)).item()),
                    "torque_saturation_rate": float(
                        torch.mean(sample_torque_saturation.float()).item()
                    ),
                    "torque_clip_excess_mean": float(
                        torch.mean(sample_torque_clip_excess).item()
                    ),
                }
                if sample_reset:
                    topology_rows.append({"topology_reset": True})
                    foot_review_skipped_steps += 1
                    foot_review_reset_count += 1
                elif foot_review_ready:
                    topology_rows.append(trajectory_row)
                else:
                    foot_review_skipped_steps += 1

                if (
                    foot_review_ready
                    and total_steps % max(1, custom.trajectory_stride) == 0
                ):
                    sample.append(
                        {
                            "step": total_steps,
                            "command_index": command_index,
                            "command_name": command.get("name", f"command_{command_index}"),
                            "command": {"vx": vx, "vy": vy, "yaw": yaw},
                            "base_pos": to_list(env.base_pos[sample_env]),
                            "base_rpy": to_list(env.rpy[sample_env]),
                            "base_lin_vel": to_list(env.base_lin_vel[sample_env]),
                            "base_ang_vel": to_list(env.base_ang_vel[sample_env]),
                            "left_foot_pos": to_list(sample_foot_pos[0]),
                            "right_foot_pos": to_list(sample_foot_pos[1]),
                            "left_ref_z": float(sample_ref_z[0].item()),
                            "right_ref_z": float(sample_ref_z[1].item()),
                            "left_contact": int(bool(sample_contacts[0].item())),
                            "right_contact": int(bool(sample_contacts[1].item())),
                            "left_contact_debounced": int(bool(sample_debounced_contacts[0].item())),
                            "right_contact_debounced": int(bool(sample_debounced_contacts[1].item())),
                            "dof_pos": to_list(env.dof_pos[sample_env]),
                        }
                    )
                total_steps += 1

    denom = max(1, total_steps)
    reference_gait_cycle_count = reference_swing_completion_events / 2.0
    debounced_same_foot_repeat_rate = (
        debounced_same_foot_repeats / max(1, debounced_alternation_opportunities)
    )
    debounced_foot_alternation_score = (
        max(0.0, 1.0 - debounced_same_foot_repeat_rate)
        if debounced_alternation_opportunities > 0
        else 0.0
    )
    if foot_stance_z is None:
        foot_stance_z = torch.zeros(env.num_envs, 2, device=env.device)
    metrics = {
        "steps": total_steps,
        "num_envs": int(env.num_envs),
        "mean_lin_vel_error": sums["lin_vel_error"] / denom,
        "mean_yaw_error": sums["yaw_error"] / denom,
        "mean_height_error": sums["height_error"] / denom,
        "mean_tilt_error": sums["tilt_error"] / denom,
        "mean_foot_ref_error": sums["foot_ref_error"] / max(1, foot_ref_valid_steps),
        "mean_abs_torque": sums["abs_torque"] / denom,
        "mean_raw_abs_torque": sums["raw_abs_torque"] / denom,
        "mean_raw_torque_abs_max": sums["raw_torque_abs_max"] / denom,
        "torque_saturation_rate": sums["torque_saturation_rate"] / denom,
        "mean_torque_clip_excess": sums["torque_clip_excess"] / denom,
        "mean_action_abs": sums["action_abs"] / denom,
        "mean_action_abs_max": sums["action_abs_max"] / denom,
        "mean_action_rate": sums["action_rate"] / denom,
        "mean_reward": sums["reward"] / denom,
        "done_rate": sums["done_rate"] / denom,
        "timeout_rate": sums["timeout_rate"] / denom,
        "terminal_fall_rate": sums["terminal_fall_rate"] / denom,
        "fall_rate": sums["fall_rate"] / denom,
        "fall_detected": bool(fall_detected),
        "fall_count": int(fall_count),
        "fall_env_count": int(torch.sum(fall_seen_by_env).item()),
        "fall_env_rate": tensor_mean(fall_seen_by_env.float()),
        "fall_episode_count": int(fall_episode_count),
        "episode_exposure_count": int(env.num_envs + episode_reset_count),
        "fall_episode_rate": fall_episode_count
        / max(1, int(env.num_envs + episode_reset_count)),
        "both_feet_air_rate": sums["both_air_rate"] / max(1, foot_contact_valid_steps),
        "both_feet_contact_rate": sums["both_contact_rate"] / max(1, foot_contact_valid_steps),
        "lift_event_count": int(lift_events),
        "double_lift_violation_count": int(double_lift_violations),
        "double_lift_violation_rate": double_lift_violations / max(1, lift_events),
        "contact_debounce_seconds": CONTACT_DEBOUNCE_SECONDS,
        "contact_debounce_steps": contact_debounce_steps,
        "debounced_phase_contact_accuracy": sums[
            "debounced_phase_contact_accuracy"
        ]
        / max(1, foot_contact_valid_steps),
        "debounced_both_feet_air_rate": sums["debounced_both_air_rate"]
        / max(1, foot_contact_valid_steps),
        "debounced_both_feet_contact_rate": sums["debounced_both_contact_rate"]
        / max(1, foot_contact_valid_steps),
        "reference_gait_cycle_count": reference_gait_cycle_count,
        "debounced_lift_event_count": int(debounced_lift_events),
        "debounced_lift_events_per_cycle": debounced_lift_events
        / max(1.0, reference_gait_cycle_count),
        "debounced_alternation_opportunity_count": int(
            debounced_alternation_opportunities
        ),
        "debounced_same_foot_repeat_count": int(debounced_same_foot_repeats),
        "debounced_same_foot_repeat_rate": debounced_same_foot_repeat_rate,
        "debounced_same_foot_repeats_per_cycle": debounced_same_foot_repeats
        / max(1.0, reference_gait_cycle_count),
        "debounced_foot_alternation_score": debounced_foot_alternation_score,
        "debounced_simultaneous_lift_event_count": int(
            debounced_simultaneous_lift_events
        ),
        "debounced_touchdown_event_count": int(debounced_touchdown_events),
        "mean_touchdown_peak_normal_force": debounced_touchdown_peak_force_sum
        / max(1, debounced_touchdown_events),
        "max_touchdown_peak_normal_force": debounced_touchdown_peak_force_max,
        "mean_touchdown_confirmation_impulse": debounced_touchdown_impulse_sum
        / max(1, debounced_touchdown_events),
        "left_reference_stance_z": tensor_mean(foot_stance_z[:, 0]),
        "right_reference_stance_z": tensor_mean(foot_stance_z[:, 1]),
        "reference_swing_target_z": swing_height_target,
    }
    for group_name, group_indices in torque_group_indices.items():
        if group_indices.numel() == 0:
            continue
        metrics[f"{group_name}_torque_saturation_rate"] = (
            sums[f"{group_name}_torque_saturation_rate"] / denom
        )
        metrics[f"{group_name}_mean_raw_abs_torque"] = (
            sums[f"{group_name}_raw_abs_torque"] / denom
        )
    if lift_events < 2:
        metrics["foot_alternation_score"] = 0.0
    else:
        alternating = max(0.0, 1.0 - (double_lift_violations / max(lift_events, 1)))
        metrics["foot_alternation_score"] = alternating * math.exp(-metrics["both_feet_air_rate"] / 0.25)
    topology = summarize_single_swing_topology(
        topology_rows,
        swing_threshold,
        float(getattr(env.cfg.rewards, "single_swing_peak_min_clearance", 0.05)),
        float(getattr(env.cfg.rewards, "single_swing_peak_min_separation", 0.15)),
        float(getattr(env.cfg.rewards, "single_swing_m_drop_threshold", 0.010)),
    )
    continuous_m_shape = summarize_continuous_m_shape(topology_rows, swing_threshold)
    metrics.update(
        {
            "single_swing_topology": topology,
            "single_swing_topology_eligible": topology["eligible"],
            "single_swing_complete_count": topology["complete_count"],
            "single_swing_multi_peak_count": topology["multi_peak_count"],
            "single_swing_multi_peak_rate": topology["multi_peak_rate"],
            "single_swing_worst_foot": topology["worst_foot"],
            "single_swing_worst_foot_multi_peak_rate": topology["worst_foot_multi_peak_rate"],
            "single_swing_worst_foot_mid_drop_max": topology["worst_foot_mid_drop_max"],
            "single_swing_mid_drop_mean": topology["mid_drop_mean"],
            "single_swing_mid_drop_max": topology["mid_drop_max"],
            "single_swing_topology_score": topology["score"],
            "left_single_swing_multi_peak_count": topology["per_foot"]["left"]["multi_peak_count"],
            "right_single_swing_multi_peak_count": topology["per_foot"]["right"]["multi_peak_count"],
            "continuous_m_shape": continuous_m_shape,
            "continuous_m_shape_eligible": continuous_m_shape["eligible"],
            "continuous_m_shape_event_count": continuous_m_shape["event_count"],
            "continuous_m_shape_max_drop": continuous_m_shape["max_drop"],
            "continuous_m_shape_mean_drop": continuous_m_shape["mean_drop"],
            "continuous_m_shape_worst_foot": continuous_m_shape["worst_foot"],
            "continuous_m_shape_stable_m_risk": continuous_m_shape["stable_m_risk"],
            "foot_review_warmup_seconds": foot_review_warmup_seconds,
            "foot_review_min_completed_swings": foot_review_min_completed_swings,
            "foot_review_skipped_steps": foot_review_skipped_steps,
            "foot_review_reset_count": foot_review_reset_count,
            "foot_review_rows": len(topology_rows) - foot_review_reset_count - 1,
            "foot_review_valid_contact_steps": foot_contact_valid_steps,
        }
    )

    output_path = Path(custom.output)
    trajectory_output = Path(custom.trajectory_output) if custom.trajectory_output else output_path.with_name("trajectory.csv")
    plot_output = Path(custom.plot_output) if custom.plot_output else output_path.with_name("trajectory.svg")
    review_rows = [row for row in topology_rows if not row.get("topology_reset")]
    sample_summary = summarize_sample_trajectory(review_rows, base_height_target)
    metrics["sample_same_foot_repeat_count"] = int(
        sample_summary.get("same_foot_repeat_count", 0) or 0
    )
    metrics["sample_lift_event_count"] = len(sample_summary.get("lift_sequence", []))
    write_trajectory_csv(trajectory_output, review_rows)
    write_trajectory_svg(plot_output, review_rows, base_height_target, swing_height_target)

    score = score_metrics(
        metrics,
        score_config,
        allow_topology_gate=not topology_diagnostic_only,
    )
    result = {
        "task": args.task,
        "load_run": args.load_run,
        "checkpoint": args.checkpoint,
        "milestone_iteration": milestone_iteration,
        "command_plan": command_plan,
        "metrics": metrics,
        "score": score,
        "simlog": build_simlog(metrics, score, review_rows, base_height_target, swing_height_target),
        "trajectory_csv": str(trajectory_output),
        "trajectory_plot": str(plot_output),
        "trajectory_sample": sample,
    }
    write_json(custom.output, result)


if __name__ == "__main__":
    custom_args, passthrough = parse_custom_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *passthrough]
    evaluate(get_args(), custom_args)
