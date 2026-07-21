import argparse
import concurrent.futures
import copy
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from auto_train_common import git_commit_paths, load_json, repo_root_from_script, timestamp_id, write_json


DEFAULT_CONFIG = {
    "task": "openduckpro2",
    "experiment_name": "openduckpro2",
    "gpus": [0, 1],
    "max_parallel_jobs": 2,
    "workspace": {
        "enabled": True,
        "root": "auto_train_workspaces",
        "snapshot_dirty_state": True,
    },
    "cycles": None,
    "control": {
        "stop_files": [],
    },
    "codex_control": {
        "enabled": True,
        "wait_for_decision": True,
        "decision_filename": "codex_decision.json",
        "heartbeat_interval_sec": 30,
        "allow_auto_mutation_fallback": False,
    },
    "streaming_candidates": {
        "enabled": True,
        "suggestions_per_completion": 1,
        "prefer_codex_candidates": True,
    },
    "continuous_training": {
        "enabled": True,
        "evaluate_milestones_after_training": True,
    },
    "termination": {
        "score_target": 0.92,
        "min_cycles": 4,
        "patience_cycles": 3,
        "min_delta": 0.005,
    },
    "dynamic_iterations": {
        "enabled": False,
        "increment": 1000,
        "max_iterations": 10000,
        "score_threshold": 0.90,
        "plateau_delta": 0.005,
    },
    "eval_milestones": [
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
    ],
    "train": {
        "num_envs": 4096,
        "max_iterations": 10000,
        "headless": True,
        "python": sys.executable,
    },
    "eval": {
        "num_envs": 32,
        "steps": 1000,
        "trajectory_stride": 10,
    },
    "candidates": [],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run parallel reward-search training cycles.")
    parser.add_argument("--config", default="configs/auto_train/openduckpro2_default.json")
    parser.add_argument("--run-id")
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-commit", action="store_true")
    return parser.parse_args()


def merged_config(path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    disk = load_json(path, default={}) or {}
    deep_update(config, disk)
    return config


def deep_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


G1_POSITIVE_REWARD_SCALES = {
    "tracking_lin_vel",
    "tracking_ang_vel",
    "alive",
    "contact",
}
G1_NEGATIVE_REWARD_SCALES = {
    "lin_vel_z",
    "ang_vel_xy",
    "orientation",
    "base_height",
    "torques",
    "dof_acc",
    "dof_vel",
    "action_rate",
    "dof_pos_limits",
    "hip_pos",
    "contact_no_vel",
    "feet_swing_height",
}


def validate_continuous_training_config(config):
    continuous = config.get("continuous_training", {})
    if not continuous.get("enabled", False):
        return

    max_iterations = int(config["train"]["max_iterations"])
    if max_iterations != 10000:
        raise ValueError(
            "continuous_training requires train.max_iterations=10000; "
            f"got {max_iterations}"
        )

    expected_milestones = list(range(1000, 10000, 1000))
    configured_milestones = sorted(
        {int(value) for value in config.get("eval_milestones", [])}
    )
    if configured_milestones != expected_milestones:
        raise ValueError(
            "continuous_training requires eval_milestones at every 1K "
            "from 1000 through 9000"
        )

    candidates = config.get("candidates", [])
    if not candidates:
        raise ValueError("continuous_training requires at least one explicit candidate")

    allowed_scales = G1_POSITIVE_REWARD_SCALES | G1_NEGATIVE_REWARD_SCALES
    for candidate in candidates:
        candidate_name = candidate.get("name", "candidate")
        scales = candidate.get("reward_scales", {})
        unknown = sorted(set(scales) - allowed_scales)
        if unknown:
            raise ValueError(
                f"Continuous candidate {candidate_name} contains non-G1 rewards: "
                + ", ".join(unknown)
            )
        for reward_name, raw_value in scales.items():
            value = float(raw_value)
            if value == 0:
                raise ValueError(
                    f"Continuous candidate {candidate_name} sets "
                    f"{reward_name}=0; G1 nonzero rewards cannot be disabled"
                )
            if reward_name in G1_POSITIVE_REWARD_SCALES and value < 0:
                raise ValueError(
                    f"Continuous candidate {candidate_name} reverses the sign "
                    f"of positive G1 reward {reward_name}"
                )
            if reward_name in G1_NEGATIVE_REWARD_SCALES and value > 0:
                raise ValueError(
                    f"Continuous candidate {candidate_name} reverses the sign "
                    f"of negative G1 reward {reward_name}"
                )

        training_pr = candidate.get("training_pr") or {}
        required_pr_fields = {
            "repo",
            "number",
            "url",
            "head_ref",
            "base_ref",
            "created_with",
            "state",
            "is_draft",
        }
        missing_pr_fields = sorted(required_pr_fields - set(training_pr))
        if missing_pr_fields:
            raise ValueError(
                f"Continuous candidate {candidate_name} is missing training PR "
                "metadata: " + ", ".join(missing_pr_fields)
            )
        if (
            training_pr.get("repo") != "orbbecwuxin/openduck-training-control"
            or training_pr.get("created_with") != "gh"
            or training_pr.get("state") != "OPEN"
            or training_pr.get("is_draft") is not True
        ):
            raise ValueError(
                f"Continuous candidate {candidate_name} must reference an open "
                "draft PR created with gh in the training control repository"
            )

    resumed = [
        candidate.get("name", "candidate")
        for candidate in candidates
        if int((candidate.get("resume") or {}).get("checkpoint", 0) or 0) > 0
    ]
    if resumed:
        raise ValueError(
            "continuous_training candidates must start fresh; checkpoint resume "
            "would create another TensorBoard run: " + ", ".join(resumed)
        )


def safe_name(value):
    allowed = []
    for ch in value.lower():
        if ch.isalnum() or ch in ("-", "_"):
            allowed.append(ch)
        elif ch in (" ", ".", "/"):
            allowed.append("_")
    return "".join(allowed).strip("_") or "candidate"


def write_heartbeat(path, state, **details):
    payload = {
        "pid": os.getpid(),
        "state": state,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    payload.update(details)
    write_json(path, payload)


def run_command(command, cwd, log_path, env=None, dry_run=False, stop_check=None, heartbeat=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# cwd: " + str(cwd) + "\n")
        log.write("$ " + " ".join(command) + "\n")
        if dry_run:
            log.write("DRY RUN: command not executed.\n")
            return 0
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            if heartbeat:
                heartbeat(process.pid)
            reason = stop_check() if stop_check else None
            if reason:
                log.write("\nSTOP REQUESTED: " + str(reason) + "\n")
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    log.write("Process did not terminate in 30s; killing.\n")
                    process.kill()
                    process.wait()
                return 130
            time.sleep(5)
        return process.returncode


def build_train_command(config, candidate, candidate_dir, gpu, run_name, segment_iterations=None):
    train_cfg = config["train"]
    resume = candidate.get("resume") or {}
    max_iterations = int(segment_iterations if segment_iterations is not None else train_cfg["max_iterations"])
    command = [
        train_cfg.get("python", sys.executable),
        "legged_gym/scripts/train_reward_variant.py",
        "--reward-overrides",
        str(candidate_dir / "reward_overrides.json"),
        "--auto-train-meta",
        str(candidate_dir / "train_meta.json"),
        f"--task={config['task']}",
        f"--experiment_name={config['experiment_name']}",
        f"--sim_device=cuda:{gpu}",
        f"--rl_device=cuda:{gpu}",
        f"--num_envs={int(train_cfg['num_envs'])}",
        f"--max_iterations={max_iterations}",
        f"--run_name={run_name}",
    ]
    if resume:
        load_run = resume.get("load_run")
        checkpoint = resume.get("checkpoint")
        if not load_run or checkpoint is None:
            raise ValueError("resume candidate requires resume.load_run and resume.checkpoint")
        command.extend(
            [
                "--resume",
                f"--load_run={load_run}",
                f"--checkpoint={int(checkpoint)}",
            ]
        )
    if train_cfg.get("headless", True):
        command.append("--headless")
    if "seed" in candidate:
        command.append(f"--seed={int(candidate['seed'])}")
    return command


def build_eval_command(
    config,
    candidate_dir,
    gpu,
    train_meta,
    milestone_iteration,
    artifact_dir=None,
):
    eval_cfg = config["eval"]
    artifact_dir = Path(artifact_dir or candidate_dir)
    command = [
        eval_cfg.get("python", config["train"].get("python", sys.executable)),
        "legged_gym/scripts/evaluate_policy.py",
        "--reward-overrides",
        str(candidate_dir / "reward_overrides.json"),
        "--output",
        str(artifact_dir / "evaluation.json"),
        f"--task={config['task']}",
        f"--experiment_name={config['experiment_name']}",
        f"--sim_device=cuda:{gpu}",
        f"--rl_device=cuda:{gpu}",
        f"--num_envs={int(eval_cfg['num_envs'])}",
        f"--load_run={train_meta['load_run']}",
        f"--checkpoint={int(milestone_iteration)}",
        "--headless",
        f"--milestone-iteration={int(milestone_iteration)}",
        f"--steps={int(eval_cfg.get('steps', 1000))}",
        f"--trajectory-stride={int(eval_cfg.get('trajectory_stride', 10))}",
    ]
    if eval_cfg.get("command_plan"):
        command.extend(["--command-plan", eval_cfg["command_plan"]])
    if eval_cfg.get("score_config"):
        command.extend(["--score-config", eval_cfg["score_config"]])
    return command


def eval_targets(config):
    max_iterations = int(config["train"]["max_iterations"])
    milestones = config.get("eval_milestones") or []
    targets = sorted({int(value) for value in milestones if 0 < int(value) < max_iterations})
    targets.append(max_iterations)
    return targets


def next_eval_target(config, start_iteration):
    start_iteration = int(start_iteration)
    for target in eval_targets(config):
        if target > start_iteration:
            return int(target)
    return int(config["train"]["max_iterations"])


def continuation_candidate(result):
    train_meta = result.get("train_meta") or {}
    checkpoint = int(result["target_iteration"])
    next_target = next_eval_target({"train": {"max_iterations": result["train_max_iterations"]}, "eval_milestones": result["eval_milestones"]}, checkpoint)
    base_candidate = result.get("candidate") or {}
    base_name = safe_name(base_candidate.get("name", "candidate"))
    return {
        "name": f"{base_name}_continue_{next_target}",
        "reward_scales": dict(base_candidate.get("reward_scales", {})),
        "source_candidate": base_candidate.get("name"),
        "source_milestone_iteration": checkpoint,
        "resume": {
            "load_run": train_meta.get("load_run"),
            "checkpoint": checkpoint,
            "latest_checkpoint": train_meta.get("latest_checkpoint"),
        },
    }


def run_git(args, cwd, **kwargs):
    return subprocess.run(["git", *args], cwd=cwd, check=True, **kwargs)


def git_output(args, cwd):
    result = run_git(args, cwd, text=True, capture_output=True)
    return result.stdout


def dirty_overlay_paths(repo_root):
    output = git_output(["ls-files", "-m", "-o", "--exclude-standard", "-z"], repo_root)
    paths = []
    for raw in output.split("\0"):
        if not raw:
            continue
        path = Path(raw)
        if path.parts and path.parts[0] in {"auto_train_runs", "auto_train_workspaces", "logs"}:
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix == ".pyc":
            continue
        if (repo_root / path).is_file():
            if path.name == "STOP_AUTO_TRAIN":
                continue
            paths.append(path)
    return paths


def copy_dirty_overlay(repo_root, workspace_dir):
    copied = []
    for rel_path in dirty_overlay_paths(repo_root):
        src = repo_root / rel_path
        dst = workspace_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(rel_path))
    return copied


def prepare_workspace(repo_root, config, run_id, cycle_index, candidate_index, candidate_name, dry_run):
    workspace_cfg = config.get("workspace", {})
    if not workspace_cfg.get("enabled", False):
        return {
            "cwd": repo_root,
            "env": os.environ.copy(),
            "metadata": {"enabled": False, "path": str(repo_root)},
        }

    root = repo_root / workspace_cfg.get("root", "auto_train_workspaces")
    workspace_dir = root / run_id / f"cycle_{cycle_index:03d}" / f"{candidate_index:02d}_{candidate_name}"
    branch = f"auto-train/{safe_name(run_id)}/c{cycle_index:03d}/{candidate_index:02d}-{candidate_name}"
    metadata = {
        "enabled": True,
        "path": str(workspace_dir),
        "branch": branch,
        "source_repo": str(repo_root),
    }

    if dry_run:
        metadata["dry_run"] = True
    else:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.parent.mkdir(parents=True, exist_ok=True)
        run_git(["clone", "--shared", str(repo_root), str(workspace_dir)], repo_root)
        run_git(["checkout", "-B", branch], workspace_dir)
        copied = []
        if workspace_cfg.get("snapshot_dirty_state", True):
            copied = copy_dirty_overlay(repo_root, workspace_dir)
        run_git(["add", "-A"], workspace_dir)
        run_git(
            [
                "-c",
                "user.name=Auto Train",
                "-c",
                "user.email=auto-train@local",
                "commit",
                "--allow-empty",
                "-m",
                f"auto-train: snapshot {run_id} cycle {cycle_index} candidate {candidate_index}",
            ],
            workspace_dir,
        )
        metadata["dirty_overlay_files"] = copied
        metadata["commit"] = git_output(["rev-parse", "HEAD"], workspace_dir).strip()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return {"cwd": workspace_dir, "env": env, "metadata": metadata}


def stage_resume_checkpoint(candidate, workspace_dir, experiment_name):
    resume = candidate.get("resume") or {}
    if not resume:
        return None

    latest_checkpoint = resume.get("latest_checkpoint")
    if not latest_checkpoint:
        return None

    load_run = resume.get("load_run")
    checkpoint = resume.get("checkpoint")
    if not load_run or checkpoint is None:
        raise ValueError("resume.latest_checkpoint requires resume.load_run and resume.checkpoint")

    checkpoint = int(checkpoint)
    expected_name = f"model_{checkpoint}.pt"
    source_checkpoint = Path(latest_checkpoint)
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {source_checkpoint}")
    if source_checkpoint.name != expected_name:
        raise ValueError(
            f"resume checkpoint mismatch: expected {expected_name}, got {source_checkpoint.name}"
        )

    source_run_dir = source_checkpoint.parent
    target_log_root = Path(workspace_dir) / "logs" / experiment_name
    target_run_dir = target_log_root / str(load_run)
    target_checkpoint = target_run_dir / expected_name
    target_log_root.mkdir(parents=True, exist_ok=True)

    if target_checkpoint.exists():
        mode = "existing"
    else:
        if target_run_dir.exists() or target_run_dir.is_symlink():
            if target_run_dir.is_symlink() or target_run_dir.is_file():
                target_run_dir.unlink()
            else:
                shutil.rmtree(target_run_dir)
        try:
            target_run_dir.symlink_to(source_run_dir, target_is_directory=True)
            mode = "symlink_run_dir"
        except OSError:
            target_run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_checkpoint, target_checkpoint)
            mode = "copy_checkpoint"

    if not target_checkpoint.exists():
        raise FileNotFoundError(f"staged resume checkpoint not found: {target_checkpoint}")

    return {
        "mode": mode,
        "load_run": str(load_run),
        "checkpoint": checkpoint,
        "source_run_dir": str(source_run_dir),
        "source_checkpoint": str(source_checkpoint),
        "staged_run_dir": str(target_run_dir),
        "staged_checkpoint": str(target_checkpoint),
    }


def run_candidate(repo_root, config, run_id, cycle_index, candidate_index, candidate, gpu, dry_run, stop_check=None):
    candidate_name = safe_name(candidate["name"])
    cycle_dir = repo_root / "auto_train_runs" / run_id / f"cycle_{cycle_index:03d}"
    candidate_dir = cycle_dir / f"{candidate_index:02d}_{candidate_name}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    workspace = prepare_workspace(
        repo_root=repo_root,
        config=config,
        run_id=run_id,
        cycle_index=cycle_index,
        candidate_index=candidate_index,
        candidate_name=candidate_name,
        dry_run=dry_run,
    )
    resume_staging = stage_resume_checkpoint(candidate, workspace["cwd"], config["experiment_name"])
    if resume_staging:
        workspace["metadata"]["resume_checkpoint"] = resume_staging
    write_json(candidate_dir / "workspace.json", workspace["metadata"])

    reward_overrides = {
        "name": candidate["name"],
        "cycle": cycle_index,
        "reward_scales": candidate.get("reward_scales", {}),
    }
    write_json(candidate_dir / "reward_overrides.json", reward_overrides)

    run_name = f"{run_id}_c{cycle_index:03d}_{candidate_index:02d}_{candidate_name}"
    resume = candidate.get("resume") or {}
    start_iteration = int(resume.get("checkpoint", 0) or 0)
    continuous_training = bool(
        config.get("continuous_training", {}).get("enabled", False)
    )
    if continuous_training and start_iteration != 0:
        raise ValueError(
            f"Continuous candidate {candidate_name} cannot resume from "
            f"iteration {start_iteration}; start a new candidate instead"
        )
    target_iteration = (
        int(config["train"]["max_iterations"])
        if continuous_training
        else next_eval_target(config, start_iteration)
    )
    segment_iterations = target_iteration - start_iteration
    if segment_iterations <= 0:
        raise ValueError(
            f"Invalid milestone segment for {candidate_name}: start={start_iteration}, target={target_iteration}"
        )
    train_command = build_train_command(
        config,
        candidate,
        candidate_dir,
        gpu,
        run_name,
        segment_iterations=segment_iterations,
    )
    write_heartbeat(
        candidate_dir / "heartbeat.json",
        "training",
        run_id=run_id,
        cycle=cycle_index,
        candidate=candidate["name"],
        gpu=gpu,
        start_iteration=start_iteration,
        target_iteration=target_iteration,
        train_max_iterations=int(config["train"]["max_iterations"]),
        continuous_training=continuous_training,
        run_name=run_name,
    )
    train_status = run_command(
        train_command,
        cwd=workspace["cwd"],
        log_path=candidate_dir / "train.log",
        env=workspace["env"],
        dry_run=dry_run,
        stop_check=stop_check,
        heartbeat=lambda pid: write_heartbeat(
            candidate_dir / "heartbeat.json",
            "training",
            run_id=run_id,
            cycle=cycle_index,
            candidate=candidate["name"],
            gpu=gpu,
            child_pid=pid,
            start_iteration=start_iteration,
            target_iteration=target_iteration,
            train_max_iterations=int(config["train"]["max_iterations"]),
            continuous_training=continuous_training,
            run_name=run_name,
        ),
    )

    result = {
        "candidate": candidate,
        "artifact_dir": str(candidate_dir),
        "gpu": gpu,
        "run_name": run_name,
        "workspace": workspace["metadata"],
        "resume_checkpoint_staging": resume_staging,
        "train_command": train_command,
        "train_returncode": train_status,
        "start_iteration": start_iteration,
        "target_iteration": target_iteration,
        "segment_iterations": segment_iterations,
        "train_max_iterations": int(config["train"]["max_iterations"]),
        "eval_milestones": eval_targets(config)[:-1],
        "continuous_training": continuous_training,
    }
    if train_status == 0 and not dry_run:
        train_meta = load_json(candidate_dir / "train_meta.json")
        result["train_meta"] = train_meta
        if train_meta.get("latest_checkpoint"):
            evaluation_iterations = (
                [
                    iteration
                    for iteration in eval_targets(config)
                    if start_iteration < iteration <= target_iteration
                ]
                if continuous_training
                and config.get("continuous_training", {}).get(
                    "evaluate_milestones_after_training", True
                )
                else [target_iteration]
            )
            milestone_evaluations = {}
            run_dir = Path(train_meta["run_dir"])
            for evaluation_iteration in evaluation_iterations:
                is_final_evaluation = evaluation_iteration == target_iteration
                artifact_dir = (
                    candidate_dir
                    if is_final_evaluation
                    else candidate_dir
                    / "milestones"
                    / f"{evaluation_iteration:05d}"
                )
                artifact_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = run_dir / f"model_{evaluation_iteration}.pt"
                milestone_result = {
                    "iteration": evaluation_iteration,
                    "checkpoint_path": str(checkpoint_path),
                    "artifact_dir": str(artifact_dir),
                }
                if not checkpoint_path.exists():
                    milestone_result["status"] = "missing_checkpoint"
                    milestone_evaluations[str(evaluation_iteration)] = milestone_result
                    continue
                eval_command = build_eval_command(
                    config,
                    candidate_dir,
                    gpu,
                    train_meta,
                    evaluation_iteration,
                    artifact_dir=artifact_dir,
                )
                write_heartbeat(
                    candidate_dir / "heartbeat.json",
                    "evaluating",
                    run_id=run_id,
                    cycle=cycle_index,
                    candidate=candidate["name"],
                    gpu=gpu,
                    start_iteration=start_iteration,
                    target_iteration=target_iteration,
                    evaluation_iteration=evaluation_iteration,
                    train_max_iterations=int(config["train"]["max_iterations"]),
                )
                eval_status = run_command(
                    eval_command,
                    cwd=workspace["cwd"],
                    log_path=artifact_dir / "evaluate.log",
                    env=workspace["env"],
                    dry_run=dry_run,
                    stop_check=stop_check,
                    heartbeat=lambda pid, iteration=evaluation_iteration: write_heartbeat(
                        candidate_dir / "heartbeat.json",
                        "evaluating",
                        run_id=run_id,
                        cycle=cycle_index,
                        candidate=candidate["name"],
                        gpu=gpu,
                        child_pid=pid,
                        start_iteration=start_iteration,
                        target_iteration=target_iteration,
                        evaluation_iteration=iteration,
                        train_max_iterations=int(config["train"]["max_iterations"]),
                    ),
                )
                milestone_result["eval_command"] = eval_command
                milestone_result["eval_returncode"] = eval_status
                milestone_result["status"] = (
                    "evaluated" if eval_status == 0 else "evaluation_failed"
                )
                if eval_status == 0:
                    milestone_result["evaluation"] = load_json(
                        artifact_dir / "evaluation.json"
                    )
                milestone_evaluations[str(evaluation_iteration)] = milestone_result
                if is_final_evaluation:
                    result["eval_command"] = eval_command
                    result["eval_returncode"] = eval_status
                    if eval_status == 0:
                        result["evaluation"] = milestone_result["evaluation"]
            result["milestone_evaluations"] = milestone_evaluations
        result["retained_model"] = train_meta.get("latest_checkpoint")
        if (
            not continuous_training
            and result.get("evaluation")
            and target_iteration < int(config["train"]["max_iterations"])
            and candidate_is_admissible(result)
        ):
            result["suggested_continuation"] = continuation_candidate(result)
    write_heartbeat(
        candidate_dir / "heartbeat.json",
        "candidate_done",
        run_id=run_id,
        cycle=cycle_index,
        candidate=candidate["name"],
        gpu=gpu,
        train_returncode=train_status,
        eval_returncode=result.get("eval_returncode"),
        target_iteration=target_iteration,
        train_max_iterations=int(config["train"]["max_iterations"]),
        continuous_training=continuous_training,
        run_name=run_name,
    )
    write_json(candidate_dir / "candidate_result.json", result)
    return result


def signed_scale(value, factor):
    if value < 0:
        return value * factor
    return value * factor


def evaluation_topology_counts(evaluation):
    metrics = (evaluation or {}).get("metrics") or {}
    score = (evaluation or {}).get("score") or {}
    simlog = (evaluation or {}).get("simlog") or {}
    sample = simlog.get("sample_trajectory") or {}
    sample_same = metrics.get("sample_same_foot_repeat_count")
    if sample_same is None:
        sample_same = sample.get("same_foot_repeat_count", 0)
    return {
        "fall": bool(score.get("fall_gate") or metrics.get("fall_detected")),
        "double_lift": int(metrics.get("double_lift_violation_count", 0) or 0),
        "sample_same_foot": int(sample_same or 0),
        "m_events": int(metrics.get("continuous_m_shape_event_count", 0) or 0),
        "single_swing_multi_peak": int(metrics.get("single_swing_multi_peak_count", 0) or 0),
        "total_score": float(score.get("total_score", 0.0) or 0.0),
        "topology_admissible": score.get("topology_admissible"),
        "topology_strict_champion": score.get("topology_strict_champion"),
        "topology_gate_reasons": list(score.get("topology_gate_reasons") or []),
    }


def evaluation_is_admissible(evaluation):
    counts = evaluation_topology_counts(evaluation)
    explicit = counts["topology_admissible"]
    if explicit is not None:
        return bool(explicit) and not counts["fall"]
    return (
        not counts["fall"]
        and counts["double_lift"] <= 100
        and counts["sample_same_foot"] <= 2
        and counts["m_events"] == 0
        and counts["single_swing_multi_peak"] == 0
    )


def candidate_is_admissible(result):
    return bool(result.get("evaluation")) and evaluation_is_admissible(result.get("evaluation"))


def candidate_quality_key(result):
    evaluation = result.get("evaluation") or {}
    counts = evaluation_topology_counts(evaluation)
    admissible = candidate_is_admissible(result)
    strict = bool(counts["topology_strict_champion"])
    if counts["topology_strict_champion"] is None:
        strict = (
            admissible
            and counts["double_lift"] <= 25
            and counts["sample_same_foot"] == 0
            and counts["m_events"] == 0
            and counts["single_swing_multi_peak"] == 0
        )
    return (
        1 if result.get("evaluation") else 0,
        0 if counts["fall"] else 1,
        1 if admissible else 0,
        1 if strict else 0,
        -counts["sample_same_foot"],
        -counts["double_lift"],
        -counts["m_events"],
        -counts["single_swing_multi_peak"],
        counts["total_score"],
    )


def candidate_quality_summary(result):
    evaluation = result.get("evaluation") or {}
    counts = evaluation_topology_counts(evaluation)
    return {
        "admissible": candidate_is_admissible(result),
        "strict_champion": bool(counts["topology_strict_champion"]),
        "fall": counts["fall"],
        "double_lift_violation_count": counts["double_lift"],
        "sample_same_foot_repeat_count": counts["sample_same_foot"],
        "continuous_m_shape_event_count": counts["m_events"],
        "single_swing_multi_peak_count": counts["single_swing_multi_peak"],
        "topology_gate_reasons": counts["topology_gate_reasons"],
        "selection_key": list(candidate_quality_key(result)),
    }


def clean_checkpoint_record(result):
    if not candidate_is_admissible(result):
        return None
    target_iteration = int(result.get("target_iteration") or 0)
    if target_iteration != 4000:
        return None
    candidate = result.get("candidate") or {}
    evaluation = result.get("evaluation") or {}
    quality = candidate_quality_summary(result)
    return {
        "candidate": candidate.get("name"),
        "artifact_dir": result.get("artifact_dir"),
        "checkpoint_path": result.get("retained_model"),
        "milestone_iteration": target_iteration,
        "gpu": result.get("gpu"),
        "quality": quality,
        "score": (evaluation.get("score") or {}).get("total_score"),
        "reward_scales": dict(candidate.get("reward_scales", {})),
        "resume": {
            "load_run": (result.get("train_meta") or {}).get("load_run"),
            "checkpoint": target_iteration,
            "latest_checkpoint": result.get("retained_model"),
        },
        "source_candidate": candidate.get("name"),
        "source_milestone_iteration": target_iteration,
    }


def clean_checkpoint_sort_key(record):
    quality = record.get("quality") or {}
    return (
        1 if quality.get("admissible") else 0,
        1 if quality.get("strict_champion") else 0,
        -int(quality.get("sample_same_foot_repeat_count") or 0),
        -int(quality.get("double_lift_violation_count") or 0),
        -int(quality.get("continuous_m_shape_event_count") or 0),
        -int(quality.get("single_swing_multi_peak_count") or 0),
        float(record.get("score") or 0.0),
    )


def load_clean_checkpoint_archive(run_root):
    default = {"records": [], "best_4k": None}
    return load_json(run_root / "clean_checkpoint_archive.json", default=default) or default


def update_clean_checkpoint_archive(repo_root, run_root, result, no_commit):
    record = clean_checkpoint_record(result)
    if not record:
        return None

    archive = load_clean_checkpoint_archive(run_root)
    records = [item for item in archive.get("records", []) if item.get("candidate") != record.get("candidate")]
    records.append(record)
    records.sort(key=clean_checkpoint_sort_key, reverse=True)
    archive = {
        "policy": "4K admissible checkpoints are protected rollback parents; 5K is validation and must not overwrite a clean 4K champion.",
        "best_4k": records[0] if records else None,
        "records": records[:20],
    }
    archive_path = run_root / "clean_checkpoint_archive.json"
    write_json(archive_path, archive)
    if not no_commit:
        git_commit_paths(repo_root, [archive_path], f"auto-train: update clean checkpoint archive {run_root.name}")
    return archive


def mutate_candidates(best_candidate, best_evaluation, count, source_cycle):
    if not evaluation_is_admissible(best_evaluation):
        return []
    base = dict(best_candidate.get("reward_scales", {}))
    nodes = best_evaluation.get("score", {}).get("nodes", {})
    children = []
    for index in range(count):
        scales = dict(base)
        multiplier = 1.0 + 0.05 * (index + 1)
        if nodes.get("survival", {}).get("score", 1.0) < 0.95:
            scales["orientation"] = signed_scale(scales.get("orientation", -1.0), 1.15 * multiplier)
            scales["base_height"] = signed_scale(scales.get("base_height", -10.0), 1.15 * multiplier)
            scales["contact"] = signed_scale(scales.get("contact", 0.18), 1.10 * multiplier)
        if nodes.get("velocity_tracking", {}).get("score", 1.0) < 0.75:
            scales["tracking_lin_vel"] = signed_scale(scales.get("tracking_lin_vel", 1.0), 1.10 * multiplier)
            scales["tracking_ang_vel"] = signed_scale(scales.get("tracking_ang_vel", 0.5), 1.05 * multiplier)
        if nodes.get("height_stability", {}).get("score", 1.0) < 0.75:
            scales["base_height"] = signed_scale(scales.get("base_height", -10.0), 1.10 * multiplier)
        if nodes.get("upright", {}).get("score", 1.0) < 0.75:
            scales["orientation"] = signed_scale(scales.get("orientation", -1.0), 1.10 * multiplier)
            scales["landing_foot_posture"] = signed_scale(scales.get("landing_foot_posture", 0.35), 1.05 * multiplier)
        if nodes.get("energy", {}).get("score", 1.0) < 0.75:
            scales["torques"] = signed_scale(scales.get("torques", -0.00001), 1.10 * multiplier)
            scales["dof_acc"] = signed_scale(scales.get("dof_acc", -0.00000025), 1.05 * multiplier)
        if nodes.get("smoothness", {}).get("score", 1.0) < 0.75:
            scales["action_rate"] = signed_scale(scales.get("action_rate", -0.01), 1.10 * multiplier)
            scales["dof_vel"] = signed_scale(scales.get("dof_vel", -0.001), 1.05 * multiplier)
        if nodes.get("foot_trajectory", {}).get("score", 1.0) < 0.80:
            scales["feet_swing_height"] = signed_scale(scales.get("feet_swing_height", -20.0), 1.10 * multiplier)
            scales["landing_foot_posture"] = signed_scale(scales.get("landing_foot_posture", 0.35), 1.05 * multiplier)
        if nodes.get("foot_alternation", {}).get("score", 1.0) < 0.80:
            scales["contact"] = signed_scale(scales.get("contact", 0.18), 1.12 * multiplier)
            scales["feet_swing_height"] = signed_scale(scales.get("feet_swing_height", -20.0), 1.08 * multiplier)
        children.append({"name": f"mutated_c{source_cycle:03d}_{index:02d}", "reward_scales": scales})
    return children


def next_cycle_candidates(best_result, current_count, cycle_index):
    if not candidate_is_admissible(best_result):
        return []
    best_candidate = best_result["candidate"]
    best_evaluation = best_result["evaluation"]
    next_candidates = [
        {
            "name": f"elite_c{cycle_index:03d}_{safe_name(best_candidate['name'])}",
            "reward_scales": dict(best_candidate.get("reward_scales", {})),
            "source_candidate": best_candidate["name"],
            "source_score": best_evaluation["score"]["total_score"],
        }
    ]
    mutation_count = max(1, current_count - 1)
    next_candidates.extend(mutate_candidates(best_candidate, best_evaluation, mutation_count, cycle_index))
    return next_candidates


def summarize_cycle(cycle_dir, results):
    scored = []
    for result in results:
        evaluation = result.get("evaluation")
        if not evaluation:
            continue
        scored.append((candidate_quality_key(result), result))
    scored.sort(key=lambda item: item[0], reverse=True)
    clean_records = [clean_checkpoint_record(result) for _, result in scored]
    clean_records = [record for record in clean_records if record]
    clean_records.sort(key=clean_checkpoint_sort_key, reverse=True)
    summary = {
        "candidates": results,
        "best": scored[0][1] if scored else None,
        "best_clean_checkpoint": clean_records[0] if clean_records else None,
        "best_4k_checkpoint": clean_records[0] if clean_records else None,
        "clean_checkpoint_candidates": clean_records,
        "selection_order": [
            {
                "candidate": result.get("candidate", {}).get("name"),
                "total_score": result.get("evaluation", {}).get("score", {}).get("total_score"),
                "quality": candidate_quality_summary(result),
            }
            for _, result in scored
        ],
    }
    write_json(cycle_dir / "cycle_summary.json", summary)

    lines = ["# Auto Train Cycle Summary", ""]
    if summary.get("best_clean_checkpoint"):
        clean = summary["best_clean_checkpoint"]
        quality = clean.get("quality") or {}
        lines.append(
            f"- Protected 4K clean checkpoint: {clean.get('candidate')} "
            f"score={clean.get('score')} strict={quality.get('strict_champion')} "
            f"same_foot={quality.get('sample_same_foot_repeat_count')} "
            f"double_lift={quality.get('double_lift_violation_count')}"
        )

    for _, result in scored:
        score = result.get("evaluation", {}).get("score", {}).get("total_score", 0.0)
        quality = candidate_quality_summary(result)
        gate = ",".join(quality["topology_gate_reasons"]) or "none"
        lines.append(
            f"- {result['candidate']['name']}: {score:.4f} "
            f"admissible={quality['admissible']} strict={quality['strict_champion']} "
            f"same_foot={quality['sample_same_foot_repeat_count']} "
            f"double_lift={quality['double_lift_violation_count']} gate={gate}"
        )
    if not scored:
        lines.append("- No successful evaluations.")
    (cycle_dir / "cycle_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def configured_max_cycles(config):
    cycles = config.get("cycles")
    if cycles is None:
        return None
    return int(cycles)


def stop_paths(repo_root, run_root, config):
    paths = [run_root / "STOP", repo_root / "STOP_AUTO_TRAIN"]
    for raw_path in config.get("control", {}).get("stop_files", []):
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        paths.append(path)
    return paths


def user_stop_reason(repo_root, run_root, config):
    for path in stop_paths(repo_root, run_root, config):
        if path.exists():
            return {"reason": "user_stop", "path": str(path)}
    return None


def convergence_stop_reason(config, best_scores):
    if not best_scores:
        return None

    termination = config.get("termination", {})
    latest = best_scores[-1]
    score_target = float(termination.get("score_target", 0.92))
    if latest >= score_target:
        return {
            "reason": "score_target_reached",
            "latest_score": latest,
            "score_target": score_target,
        }

    min_cycles = int(termination.get("min_cycles", 4))
    patience = int(termination.get("patience_cycles", 3))
    min_delta = float(termination.get("min_delta", 0.005))
    if len(best_scores) < min_cycles or len(best_scores) <= patience:
        return None

    previous_best = max(best_scores[:-patience])
    recent_best = max(best_scores[-patience:])
    improvement = recent_best - previous_best
    if improvement < min_delta:
        return {
            "reason": "score_plateau",
            "previous_best": previous_best,
            "recent_best": recent_best,
            "improvement": improvement,
            "min_delta": min_delta,
            "patience_cycles": patience,
        }
    return None


def next_iteration_budget(config, current_iterations, best_scores):
    dynamic = config.get("dynamic_iterations", {})
    if not dynamic.get("enabled", True):
        return current_iterations

    max_iterations = int(dynamic.get("max_iterations", current_iterations))
    increment = int(dynamic.get("increment", 0))
    if current_iterations >= max_iterations or increment <= 0:
        return current_iterations

    latest = best_scores[-1] if best_scores else 0.0
    score_threshold = float(dynamic.get("score_threshold", 0.90))
    previous_best = max(best_scores[:-1]) if len(best_scores) > 1 else None
    improvement = latest - previous_best if previous_best is not None else None
    plateau_delta = float(dynamic.get("plateau_delta", 0.005))

    should_extend = latest < score_threshold
    if improvement is not None and improvement < plateau_delta:
        should_extend = True
    if not should_extend:
        return current_iterations
    return min(max_iterations, current_iterations + increment)


def candidate_review(result):
    evaluation = result.get("evaluation") or {}
    score = evaluation.get("score") or {}
    simlog = evaluation.get("simlog") or {}
    return {
        "candidate": result.get("candidate", {}).get("name"),
        "gpu": result.get("gpu"),
        "artifact_dir": result.get("artifact_dir"),
        "retained_model": result.get("retained_model"),
        "checkpoint_path": result.get("retained_model"),
        "start_iteration": result.get("start_iteration"),
        "milestone_iteration": result.get("target_iteration"),
        "train_max_iterations": result.get("train_max_iterations"),
        "eval_milestones": result.get("eval_milestones"),
        "suggested_continuation": result.get("suggested_continuation"),
        "total_score": score.get("total_score"),
        "fall_gate": score.get("fall_gate"),
        "score_nodes": score.get("nodes"),
        "metrics": evaluation.get("metrics"),
        "quality": candidate_quality_summary(result) if result.get("evaluation") else None,
        "topology_admissible": score.get("topology_admissible"),
        "topology_strict_champion": score.get("topology_strict_champion"),
        "topology_gate_reasons": score.get("topology_gate_reasons"),
        "simlog_issues": simlog.get("issues", []),
        "trajectory_csv": evaluation.get("trajectory_csv"),
        "trajectory_plot": evaluation.get("trajectory_plot"),
        "train_log": str(Path(result.get("artifact_dir", "")) / "train.log") if result.get("artifact_dir") else None,
        "evaluate_log": str(Path(result.get("artifact_dir", "")) / "evaluate.log") if result.get("artifact_dir") else None,
    }


def write_codex_review_request(
    run_root,
    cycle_dir,
    cycle_index,
    summary,
    best_scores,
    suggested_candidates,
    suggested_iterations,
    decision_filename,
):
    reviews = [candidate_review(result) for result in summary.get("candidates", [])]
    best = candidate_review(summary["best"]) if summary.get("best") else None
    clean_archive = load_clean_checkpoint_archive(run_root)
    decision_path = cycle_dir / decision_filename
    request = {
        "owner": "codex",
        "state": "waiting_for_codex_decision",
        "cycle": cycle_index,
        "decision_path": str(decision_path),
        "decision_scope": "slot-local",
        "stop_scope": "slot-local",
        "best_scores": best_scores,
        "best": best,
        "candidates": reviews,
        "clean_checkpoint_archive": clean_archive,
        "protected_4k_checkpoint": clean_archive.get("best_4k"),
        "summary_best_clean_checkpoint": summary.get("best_clean_checkpoint"),
        "suggested_train_max_iterations": suggested_iterations,
        "suggested_next_candidates": suggested_candidates,
        "rules": [
            "Codex is the optimization owner; auto_train executes one continuous 10K training process per logical candidate and evaluates saved checkpoints afterward.",
            "A logical candidate must keep one run_name, one log directory, and one TensorBoard series; 1K-9K checkpoints never trigger stop/resume segmentation.",
            "OpenDuckPro2 reward names, formulas, masks, targets, and phase logic remain aligned with G1.",
            "Candidate search may modify only existing nonzero G1 reward scales; reward source changes are forbidden.",
            "Intermediate 1K-9K evaluations are trend evidence only. The Codex decision is made after the continuous 10K run and post-hoc milestone evaluation complete.",
            "Mature gait admission uses episode fall rate plus 60 ms debounced, cycle-normalized contact metrics; legacy raw contact counts remain diagnostic.",
            "Dirty candidates must not be used as mutation parents.",
            "A replacement candidate requires its own draft PR, descriptive run name, independent TensorBoard directory, and full 10K budget.",
            "candidate-local stop is allowed only for runtime failure or an explicit user stop, not ordinary immature milestone behavior.",
        ],
        "decision_schema": {
            "action": "continue | auto_mutate | stop",
            "decision_scope": "slot-local; stop means stop current candidate/slot review, not the whole run",
            "train_max_iterations": "optional int; defaults to suggested_train_max_iterations",
            "next_candidates": "required for action=continue unless allow_auto_mutation_fallback is enabled; optional for candidate-local stop",
            "reward_source_commit": "required unchanged G1-aligned training source git sha",
            "reward_source_changes": {
                "added_rewards": "must be an empty list",
                "removed_rewards": "must be an empty list",
                "modified_rewards": "must be an empty list",
                "scale_only": "must be true",
                "evidence": "short mapping from post-hoc milestone metrics to the scale-only decision",
            },
            "notes": "required short explanation of the 10K evidence and chosen scale-only action",
        },
    }
    write_json(cycle_dir / "codex_review_request.json", request)

    lines = [
        "# Codex Review Request",
        "",
        f"- Run: `{run_root.name}`",
        f"- Cycle: `{cycle_index}`",
        f"- Decision file: `{decision_path}`",
        f"- Suggested next train iterations: `{suggested_iterations}`",
        "",
        "## Best",
    ]
    if best:
        lines.append(f"- Candidate: `{best['candidate']}`")
        lines.append(f"- Milestone iteration: `{best.get('milestone_iteration')}`")
        lines.append(f"- Checkpoint: `{best.get('checkpoint_path')}`")
        lines.append(f"- Score: `{best['total_score']}`")
        lines.append(f"- Fall gate: `{best['fall_gate']}`")
        lines.append(f"- Trajectory plot: `{best['trajectory_plot']}`")
        for issue in best.get("simlog_issues", []):
            lines.append(f"- Issue `{issue['code']}`: {issue['codex_reward_action']}")
    else:
        lines.append("- No successful evaluation.")

    protected = clean_archive.get("best_4k") if isinstance(clean_archive, dict) else None
    if protected:
        quality = protected.get("quality") or {}
        lines.extend(
            [
                "",
                "## Protected 4K Checkpoint",
                f"- Candidate: `{protected.get('candidate')}`",
                f"- Checkpoint: `{protected.get('checkpoint_path')}`",
                f"- Score: `{protected.get('score')}`",
                f"- Strict champion: `{quality.get('strict_champion')}`",
                f"- Same-foot repeats: `{quality.get('sample_same_foot_repeat_count')}`",
                f"- Double-lift violations: `{quality.get('double_lift_violation_count')}`",
                "- Policy: 5K is validation; if 5K regresses, prefer this clean 4K parent or candidate-local stop.",
            ]
        )
    lines.extend(
        [
            "",
            "## Required Codex Action",
            "",
            "Inspect the simlog artifacts, edit reward source logic when needed, commit the change, then write `codex_decision.json`.",
            "Use reward scale changes only for small retuning or when explicitly justified by the simlog.",
            "",
            "Example decision:",
            "",
            "```json",
            "{",
            '  "action": "continue",',
            f'  "train_max_iterations": {suggested_iterations},',
            f'  "milestone_iteration": {best.get("milestone_iteration") if best else None},',
            f'  "checkpoint_path": "{best.get("checkpoint_path") if best else ""}",',
            '  "reward_source_commit": "<git-sha-or-empty>",',
            '  "reward_source_changes": {',
            '    "added_rewards": ["alternating_lift_sequence"],',
            '    "removed_rewards": [],',
            '    "modified_rewards": ["feet_swing_height"],',
            '    "scale_only": false,',
            '    "evidence": "simlog showed repeated same-foot lift-offs and foot reference error"',
            "  },",
            '  "notes": "Codex changed reward logic because simlog showed repeated same-foot lift-offs.",',
            '  "next_candidates": [',
            '    {"name": "codex_reward_fix_c001", "reward_scales": {}}',
            "  ]",
            "}",
            "```",
        ]
    )
    (cycle_dir / "codex_review_request.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return request


def load_codex_decision(path):
    decision = load_json(path)
    action = decision.get("action")
    if action not in {"continue", "auto_mutate", "stop"}:
        raise ValueError(f"Invalid codex decision action: {action}")
    return decision


def wait_for_codex_decision(repo_root, run_root, cycle_dir, config, review_request, no_commit):
    codex_cfg = config.get("codex_control", {})
    decision_path = Path(review_request["decision_path"])
    heartbeat_interval = max(5, int(codex_cfg.get("heartbeat_interval_sec", 30)))
    last_heartbeat = 0.0
    error_path = cycle_dir / "codex_decision_error.json"

    while True:
        stop_reason = user_stop_reason(repo_root, run_root, config)
        if stop_reason:
            return {"action": "stop", "stop_reason": stop_reason}

        now = time.time()
        if now - last_heartbeat >= heartbeat_interval:
            write_heartbeat(
                run_root / "heartbeat.json",
                "waiting_for_codex_decision",
                cycle=review_request["cycle"],
                decision_path=str(decision_path),
                best_score=(review_request.get("best") or {}).get("total_score"),
            )
            last_heartbeat = now

        if decision_path.exists():
            try:
                decision = load_codex_decision(decision_path)
                if (
                    decision["action"] == "continue"
                    and not decision.get("next_candidates")
                    and not codex_cfg.get("allow_auto_mutation_fallback", False)
                ):
                    raise ValueError("action=continue requires next_candidates")
                if decision["action"] == "continue":
                    changes = decision.get("reward_source_changes")
                    if not isinstance(changes, dict):
                        raise ValueError("action=continue requires reward_source_changes")
                    if not changes.get("scale_only", False) and not decision.get("reward_source_commit"):
                        raise ValueError("source reward changes require reward_source_commit")
            except Exception as exc:  # noqa: BLE001
                write_json(
                    error_path,
                    {
                        "error": str(exc),
                        "decision_path": str(decision_path),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    },
                )
                time.sleep(heartbeat_interval)
                continue

            write_json(cycle_dir / "codex_decision_accepted.json", decision)
            if not no_commit:
                git_commit_paths(
                    repo_root,
                    [decision_path, cycle_dir / "codex_decision_accepted.json"],
                    f"auto-train: accept {run_root.name} cycle {review_request['cycle']} codex decision",
                )
            return decision

        time.sleep(5)


def streaming_suggested_candidates(result, count, source_index):
    if result.get("continuous_training") or int(count) <= 0:
        return []
    if not result.get("evaluation") or not candidate_is_admissible(result):
        return []
    if result.get("suggested_continuation"):
        return [result["suggested_continuation"]]
    return mutate_candidates(
        result["candidate"],
        result["evaluation"],
        max(1, int(count)),
        source_index,
    )


def candidates_from_decision(decision, suggested_candidates, config):
    codex_cfg = config.get("codex_control", {})
    if decision["action"] == "auto_mutate":
        return decision.get("next_candidates") or suggested_candidates
    candidates = decision.get("next_candidates")
    if not candidates and codex_cfg.get("allow_auto_mutation_fallback", False):
        candidates = suggested_candidates
    return candidates or []


def request_running_jobs_stop(run_root):
    try:
        (run_root / "STOP").touch()
    except OSError:
        pass


def write_streaming_state(run_root, run_id, active, queued_count, completed_count, current_iterations):
    write_heartbeat(
        run_root / "heartbeat.json",
        "streaming_running",
        run_id=run_id,
        active_jobs=len(active),
        queued_candidates=queued_count,
        completed_candidates=completed_count,
        train_max_iterations=current_iterations,
    )


def main_streaming(args, repo_root, config, run_id, run_root):
    stream_cfg = config.get("streaming_candidates", {})
    gpus = config.get("gpus", [0])
    max_workers = min(int(config.get("max_parallel_jobs", len(gpus))), len(gpus))
    current_iterations = int(config["train"]["max_iterations"])
    best_scores = []
    completed_results = []
    candidate_queue = list(config.get("candidates", []))
    candidate_counter = 0
    cycle_index = 0
    stop_check = lambda: user_stop_reason(repo_root, run_root, config)

    stream_root = run_root / "streaming"
    stream_root.mkdir(parents=True, exist_ok=True)
    write_json(
        stream_root / "streaming_config.json",
        {
            "enabled": True,
            "max_workers": max_workers,
            "gpus": gpus,
            "initial_candidates": candidate_queue,
            "suggestions_per_completion": int(stream_cfg.get("suggestions_per_completion", 1)),
        },
    )
    if not args.no_commit:
        git_commit_paths(
            repo_root,
            [stream_root / "streaming_config.json"],
            f"auto-train: enable streaming candidates {run_id}",
        )

    def launch_candidate(executor, active, slot_index, gpu, candidate):
        nonlocal candidate_counter
        candidate_index = candidate_counter
        candidate_counter += 1
        run_config = copy.deepcopy(config)
        run_config["train"]["max_iterations"] = current_iterations
        future = executor.submit(
            run_candidate,
            repo_root,
            run_config,
            run_id,
            cycle_index,
            candidate_index,
            candidate,
            gpu,
            args.dry_run,
            stop_check,
        )
        active[future] = {
            "slot_index": slot_index,
            "gpu": gpu,
            "candidate_index": candidate_index,
            "candidate": candidate,
        }
        write_json(
            stream_root / f"launch_{candidate_index:04d}.json",
            {
                "candidate_index": candidate_index,
                "slot_index": slot_index,
                "gpu": gpu,
                "candidate": candidate,
                "train_max_iterations": current_iterations,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        return future

    def record_slot_event(candidate_index, metadata, reason, candidate_dir=None):
        path = stream_root / f"candidate_{candidate_index:04d}_slot_event.json"
        payload = {
            "candidate_index": candidate_index,
            "slot_index": metadata.get("slot_index"),
            "gpu": metadata.get("gpu"),
            "candidate": metadata.get("candidate"),
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        write_json(path, payload)
        if not args.no_commit:
            paths = [path]
            if candidate_dir is not None:
                paths.append(candidate_dir)
            git_commit_paths(
                repo_root,
                paths,
                f"auto-train: record {run_id} candidate {candidate_index} slot event",
            )

    def refill_slot(executor, active, metadata):
        if candidate_queue:
            launch_candidate(
                executor,
                active,
                metadata["slot_index"],
                metadata["gpu"],
                candidate_queue.pop(0),
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        active = {}
        for slot_index, gpu in enumerate(gpus[:max_workers]):
            if not candidate_queue:
                break
            launch_candidate(executor, active, slot_index, gpu, candidate_queue.pop(0))

        while active:
            stop_reason = user_stop_reason(repo_root, run_root, config)
            if stop_reason:
                request_running_jobs_stop(run_root)
                write_stop_reason(repo_root, run_root, stop_reason, args.no_commit)
                return

            write_streaming_state(
                run_root,
                run_id,
                active,
                len(candidate_queue),
                len(completed_results),
                current_iterations,
            )
            done, _ = concurrent.futures.wait(
                active.keys(),
                timeout=5,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                continue

            for future in done:
                metadata = active.pop(future)
                candidate_index = metadata["candidate_index"]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    record_slot_event(
                        candidate_index,
                        metadata,
                        {
                            "type": "candidate_exception",
                            "error": str(exc),
                        },
                    )
                    refill_slot(executor, active, metadata)
                    continue

                completed_results.append(result)
                candidate_dir = Path(result["artifact_dir"])
                if not args.no_commit:
                    git_commit_paths(
                        repo_root,
                        [candidate_dir],
                        f"auto-train: record {run_id} candidate {candidate_index} result",
                    )

                if not result.get("evaluation"):
                    record_slot_event(
                        candidate_index,
                        metadata,
                        {
                            "type": "candidate_without_evaluation",
                            "candidate_index": candidate_index,
                            "candidate": result.get("candidate", {}).get("name"),
                            "train_returncode": result.get("train_returncode"),
                            "eval_returncode": result.get("eval_returncode"),
                        },
                        candidate_dir,
                    )
                    refill_slot(executor, active, metadata)
                    continue

                update_clean_checkpoint_archive(repo_root, run_root, result, args.no_commit)

                score = result["evaluation"]["score"]["total_score"]
                best_scores.append(score)
                is_final_target = int(result.get("target_iteration") or 0) >= int(result.get("train_max_iterations") or current_iterations)
                if is_final_target and stream_cfg.get("allow_convergence_stop", False):
                    stop_reason = convergence_stop_reason(config, best_scores)
                    if stop_reason:
                        request_running_jobs_stop(run_root)
                        stop_reason["candidate_index"] = candidate_index
                        stop_reason["best_scores"] = best_scores
                        write_stop_reason(repo_root, run_root, stop_reason, args.no_commit)
                        return

                suggested_iterations = (
                    next_iteration_budget(config, current_iterations, best_scores)
                    if is_final_target
                    else current_iterations
                )
                suggested_candidates = streaming_suggested_candidates(
                    result,
                    int(stream_cfg.get("suggestions_per_completion", 1)),
                    candidate_index,
                )

                codex_cfg = config.get("codex_control", {})
                if codex_cfg.get("enabled", True):
                    decision_filename = codex_cfg.get("decision_filename", "codex_decision.json")
                    summary = {"candidates": [result], "best": result}
                    review_request = write_codex_review_request(
                        run_root=run_root,
                        cycle_dir=candidate_dir,
                        cycle_index=candidate_index,
                        summary=summary,
                        best_scores=best_scores,
                        suggested_candidates=suggested_candidates,
                        suggested_iterations=suggested_iterations,
                        decision_filename=decision_filename,
                    )
                    if not args.no_commit:
                        git_commit_paths(
                            repo_root,
                            [candidate_dir / "codex_review_request.json", candidate_dir / "codex_review_request.md"],
                            f"auto-train: request codex review {run_id} candidate {candidate_index}",
                        )

                    if not codex_cfg.get("wait_for_decision", True) or args.dry_run:
                        record_slot_event(
                            candidate_index,
                            metadata,
                            {
                                "type": "codex_decision_required",
                                "candidate_index": candidate_index,
                                "decision_path": review_request["decision_path"],
                            },
                            candidate_dir,
                        )
                        refill_slot(executor, active, metadata)
                        continue

                    decision = wait_for_codex_decision(
                        repo_root,
                        run_root,
                        candidate_dir,
                        config,
                        review_request,
                        args.no_commit,
                    )
                    if decision.get("action") == "stop":
                        reason = decision.get("stop_reason") or {
                            "type": "codex_stop",
                            "candidate_index": candidate_index,
                            "notes": decision.get("notes"),
                        }
                        record_slot_event(candidate_index, metadata, reason, candidate_dir)
                        replacement_candidates = decision.get("next_candidates") or []
                        if stream_cfg.get("prefer_codex_candidates", True):
                            candidate_queue[:0] = replacement_candidates
                        else:
                            candidate_queue.extend(replacement_candidates)
                        refill_slot(executor, active, metadata)
                        continue

                    current_iterations = int(decision.get("train_max_iterations", suggested_iterations))
                    next_candidates = candidates_from_decision(decision, suggested_candidates, config)
                    if not next_candidates:
                        record_slot_event(
                            candidate_index,
                            metadata,
                            {
                                "type": "codex_decision_missing_next_candidates",
                                "candidate_index": candidate_index,
                                "decision_path": review_request["decision_path"],
                            },
                            candidate_dir,
                        )
                        refill_slot(executor, active, metadata)
                        continue
                    if stream_cfg.get("prefer_codex_candidates", True):
                        candidate_queue[:0] = next_candidates
                    else:
                        candidate_queue.extend(next_candidates)
                else:
                    current_iterations = suggested_iterations
                    candidate_queue[:0] = suggested_candidates

                refill_slot(executor, active, metadata)

    write_stop_reason(
        repo_root,
        run_root,
        {
            "reason": "streaming_no_active_jobs",
            "completed_candidates": len(completed_results),
            "queued_candidates": len(candidate_queue),
        },
        args.no_commit,
    )


def write_stop_reason(repo_root, run_root, reason, no_commit):
    write_json(run_root / "stop_reason.json", reason)
    if not no_commit:
        git_commit_paths(repo_root, [run_root / "stop_reason.json"], f"auto-train: stop {run_root.name} {reason['reason']}")


def main():
    args = parse_args()
    repo_root = repo_root_from_script(__file__)
    config = merged_config(repo_root / args.config)
    if args.cycles is not None:
        config["cycles"] = args.cycles
    validate_continuous_training_config(config)

    run_id = args.run_id or timestamp_id("openduck_auto")
    run_root = repo_root / "auto_train_runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run_config.json", config)
    write_heartbeat(run_root / "heartbeat.json", "started", run_id=run_id)

    if not args.no_commit:
        git_commit_paths(repo_root, [run_root / "run_config.json"], f"auto-train: start {run_id}")

    if config.get("streaming_candidates", {}).get("enabled", False):
        main_streaming(args, repo_root, config, run_id, run_root)
        return

    candidates = config["candidates"]
    gpus = config.get("gpus", [0])
    max_workers = min(int(config.get("max_parallel_jobs", len(gpus))), len(gpus))
    max_cycle_count = configured_max_cycles(config)
    current_iterations = int(config["train"]["max_iterations"])
    best_scores = []
    cycle_index = 0

    while True:
        if max_cycle_count is not None and cycle_index >= max_cycle_count:
            write_stop_reason(
                repo_root,
                run_root,
                {"reason": "cycle_cap_reached", "max_cycles": max_cycle_count},
                args.no_commit,
            )
            break

        stop_reason = user_stop_reason(repo_root, run_root, config)
        if stop_reason:
            write_stop_reason(repo_root, run_root, stop_reason, args.no_commit)
            break

        cycle_config = copy.deepcopy(config)
        cycle_config["train"]["max_iterations"] = current_iterations
        cycle_dir = run_root / f"cycle_{cycle_index:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        write_heartbeat(
            run_root / "heartbeat.json",
            "cycle_running",
            run_id=run_id,
            cycle=cycle_index,
            train_max_iterations=current_iterations,
            candidate_count=len(candidates),
        )
        write_json(
            cycle_dir / "cycle_config.json",
            {
                "cycle": cycle_index,
                "train_max_iterations": current_iterations,
                "stop_files": [str(path) for path in stop_paths(repo_root, run_root, config)],
            },
        )
        write_json(cycle_dir / "candidates.json", candidates)
        if not args.no_commit:
            git_commit_paths(
                repo_root,
                [cycle_dir / "cycle_config.json", cycle_dir / "candidates.json"],
                f"auto-train: add {run_id} cycle {cycle_index} candidates",
            )

        results = []
        stop_check = lambda: user_stop_reason(repo_root, run_root, config)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, candidate in enumerate(candidates):
                gpu = gpus[idx % len(gpus)]
                futures.append(
                    executor.submit(
                        run_candidate,
                        repo_root,
                        cycle_config,
                        run_id,
                        cycle_index,
                        idx,
                        candidate,
                        gpu,
                        args.dry_run,
                        stop_check,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        for result in results:
            if result.get("evaluation"):
                update_clean_checkpoint_archive(repo_root, run_root, result, args.no_commit)

        summary = summarize_cycle(cycle_dir, results)
        write_heartbeat(
            run_root / "heartbeat.json",
            "cycle_evaluated",
            run_id=run_id,
            cycle=cycle_index,
            successful_evaluations=len([result for result in results if result.get("evaluation")]),
        )
        if not args.no_commit:
            paths = [cycle_dir]
            git_commit_paths(repo_root, paths, f"auto-train: record {run_id} cycle {cycle_index} results")

        best = summary.get("best")
        stop_reason = user_stop_reason(repo_root, run_root, config)
        if stop_reason:
            write_stop_reason(repo_root, run_root, stop_reason, args.no_commit)
            break

        if best is None:
            write_stop_reason(
                repo_root,
                run_root,
                {"reason": "no_successful_evaluations", "cycle": cycle_index},
                args.no_commit,
            )
            break

        best_scores.append(best["evaluation"]["score"]["total_score"])

        stop_reason = convergence_stop_reason(config, best_scores)
        if stop_reason:
            stop_reason["cycle"] = cycle_index
            stop_reason["best_scores"] = best_scores
            write_stop_reason(repo_root, run_root, stop_reason, args.no_commit)
            break

        suggested_iterations = next_iteration_budget(config, current_iterations, best_scores)
        suggested_candidates = next_cycle_candidates(best, len(candidates), cycle_index)
        codex_cfg = config.get("codex_control", {})
        if codex_cfg.get("enabled", True):
            decision_filename = codex_cfg.get("decision_filename", "codex_decision.json")
            review_request = write_codex_review_request(
                run_root=run_root,
                cycle_dir=cycle_dir,
                cycle_index=cycle_index,
                summary=summary,
                best_scores=best_scores,
                suggested_candidates=suggested_candidates,
                suggested_iterations=suggested_iterations,
                decision_filename=decision_filename,
            )
            if not args.no_commit:
                git_commit_paths(
                    repo_root,
                    [cycle_dir / "codex_review_request.json", cycle_dir / "codex_review_request.md"],
                    f"auto-train: request codex review {run_id} cycle {cycle_index}",
                )

            if not codex_cfg.get("wait_for_decision", True) or args.dry_run:
                write_stop_reason(
                    repo_root,
                    run_root,
                    {
                        "reason": "codex_decision_required",
                        "cycle": cycle_index,
                        "decision_path": review_request["decision_path"],
                    },
                    args.no_commit,
                )
                break

            decision = wait_for_codex_decision(repo_root, run_root, cycle_dir, config, review_request, args.no_commit)
            if decision.get("action") == "stop":
                reason = decision.get("stop_reason") or {
                    "reason": "codex_stop",
                    "cycle": cycle_index,
                    "notes": decision.get("notes"),
                }
                write_stop_reason(repo_root, run_root, reason, args.no_commit)
                break

            current_iterations = int(decision.get("train_max_iterations", suggested_iterations))
            if decision["action"] == "auto_mutate":
                candidates = decision.get("next_candidates") or suggested_candidates
            else:
                candidates = decision.get("next_candidates")
                if not candidates and codex_cfg.get("allow_auto_mutation_fallback", False):
                    candidates = suggested_candidates
                if not candidates:
                    write_stop_reason(
                        repo_root,
                        run_root,
                        {
                            "reason": "codex_decision_missing_next_candidates",
                            "cycle": cycle_index,
                            "decision_path": review_request["decision_path"],
                        },
                        args.no_commit,
                    )
                    break
        else:
            current_iterations = suggested_iterations
            candidates = suggested_candidates
        cycle_index += 1


if __name__ == "__main__":
    main()
