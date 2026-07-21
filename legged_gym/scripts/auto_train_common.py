import json
import os
import re
import subprocess
import time
from pathlib import Path


MODEL_RE = re.compile(r"model_(\d+)\.pt$")


def repo_root_from_script(script_file):
    return Path(script_file).resolve().parents[2]


def load_json(path, default=None):
    if path is None:
        return default
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def read_reward_scales(env_cfg):
    scales = {}
    for name in dir(env_cfg.rewards.scales):
        if name.startswith("_"):
            continue
        value = getattr(env_cfg.rewards.scales, name)
        if isinstance(value, (int, float)):
            scales[name] = float(value)
    return scales


def apply_reward_overrides(env_cfg, override_path=None, overrides=None):
    if overrides is None:
        overrides = load_json(override_path, default={}) or {}
    reward_scales = overrides.get("reward_scales")
    if reward_scales is None:
        reward_scales = overrides.get("rewards", {}).get("scales", {})
    for name, value in reward_scales.items():
        if not hasattr(env_cfg.rewards.scales, name):
            raise ValueError(f"Unknown reward scale: {name}")
        setattr(env_cfg.rewards.scales, name, float(value))
    return reward_scales


def latest_run_dir(log_root, run_name=None, min_mtime=0.0):
    log_root = Path(log_root)
    if not log_root.exists():
        return None
    candidates = []
    for child in log_root.iterdir():
        if not child.is_dir() or child.name == "exported":
            continue
        if run_name and not child.name.endswith("_" + run_name):
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        if mtime >= min_mtime - 1.0:
            candidates.append((mtime, child))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def latest_checkpoint(run_dir):
    run_dir = Path(run_dir)
    models = []
    for path in run_dir.glob("model_*.pt"):
        match = MODEL_RE.match(path.name)
        if match:
            models.append((int(match.group(1)), path))
    if not models:
        return None
    models.sort(key=lambda item: item[0])
    return models[-1][1]


def git_commit_paths(repo_root, paths, message):
    repo_root = Path(repo_root).resolve()
    rel_paths = []
    for path in paths:
        resolved = Path(path).resolve()
        try:
            rel_paths.append(str(resolved.relative_to(repo_root)))
        except ValueError as exc:
            raise ValueError(f"Refusing to commit path outside repo: {resolved}") from exc

    # Training runs contain checkpoints and logs, not versioned source.  They are
    # deliberately ignored so a completed candidate cannot terminate the
    # streaming controller merely because there is nothing safe to stage.
    generated_roots = ("auto_train_runs", "auto_train_workspaces")
    stageable_paths = [
        path
        for path in rel_paths
        if not any(path == root or path.startswith(root + "/") for root in generated_roots)
    ]
    if not stageable_paths:
        return {"committed": False, "reason": "generated artifacts are not versioned"}

    subprocess.run(["git", "add", "--", *stageable_paths], cwd=repo_root, check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_root)
    if diff.returncode == 0:
        return {"committed": False, "reason": "no staged changes"}

    cmd = [
        "git",
        "-c",
        "user.name=Auto Train",
        "-c",
        "user.email=auto-train@local",
        "commit",
        "-m",
        message,
    ]
    result = subprocess.run(cmd, cwd=repo_root, check=True, text=True, capture_output=True)
    return {"committed": True, "output": result.stdout.strip()}


def timestamp_id(prefix="auto"):
    return time.strftime(prefix + "_%Y%m%d_%H%M%S")
