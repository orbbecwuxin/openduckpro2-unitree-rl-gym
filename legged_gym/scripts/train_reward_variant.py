import argparse
import sys
import time
from pathlib import Path

import isaacgym  # noqa: F401

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

from auto_train_common import (
    apply_reward_overrides,
    latest_checkpoint,
    latest_run_dir,
    load_json,
    read_reward_scales,
    write_json,
)


def parse_custom_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reward-overrides")
    parser.add_argument("--auto-train-meta")
    custom, remaining = parser.parse_known_args(argv)
    return custom, remaining


def train(args, custom):
    started_at = time.time()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    requested_overrides = apply_reward_overrides(env_cfg, custom.reward_overrides)

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )

    log_root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name
    run_dir = latest_run_dir(log_root, train_cfg.runner.run_name, min_mtime=started_at)
    checkpoint = latest_checkpoint(run_dir) if run_dir is not None else None

    if custom.auto_train_meta:
        write_json(
            custom.auto_train_meta,
            {
                "task": args.task,
                "experiment_name": train_cfg.runner.experiment_name,
                "run_name": train_cfg.runner.run_name,
                "run_dir": str(run_dir) if run_dir is not None else None,
                "load_run": run_dir.name if run_dir is not None else None,
                "latest_checkpoint": str(checkpoint) if checkpoint is not None else None,
                "requested_reward_overrides": requested_overrides,
                "effective_reward_scales": read_reward_scales(env_cfg),
            },
        )


if __name__ == "__main__":
    custom_args, passthrough = parse_custom_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *passthrough]
    train(get_args(), custom_args)
