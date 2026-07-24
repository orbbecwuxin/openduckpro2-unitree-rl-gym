#!/usr/bin/env python3
"""Finite OpenDuckPro3 Isaac Lab environment smoke test."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-OpenDuckPro3-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=32)
parser.add_argument(
    "--events",
    choices=("all", "none", "friction", "mass", "push"),
    default="all",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_openduck  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs, use_fabric=True
    )
    if args.events == "none":
        env_cfg.events = None
    elif args.events != "all":
        env_cfg.events.physics_material = (
            env_cfg.events.physics_material if args.events == "friction" else None
        )
        env_cfg.events.add_base_mass = (
            env_cfg.events.add_base_mass if args.events == "mass" else None
        )
        env_cfg.events.push_robot = (
            env_cfg.events.push_robot if args.events == "push" else None
        )

    env = gym.make(args.task, cfg=env_cfg)
    observations, _ = env.reset()
    for _ in range(args.steps):
        actions = torch.zeros(env.unwrapped.num_envs, 10, device=env.unwrapped.device)
        observations, rewards, terminated, truncated, _ = env.step(actions)
        assert torch.isfinite(rewards).all()
        assert not torch.any(terminated & truncated)

    policy = observations["policy"]
    critic = observations["critic"]
    print(
        "SMOKE_OK "
        f"events={args.events} steps={args.steps} "
        f"policy={tuple(policy.shape)} critic={tuple(critic.shape)} "
        f"reward_mean={rewards.mean().item():.6f}"
    )
    print(f"JOINTS={env.unwrapped._robot.data.joint_names}")
    print(f"FEET={env.unwrapped._feet_names}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
