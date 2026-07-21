import os

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args()
    env, _ = task_registry.make_env(name=args.task, args=args)
    max_steps = int(os.environ.get("MAX_STAND_STEPS", "1000000"))
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    with torch.no_grad():
        for _ in range(max_steps):
            env.step(actions)


if __name__ == "__main__":
    main()
