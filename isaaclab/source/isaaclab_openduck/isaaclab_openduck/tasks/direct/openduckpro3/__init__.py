"""OpenDuckPro3 velocity tracking environment."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-OpenDuckPro3-Direct-v0",
    entry_point=f"{__name__}.openduckpro3_env:OpenDuckPro3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openduckpro3_env_cfg:OpenDuckPro3EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:OpenDuckPro3PPORunnerCfg"
        ),
    },
)
