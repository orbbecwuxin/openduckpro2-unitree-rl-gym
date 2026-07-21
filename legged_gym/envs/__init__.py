from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.open_duck.open_duck_config import OpenDuckMiniRoughCfg, OpenDuckMiniRoughCfgPPO
from legged_gym.envs.open_duck.open_duck_env import OpenDuckMiniRobot
from legged_gym.envs.openduckpro2.openduckpro2_config import OpenDuckPro2RoughCfg, OpenDuckPro2RoughCfgPPO
from legged_gym.envs.openduckpro2.openduckpro2_env import OpenDuckPro2Robot
from legged_gym.utils.task_registry import task_registry


task_registry.register("open_duck", OpenDuckMiniRobot, OpenDuckMiniRoughCfg(), OpenDuckMiniRoughCfgPPO())
task_registry.register("openduckpro2", OpenDuckPro2Robot, OpenDuckPro2RoughCfg(), OpenDuckPro2RoughCfgPPO())
