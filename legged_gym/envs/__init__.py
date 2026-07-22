from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.open_duck.open_duck_config import OpenDuckMiniRoughCfg, OpenDuckMiniRoughCfgPPO
from legged_gym.envs.open_duck.open_duck_env import OpenDuckMiniRobot
from legged_gym.envs.openduckpro3.openduckpro3_config import openduckpro3RoughCfg, openduckpro3RoughCfgPPO
from legged_gym.envs.openduckpro3.openduckpro3_env import openduckpro3Robot
from legged_gym.envs.openduckpro3.openduckpro3_config import OpenDuckPro3RoughCfg, OpenDuckPro3RoughCfgPPO
from legged_gym.utils.task_registry import task_registry


task_registry.register("open_duck", OpenDuckMiniRobot, OpenDuckMiniRoughCfg(), OpenDuckMiniRoughCfgPPO())
task_registry.register("openduckpro3", openduckpro3Robot, openduckpro3RoughCfg(), openduckpro3RoughCfgPPO())
task_registry.register("openduckpro3", openduckpro3Robot, OpenDuckPro3RoughCfg(), OpenDuckPro3RoughCfgPPO())
