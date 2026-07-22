from legged_gym.envs.openduckpro3.openduckpro3_config import (
    openduckpro3RoughCfg,
    openduckpro3RoughCfgPPO,
)


class OpenDuckPro3RoughCfg(openduckpro3RoughCfg):
    """openduckpro3 training settings with the frame-aligned Pro3 model."""

    class init_state(openduckpro3RoughCfg.init_state):
        default_joint_angles = {
            "left_leg_yaw_joint": 0.0,
            "left_leg_roll_joint": 0.0,
            "left_leg_pitch_joint": 0.40,
            "left_knee_pitch_joint": -0.93,
            "left_ankle_pitch_joint": 0.53,
            "right_leg_yaw_joint": 0.0,
            "right_leg_roll_joint": 0.0,
            "right_leg_pitch_joint": 0.40,
            "right_knee_pitch_joint": -0.93,
            "right_ankle_pitch_joint": 0.53,
        }

    class asset(openduckpro3RoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/openduckpro3/urdf/openduckpro3.urdf"
        name = "openduckpro3"


class OpenDuckPro3RoughCfgPPO(openduckpro3RoughCfgPPO):
    class runner(openduckpro3RoughCfgPPO.runner):
        experiment_name = "openduckpro3"
