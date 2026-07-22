from legged_gym.envs.openduckpro2.openduckpro2_config import (
    OpenDuckPro2RoughCfg,
    OpenDuckPro2RoughCfgPPO,
)


class OpenDuckPro3RoughCfg(OpenDuckPro2RoughCfg):
    """OpenDuckPro2 training settings with the frame-aligned Pro3 model."""

    class init_state(OpenDuckPro2RoughCfg.init_state):
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

    class asset(OpenDuckPro2RoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/openduckpro2/urdf/openduckpro3.urdf"
        name = "openduckpro3"


class OpenDuckPro3RoughCfgPPO(OpenDuckPro2RoughCfgPPO):
    class runner(OpenDuckPro2RoughCfgPPO.runner):
        experiment_name = "openduckpro3"
