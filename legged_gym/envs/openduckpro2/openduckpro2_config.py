from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class OpenDuckPro2RoughCfg(LeggedRobotCfg):
    """Isaac Gym training config for the 10-DOF OpenDuckPro2 URDF export."""

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.347]
        default_joint_angles = {
            "left_leg_yaw_joint": 0.0,
            "left_leg_roll_joint": 0.0,
            "left_leg_pitch_joint": 0.40,
            "left_knee_pitch_joint": -0.93,
            "left_ankle_pitch_joint": 0.53,
            "right_leg_yaw_joint": 0.0,
            "right_leg_roll_joint": 0.0,
            "right_leg_pitch_joint": -0.40,
            "right_knee_pitch_joint": 0.93,
            "right_ankle_pitch_joint": -0.53,
        }

    class env(LeggedRobotCfg.env):
        # 3 ang_vel + 3 gravity + 3 commands + 10 dof pos + 10 dof vel
        # + 10 previous actions + 2 gait phase terms
        num_observations = 41
        num_privileged_obs = 44  # + 3 base_lin_vel for the critic
        num_actions = 10

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-0.5, 0.5]
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 0.4

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {
            "leg_yaw": 20.0,
            "leg_roll": 30.0,
            "leg_pitch": 30.0,
            "knee": 30.0,
            "ankle": 15.0,
        }
        damping = {
            "leg_yaw": 0.6,
            "leg_roll": 1.1,
            "leg_pitch": 1.0,
            "knee": 1.0,
            "ankle": 0.5,
        }
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/openduckpro2/urdf/openduckpro2.urdf"
        name = "openduckpro2"
        foot_name = "ankle_pitch_link"
        penalize_contacts_on = ["knee_pitch_link", "leg_pitch_link"]
        terminate_after_contacts_on = ["base_link"]
        default_dof_drive_mode = 3
        self_collisions = 0
        flip_visual_attachments = False

    class commands(LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.45, 0.45]
            lin_vel_y = [-0.18, 0.18]
            ang_vel_yaw = [-0.36, 0.36]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.338
        # swing_phase_threshold = 0.55

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            # torques = -0.00001
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

    # class viewer(LeggedRobotCfg.viewer):
    #     pos = [2.0, -3.0, 1.4]
    #     lookat = [0.0, 0.0, 0.25]


class OpenDuckPro2RoughCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = "elu"
        rnn_type = "lstm"
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ""
        experiment_name = "openduckpro2"
