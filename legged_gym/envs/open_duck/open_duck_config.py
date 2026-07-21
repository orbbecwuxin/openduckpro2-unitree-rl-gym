from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class OpenDuckMiniRoughCfg(LeggedRobotCfg):
    """legged_gym config for Open Duck Mini v2.

    Physical parameters (default pose, PD gains, torque/velocity limits,
    action scale, control rate, armature) are copied from the Open Duck
    Playground training model (open_duck_mini_v2.xml + joystick.py) so the
    dynamics match the original MuJoCo training setup. Reward / command /
    domain-randomization / PPO structure follows the unitree_rl_gym G1
    humanoid task (phase-based walking); only dimensions and joint/foot
    names were adapted for this robot.
    """

    class env(LeggedRobotCfg.env):
        # 3 (ang_vel) + 3 (gravity) + 3 (commands) + 14*3 (dof pos/vel/action) + 2 (phase)
        num_observations = 53
        num_privileged_obs = 56  # + 3 base_lin_vel
        num_actions = 14

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.15]  # keyframe "home" floating-base height
        dof_pos_noise = 0.0
        root_xy_noise = 0.0
        root_vel_noise = 0.0
        default_joint_angles = {  # target angles [rad] when action = 0.0 (keyframe "home")
            "left_hip_yaw": 0.002,
            "left_hip_roll": 0.053,
            "left_hip_pitch": -0.63,
            "left_knee": 1.368,
            "left_ankle": -0.784,
            "neck_pitch": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            "right_hip_yaw": -0.003,
            "right_hip_roll": -0.065,
            "right_hip_pitch": 0.635,
            "right_knee": 1.379,
            "right_ankle": -0.796,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        # sts3215 servo: position kp=13.37, joint damping=0.56 (see MJCF sts3215 class).
        # Substring-matched against dof names; every joint uses the same servo.
        stiffness = {
            "hip": 13.37,
            "knee": 13.37,
            "ankle": 13.37,
            "neck": 13.37,
            "head": 13.37,
        }  # [N*m/rad]
        damping = {
            "hip": 0.56,
            "knee": 0.56,
            "ankle": 0.56,
            "neck": 0.56,
            "head": 0.56,
        }  # [N*m*s/rad]
        action_scale = 0.25  # matches Playground action_scale
        head_action_scale = 0.05
        decimation = 10  # sim_dt 0.002 * 10 = ctrl_dt 0.02

    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # matches Playground sim_dt

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/open_duck_mini/open_duck_mini.urdf"
        name = "open_duck_mini"
        foot_name = "foot_assembly"  # matches foot_assembly (L) and foot_assembly_2 (R)
        penalize_contacts_on = ["knee_and_ankle", "roll_to_pitch"]
        terminate_after_contacts_on = ["trunk_assembly"]
        default_dof_drive_mode = 1  # Isaac Gym position target drive.
        # armature = reflected rotor inertia of the sts3215 servo (0.027), applied to all dofs
        armature = 0.027
        self_collisions = 1  # 1 = disable self collision (original model has feet-only contact)
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.1, 1.25]
        randomize_base_mass = False
        added_mass_range = [-0.5, 0.5]
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 1.0

    class commands(LeggedRobotCfg.commands):
        heading_command = False
        min_command_norm = 0.03
        straight_command_prob = 0.35
        straight_command_min_abs_vx = 0.06

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.15, 0.15]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        base_height_target = 0.15
        swing_phase_threshold = 0.55
        swing_height_target = 0.08
        straight_command_vx_threshold = 0.04
        straight_command_vy_threshold = 0.03
        straight_command_yaw_threshold = 0.05

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            head_pos = -0.5
            head_vel = -0.02
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            swing_contact = -0.25
            straight_line_y = -2.0
            straight_line_yaw = -0.75
            contact = 0.18


class OpenDuckMiniRoughCfgPPO(LeggedRobotCfgPPO):
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
        experiment_name = "open_duck_mini"
