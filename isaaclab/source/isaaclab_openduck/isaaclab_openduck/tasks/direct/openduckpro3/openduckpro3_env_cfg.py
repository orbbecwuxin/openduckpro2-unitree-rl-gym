from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


REPOSITORY_ROOT = Path(__file__).resolve().parents[7]
OPENDUCKPRO3_URDF = REPOSITORY_ROOT / "openduckpro3" / "urdf" / "openduckpro3.urdf"


OPENDUCKPRO3_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(OPENDUCKPRO3_URDF),
        fix_base=False,
        merge_fixed_joints=True,
        make_instanceable=False,
        self_collision=False,
        activate_contact_sensors=True,
        joint_drive=None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*_leg_yaw_joint": 0.0,
            ".*_leg_roll_joint": 0.0,
            ".*_leg_pitch_joint": 0.55,
            ".*_knee_pitch_joint": -1.22,
            ".*_ankle_pitch_joint": 0.67,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_yaw_joint",
                ".*_leg_roll_joint",
                ".*_leg_pitch_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit_sim=30.0,
            velocity_limit_sim=21.0,
            stiffness={
                ".*_leg_yaw_joint": 15.0,
                ".*_leg_roll_joint": 20.0,
                ".*_leg_pitch_joint": 20.0,
                ".*_knee_pitch_joint": 20.0,
            },
            damping={
                ".*_leg_yaw_joint": 0.6,
                ".*_leg_roll_joint": 0.8,
                ".*_leg_pitch_joint": 0.8,
                ".*_knee_pitch_joint": 0.8,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit_sim=30.0,
            velocity_limit_sim=21.0,
            stiffness=10.0,
            damping=0.5,
        ),
    },
)


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 1.25),
            "dynamic_friction_range": (0.1, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base.*"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={
            "velocity_range": {
                "x": (-0.4, 0.4),
                "y": (-0.4, 0.4),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        },
    )


@configclass
class OpenDuckPro3EnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_space = 10
    observation_space = 41
    state_space = 44

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            min_position_iteration_count=4,
            max_position_iteration_count=4,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
            gpu_max_rigid_contact_count=2**23,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = OPENDUCKPRO3_CFG
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=2,
        update_period=0.005,
        track_air_time=True,
    )
    events: EventCfg = EventCfg()

    action_scale = 0.25
    command_resampling_time_s = 10.0
    command_lin_vel_x = (0.15, 0.30)
    gait_period_s = 0.56
    stance_phase_ratio = 0.55

    tracking_sigma = 0.25
    base_height_target = 0.348
    swing_height_min = 0.03
    swing_height_max = 0.05
    contact_mismatch_penalty = 0.3
    only_positive_rewards = True

    tracking_lin_vel_scale = 1.0
    tracking_ang_vel_scale = 0.5
    lin_vel_z_scale = -2.0
    ang_vel_xy_scale = -0.05
    orientation_scale = -1.0
    base_height_scale = -10.0
    dof_acc_scale = -2.5e-7
    dof_vel_scale = -1.0e-3
    action_rate_scale = -0.01
    dof_pos_limits_scale = -5.0
    alive_scale = 0.15
    hip_pos_scale = -1.0
    contact_no_vel_scale = -0.5
    feet_swing_height_scale = 1.0
    contact_scale = 1.0
