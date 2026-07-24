from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .openduckpro3_env_cfg import OpenDuckPro3EnvCfg


class OpenDuckPro3Env(DirectRLEnv):
    """Isaac Lab port of the OpenDuckPro3 Isaac Gym task."""

    cfg: OpenDuckPro3EnvCfg

    def __init__(self, cfg: OpenDuckPro3EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._previous_joint_vel = torch.zeros_like(self._robot.data.joint_vel)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self._feet_ids, self._feet_names = self._contact_sensor.find_bodies(".*_foot_link")
        self._base_ids, _ = self._contact_sensor.find_bodies("base.*")
        self._hip_ids, _ = self._robot.find_joints(
            [".*_leg_yaw_joint", ".*_leg_roll_joint"], preserve_order=True
        )
        if len(self._feet_ids) != 2 or not self._base_ids or len(self._hip_ids) != 4:
            raise RuntimeError(
                "Unexpected OpenDuckPro3 import: "
                f"feet={self._feet_names}, base_ids={self._base_ids}, hip_ids={self._hip_ids}"
            )

        self._foot_phase_offsets = torch.tensor(
            [0.0 if name.startswith("left_") else 0.5 for name in self._feet_names],
            device=self.device,
        )
        self._last_feet_z = self._robot.data.body_pos_w[:, self._feet_ids, 2].clone()
        self._feet_clearance = torch.zeros_like(self._last_feet_z)
        self._reset_feet_clearance = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        reward_names = [
            "tracking_lin_vel",
            "tracking_ang_vel",
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
            "base_height",
            "dof_acc",
            "dof_vel",
            "action_rate",
            "dof_pos_limits",
            "alive",
            "hip_pos",
            "contact_no_vel",
            "feet_swing_height",
            "contact",
        ]
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in reward_names
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = (
            self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _leg_phase(self) -> torch.Tensor:
        phase = (self.episode_length_buf * self.step_dt) % self.cfg.gait_period_s
        phase = phase / self.cfg.gait_period_s
        return (phase.unsqueeze(1) + self._foot_phase_offsets.unsqueeze(0)) % 1.0

    def _resample_commands(self, env_ids: torch.Tensor):
        self._commands[env_ids, 0].uniform_(*self.cfg.command_lin_vel_x)
        self._commands[env_ids, 1:] = 0.0

    def _get_observations(self) -> dict:
        resample_steps = round(self.cfg.command_resampling_time_s / self.step_dt)
        resample_ids = torch.nonzero(
            self.episode_length_buf % resample_steps == 0, as_tuple=False
        ).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)

        phase = (self.episode_length_buf * self.step_dt) % self.cfg.gait_period_s
        phase = phase / self.cfg.gait_period_s
        sin_phase = torch.sin(2.0 * torch.pi * phase).unsqueeze(1)
        cos_phase = torch.cos(2.0 * torch.pi * phase).unsqueeze(1)
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos

        policy = torch.cat(
            (
                self._robot.data.root_ang_vel_b * 0.25,
                self._robot.data.projected_gravity_b,
                self._commands * torch.tensor((2.0, 2.0, 0.25), device=self.device),
                joint_pos_rel,
                self._robot.data.joint_vel * 0.05,
                self._actions,
                sin_phase,
                cos_phase,
            ),
            dim=-1,
        )
        critic = torch.cat(
            (
                self._robot.data.root_lin_vel_b * 2.0,
                self._robot.data.root_ang_vel_b * 0.25,
                self._robot.data.projected_gravity_b,
                self._commands * torch.tensor((2.0, 2.0, 0.25), device=self.device),
                joint_pos_rel,
                self._robot.data.joint_vel * 0.05,
                self._actions,
                sin_phase,
                cos_phase,
            ),
            dim=-1,
        )

        if self.cfg.observation_noise_model is None:
            noise = torch.zeros_like(policy)
            noise[:, :3].uniform_(-0.05, 0.05)
            noise[:, 3:6].uniform_(-0.05, 0.05)
            noise[:, 9:19].uniform_(-0.01, 0.01)
            noise[:, 19:29].uniform_(-0.075, 0.075)
            policy += noise

        self._previous_actions.copy_(self._actions)
        self._previous_joint_vel.copy_(self._robot.data.joint_vel)
        return {"policy": policy, "critic": critic}

    def _contact(self) -> torch.Tensor:
        return self._contact_sensor.data.net_forces_w[:, self._feet_ids, 2] > 1.0

    def _get_rewards(self) -> torch.Tensor:
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - root_lin_vel_b[:, :2]), dim=1
        )
        ang_vel_error = torch.square(self._commands[:, 2] - root_ang_vel_b[:, 2])
        soft_limits = self._robot.data.soft_joint_pos_limits
        below = -(joint_pos - soft_limits[:, :, 0]).clip(max=0.0)
        above = (joint_pos - soft_limits[:, :, 1]).clip(min=0.0)
        contact = self._contact()
        leg_phase = self._leg_phase()
        is_stance = leg_phase < self.cfg.stance_phase_ratio

        feet_pos_z = self._robot.data.body_pos_w[:, self._feet_ids, 2]
        delta_z = feet_pos_z - self._last_feet_z
        delta_z *= ~self._reset_feet_clearance.unsqueeze(1)
        self._reset_feet_clearance.fill_(False)
        self._feet_clearance += delta_z
        self._last_feet_z.copy_(feet_pos_z)
        clearance = torch.clamp(self._feet_clearance, min=0.0)
        ramp_width = max(
            self.cfg.swing_height_max - self.cfg.swing_height_min, 1.0e-6
        )
        lift_reward = torch.clamp(clearance / self.cfg.swing_height_min, 0.0, 1.0)
        excess_reward = torch.clamp(
            (self.cfg.swing_height_max + ramp_width - clearance) / ramp_width,
            0.0,
            1.0,
        )
        swing_reward = torch.sum(lift_reward * excess_reward * ~is_stance, dim=1)
        self._feet_clearance *= ~contact

        matches_phase = contact == is_stance
        contact_reward = torch.mean(
            matches_phase.float()
            - (~matches_phase).float() * self.cfg.contact_mismatch_penalty,
            dim=1,
        )
        horizontal_foot_speed = torch.sum(
            torch.square(self._robot.data.body_lin_vel_w[:, self._feet_ids, :2]), dim=2
        )

        raw = {
            "tracking_lin_vel": torch.exp(-lin_vel_error / self.cfg.tracking_sigma),
            "tracking_ang_vel": torch.exp(-ang_vel_error / self.cfg.tracking_sigma),
            "lin_vel_z": torch.square(root_lin_vel_b[:, 2]),
            "ang_vel_xy": torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1),
            "orientation": torch.sum(
                torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1
            ),
            "base_height": torch.square(
                self._robot.data.root_pos_w[:, 2] - self.cfg.base_height_target
            ),
            "dof_acc": torch.sum(
                torch.square((self._previous_joint_vel - joint_vel) / self.step_dt),
                dim=1,
            ),
            "dof_vel": torch.sum(torch.square(joint_vel), dim=1),
            "action_rate": torch.sum(
                torch.square(self._previous_actions - self._actions), dim=1
            ),
            "dof_pos_limits": torch.sum(below + above, dim=1),
            "alive": torch.ones(self.num_envs, device=self.device),
            "hip_pos": torch.sum(torch.square(joint_pos[:, self._hip_ids]), dim=1),
            "contact_no_vel": torch.sum(horizontal_foot_speed * contact, dim=1),
            "feet_swing_height": swing_reward,
            "contact": contact_reward,
        }
        scales = {
            "tracking_lin_vel": self.cfg.tracking_lin_vel_scale,
            "tracking_ang_vel": self.cfg.tracking_ang_vel_scale,
            "lin_vel_z": self.cfg.lin_vel_z_scale,
            "ang_vel_xy": self.cfg.ang_vel_xy_scale,
            "orientation": self.cfg.orientation_scale,
            "base_height": self.cfg.base_height_scale,
            "dof_acc": self.cfg.dof_acc_scale,
            "dof_vel": self.cfg.dof_vel_scale,
            "action_rate": self.cfg.action_rate_scale,
            "dof_pos_limits": self.cfg.dof_pos_limits_scale,
            "alive": self.cfg.alive_scale,
            "hip_pos": self.cfg.hip_pos_scale,
            "contact_no_vel": self.cfg.contact_no_vel_scale,
            "feet_swing_height": self.cfg.feet_swing_height_scale,
            "contact": self.cfg.contact_scale,
        }
        terms = {
            name: value * scales[name] * self.step_dt for name, value in raw.items()
        }
        reward = torch.sum(torch.stack(list(terms.values())), dim=0)
        if self.cfg.only_positive_rewards:
            reward.clamp_(min=0.0)
        for name, value in terms.items():
            self._episode_sums[name] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        force_history = self._contact_sensor.data.net_forces_w_history
        base_force = torch.norm(force_history[:, :, self._base_ids], dim=-1)
        died = torch.any(torch.max(base_force, dim=1).values > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_joint_vel[env_ids] = 0.0
        self._feet_clearance[env_ids] = 0.0
        self._reset_feet_clearance[env_ids] = True
        self._resample_commands(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        logs = {}
        for name, episode_sum in self._episode_sums.items():
            logs[f"Episode_Reward/{name}"] = (
                torch.mean(episode_sum[env_ids]) / self.max_episode_length_s
            )
            episode_sum[env_ids] = 0.0
        logs["Episode_Termination/base_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        logs["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"] = logs
