import numpy as np
import time

from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch


class OpenDuckMiniRobot(LeggedRobot):
    """Open Duck Mini v2 walking task.

    Mirrors the unitree_rl_gym G1 phase-based humanoid environment. The only
    robot-specific change is that the hip-position penalty resolves the hip
    yaw/roll DOF indices from the joint names at runtime (instead of the
    hard-coded G1 indices), so it works regardless of Isaac Gym's DOF ordering.
    """

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.  # commands
        noise_vec[9:9 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9 + self.num_actions:9 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9 + 2 * self.num_actions:9 + 3 * self.num_actions] = 0.  # previous actions
        noise_vec[9 + 3 * self.num_actions:9 + 3 * self.num_actions + 2] = 0.  # sin/cos phase
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _init_buffers(self):
        super()._init_buffers()
        # resolve hip yaw/roll DOF indices by name (robust to DOF ordering)
        hip_names = ["left_hip_yaw", "left_hip_roll", "right_hip_yaw", "right_hip_roll"]
        self.hip_indices = torch.tensor(
            [self.dof_names.index(n) for n in hip_names],
            dtype=torch.long, device=self.device,
        )
        head_names = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
        self.head_indices = torch.tensor(
            [self.dof_names.index(n) for n in head_names],
            dtype=torch.long, device=self.device,
        )
        self.action_scales = torch.ones(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        ) * self.cfg.control.action_scale
        self.action_scales[self.head_indices] = getattr(
            self.cfg.control, "head_action_scale", self.cfg.control.action_scale
        )
        self.dof_pos_targets = torch.zeros_like(self.dof_pos)
        self.phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.phase_left = torch.zeros_like(self.phase)
        self.phase_right = torch.zeros_like(self.phase)
        self.leg_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        for i, name in enumerate(self.dof_names):
            props["driveMode"][i] = gymapi.DOF_MODE_POS
            props["stiffness"][i] = 0.0
            props["damping"][i] = 0.0
            for dof_name, stiffness in self.cfg.control.stiffness.items():
                if dof_name in name:
                    props["stiffness"][i] = stiffness
                    props["damping"][i] = self.cfg.control.damping[dof_name]
                    break
        return props

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        min_norm = getattr(self.cfg.commands, "min_command_norm", 0.0)
        if min_norm > 0.0:
            self.commands[env_ids, :2] *= (
                torch.norm(self.commands[env_ids, :2], dim=1) > min_norm
            ).unsqueeze(1)

        straight_prob = getattr(self.cfg.commands, "straight_command_prob", 0.0)
        if straight_prob > 0.0:
            straight_mask = torch.rand(len(env_ids), device=self.device) < straight_prob
            straight_env_ids = env_ids[straight_mask]
            if len(straight_env_ids) > 0:
                self.commands[straight_env_ids, 1] = 0.0
                self.commands[straight_env_ids, 2] = 0.0

                min_abs_vx = getattr(self.cfg.commands, "straight_command_min_abs_vx", 0.0)
                if min_abs_vx > 0.0:
                    small_vx = torch.abs(self.commands[straight_env_ids, 0]) < min_abs_vx
                    if torch.any(small_vx):
                        num_small = int(torch.sum(small_vx).item())
                        vx_sign = torch.where(
                            torch.rand(num_small, device=self.device) > 0.5,
                            torch.ones(num_small, device=self.device),
                            -torch.ones(num_small, device=self.device),
                        )
                        self.commands[straight_env_ids[small_vx], 0] = vx_sign * min_abs_vx

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_pos_targets[env_ids] = self.default_dof_pos
        noise = getattr(self.cfg.init_state, "dof_pos_noise", 0.0)
        if noise > 0.0:
            self.dof_pos[env_ids] += torch_rand_float(
                -noise, noise, (len(env_ids), self.num_dof), device=self.device
            )
            self.dof_pos_targets[env_ids] = self.dof_pos[env_ids]
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]

        xy_noise = getattr(self.cfg.init_state, "root_xy_noise", 0.0)
        if xy_noise > 0.0:
            self.root_states[env_ids, :2] += torch_rand_float(
                -xy_noise, xy_noise, (len(env_ids), 2), device=self.device
            )

        vel_noise = getattr(self.cfg.init_state, "root_vel_noise", 0.0)
        if vel_noise > 0.0:
            self.root_states[env_ids, 7:13] = torch_rand_float(
                -vel_noise, vel_noise, (len(env_ids), 6), device=self.device
            )
        else:
            self.root_states[env_ids, 7:13] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.dof_pos_targets[:] = self.default_dof_pos + self.actions * self.action_scales
        self.dof_pos_targets[:] = torch.max(
            torch.min(self.dof_pos_targets, self.dof_pos_limits[:, 1]),
            self.dof_pos_limits[:, 0],
        )
        self.torques[:] = 0.0

        self.render()
        for _ in range(self.cfg.control.decimation):
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.dof_pos_targets)
            )
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)

            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        return super()._post_physics_step_callback()

    def compute_observations(self):
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3] * self.commands_scale,
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions,
                                  sin_phase,
                                  cos_phase
                                  ), dim=-1)
        self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                              self.base_ang_vel * self.obs_scales.ang_vel,
                                              self.projected_gravity,
                                              self.commands[:, :3] * self.commands_scale,
                                              (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                              self.dof_vel * self.obs_scales.dof_vel,
                                              self.actions,
                                              sin_phase,
                                              cos_phase
                                              ), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def _reward_feet_swing_height(self):
        swing = self.leg_phase >= getattr(self.cfg.rewards, "swing_phase_threshold", 0.55)
        target_height = getattr(self.cfg.rewards, "swing_height_target", 0.08)
        pos_error = torch.square(self.feet_pos[:, :, 2] - target_height) * swing
        return torch.sum(pos_error, dim=1)

    def _reward_swing_contact(self):
        swing = self.leg_phase >= getattr(self.cfg.rewards, "swing_phase_threshold", 0.55)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.sum(contact * swing, dim=1)

    def _straight_command_mask(self):
        vx_threshold = getattr(self.cfg.rewards, "straight_command_vx_threshold", 0.04)
        vy_threshold = getattr(self.cfg.rewards, "straight_command_vy_threshold", 0.03)
        yaw_threshold = getattr(self.cfg.rewards, "straight_command_yaw_threshold", 0.05)
        return (
            (torch.abs(self.commands[:, 0]) > vx_threshold)
            & (torch.abs(self.commands[:, 1]) < vy_threshold)
            & (torch.abs(self.commands[:, 2]) < yaw_threshold)
        )

    def _reward_straight_line_y(self):
        return torch.square(self.base_lin_vel[:, 1]) * self._straight_command_mask()

    def _reward_straight_line_yaw(self):
        return torch.square(self.base_ang_vel[:, 2]) * self._straight_command_mask()

    def _reward_alive(self):
        return 1.0

    def _reward_contact_no_vel(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices]), dim=1)

    def _reward_head_pos(self):
        head_error = self.dof_pos[:, self.head_indices] - self.default_dof_pos[:, self.head_indices]
        return torch.sum(torch.square(head_error), dim=1)

    def _reward_head_vel(self):
        return torch.sum(torch.square(self.dof_vel[:, self.head_indices]), dim=1)
