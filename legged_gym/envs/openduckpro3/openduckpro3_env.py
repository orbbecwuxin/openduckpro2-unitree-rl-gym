import numpy as np
import torch

from isaacgym import gymtorch

from legged_gym.envs.base.legged_robot import LeggedRobot


class OpenDuckPro3Robot(LeggedRobot):

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase

        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        self.last_feet_z = self.feet_pos[:, :, 2].clone()
        self.feet_clearance = torch.zeros_like(self.last_feet_z)
        self.reset_feet_clearance = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "reset_feet_clearance"):
            self.feet_clearance[env_ids] = 0.0
            self.reset_feet_clearance[env_ids] = True

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
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def _reward_contact(self):
        is_stance = self.leg_phase < 0.55
        contact = self.contact_forces[:, self.feet_indices, 2] > 1
        matches_phase = contact == is_stance
        mismatch = self.cfg.rewards.contact_mismatch_penalty
        reward = matches_phase.float() - (~matches_phase).float() * mismatch
        return torch.mean(reward, dim=1)

    def _reward_feet_swing_height(self):
        # Accumulate clearance from the latest contact instead of world-frame height.
        delta_z = self.feet_pos[:, :, 2] - self.last_feet_z
        delta_z *= ~self.reset_feet_clearance.unsqueeze(1)
        self.reset_feet_clearance.fill_(False)
        self.feet_clearance += delta_z
        self.last_feet_z.copy_(self.feet_pos[:, :, 2])

        contact = self.contact_forces[:, self.feet_indices, 2] > 1
        is_swing = self.leg_phase >= 0.55
        clearance = torch.clamp(self.feet_clearance, min=0.0)
        min_height = self.cfg.rewards.swing_height_min
        max_height = self.cfg.rewards.swing_height_max
        ramp_width = max(max_height - min_height, 1e-6)
        lift_reward = torch.clamp(clearance / min_height, 0.0, 1.0)
        excess_reward = torch.clamp(
            (max_height + ramp_width - clearance) / ramp_width, 0.0, 1.0
        )
        reward = lift_reward * excess_reward * is_swing
        self.feet_clearance *= ~contact
        return torch.sum(reward, dim=1)

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0

    def _reward_contact_no_vel(self):
        # Penalize horizontal foot slip while in contact.
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        horizontal_speed = torch.sum(torch.square(self.feet_vel[:, :, :2]), dim=2)
        return torch.sum(horizontal_speed * contact, dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [0, 1, 5, 6]]), dim=1)
