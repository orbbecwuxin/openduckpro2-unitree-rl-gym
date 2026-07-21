import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

import isaacgym  # noqa: F401
from isaacgym import gymapi, gymutil
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


DEFAULT_COMMAND_PLAN = [
    {"name": "forward_slow", "vx": 0.12, "vy": 0.0, "yaw": 0.0, "seconds": 4.0},
    {"name": "forward_turn", "vx": 0.10, "vy": 0.0, "yaw": 0.35, "seconds": 3.0},
    {"name": "lateral", "vx": 0.0, "vy": 0.08, "yaw": 0.0, "seconds": 3.0},
]


def parse_custom_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--sample-env", type=int, default=0)
    parser.add_argument("--camera-distance", type=float, default=2.2)
    parser.add_argument("--camera-side", type=float, default=-1.7)
    parser.add_argument("--camera-height", type=float, default=1.0)
    parser.add_argument("--fallback-only", action="store_true")
    custom, remaining = parser.parse_known_args(argv)
    return custom, remaining


def patch_base_task_for_headless_camera():
    from legged_gym.envs.base import base_task

    def patched_init(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        self.graphics_device_id = self.sim_device_id
        if self.headless and os.environ.get("OPENDUCK_RECORD_HEADLESS_CAMERA", "1") != "1":
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float
            )
        else:
            self.privileged_obs_buf = None

        self.extras = {}
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.enable_viewer_sync = True
        self.viewer = None

        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    base_task.BaseTask.__init__ = patched_init


def actor_obs(observations):
    if isinstance(observations, tuple):
        return observations[0]
    return observations


def unpack_step(step_result):
    if len(step_result) == 5:
        obs, _, rews, dones, infos = step_result
        return obs, rews, dones, infos
    return step_result


def clamp(value, value_range):
    return max(value_range[0], min(value_range[1], value))


def command_for_time(t, duration_s):
    elapsed = 0.0
    for command in DEFAULT_COMMAND_PLAN:
        end = elapsed + command["seconds"]
        if t < end or command is DEFAULT_COMMAND_PLAN[-1]:
            return command
        elapsed = end
    return DEFAULT_COMMAND_PLAN[-1]


def apply_command(env, t, duration_s):
    command = command_for_time(t, duration_s)
    env.commands[:, 0] = clamp(float(command["vx"]), env.command_ranges["lin_vel_x"])
    env.commands[:, 1] = clamp(float(command["vy"]), env.command_ranges["lin_vel_y"])
    env.commands[:, 2] = clamp(float(command["yaw"]), env.command_ranges["ang_vel_yaw"])


def configure_env(args, custom):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1 if args.num_envs is None else max(1, int(args.num_envs))
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = False
    env_cfg.commands.resampling_time = 1e9
    args.headless = True
    return env_cfg, train_cfg


def load_policy(env, args, train_cfg, checkpoint_path):
    train_cfg.runner.resume = False
    ppo_runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    ppo_runner.load(str(checkpoint_path))
    return ppo_runner.get_inference_policy(device=env.device)


def create_camera(env, custom):
    props = gymapi.CameraProperties()
    props.width = int(custom.width)
    props.height = int(custom.height)
    camera = env.gym.create_camera_sensor(env.envs[int(custom.sample_env)], props)
    if camera < 0:
        raise RuntimeError("create_camera_sensor returned an invalid camera handle")
    return camera


def update_camera(env, camera, custom):
    index = int(custom.sample_env)
    base = env.root_states[index, :3].detach().cpu().numpy()
    cam_pos = gymapi.Vec3(
        float(base[0] - custom.camera_distance),
        float(base[1] + custom.camera_side),
        float(base[2] + custom.camera_height),
    )
    cam_target = gymapi.Vec3(float(base[0] + 0.2), float(base[1]), float(base[2] + 0.15))
    env.gym.set_camera_location(camera, env.envs[index], cam_pos, cam_target)


def capture_camera_frame(env, camera, custom):
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    image = env.gym.get_camera_image(
        env.sim,
        env.envs[int(custom.sample_env)],
        camera,
        gymapi.IMAGE_COLOR,
    )
    array = np.asarray(image, dtype=np.uint8)
    array = array.reshape((int(custom.height), int(custom.width), 4))
    return array[:, :, :3].copy()


def rollout(env, policy, custom, capture_mode):
    sim_dt = float(env.dt)
    steps = int(math.ceil(float(custom.duration_s) / sim_dt))
    obs = actor_obs(env.get_observations())
    camera = None
    camera_frames = []
    state_rows = []

    if capture_mode == "camera":
        camera = create_camera(env, custom)

    for step in range(steps):
        t = step * sim_dt
        apply_command(env, t, float(custom.duration_s))
        env.compute_observations()
        obs = actor_obs(env.get_observations())
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, dones, _ = unpack_step(env.step(actions.detach()))

        sample = int(custom.sample_env)
        row = {
            "t": t,
            "base": env.root_states[sample, :3].detach().cpu().numpy().copy(),
            "rpy": env.rpy[sample, :3].detach().cpu().numpy().copy(),
            "feet": env.feet_pos[sample, :, :3].detach().cpu().numpy().copy(),
            "command": env.commands[sample, :3].detach().cpu().numpy().copy(),
            "done": bool(dones[sample].item()) if hasattr(dones, "__getitem__") else bool(dones),
        }
        state_rows.append(row)

        if capture_mode == "camera":
            update_camera(env, camera, custom)
            camera_frames.append(capture_camera_frame(env, camera, custom))

    return camera_frames, state_rows


def resample_indices(source_count, target_count):
    if source_count <= 0:
        return []
    if target_count <= 1:
        return [0]
    return np.linspace(0, source_count - 1, target_count).round().astype(int).tolist()


def resample_camera_frames(frames, target_count):
    if not frames:
        return []
    if len(frames) == target_count:
        return list(frames)
    positions = np.linspace(0, len(frames) - 1, target_count)
    output = []
    for position in positions:
        low = int(math.floor(float(position)))
        high = min(low + 1, len(frames) - 1)
        alpha = float(position - low)
        if high == low or alpha <= 0.0:
            output.append(frames[low].copy())
            continue
        low_frame = frames[low].astype(np.float32)
        high_frame = frames[high].astype(np.float32)
        frame = (1.0 - alpha) * low_frame + alpha * high_frame
        output.append(np.clip(frame, 0, 255).astype(np.uint8))
    return output


def gif_durations_ms(frame_count, duration_s):
    total_centiseconds = int(round(float(duration_s) * 100.0))
    base = max(1, total_centiseconds // max(1, frame_count))
    extra = max(0, total_centiseconds - base * frame_count)
    durations = []
    carry = 0
    for _ in range(frame_count):
        centiseconds = base
        carry += extra
        if carry >= frame_count:
            centiseconds += 1
            carry -= frame_count
        durations.append(int(centiseconds * 10))
    return durations


def encode_gif(frames, output, duration_s):
    from PIL import Image

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB") for frame in frames]
    durations = gif_durations_ms(len(images), duration_s)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )


def rotation_matrix_from_rpy(rpy):
    roll, pitch, yaw = [float(x) for x in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def draw_fallback_frame(row, history, width, height):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    base = row["base"]
    feet = row["feet"]
    rpy = row["rpy"]
    rot = rotation_matrix_from_rpy(rpy)
    body_offsets = np.array(
        [
            [-0.12, -0.08, 0.0],
            [0.12, -0.08, 0.0],
            [0.12, 0.08, 0.0],
            [-0.12, 0.08, 0.0],
            [-0.12, -0.08, 0.0],
        ],
        dtype=np.float32,
    )
    body = base + body_offsets @ rot.T

    for foot_index, color in enumerate(["#2563eb", "#dc2626"]):
        points = np.array([base, feet[foot_index]])
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=2.5)
        ax.scatter(feet[foot_index, 0], feet[foot_index, 1], feet[foot_index, 2], color=color, s=45)
        trace = np.array([h["feet"][foot_index] for h in history], dtype=np.float32)
        if len(trace) > 1:
            ax.plot(trace[:, 0], trace[:, 1], trace[:, 2], color=color, linewidth=1.0, alpha=0.35)

    ax.plot(body[:, 0], body[:, 1], body[:, 2], color="#111827", linewidth=2.5)
    ax.scatter(base[0], base[1], base[2], color="#111827", s=35)

    trace_base = np.array([h["base"] for h in history], dtype=np.float32)
    if len(trace_base) > 1:
        ax.plot(trace_base[:, 0], trace_base[:, 1], trace_base[:, 2], color="#16a34a", linewidth=1.4, alpha=0.5)

    center = base
    span = 0.8
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(0.0, max(0.8, center[2] + 0.35))
    ax.view_init(elev=18, azim=-55)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"t={row['t']:.2f}s  cmd=({row['command'][0]:+.2f}, {row['command'][1]:+.2f}, {row['command'][2]:+.2f})",
        fontsize=9,
    )
    plt.tight_layout(pad=0.2)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape((height, width, 3)).copy()
    plt.close(fig)
    return data


def build_fallback_frames(state_rows, custom):
    target_count = int(round(float(custom.duration_s) * int(custom.fps)))
    indices = resample_indices(len(state_rows), target_count)
    frames = []
    for out_index, row_index in enumerate(indices):
        start = max(0, row_index - 80)
        history = state_rows[start : row_index + 1]
        frames.append(draw_fallback_frame(state_rows[row_index], history, int(custom.width), int(custom.height)))
        if (out_index + 1) % 60 == 0:
            print(f"fallback rendered {out_index + 1}/{target_count} frames", flush=True)
    return frames


def record(args, custom):
    checkpoint_path = Path(custom.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if custom.fps <= 0:
        raise ValueError("--fps must be positive")
    if custom.duration_s <= 0:
        raise ValueError("--duration-s must be positive")

    patch_base_task_for_headless_camera()
    env_cfg, train_cfg = configure_env(args, custom)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    policy = load_policy(env, args, train_cfg, checkpoint_path)

    capture_mode = "fallback" if custom.fallback_only else "camera"
    try:
        camera_frames, state_rows = rollout(env, policy, custom, capture_mode)
    except Exception as exc:
        if custom.fallback_only:
            raise
        print(f"camera capture failed, falling back to trajectory GIF: {exc}", flush=True)
        env_cfg, train_cfg = configure_env(args, custom)
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        policy = load_policy(env, args, train_cfg, checkpoint_path)
        camera_frames, state_rows = rollout(env, policy, custom, "fallback")

    target_count = int(round(float(custom.duration_s) * int(custom.fps)))
    if camera_frames:
        frames = resample_camera_frames(camera_frames, target_count)
        mode = "camera"
    else:
        frames = build_fallback_frames(state_rows, custom)
        mode = "fallback"

    encode_gif(frames, custom.output, float(custom.duration_s))
    print(
        f"wrote {custom.output} mode={mode} frames={len(frames)} "
        f"duration_s={custom.duration_s} fps={custom.fps}",
        flush=True,
    )


if __name__ == "__main__":
    custom_args, passthrough = parse_custom_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *passthrough]
    record(get_args(), custom_args)
