import os
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def _wrap_to_pi(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def _current_yaw(env):
    return float(env.rpy[0, 2].item())


def _subscribe_key(env, key_name, action):
    key = getattr(gymapi, key_name, None)
    if key is not None:
        env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action)


def _format_command_state(state):
    vx = state["vx_sign"] * state["vx_speed"]
    vy = state["vy_sign"] * state["vy_speed"]
    yaw = state["yaw_sign"] * state["yaw_speed"]

    return (
        f"cmd vx={vx:+.2f}, vy={vy:+.2f}, yaw={yaw:+.2f}; "
        f"speeds vx={state['vx_speed']:.2f}, vy={state['vy_speed']:.2f}, yaw={state['yaw_speed']:.2f}; "
        f"heading_hold={'on' if state['heading_hold'] else 'off'}"
    )


def _clamp_abs_speed(value, command_range):
    max_abs = max(abs(command_range[0]), abs(command_range[1]))
    return max(0.0, min(max_abs, value))


def _setup_keyboard_commands(env, args):
    if env.headless or env.viewer is None:
        print("Keyboard commands requested, but viewer is disabled.")
        return None

    key_bindings = [
        ("KEY_W", "cmd_forward"),
        ("KEY_UP", "cmd_forward"),
        ("KEY_S", "cmd_backward"),
        ("KEY_DOWN", "cmd_backward"),
        ("KEY_A", "cmd_left"),
        ("KEY_D", "cmd_right"),
        ("KEY_Q", "cmd_yaw_left"),
        ("KEY_LEFT", "cmd_yaw_left"),
        ("KEY_E", "cmd_yaw_right"),
        ("KEY_RIGHT", "cmd_yaw_right"),
        ("KEY_Z", "cmd_yaw_stop"),
        ("KEY_X", "cmd_stop"),
        ("KEY_SPACE", "cmd_stop"),
        ("KEY_R", "speed_vx_up"),
        ("KEY_F", "speed_vx_down"),
        ("KEY_T", "speed_vy_up"),
        ("KEY_G", "speed_vy_down"),
        ("KEY_Y", "speed_yaw_up"),
        ("KEY_H", "speed_yaw_down"),
    ]
    for key_name, action in key_bindings:
        _subscribe_key(env, key_name, action)

    state = {
        "vx_sign": 0.0,
        "vy_sign": 0.0,
        "yaw_sign": 0.0,
        "vx_speed": _clamp_abs_speed(args.keyboard_vx, env.command_ranges["lin_vel_x"]),
        "vy_speed": _clamp_abs_speed(args.keyboard_vy, env.command_ranges["lin_vel_y"]),
        "yaw_speed": _clamp_abs_speed(args.keyboard_yaw, env.command_ranges["ang_vel_yaw"]),
        "heading_hold": args.keyboard_heading_hold,
        "heading_kp": args.keyboard_heading_kp,
        "target_yaw": _current_yaw(env),
    }

    def print_state(prefix):
        print(f"{prefix}: {_format_command_state(state)}", flush=True)

    def adjust_speed(name, delta, command_range):
        state[name] = _clamp_abs_speed(state[name] + delta, command_range)
        print_state("Keyboard speed")

    def moving():
        return state["vx_sign"] != 0.0 or state["vy_sign"] != 0.0

    def set_translation(name, sign):
        was_moving = moving()
        state[name] = sign
        if state["heading_hold"] and not was_moving and state["yaw_sign"] == 0.0:
            state["target_yaw"] = _current_yaw(env)
        print_state("Keyboard command")

    def set_yaw(sign):
        state["yaw_sign"] = sign
        if state["heading_hold"] and sign == 0.0:
            state["target_yaw"] = _current_yaw(env)
        print_state("Keyboard command")

    def handle_viewer_event(evt):
        if evt.value <= 0:
            return

        if evt.action == "cmd_stop":
            state["vx_sign"] = 0.0
            state["vy_sign"] = 0.0
            state["yaw_sign"] = 0.0
            if state["heading_hold"]:
                state["target_yaw"] = _current_yaw(env)
            print_state("Keyboard command")
            return
        if evt.action == "cmd_forward":
            set_translation("vx_sign", 1.0)
            return
        if evt.action == "cmd_backward":
            set_translation("vx_sign", -1.0)
            return
        if evt.action == "cmd_left":
            set_translation("vy_sign", 1.0)
            return
        if evt.action == "cmd_right":
            set_translation("vy_sign", -1.0)
            return
        if evt.action == "cmd_yaw_left":
            set_yaw(1.0)
            return
        if evt.action == "cmd_yaw_right":
            set_yaw(-1.0)
            return
        if evt.action == "cmd_yaw_stop":
            set_yaw(0.0)
            return
        if evt.action == "speed_vx_up":
            adjust_speed("vx_speed", args.keyboard_vx_step, env.command_ranges["lin_vel_x"])
            return
        if evt.action == "speed_vx_down":
            adjust_speed("vx_speed", -args.keyboard_vx_step, env.command_ranges["lin_vel_x"])
            return
        if evt.action == "speed_vy_up":
            adjust_speed("vy_speed", args.keyboard_vy_step, env.command_ranges["lin_vel_y"])
            return
        if evt.action == "speed_vy_down":
            adjust_speed("vy_speed", -args.keyboard_vy_step, env.command_ranges["lin_vel_y"])
            return
        if evt.action == "speed_yaw_up":
            adjust_speed("yaw_speed", args.keyboard_yaw_step, env.command_ranges["ang_vel_yaw"])
            return
        if evt.action == "speed_yaw_down":
            adjust_speed("yaw_speed", -args.keyboard_yaw_step, env.command_ranges["ang_vel_yaw"])
            return

    env._viewer_event_handler = handle_viewer_event
    print(
        "Keyboard control: W/S or Up/Down = vx, A/D = vy, Q/E or Left/Right = yaw, "
        "Z = yaw stop, X/Space = stop, R/F = vx speed, T/G = vy speed, Y/H = yaw speed. "
        "Commands latch until another direction or stop is pressed.",
        flush=True,
    )
    print_state("Keyboard initial")
    return state


def _clamp_command(value, command_range):
    return max(command_range[0], min(command_range[1], value))


def _apply_keyboard_commands(env, state):
    vx = state["vx_sign"] * state["vx_speed"]
    vy = state["vy_sign"] * state["vy_speed"]
    yaw = state["yaw_sign"] * state["yaw_speed"]
    if state["heading_hold"] and state["yaw_sign"] == 0.0 and (vx != 0.0 or vy != 0.0):
        heading_error = _wrap_to_pi(state["target_yaw"] - _current_yaw(env))
        yaw = state["heading_kp"] * heading_error

    env.commands[:, 0] = _clamp_command(vx, env.command_ranges["lin_vel_x"])
    env.commands[:, 1] = _clamp_command(vy, env.command_ranges["lin_vel_y"])
    env.commands[:, 2] = _clamp_command(yaw, env.command_ranges["ang_vel_yaw"])


def _actor_obs(observations):
    if isinstance(observations, tuple):
        return observations[0]
    return observations


def _unpack_step(step_result):
    if len(step_result) == 5:
        obs, _, rews, dones, infos = step_result
        return obs, rews, dones, infos
    return step_result


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True
    if args.keyboard_commands:
        env_cfg.commands.resampling_time = 1e9

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    keyboard_state = _setup_keyboard_commands(env, args) if args.keyboard_commands else None
    obs = _actor_obs(env.get_observations())
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    print("Play loop started. Click the Isaac Gym viewer before pressing control keys.", flush=True)
    for i in range(10*int(env.max_episode_length)):
        obs = _actor_obs(obs)
        if keyboard_state is not None:
            _apply_keyboard_commands(env, keyboard_state)
            env.compute_observations()
            obs = _actor_obs(env.get_observations())
        actions = policy(obs.detach())
        obs, rews, dones, infos = _unpack_step(env.step(actions.detach()))

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
