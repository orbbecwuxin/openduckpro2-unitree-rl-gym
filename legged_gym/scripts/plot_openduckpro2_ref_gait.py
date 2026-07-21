#!/usr/bin/env python3
"""Render the OpenDuckPro2 phase reference gait as an animated GIF.

The script is intentionally standalone: it parses the config with ``ast`` and
the URDF with ``xml.etree`` so it can run without importing Isaac Gym.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openduckpro2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


PERIOD_S = 0.8
PHASE_OFFSET = 0.5

LEFT_CHAIN = [
    "base_link",
    "left_leg_yaw_link",
    "left_leg_roll_link",
    "left_leg_pitch_link",
    "left_knee_pitch_link",
    "left_ankle_pitch_link",
]
RIGHT_CHAIN = [
    "base_link",
    "right_leg_yaw_link",
    "right_leg_roll_link",
    "right_leg_pitch_link",
    "right_knee_pitch_link",
    "right_ankle_pitch_link",
]


@dataclass(frozen=True)
class Joint:
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def _class_node(parent: ast.AST, name: str) -> ast.ClassDef:
    for node in getattr(parent, "body", []):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise KeyError(f"class {name!r} not found")


def _assigned_literal(class_node: ast.ClassDef, name: str):
    for node in class_node.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"assignment {name!r} not found in {class_node.name}")


def load_reference_config(config_path: Path) -> dict:
    tree = ast.parse(config_path.read_text())
    cfg = _class_node(tree, "OpenDuckPro2RoughCfg")
    init_state = _class_node(cfg, "init_state")
    control = _class_node(cfg, "control")
    rewards = _class_node(cfg, "rewards")

    return {
        "base_height": float(_assigned_literal(init_state, "pos")[2]),
        "default_angles": _assigned_literal(init_state, "default_joint_angles"),
        "action_scale": float(_assigned_literal(control, "action_scale")),
        "base_height_target": float(_assigned_literal(rewards, "base_height_target")),
        "swing_phase_threshold": float(_assigned_literal(rewards, "swing_phase_threshold")),
        "swing_height_target": float(_assigned_literal(rewards, "swing_height_target")),
        "ref_hip_pitch_amp": float(_assigned_literal(rewards, "ref_hip_pitch_amp")),
        "ref_knee_pitch_amp": float(_assigned_literal(rewards, "ref_knee_pitch_amp")),
        "ref_ankle_pitch_amp": float(_assigned_literal(rewards, "ref_ankle_pitch_amp")),
    }


def parse_vector(value: str, default: str = "0 0 0") -> np.ndarray:
    return np.array([float(v) for v in value.split()]) if value else np.array(
        [float(v) for v in default.split()]
    )


def load_urdf(urdf_path: Path) -> tuple[dict[str, Joint], dict[str, tuple[Path, np.ndarray, np.ndarray]]]:
    root = ET.parse(urdf_path).getroot()
    joints: dict[str, Joint] = {}
    link_collision_meshes: dict[str, tuple[Path, np.ndarray, np.ndarray]] = {}

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        collision = link.find("collision")
        if collision is None:
            continue
        geometry = collision.find("geometry")
        mesh = geometry.find("mesh") if geometry is not None else None
        if mesh is None:
            continue
        origin = collision.find("origin")
        xyz = parse_vector(origin.attrib.get("xyz", "0 0 0")) if origin is not None else np.zeros(3)
        rpy = parse_vector(origin.attrib.get("rpy", "0 0 0")) if origin is not None else np.zeros(3)
        mesh_path = (urdf_path.parent / mesh.attrib["filename"]).resolve()
        link_collision_meshes[link_name] = (mesh_path, xyz, rpy)

    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        joints[joint.attrib["name"]] = Joint(
            parent=joint.find("parent").attrib["link"],
            child=joint.find("child").attrib["link"],
            xyz=parse_vector(origin.attrib.get("xyz", "0 0 0")) if origin is not None else np.zeros(3),
            rpy=parse_vector(origin.attrib.get("rpy", "0 0 0")) if origin is not None else np.zeros(3),
            axis=parse_vector(axis.attrib.get("xyz", "0 0 1"), "0 0 1") if axis is not None else np.array([0.0, 0.0, 1.0]),
        )

    return joints, link_collision_meshes


def load_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) >= 84:
        triangle_count = struct.unpack("<I", data[80:84])[0]
        expected_size = 84 + triangle_count * 50
        if expected_size == len(data):
            vertices = []
            offset = 84
            for _ in range(triangle_count):
                offset += 12
                for _ in range(3):
                    vertices.append(struct.unpack("<fff", data[offset:offset + 12]))
                    offset += 12
                offset += 2
            return np.asarray(vertices, dtype=float)

    vertices = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(vertices, dtype=float)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def transform_from_pose(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def transform_from_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = axis_rotation(axis, angle)
    return transform


class OpenDuckPro2Kinematics:
    def __init__(self, urdf_path: Path):
        self.joints, self.link_collision_meshes = load_urdf(urdf_path)
        self.parents = {joint.child: name for name, joint in self.joints.items()}
        self.mesh_vertices = {
            link: load_stl_vertices(mesh_path)
            for link, (mesh_path, _, _) in self.link_collision_meshes.items()
        }

    def joint_chain_to(self, link: str) -> list[str]:
        chain = []
        current = link
        while current != "base_link":
            joint_name = self.parents[current]
            chain.append(joint_name)
            current = self.joints[joint_name].parent
        return list(reversed(chain))

    def link_poses(self, qmap: dict[str, float]) -> dict[str, np.ndarray]:
        poses = {"base_link": np.eye(4)}

        def visit(link: str, parent_pose: np.ndarray):
            for joint_name, joint in self.joints.items():
                if joint.parent != link:
                    continue
                child_pose = (
                    parent_pose
                    @ transform_from_pose(joint.xyz, joint.rpy)
                    @ transform_from_axis(joint.axis, qmap.get(joint_name, 0.0))
                )
                poses[joint.child] = child_pose
                visit(joint.child, child_pose)

        visit("base_link", poses["base_link"])
        return poses

    def foot_bottom_z(self, link: str, qmap: dict[str, float], base_height: float) -> float:
        poses = self.link_poses(qmap)
        mesh_vertices = self.mesh_vertices[link]
        _, collision_xyz, collision_rpy = self.link_collision_meshes[link]
        collision_pose = poses[link] @ transform_from_pose(collision_xyz, collision_rpy)
        homogeneous = np.c_[mesh_vertices, np.ones(len(mesh_vertices))]
        world = (collision_pose @ homogeneous.T).T[:, :3]
        return float(world[:, 2].min() + base_height)


def swing_profile(phase: np.ndarray | float, threshold: float) -> np.ndarray | float:
    swing = np.clip((phase - threshold) / (1.0 - threshold), 0.0, 1.0)
    return np.sin(np.pi * swing) * (phase >= threshold)


def reference_angles(phase: float, cfg: dict) -> dict[str, float]:
    default = dict(cfg["default_angles"])
    left_phase = phase
    right_phase = (phase + PHASE_OFFSET) % 1.0
    left_amp = float(swing_profile(left_phase, cfg["swing_phase_threshold"]))
    right_amp = float(swing_profile(right_phase, cfg["swing_phase_threshold"]))

    default["left_leg_pitch_joint"] += -cfg["ref_hip_pitch_amp"] * left_amp
    default["left_knee_pitch_joint"] += cfg["ref_knee_pitch_amp"] * left_amp
    default["left_ankle_pitch_joint"] += -cfg["ref_ankle_pitch_amp"] * left_amp

    default["right_leg_pitch_joint"] += cfg["ref_hip_pitch_amp"] * right_amp
    default["right_knee_pitch_joint"] += -cfg["ref_knee_pitch_amp"] * right_amp
    default["right_ankle_pitch_joint"] += cfg["ref_ankle_pitch_amp"] * right_amp
    return default


def chain_points(poses: dict[str, np.ndarray], chain: list[str], base_height: float) -> np.ndarray:
    points = []
    for link in chain:
        point = poses[link][:3, 3].copy()
        point[2] += base_height
        points.append(point)
    return np.asarray(points)


def build_reference_samples(kin: OpenDuckPro2Kinematics, cfg: dict, frames: int):
    phases = np.linspace(0.0, 1.0, frames, endpoint=False)
    samples = []
    left_bottom = []
    right_bottom = []
    left_ankle_z = []
    right_ankle_z = []
    joint_series = {
        "left hip": [],
        "left knee": [],
        "left ankle": [],
        "right hip": [],
        "right knee": [],
        "right ankle": [],
    }

    for phase in phases:
        qmap = reference_angles(float(phase), cfg)
        poses = kin.link_poses(qmap)
        left = chain_points(poses, LEFT_CHAIN, cfg["base_height"])
        right = chain_points(poses, RIGHT_CHAIN, cfg["base_height"])
        left_bottom.append(kin.foot_bottom_z("left_ankle_pitch_link", qmap, cfg["base_height"]))
        right_bottom.append(kin.foot_bottom_z("right_ankle_pitch_link", qmap, cfg["base_height"]))
        left_ankle_z.append(left[-1, 2])
        right_ankle_z.append(right[-1, 2])
        joint_series["left hip"].append(qmap["left_leg_pitch_joint"])
        joint_series["left knee"].append(qmap["left_knee_pitch_joint"])
        joint_series["left ankle"].append(qmap["left_ankle_pitch_joint"])
        joint_series["right hip"].append(qmap["right_leg_pitch_joint"])
        joint_series["right knee"].append(qmap["right_knee_pitch_joint"])
        joint_series["right ankle"].append(qmap["right_ankle_pitch_joint"])
        samples.append({"phase": phase, "qmap": qmap, "poses": poses, "left": left, "right": right})

    return {
        "phases": phases,
        "samples": samples,
        "left_bottom": np.asarray(left_bottom),
        "right_bottom": np.asarray(right_bottom),
        "left_ankle_z": np.asarray(left_ankle_z),
        "right_ankle_z": np.asarray(right_ankle_z),
        "joint_series": {name: np.asarray(values) for name, values in joint_series.items()},
    }


def draw_chain(ax, points: np.ndarray, color: str, label: str, xy_indices: tuple[int, int]):
    x_idx, y_idx = xy_indices
    ax.plot(points[:, x_idx], points[:, y_idx], "-o", color=color, lw=2.4, ms=5, label=label)
    ax.scatter(points[-1, x_idx], points[-1, y_idx], color=color, s=70, marker="s")


def render_gif(samples: dict, cfg: dict, output_path: Path, fps: int):
    phases = samples["phases"]
    seconds = phases * PERIOD_S
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=110)
    fig.suptitle("OpenDuckPro2 Reference Gait", fontsize=14, fontweight="bold")

    left_color = "#1f77b4"
    right_color = "#d62728"

    def update(frame_index: int):
        for ax in axes.flat:
            ax.clear()

        sample = samples["samples"][frame_index]
        phase = sample["phase"]
        left = sample["left"]
        right = sample["right"]

        ax_side = axes[0, 0]
        draw_chain(ax_side, left, left_color, "left", (0, 2))
        draw_chain(ax_side, right, right_color, "right", (0, 2))
        ax_side.axhline(0.0, color="black", lw=1.0)
        ax_side.axhline(cfg["base_height_target"], color="#555555", lw=1.0, ls="--", label="base target")
        ax_side.set_title(f"Side view, phase={phase:.2f}, t={phase * PERIOD_S:.2f}s")
        ax_side.set_xlabel("x [m]")
        ax_side.set_ylabel("z [m]")
        ax_side.set_xlim(-0.16, 0.08)
        ax_side.set_ylim(-0.02, 0.36)
        ax_side.grid(True, alpha=0.25)
        ax_side.set_aspect("equal", adjustable="box")
        ax_side.legend(loc="upper right")

        ax_front = axes[0, 1]
        draw_chain(ax_front, left, left_color, "left", (1, 2))
        draw_chain(ax_front, right, right_color, "right", (1, 2))
        ax_front.axhline(0.0, color="black", lw=1.0)
        ax_front.set_title("Front view")
        ax_front.set_xlabel("y [m]")
        ax_front.set_ylabel("z [m]")
        ax_front.set_xlim(-0.18, 0.18)
        ax_front.set_ylim(-0.02, 0.36)
        ax_front.grid(True, alpha=0.25)
        ax_front.set_aspect("equal", adjustable="box")
        ax_front.legend(loc="upper right")

        ax_height = axes[1, 0]
        ax_height.plot(seconds, samples["left_ankle_z"], color=left_color, label="left ankle z")
        ax_height.plot(seconds, samples["right_ankle_z"], color=right_color, label="right ankle z")
        ax_height.plot(seconds, samples["left_bottom"], color=left_color, ls=":", label="left foot bottom")
        ax_height.plot(seconds, samples["right_bottom"], color=right_color, ls=":", label="right foot bottom")
        ax_height.axhline(cfg["swing_height_target"], color="#555555", ls="--", lw=1.0, label="ankle target")
        ax_height.axhline(0.0, color="black", lw=1.0)
        ax_height.axvline(phase * PERIOD_S, color="#222222", lw=1.0, alpha=0.7)
        ax_height.scatter([phase * PERIOD_S], [samples["left_ankle_z"][frame_index]], color=left_color, s=35)
        ax_height.scatter([phase * PERIOD_S], [samples["right_ankle_z"][frame_index]], color=right_color, s=35)
        ax_height.set_title("Foot and ankle height")
        ax_height.set_xlabel("time in gait cycle [s]")
        ax_height.set_ylabel("height [m]")
        ax_height.set_xlim(0.0, PERIOD_S)
        ax_height.set_ylim(-0.01, 0.15)
        ax_height.grid(True, alpha=0.25)
        ax_height.legend(loc="upper right", fontsize=8)

        ax_joint = axes[1, 1]
        for name, values in samples["joint_series"].items():
            color = left_color if name.startswith("left") else right_color
            linestyle = "-" if "hip" in name else "--" if "knee" in name else ":"
            ax_joint.plot(seconds, values, color=color, ls=linestyle, label=name)
        ax_joint.axhline(0.0, color="black", lw=0.8)
        ax_joint.axvline(phase * PERIOD_S, color="#222222", lw=1.0, alpha=0.7)
        ax_joint.set_title("Pitch joint references")
        ax_joint.set_xlabel("time in gait cycle [s]")
        ax_joint.set_ylabel("angle [rad]")
        ax_joint.set_xlim(0.0, PERIOD_S)
        ax_joint.set_ylim(-0.65, 0.65)
        ax_joint.grid(True, alpha=0.25)
        ax_joint.legend(loc="upper right", fontsize=8, ncol=2)

        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    animation = FuncAnimation(fig, update, frames=len(samples["samples"]), interval=1000 / fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_output = repo_root / "resources" / "reference_motion" / "openduckpro2" / "openduckpro2_ref_gait.gif"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=80, help="number of animation frames")
    parser.add_argument("--fps", type=int, default=16, help="GIF frames per second")
    parser.add_argument("--output", type=Path, default=default_output, help="output GIF path")
    args = parser.parse_args()

    config_path = repo_root / "legged_gym" / "envs" / "openduckpro2" / "openduckpro2_config.py"
    urdf_path = repo_root / "resources" / "robots" / "openduckpro2" / "urdf" / "openduckpro2.urdf"

    cfg = load_reference_config(config_path)
    kin = OpenDuckPro2Kinematics(urdf_path)
    samples = build_reference_samples(kin, cfg, args.frames)
    render_gif(samples, cfg, args.output, args.fps)

    print(args.output)
    print(f"frames={args.frames} fps={args.fps}")
    print(f"base_height={cfg['base_height']:.3f} swing_height_target={cfg['swing_height_target']:.3f}")
    print(
        "foot_bottom_range="
        f"L[{samples['left_bottom'].min():.3f}, {samples['left_bottom'].max():.3f}] "
        f"R[{samples['right_bottom'].min():.3f}, {samples['right_bottom'].max():.3f}]"
    )


if __name__ == "__main__":
    main()
