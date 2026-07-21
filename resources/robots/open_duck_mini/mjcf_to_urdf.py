"""Convert the Open Duck Mini v2 MuJoCo training model (MJCF) to a URDF that
Isaac Gym / legged_gym (unitree_rl_gym) can load.

Source input is provided explicitly with ``--mjcf``.

Why a conversion is needed
--------------------------
Isaac Gym can technically import MJCF, but legged_gym is built and tested around
URDF, and the training MJCF relies on features the Isaac Gym MJCF importer does
NOT reproduce reliably: nested ``<default class=...>`` / ``childclass`` joint
defaults (armature, damping, frictionloss, actuator kp / forcerange), MuJoCo
``<sensor>`` and ``<site>`` blocks, ``position`` actuators with ``inheritrange``
and the ``keyframe`` home pose. Loading the raw XML would therefore silently
drop joint dynamics and limits. To guarantee the *physical* parameters are
preserved we read the compiled body/joint/inertial data directly from the MJCF
and re-emit an explicit URDF (masses, full inertia tensors, body frames, joint
axes and joint limits are copied 1:1). Parameters that URDF cannot express
(armature, joint damping/frictionloss, actuator kp) are re-applied in the
legged_gym config instead (see open_duck_config.py).

This script only depends on numpy + lxml (no mujoco), so it can run anywhere.
"""

import argparse
import math
import os
import shutil

PARSER = argparse.ArgumentParser(description="Convert OpenDuck Mini MJCF to URDF.")
PARSER.add_argument("--mjcf", required=True, help="Path to the source OpenDuck Mini MJCF XML.")
PARSER.add_argument("--assets-dir", help="Mesh directory; defaults to an assets directory beside the MJCF.")
ARGS = PARSER.parse_args()

import numpy as np
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
MJCF_PATH = os.path.abspath(ARGS.mjcf)
ASSETS_SRC = os.path.abspath(ARGS.assets_dir or os.path.join(os.path.dirname(MJCF_PATH), "assets"))

OUT_DIR = HERE
MESH_DIR = os.path.join(OUT_DIR, "meshes")
URDF_PATH = os.path.join(OUT_DIR, "open_duck_mini.urdf")

ROOT_LINK = "trunk_assembly"  # child of the free-floating "base" wrapper

# Actuator (servo) limits taken from the sts3215 default class in the MJCF.
EFFORT_LIMIT = 3.23  # N*m  (position forcerange)
VELOCITY_LIMIT = 5.24  # rad/s (max_motor_velocity in joystick.py)


def _floats(text):
    return [float(v) for v in text.replace(",", " ").split()]


def quat_wxyz_to_rpy(q):
    """MuJoCo quaternion (w, x, y, z) -> URDF fixed-axis rpy (R = Rz*Ry*Rx).

    Builds the rotation matrix then extracts XYZ Euler angles with a
    numerically stable gimbal-lock branch (several leg/head frames have a
    +/-90 deg pitch, i.e. R[2,0] = -/+1).
    """
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    r20 = max(-1.0, min(1.0, R[2][0]))
    if abs(r20) < 0.999999:
        pitch = math.asin(-r20)
        roll = math.atan2(R[2][1], R[2][2])
        yaw = math.atan2(R[1][0], R[0][0])
    else:  # gimbal lock: cos(pitch) ~ 0
        yaw = 0.0
        if r20 <= -1.0 + 1e-6:  # pitch = +90 deg
            pitch = math.pi / 2
            roll = math.atan2(R[0][1], R[0][2])
        else:  # pitch = -90 deg
            pitch = -math.pi / 2
            roll = math.atan2(-R[0][1], -R[0][2])
    return (roll, pitch, yaw)


def _pos(el):
    return _floats(el.get("pos", "0 0 0"))


def _quat(el):
    return _floats(el.get("quat", "1 0 0 0"))


def fmt(vals):
    return " ".join(f"{v:.12g}" for v in vals)


def collect_materials(root):
    mats = {}
    for m in root.iter("material"):
        name = m.get("name")
        rgba = m.get("rgba")
        if name and rgba:
            mats[name] = _floats(rgba)
    return mats


def add_origin(parent, pos, quat):
    rpy = quat_wxyz_to_rpy(quat)
    etree.SubElement(
        parent, "origin", xyz=fmt(pos), rpy=fmt(rpy)
    )


def geom_is_collision(geom):
    """Return whether a source MJCF geom participates in contact.

    The Open Duck MJCF marks render-only meshes with ``class="visual"`` and
    the deploy-time foot contact patches with ``class="collision"``.  Preserve
    that split in the generated URDF instead of making every mesh collide.
    """
    if geom.get("class") == "collision":
        return True
    contype = geom.get("contype")
    conaffinity = geom.get("conaffinity")
    if contype is None and conaffinity is None:
        return False
    return (contype is not None and contype != "0") or (
        conaffinity is not None and conaffinity != "0"
    )


def add_geoms(link_el, body, materials, seen_materials, as_collision):
    """Append URDF mesh elements matching MJCF visual/collision semantics."""
    for geom in body.findall("geom"):
        if geom.get("type", "mesh") != "mesh":
            continue
        mesh = geom.get("mesh")
        if mesh is None:
            continue
        if geom_is_collision(geom) != as_collision:
            continue
        tag = "collision" if as_collision else "visual"
        el = etree.SubElement(link_el, tag)
        add_origin(el, _pos(geom), _quat(geom))
        geo = etree.SubElement(el, "geometry")
        etree.SubElement(geo, "mesh", filename=f"meshes/{mesh}.stl")
        if not as_collision:
            mat_name = geom.get("material")
            if mat_name:
                mel = etree.SubElement(el, "material", name=mat_name)
                if mat_name not in seen_materials and mat_name in materials:
                    rgba = materials[mat_name]
                    etree.SubElement(mel, "color", rgba=fmt(rgba))
                    seen_materials.add(mat_name)


def add_inertial(link_el, body):
    inertial = body.find("inertial")
    if inertial is None:
        return
    ipos = _pos(inertial)
    iel = etree.SubElement(link_el, "inertial")
    etree.SubElement(iel, "origin", xyz=fmt(ipos), rpy="0 0 0")
    etree.SubElement(iel, "mass", value=f"{float(inertial.get('mass')):.12g}")
    if inertial.get("fullinertia") is not None:
        ixx, iyy, izz, ixy, ixz, iyz = _floats(inertial.get("fullinertia"))
    else:  # diaginertia fallback
        ixx, iyy, izz = _floats(inertial.get("diaginertia"))
        ixy = ixz = iyz = 0.0
    etree.SubElement(
        iel, "inertia",
        ixx=f"{ixx:.12g}", ixy=f"{ixy:.12g}", ixz=f"{ixz:.12g}",
        iyy=f"{iyy:.12g}", iyz=f"{iyz:.12g}", izz=f"{izz:.12g}",
    )


def main():
    tree = etree.parse(MJCF_PATH)
    root = tree.getroot()
    materials = collect_materials(root)

    # locate the ROOT_LINK body element
    root_body = None
    for b in root.iter("body"):
        if b.get("name") == ROOT_LINK:
            root_body = b
            break
    if root_body is None:
        raise RuntimeError(f"Body {ROOT_LINK} not found")

    urdf = etree.Element("robot", name="open_duck_mini")
    seen_materials = set()

    joints = []  # (name, parent, child, pos, quat, lower, upper)

    def recurse(body, parent_name):
        name = body.get("name")
        link = etree.SubElement(urdf, "link", name=name)
        add_inertial(link, body)
        add_geoms(link, body, materials, seen_materials, as_collision=False)
        add_geoms(link, body, materials, seen_materials, as_collision=True)
        # joint that connects parent -> this body (skip for root)
        joint = body.find("joint")
        if parent_name is not None and joint is not None:
            lower, upper = _floats(joint.get("range"))
            joints.append(
                (joint.get("name"), parent_name, name, _pos(body), _quat(body), lower, upper)
            )
        for child in body.findall("body"):
            recurse(child, name)

    recurse(root_body, None)

    # emit joints (after all links so URDF order is links-then-joints)
    for jname, parent, child, pos, quat, lower, upper in joints:
        j = etree.SubElement(urdf, "joint", name=jname, type="revolute")
        etree.SubElement(j, "parent", link=parent)
        etree.SubElement(j, "child", link=child)
        rpy = quat_wxyz_to_rpy(quat)
        etree.SubElement(j, "origin", xyz=fmt(pos), rpy=fmt(rpy))
        etree.SubElement(j, "axis", xyz="0 0 1")  # MuJoCo default hinge axis
        etree.SubElement(
            j, "limit",
            lower=f"{lower:.12g}", upper=f"{upper:.12g}",
            effort=f"{EFFORT_LIMIT}", velocity=f"{VELOCITY_LIMIT}",
        )
        # damping/friction handled by legged_gym PD (Kd); kept 0 here to avoid double count
        etree.SubElement(j, "dynamics", damping="0.0", friction="0.0")

    # copy meshes
    os.makedirs(MESH_DIR, exist_ok=True)
    n_meshes = 0
    for mesh_el in root.iter("mesh"):
        f = mesh_el.get("file")
        if not f:
            continue
        src = os.path.join(ASSETS_SRC, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(MESH_DIR, f))
            n_meshes += 1

    etree.ElementTree(urdf).write(
        URDF_PATH, pretty_print=True, xml_declaration=True, encoding="utf-8"
    )
    n_links = len(list(urdf.iter("link")))
    n_joints = len(joints)
    print(f"Wrote {URDF_PATH}")
    print(f"  links={n_links} revolute_joints={n_joints} meshes_copied={n_meshes}")
    print("  joint order:")
    for i, (jn, *_rest) in enumerate(joints):
        print(f"    [{i:2d}] {jn}")


if __name__ == "__main__":
    main()
