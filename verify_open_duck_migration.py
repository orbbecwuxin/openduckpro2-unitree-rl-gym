"""Offline verification of the Open Duck Mini -> unitree_rl_gym migration.

Isaac Gym is not required. This cross-checks the generated URDF against the
source MJCF and the legged_gym config, and does a small forward-kinematics pass
to confirm the default ("home") pose does not penetrate the ground (so `reset`
will not bounce the robot up).

Run:  python3 verify_open_duck_migration.py --mjcf /path/to/open_duck_mini_v2.xml
"""

import argparse
import ast
import math
import os

PARSER = argparse.ArgumentParser(description="Verify the OpenDuck Mini URDF migration.")
PARSER.add_argument("--mjcf", required=True, help="Path to the source OpenDuck Mini MJCF XML.")
ARGS = PARSER.parse_args()

import numpy as np
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "resources", "robots", "open_duck_mini", "open_duck_mini.urdf")
MJCF = os.path.abspath(ARGS.mjcf)
CONFIG = os.path.join(HERE, "legged_gym", "envs", "open_duck", "open_duck_config.py")

EXPECTED_ORDER = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
INIT_Z = 0.15
FEET_LINKS = ["foot_assembly", "foot_assembly_2"]

ok = True


def check(cond, msg):
    global ok
    print(("  PASS" if cond else "  FAIL") + f"  {msg}")
    ok = ok and cond


def rpy_to_mat(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def T(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


# ---------------------------------------------------------------- parse URDF
utree = etree.parse(URDF)
uroot = utree.getroot()
ujoints = [j for j in uroot.findall("joint") if j.get("type") == "revolute"]
ujoint_names = [j.get("name") for j in ujoints]
ulinks = [l.get("name") for l in uroot.findall("link")]

print("[1] URDF joint count / order")
check(len(ujoints) == 14, f"14 revolute joints (got {len(ujoints)})")
check(ujoint_names == EXPECTED_ORDER, "joint declaration order matches MJCF actuator order")

# ---------------------------------------------------------------- MJCF limits
mroot = etree.parse(MJCF).getroot()
mjcf_ranges = {}
for jn in mroot.iter("joint"):
    name = jn.get("name")
    rng = jn.get("range")
    if name and rng:
        mjcf_ranges[name] = [float(v) for v in rng.split()]

print("[2] Joint limits equal to MJCF")
lim_ok = True
for j in ujoints:
    lim = j.find("limit")
    lo, hi = float(lim.get("lower")), float(lim.get("upper"))
    mlo, mhi = mjcf_ranges[j.get("name")]
    if abs(lo - mlo) > 1e-6 or abs(hi - mhi) > 1e-6:
        lim_ok = False
        print(f"      mismatch {j.get('name')}: urdf[{lo},{hi}] mjcf[{mlo},{mhi}]")
check(lim_ok, "all 14 joint ranges match")
eff = {float(j.find("limit").get("effort")) for j in ujoints}
vel = {float(j.find("limit").get("velocity")) for j in ujoints}
check(eff == {3.23}, f"effort/torque limit = 3.23 Nm (got {eff})")
check(vel == {5.24}, f"velocity limit = 5.24 rad/s (got {vel})")

# ---------------------------------------------------------------- feet bodies
print("[3] Feet bodies + collision")
foot_name = "foot_assembly"
matched_feet = [l for l in ulinks if foot_name in l]
check(sorted(matched_feet) == sorted(FEET_LINKS),
      f"foot_name='{foot_name}' matches exactly {FEET_LINKS} (got {matched_feet})")
for fl in FEET_LINKS:
    link_el = next(l for l in uroot.findall("link") if l.get("name") == fl)
    has_col = link_el.find("collision") is not None
    check(has_col, f"{fl} has collision geometry")
# left/right disambiguation via kinematic chain
left_chain = "left" in [j.get("name") for j in ujoints if j.find("child").get("link") == "foot_assembly"][0]
right_chain = "right" in [j.get("name") for j in ujoints if j.find("child").get("link") == "foot_assembly_2"][0]
check(left_chain, "foot_assembly is on the LEFT leg (left_ankle joint)")
check(right_chain, "foot_assembly_2 is on the RIGHT leg (right_ankle joint)")

# ---------------------------------------------------------------- config (ast)
print("[4] legged_gym config dims / default pose")
with open(CONFIG) as f:
    ctree = ast.parse(f.read())

cfg_vals = {}
default_pose = {}
for node in ast.walk(ctree):
    if isinstance(node, ast.Assign):
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name) and isinstance(node.value, ast.Constant):
            cfg_vals[tgt.id] = node.value.value
        if isinstance(tgt, ast.Name) and tgt.id == "default_joint_angles" and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                val = v.value if isinstance(v, ast.Constant) else -v.operand.value
                default_pose[k.value] = float(val)

check(cfg_vals.get("num_actions") == 14, f"num_actions=14 (got {cfg_vals.get('num_actions')})")
expected_obs = 3 + 3 + 3 + 3 * 14 + 2
check(cfg_vals.get("num_observations") == expected_obs,
      f"num_observations={expected_obs} (got {cfg_vals.get('num_observations')})")
check(cfg_vals.get("num_privileged_obs") == expected_obs + 3,
      f"num_privileged_obs={expected_obs + 3} (got {cfg_vals.get('num_privileged_obs')})")
check(set(default_pose.keys()) == set(EXPECTED_ORDER),
      "default_joint_angles has an entry for every DOF (no KeyError at init)")

# ---------------------------------------------------------------- FK ground check
print("[5] Default pose ground clearance (reset does not bounce)")
# build joint chain lookup
by_child = {j.find("child").get("link"): j for j in ujoints}

# foot contact frames (soles) from the MJCF sites, expressed in the foot body frame
foot_sites = {}
for site in mroot.iter("site"):
    if site.get("name") in ("left_foot", "right_foot"):
        foot_sites[site.get("name")] = np.array(
            [float(v) for v in site.get("pos").split()])
site_of = {"foot_assembly": "left_foot", "foot_assembly_2": "right_foot"}


def foot_sole_world_z(link):
    """world z of the foot sole site at the default pose, base at INIT_Z."""
    M = np.eye(4)
    cur = link
    chain = []
    while cur in by_child:
        j = by_child[cur]
        chain.append(j)
        cur = j.find("parent").get("link")
    for j in reversed(chain):
        o = j.find("origin")
        xyz = np.array([float(v) for v in o.get("xyz").split()])
        rpy = [float(v) for v in o.get("rpy").split()]
        M = M @ T(rpy_to_mat(rpy), xyz)
        ang = default_pose[j.get("name")]
        M = M @ T(rotz(ang), np.zeros(3))
    sole_local = foot_sites[site_of[link]]
    world = np.array([0, 0, INIT_Z]) + (M[:3, :3] @ sole_local + M[:3, 3])
    return world[2]

for fl in FEET_LINKS:
    z = foot_sole_world_z(fl)
    # soles should rest at/near the ground (~0), not deeply below it
    check(-0.03 < z < 0.05, f"{site_of[fl]} sole world z = {z:.4f} m (rests near ground)")

print()
print("RESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
