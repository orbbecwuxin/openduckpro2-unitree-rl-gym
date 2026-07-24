#!/usr/bin/env python3
"""Run an Isaac Lab script after injecting the OpenDuck task registration."""

import argparse
import os
from pathlib import Path
import sys


SCRIPT_PATHS = {
    "train": "scripts/reinforcement_learning/rsl_rl/train.py",
    "play": "scripts/reinforcement_learning/rsl_rl/play.py",
    "zero": "scripts/environments/zero_agent.py",
    "random": "scripts/environments/random_agent.py",
}
MARKER = "# PLACEHOLDER: Extension template (do not remove this comment)"


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("script", choices=SCRIPT_PATHS)
    args, forwarded = parser.parse_known_args()

    isaaclab_root = Path(
        os.environ.get("ISAACLAB_ROOT", "/data2/wuxin/IsaacLab-2.3.2")
    ).resolve()
    script_path = isaaclab_root / SCRIPT_PATHS[args.script]
    source = script_path.read_text(encoding="utf-8")
    if MARKER not in source:
        raise RuntimeError(f"Isaac Lab script injection marker missing: {script_path}")
    source = source.replace(MARKER, f"{MARKER}\nimport isaaclab_openduck", 1)

    sys.path.insert(0, str(script_path.parent))
    sys.argv = [str(script_path), *forwarded]
    namespace = {"__name__": "__main__", "__file__": str(script_path)}
    exec(compile(source, str(script_path), "exec"), namespace)


if __name__ == "__main__":
    main()
