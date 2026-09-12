"""Raiden - Toolkit for policy learning with YAM bimanual robot arms"""

import os
import sys

# MuJoCo picks its GL backend when first imported.  Without a display (ssh, a headless
# `rd convert` rendering sim frames) only EGL can render; with one, keep MuJoCo's default.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

__version__ = "0.1.0"
