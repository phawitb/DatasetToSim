"""Reproducible replay and calibration utilities for real SO-101 data."""

import os
import platform

# GLFW can terminate a headless macOS process before Python receives an error.
# CGL instead raises a catchable error when no desktop graphics connection exists.
if platform.system() == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")
