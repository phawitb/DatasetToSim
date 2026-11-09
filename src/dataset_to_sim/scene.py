"""Build a calibrated workcell around an immutable SO-101 robot model."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .calibration import Calibration


def _numbers(values) -> str:
    return " ".join(str(float(value)) for value in values)


def _look_at_quaternion(position, target) -> list[float]:
    """Return MuJoCo camera quaternion (wxyz) for camera -Z facing target."""
    position = np.asarray(position, dtype=float)
    forward = np.asarray(target, dtype=float) - position
    length = np.linalg.norm(forward)
    if length == 0:
        raise ValueError("camera position and target must differ")
    forward /= length
    reference_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, reference_up))) > 0.99:
        reference_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, reference_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    matrix = np.column_stack((right, up, -forward))
    return _matrix_to_quaternion(matrix)


def _matrix_to_quaternion(matrix: np.ndarray) -> list[float]:
    trace = float(np.trace(matrix))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [(s / 4), (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s]
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            q = [(matrix[2, 1] - matrix[1, 2]) / s, s / 4, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s]
        elif index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            q = [(matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, s / 4, (matrix[1, 2] + matrix[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            q = [(matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, s / 4]
    return [float(value) for value in q]


def build_scene_xml(calibration: Calibration, robot_xml: str | Path) -> str:
    """Return scene XML while making the source robot and its meshes read-only inputs."""
    robot_path = Path(robot_xml).expanduser().resolve()
    if not robot_path.is_file():
        raise FileNotFoundError(f"SO-101 XML does not exist: {robot_path}")
    root = ET.parse(robot_path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str((robot_path.parent / "assets").resolve()))
    world = root.find("worldbody")
    if world is None:
        raise ValueError("robot XML has no worldbody")
    base = world.find("body[@name='base']")
    if base is not None:
        base_config = calibration.world["robot_base"]
        base.set("pos", _numbers(base_config["position"]))
        base.set("quat", _numbers(base_config["quaternion"]))
    table = calibration.world["table"]
    width, depth = table["size"]
    height = table["height"]
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 .01", rgba=".87 .87 .87 1")
    ET.SubElement(world, "geom", name="table", type="box", pos=_numbers([0, 0, height - 0.018]), size=_numbers([width / 2, depth / 2, 0.018]), rgba=".94 .94 .94 1", friction=".8 .005 .0001")
    mat = calibration.world["mat"]
    mat_geom = ET.SubElement(world, "geom", name="work_mat", type="box", pos=_numbers(mat["position"]), size=_numbers([mat["size"][0] / 2, mat["size"][1] / 2, mat["size"][2] / 2]), rgba=".05 .05 .06 1", friction=".95 .005 .0001")
    if "quaternion" in mat:
        mat_geom.set("quat", _numbers(mat["quaternion"]))
    cube = calibration.world["cube"]
    cube_body = ET.SubElement(world, "body", name="cube", pos=_numbers(cube["position"]))
    ET.SubElement(cube_body, "freejoint", name="cube_free")
    ET.SubElement(cube_body, "geom", name="cube_geom", type="box", size=_numbers([value / 2 for value in cube.get("size_xyz", [cube["size"]] * 3)]), mass=".04", rgba=".82 .07 .07 1", friction=".9 .005 .0001")
    bowl = calibration.world["bowl"]
    bowl_body = ET.SubElement(world, "body", name="bowl", pos=_numbers(bowl["position"]))
    ET.SubElement(bowl_body, "geom", name="bowl_base", type="cylinder", pos=_numbers([0, 0, 0.003]), size=_numbers([bowl["outer_radius"], 0.003]), rgba=".50 .70 .65 1", friction=".8 .005 .0001")
    for step in range(20):
        angle = 2 * math.pi * step / 20
        ET.SubElement(bowl_body, "geom", name=f"bowl_wall_{step}", type="box", pos=_numbers([bowl["outer_radius"] * math.cos(angle), bowl["outer_radius"] * math.sin(angle), bowl["height"] / 2]), size=_numbers([0.003, 0.012, bowl["height"] / 2]), euler=_numbers([0, 0, angle]), rgba=".50 .70 .65 1", friction=".8 .005 .0001")
    top = calibration.camera("top")
    ET.SubElement(world, "camera", name="top_cam", mode="fixed", pos=_numbers(top["position"]), quat=_numbers(_look_at_quaternion(top["position"], top["target"])), fovy=str(float(top["fovy"])))
    return ET.tostring(root, encoding="unicode")
