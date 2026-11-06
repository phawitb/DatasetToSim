"""Versioned, explicit calibration inputs for dataset replay."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .joints import JointConvention


class CalibrationError(ValueError):
    """A calibration cannot support a reproducible replay."""


def _numbers(value: Any, name: str, length: int | None = None) -> list[float]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        expected = f"a list of {length}" if length is not None else "a list"
        raise CalibrationError(f"{name} must be {expected}")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"{name} must contain numbers") from error
    if not all(math.isfinite(item) for item in result):
        raise CalibrationError(f"{name} must contain finite numbers")
    return result


@dataclass(frozen=True)
class Calibration:
    path: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        calibration_path = Path(path).expanduser().resolve()
        try:
            data = json.loads(calibration_path.read_text())
        except FileNotFoundError as error:
            raise CalibrationError(f"calibration file does not exist: {calibration_path}") from error
        except json.JSONDecodeError as error:
            raise CalibrationError(f"invalid calibration JSON: {error}") from error
        calibration = cls(path=calibration_path, data=data)
        calibration.validate()
        return calibration

    @property
    def status(self) -> str:
        return self.data["status"]

    @property
    def world(self) -> dict[str, Any]:
        return self.data["world"]

    @property
    def joint_convention(self) -> JointConvention:
        source = self.data["joint_convention"]
        return JointConvention(sign=source["sign"], offset_deg=source["offset_deg"])

    def camera(self, view: str) -> dict[str, Any]:
        try:
            return self.data["cameras"][view]
        except KeyError as error:
            raise CalibrationError(f"missing {view} camera configuration") from error

    def validate(self) -> None:
        if not isinstance(self.data, dict) or self.data.get("schema_version") != 1:
            raise CalibrationError("schema_version must be 1")
        if self.data.get("status") not in {"draft", "accepted"}:
            raise CalibrationError("status must be 'draft' or 'accepted'")
        if self.status == "accepted" and not isinstance(self.data.get("residuals"), dict):
            raise CalibrationError("accepted calibration requires residuals evidence")
        convention = self.data.get("joint_convention")
        if not isinstance(convention, dict):
            raise CalibrationError("missing joint_convention")
        try:
            JointConvention(sign=convention["sign"], offset_deg=convention["offset_deg"])
        except (KeyError, ValueError) as error:
            raise CalibrationError(f"invalid joint_convention: {error}") from error
        world = self.data.get("world")
        if not isinstance(world, dict):
            raise CalibrationError("missing world")
        base = _object(world, "robot_base")
        _numbers(base.get("position"), "world.robot_base.position", 3)
        _numbers(base.get("quaternion"), "world.robot_base.quaternion", 4)
        table = _object(world, "table")
        _positive(_numbers(table.get("size"), "world.table.size", 2), "world.table.size")
        _positive([_number(table.get("height"), "world.table.height")], "world.table.height")
        mat = _object(world, "mat")
        _numbers(mat.get("position"), "world.mat.position", 3)
        _positive(_numbers(mat.get("size"), "world.mat.size", 3), "world.mat.size")
        cube = _object(world, "cube")
        _numbers(cube.get("position"), "world.cube.position", 3)
        _positive([_number(cube.get("size"), "world.cube.size")], "world.cube.size")
        bowl = _object(world, "bowl")
        _numbers(bowl.get("position"), "world.bowl.position", 3)
        _positive(
            [
                _number(bowl.get("outer_radius"), "world.bowl.outer_radius"),
                _number(bowl.get("height"), "world.bowl.height"),
            ],
            "world.bowl",
        )
        top = self.camera("top")
        _numbers(top.get("position"), "cameras.top.position", 3)
        _numbers(top.get("target"), "cameras.top.target", 3)
        _positive([_number(top.get("fovy"), "cameras.top.fovy")], "cameras.top.fovy")
        wrist = self.camera("wrist")
        if not isinstance(wrist.get("camera_name"), str) or not wrist["camera_name"]:
            raise CalibrationError("cameras.wrist.camera_name must be a non-empty string")
        acceptance = self.data.get("acceptance")
        if not isinstance(acceptance, dict):
            raise CalibrationError("missing acceptance thresholds")
        _positive(
            [
                _number(acceptance.get("top_rmse_max"), "acceptance.top_rmse_max"),
                _number(acceptance.get("wrist_rmse_max"), "acceptance.wrist_rmse_max"),
                _number(acceptance.get("top_mask_iou_min"), "acceptance.top_mask_iou_min"),
            ],
            "acceptance",
        )


def _object(parent: dict[str, Any], name: str) -> dict[str, Any]:
    value = parent.get(name)
    if not isinstance(value, dict):
        raise CalibrationError(f"missing world.{name}")
    return value


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"{name} must be a number") from error
    if not math.isfinite(result):
        raise CalibrationError(f"{name} must be finite")
    return result


def _positive(values: list[float], name: str) -> None:
    if not all(value > 0 for value in values):
        raise CalibrationError(f"{name} must be positive")
