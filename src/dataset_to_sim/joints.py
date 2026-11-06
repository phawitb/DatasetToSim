"""Explicit conversion between the recorded SO-101 convention and MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
FEATURE_JOINT_NAMES = tuple(f"{name}.pos" for name in JOINT_NAMES)


class JointRangeError(ValueError):
    """A converted recorded position cannot be represented by the model."""


@dataclass(frozen=True)
class JointConvention:
    """Recorded degrees plus mounting/calibration corrections to model radians."""

    sign: Sequence[float] = field(default_factory=lambda: [1.0] * len(JOINT_NAMES))
    offset_deg: Sequence[float] = field(default_factory=lambda: [0.0] * len(JOINT_NAMES))

    def __post_init__(self) -> None:
        if len(self.sign) != len(JOINT_NAMES) or len(self.offset_deg) != len(JOINT_NAMES):
            raise ValueError(f"joint convention requires exactly {len(JOINT_NAMES)} entries")
        if not np.isfinite(self.sign).all() or not np.isfinite(self.offset_deg).all():
            raise ValueError("joint convention values must be finite")

    def to_radians(self, values_deg: Sequence[float]) -> np.ndarray:
        values = np.asarray(values_deg, dtype=float)
        if values.shape[-1:] != (len(JOINT_NAMES),):
            raise ValueError(f"expected final joint dimension {len(JOINT_NAMES)}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("recorded joint values must be finite")
        return np.deg2rad((values + np.asarray(self.offset_deg)) * np.asarray(self.sign))

    def validate_model_ranges(
        self,
        values_deg: Sequence[float],
        model_ranges_rad: Sequence[Sequence[float]],
        *,
        frame: int | None = None,
    ) -> np.ndarray:
        radians = self.to_radians(values_deg)
        ranges = np.asarray(model_ranges_rad, dtype=float)
        if ranges.shape != (len(JOINT_NAMES), 2):
            raise ValueError(f"expected model ranges shape ({len(JOINT_NAMES)}, 2), got {ranges.shape}")
        for index, (value, limits) in enumerate(zip(radians, ranges, strict=True)):
            if not limits[0] <= value <= limits[1]:
                suffix = f" at frame {frame}" if frame is not None else ""
                raise JointRangeError(
                    f"{JOINT_NAMES[index]}={value:.6f} rad is outside "
                    f"[{limits[0]:.6f}, {limits[1]:.6f}]{suffix}"
                )
        return radians
