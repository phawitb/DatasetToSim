"""Deterministic state and controller replay in a calibrated MuJoCo workcell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .calibration import Calibration
from .dataset import Episode
from .scene import build_scene_xml


@dataclass(frozen=True)
class ReplayFrame:
    mode: str
    timestamp: float
    qpos: np.ndarray
    images: dict[str, np.ndarray]


class ReplaySession:
    """Owns one compiled scene and makes replay mode explicit in every result."""

    def __init__(
        self,
        episode: Episode,
        calibration: Calibration,
        robot_xml: str | Path,
        *,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.episode = episode
        self.calibration = calibration
        self.model = mujoco.MjModel.from_xml_string(build_scene_xml(calibration, robot_xml))
        self.width, self.height = width, height
        self._joint_ranges = self.model.jnt_range[:6].copy()
        self._control_ranges = self.model.actuator_ctrlrange[:6].copy()
        if self.model.nu < 6 or self.model.nq < 6:
            raise ValueError("SO-101 model must expose six joints and actuators")
        self._validate_kinematic_source()
        self._renderer: mujoco.Renderer | None = None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "ReplaySession":
        return self

    def __exit__(self, *_unused) -> None:
        self.close()

    def kinematic_frame(self, frame: int, *, render: bool = True) -> ReplayFrame:
        """Replay measured state exactly; this is the visual-ground-truth mode."""
        self._check_frame(frame)
        data = mujoco.MjData(self.model)
        qpos = self.calibration.joint_convention.validate_model_ranges(
            self.episode.state_deg[frame], self._joint_ranges, frame=frame
        )
        data.qpos[:6] = qpos
        mujoco.mj_forward(self.model, data)
        return ReplayFrame(
            mode="kinematic",
            timestamp=float(self.episode.timestamps[frame]),
            qpos=data.qpos[:6].copy(),
            images=self._capture(data) if render else {},
        )

    def pd_frame(self, frame: int, *, render: bool = True) -> ReplayFrame:
        """Replay recorded command targets through model actuators, never as state replay."""
        self._check_frame(frame)
        data = mujoco.MjData(self.model)
        initial = self.calibration.joint_convention.validate_model_ranges(
            self.episode.state_deg[0], self._joint_ranges, frame=0
        )
        data.qpos[:6] = initial
        data.ctrl[:6] = self._action_target(0)
        mujoco.mj_forward(self.model, data)
        for current in range(1, frame + 1):
            data.ctrl[:6] = self._action_target(current)
            duration = float(self.episode.timestamps[current] - self.episode.timestamps[current - 1])
            steps = max(1, round(duration / self.model.opt.timestep))
            for _ in range(steps):
                mujoco.mj_step(self.model, data)
        return ReplayFrame(
            mode="pd",
            timestamp=float(self.episode.timestamps[frame]),
            qpos=data.qpos[:6].copy(),
            images=self._capture(data) if render else {},
        )

    def _validate_kinematic_source(self) -> None:
        for frame, values in enumerate(self.episode.state_deg):
            self.calibration.joint_convention.validate_model_ranges(
                values, self._joint_ranges, frame=frame
            )

    def _action_target(self, frame: int) -> np.ndarray:
        return self.calibration.joint_convention.validate_model_ranges(
            self.episode.action_deg[frame], self._control_ranges, frame=frame
        )

    def _check_frame(self, frame: int) -> None:
        if not 0 <= frame < self.episode.frame_count:
            raise IndexError(f"frame {frame} is outside episode range 0..{self.episode.frame_count - 1}")

    def _capture(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            except Exception as error:
                raise RuntimeError(
                    "MuJoCo could not create an OpenGL renderer. On macOS, run this "
                    "from a logged-in desktop session (MUJOCO_GL=cgl); use replay "
                    "--no-render only to validate trajectory state in a headless shell."
                ) from error
        images: dict[str, np.ndarray] = {}
        for view, camera in (("top", "top_cam"), ("wrist", self.calibration.camera("wrist")["camera_name"])):
            self._renderer.update_scene(data, camera=camera)
            images[view] = self._renderer.render().copy()
        return images
