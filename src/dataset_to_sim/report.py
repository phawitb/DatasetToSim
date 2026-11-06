"""Visual evidence generation for real-to-sim calibration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration
from .dataset import VideoSlice


def image_metrics(reference: np.ndarray, simulation: np.ndarray) -> dict[str, float]:
    """Return normalized, shape-safe image residuals and simple colour-mask IoU."""
    if reference.shape != simulation.shape:
        raise ValueError(f"reference and simulation image shape differ: {reference.shape} != {simulation.shape}")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("images must be H×W×3 RGB arrays")
    delta = reference.astype(np.float64) / 255.0 - simulation.astype(np.float64) / 255.0
    return {
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "mask_iou": _mask_iou(_object_mask(reference), _object_mask(simulation)),
    }


def evaluate_calibration(calibration: Calibration, metrics_by_view: dict[str, dict[str, float]]) -> dict[str, object]:
    """Apply thresholds without ever promoting a draft to an accepted result."""
    if calibration.status != "accepted":
        return {"accepted": False, "reason": f"calibration status is {calibration.status}"}
    thresholds = calibration.data["acceptance"]
    top, wrist = metrics_by_view["top"], metrics_by_view["wrist"]
    checks = {
        "top_rmse": top["rmse"] <= thresholds["top_rmse_max"],
        "wrist_rmse": wrist["rmse"] <= thresholds["wrist_rmse_max"],
        "top_mask_iou": top["mask_iou"] >= thresholds["top_mask_iou_min"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"accepted": not failed, "reason": "thresholds passed" if not failed else f"failed: {', '.join(failed)}", "checks": checks}


def decode_video_frame(video: VideoSlice, episode_timestamp: float) -> np.ndarray:
    """Decode one RGB source frame at the global timestamp recorded in metadata."""
    capture = cv2.VideoCapture(str(video.path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {video.path}")
    requested_seconds = video.start_time + float(episode_timestamp)
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, requested_seconds * 1000)
        ok, bgr = capture.read()
    finally:
        capture.release()
    if not ok or bgr is None:
        raise RuntimeError(f"cannot decode video at {requested_seconds:.6f}s: {video.path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_contact_sheet(reference: np.ndarray, simulation: np.ndarray, output: str | Path, *, label: str) -> Path:
    """Save a labelled source/render comparison without modifying either input."""
    if reference.shape != simulation.shape:
        raise ValueError("cannot create contact sheet for images of different shapes")
    height, width = reference.shape[:2]
    canvas = np.zeros((height + 32, width * 2, 3), dtype=np.uint8)
    canvas[32:, :width] = reference
    canvas[32:, width:] = simulation
    cv2.putText(canvas, f"real {label}", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"MuJoCo {label}", (width + 8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write comparison image: {destination}")
    return destination


def _object_mask(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.int16)
    red = (rgb[..., 0] > 100) & (rgb[..., 0] > rgb[..., 1] * 13 // 10) & (rgb[..., 0] > rgb[..., 2] * 13 // 10)
    green = (rgb[..., 1] > 80) & (rgb[..., 1] > rgb[..., 0] * 11 // 10) & (rgb[..., 1] > rgb[..., 2] * 11 // 10)
    return red | green


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return 1.0 if union == 0 else float(np.logical_and(left, right).sum() / union)
