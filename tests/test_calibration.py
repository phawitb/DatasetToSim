import json

import pytest

from dataset_to_sim.calibration import Calibration, CalibrationError


def valid_config():
    return {
        "schema_version": 1,
        "status": "draft",
        "joint_convention": {"sign": [1] * 6, "offset_deg": [0] * 6},
        "world": {
            "robot_base": {"position": [0, 0, 0.768], "quaternion": [1, 0, 0, 0]},
            "table": {"size": [0.85, 0.65], "height": 0.75},
            "mat": {"position": [0.1, -0.1, 0.772], "size": [0.6, 0.45, 0.004]},
            "cube": {"position": [0.12, -0.12, 0.797], "size": 0.04},
            "bowl": {"position": [0.3, -0.1, 0.775], "outer_radius": 0.075, "height": 0.055},
        },
        "cameras": {
            "top": {"position": [0.3, -1.1, 1.6], "target": [0.1, -0.1, 0.78], "fovy": 62},
            "wrist": {"camera_name": "wrist_cam"},
        },
        "acceptance": {"top_rmse_max": 0.1, "wrist_rmse_max": 0.1, "top_mask_iou_min": 0.8},
    }


def test_calibration_loads_explicit_world_camera_and_joint_convention(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(valid_config()))

    calibration = Calibration.load(path)

    assert calibration.status == "draft"
    assert calibration.joint_convention.offset_deg == [0] * 6
    assert calibration.camera("top")["fovy"] == 62


def test_calibration_rejects_an_accepted_result_without_evidence(tmp_path):
    payload = valid_config()
    payload["status"] = "accepted"
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(CalibrationError, match="residuals"):
        Calibration.load(path)
