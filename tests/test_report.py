import json

import numpy as np
import pytest

from dataset_to_sim.calibration import Calibration
from dataset_to_sim.report import evaluate_calibration, image_metrics
from tests.test_calibration import valid_config


def test_image_metrics_are_normalized_and_reject_shape_mismatch():
    reference = np.zeros((2, 2, 3), dtype=np.uint8)
    simulation = np.full((2, 2, 3), 255, dtype=np.uint8)

    metrics = image_metrics(reference, simulation)

    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == 1.0
    with pytest.raises(ValueError, match="shape"):
        image_metrics(reference, np.zeros((2, 3, 3), dtype=np.uint8))


def test_draft_calibration_cannot_be_reported_as_accepted(tmp_path):
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(valid_config()))

    verdict = evaluate_calibration(
        Calibration.load(path),
        {"top": {"rmse": 0.0, "mask_iou": 1.0}, "wrist": {"rmse": 0.0, "mask_iou": 1.0}},
    )

    assert verdict["accepted"] is False
    assert "draft" in verdict["reason"]
