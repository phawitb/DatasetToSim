"""An explicitly identified clear roll can be localized by its visible core rim."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pickplace_vision import object_pose
from pickplace import load_registry
from top_camera import project
CONFIG = json.loads((ROOT / 'calibration/native-action-current.json').read_text())


def observed_rim(xy, support, diameter=.030):
    a = np.arange(96) * 2 * np.pi / 96
    points = np.c_[xy[0] + diameter / 2 * np.cos(a), xy[1] + diameter / 2 * np.sin(a), np.full(96, support + .024)]
    mask = np.zeros((480, 640), np.uint8)
    cv2.fillConvexPoly(mask, np.round(project(CONFIG['base_calibration']['cameras']['top'], points)).astype(np.int32), 1)
    return mask > 0


def test_core_observation_localizes_full_tape_at_support_height():
    xy = [.263, -.135]
    mask = observed_rim(xy, .007)
    pose = object_pose(CONFIG, load_registry(), 'tape', mask, np.zeros((480, 640, 3), np.uint8), .007, mask_feature='core_outer_rim')
    assert np.linalg.norm(np.array(pose['xy']) - xy) < .0015
    assert pose['fit_rmse_px'] < 1.
    assert load_registry()['objects']['tape']['shape']['outer_diameter_m'] == .053


def test_core_observation_rejects_wrong_size_and_other_object_kind():
    rgb = np.zeros((480, 640, 3), np.uint8)
    with pytest.raises(ValueError):
        object_pose(CONFIG, load_registry(), 'tape', observed_rim([.263, -.135], .007, .050), rgb, .007, mask_feature='core_outer_rim')
    with pytest.raises(ValueError):
        object_pose(CONFIG, load_registry(), 'cube', observed_rim([.263, -.135], .007), rgb, .007, mask_feature='core_outer_rim')


def test_standing_pen_grasp_follows_full_orientation_not_flat_yaw():
    from pickplace import grasp_candidates
    pen = {'kind': 'pen', 'position': [.28, 0., .070], 'yaw': 0.,
           'initial_quaternion': [2 ** -.5, 0., -(2 ** -.5), 0.]}
    candidates = grasp_candidates(load_registry(), pen, [0., 0.])
    cap = next(c for c in candidates if c['grasp'] == 'Behind the cap')
    assert np.allclose(cap['xyz'], [.28, 0., .100], atol=1e-8)
    assert cap['jaw_yaw'] == pytest.approx(np.pi / 2)
