import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import import_calibration as ic  # noqa: E402


def camera(params=(.2, .05, .7, 2.3, 1.15, .3, 35.)):
    position, quat, fovy = ic.look_at(np.array(params))
    return {'position': position, 'quaternion': quat, 'fovy': fovy}


def test_look_at_points_camera_at_target():
    cam = camera()
    uv, depth = ic.project(np.array([[.2, .05, 0.]]), cam, 320, 240)
    assert depth[0] > 0
    np.testing.assert_allclose(uv[0], [160, 120], atol=1e-6)


def test_backproject_inverts_projection():
    cam = camera(); points = np.array([[.1, -.1, .003], [.3, .15, .003]])
    uv, _ = ic.project(points, cam, 320, 240)
    np.testing.assert_allclose(ic.backproject(uv, cam, 320, 240, .003), points, atol=1e-9)


def test_cma_es_finds_quadratic_minimum():
    target = np.array([.3, -.2, 1.5])
    x, value = ic.cma_es(lambda p: float(np.sum((p - target) ** 2)), np.zeros(3), np.full(3, .5),
                         np.full(3, -3.), np.full(3, 3.), iterations=200, population=12)
    np.testing.assert_allclose(x, target, atol=1e-3)


def test_box_silhouette_fit_recovers_pose():
    cam = camera(); specs = ic.library_objects(); pose = np.array([.15, .1, .6])
    blob = ic.silhouette(specs['eraser'], pose, .003, cam, (240, 320))
    fitted, iou = ic.fit_object(specs['eraser'], blob, .003, cam, np.ones((240, 320), bool))
    assert iou > .85
    np.testing.assert_allclose(fitted[:2], pose[:2], atol=.003)
