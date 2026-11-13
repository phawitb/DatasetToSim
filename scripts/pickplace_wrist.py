"""Wrist-camera check right above the object, before the grasp.

The arm stops at the Look pose of a simulated plan. The Wrist camera, about 10 cm
above the object, sees it roughly five times larger than the top camera. SAM
segments it; the measured 3D model is fitted to that silhouette with the camera
pose taken from the measured joints. The camera rides on the gripper, so errors
of the arm model and of the top camera largely cancel: what is measured is where
the object is relative to the jaws. The rest of the pick and place is then
planned and simulated again from the measured pose.
"""
import copy
import itertools
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
MAX_SHIFT_M = .025       # larger means the wrong object or a bad capture
MAX_TURN_RAD = np.pi / 4
MAX_RMSE_PX = 6.
RECHECK_SHIFT_M = .003   # the object must not move between the check and execution
RECHECK_TURN_RAD = np.deg2rad(5)


@lru_cache(maxsize=2)
def _model(config_json):
    import mujoco
    from episode0 import Calibration
    from strict_action_scene import build_model
    config = json.loads(config_json)
    cal = Calibration(ROOT / 'calibration/native-action-current.json', copy.deepcopy(config['base_calibration']))
    return build_model(cal, config)


def wrist_camera(config, q_native):
    """Wrist camera pose at measured joints, from the same calibrated robot model."""
    import mujoco
    from dataset_to_sim.strict_replay import NativeJointAdapter
    model = _model(json.dumps(config, sort_keys=True)); data = mujoco.MjData(model)
    data.qpos[:6] = NativeJointAdapter(**config['adapter']).to_model(np.asarray(q_native, float)); mujoco.mj_forward(model, data)
    i = model.camera('wrist_cam').id
    return {'position': data.cam_xpos[i].copy(), 'rotation': data.cam_xmat[i].reshape(3, 3).copy(),
            'focal': 240 / np.tan(np.deg2rad(model.cam_fovy[i] / 2))}


def project(camera, points):
    points = np.atleast_2d(points)
    local = (points - camera['position']) @ camera['rotation'] @ np.diag([1, -1, -1])
    if np.any(local[:, 2] <= .01):
        raise ValueError('Object is behind or too close to the Wrist camera')
    return local[:, :2] / local[:, 2:] * camera['focal'] + [320, 240]


def model_points(shape, pose):
    """3D outline points of a registry shape at pose (x, y, yaw), origin height z."""
    x, y, z, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw); R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if shape['type'] == 'box':
        size = np.asarray(shape['size_m'])
        local = np.array(list(itertools.product(*[(-v / 2, v / 2) for v in size])))
    elif shape['type'] == 'ring':
        r, h = shape['outer_diameter_m'] / 2, shape['height_m']
        a = np.arange(48) * 2 * np.pi / 48
        local = np.array([[r * np.cos(t), r * np.sin(t), dz] for dz in (-h / 2, h / 2) for t in a])
    else:
        r, h = shape['outer_radius_m'], shape['height_m']
        a = np.arange(64) * 2 * np.pi / 64
        local = np.array([[r * np.cos(t), r * np.sin(t), h] for t in a] + [[shape['bottom_outer_radius_m'] * np.cos(t), shape['bottom_outer_radius_m'] * np.sin(t), 0] for t in a])
    return local @ R.T + [x, y, z]


def _hull(points):
    from scipy.spatial import ConvexHull
    return points[ConvexHull(points).vertices]


def fit(camera, mask, shape, initial):
    """Least-squares XY (and yaw for boxes) so the projected model outline matches the mask."""
    from dataset_to_sim.scene_estimation import _sample, _distance
    contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    observed = cv2.convexHull(max(contours, key=cv2.contourArea))[:, 0, :].astype(float); points = _sample(observed)
    x0, y0, z, yaw0 = initial; free_yaw = shape['type'] == 'box'

    def outline(p):
        return _hull(project(camera, model_points(shape, (p[0], p[1], z, p[2] if free_yaw else yaw0))))

    def residual(p):
        h = outline(p)
        return np.r_[_distance(_sample(h), observed), _distance(points, h)]
    start = np.array([x0, y0, yaw0]); span = np.array([MAX_SHIFT_M, MAX_SHIFT_M, MAX_TURN_RAD if free_yaw else 1e-6])
    fits = []
    for turn in ((0., -np.pi / 4, np.pi / 4) if free_yaw else (0.,)):
        guess = np.clip(start + [0, 0, turn], start - span, start + span)
        fits.append(least_squares(residual, guess, bounds=(start - span, start + span), diff_step=1e-4, loss='soft_l1', f_scale=2.))
    best = min(fits, key=lambda r: r.cost)
    rmse = float(np.sqrt(np.mean(residual(best.x) ** 2)))
    return best.x, rmse, observed, outline(best.x)


def measure(client, image, config, registry, picked, q_native, decode, colour_pixels=None):
    """Measured picked-object pose from one Wrist image at the Look pose."""
    shape = registry['objects'][picked['kind']]['shape']
    camera = wrist_camera(config, q_native)
    pose = (*picked['position'][:2], picked['position'][2], picked.get('yaw', 0.))
    top = np.asarray(picked['position']) + [0, 0, (shape['size_m'][2] / 2 if shape['type'] == 'box' else shape.get('height_m', 0) / 2)]
    centre = project(camera, top[None])[0]
    expected = _hull(project(camera, model_points(shape, pose)))
    area = cv2.contourArea(expected.astype(np.float32))
    prompts = [centre] + [centre + d for d in ([0, -18], [0, 18], [-18, 0], [18, 0], [0, -36], [0, 36])]
    best = None; error = 'no usable mask'
    for point in prompts:
        if not (0 <= point[0] < 640 and 0 <= point[1] < 480):
            continue
        try:
            result = client.predict(image, [float(point[0]), float(point[1])])
        except ValueError as exc:
            error = str(exc); continue
        mask = decode(result)
        if colour_pixels is not None:
            # A painted cube on another cube: keep only its own colour (top face and sides).
            mask = mask & colour_pixels(mask)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            if n < 2:
                continue
            mask = labels == 1 + int(np.argmax(stats[1:, 4]))
        if not .3 * area <= mask.sum() <= 3. * area:
            error = 'mask size does not match the object'; continue
        try:
            fitted, rmse, observed, outline = fit(camera, mask, shape, pose)
        except ValueError as exc:
            error = str(exc); continue
        if best is None or rmse < best[1]:
            best = (fitted, rmse, observed, outline, point, result, mask)
        if rmse <= 3:
            break
    if best is None:
        raise ValueError(f'Wrist check could not see the {picked["kind"]} ({error}); capture again')
    fitted, rmse, observed, outline, point, result, mask = best
    if rmse > MAX_RMSE_PX:
        raise ValueError(f'Wrist check: the {picked["kind"]} outline does not match its model ({rmse:.1f} px); capture again')
    shift = float(np.linalg.norm(fitted[:2] - pose[:2]))
    turn = float((fitted[2] - pose[3] + np.pi / 4) % (np.pi / 2) - np.pi / 4) if shape['type'] == 'box' else 0.
    return {'position': [float(fitted[0]), float(fitted[1]), float(pose[2])], 'yaw': float(pose[3] + turn),
            'shift_m': shift, 'shift_xy_m': (fitted[:2] - pose[:2]).tolist(), 'turn_rad': turn, 'rmse_px': rmse,
            'prompt': [float(point[0]), float(point[1])], 'observed_hull': observed.tolist(), 'model_hull': outline.tolist(),
            'mask': mask, 'result': result}


def colour_filter(rgb, colour):
    """Pixels of one painted cube colour, for separating a top cube from the one below."""
    from studio_sam_geometry import _color_pixels
    if colour not in ('red', 'green', 'blue', 'yellow'):
        return None
    return lambda mask: _color_pixels(rgb, mask)[colour]
