"""Keep live top-camera images aligned with the recorded dataset (SIMCUBE_1O1B_VF).

The top camera's pose was calibrated on the recorded dataset; every pixel-to-robot
conversion uses that calibration. If the camera is bumped, the live image no longer
lines up with it. Each live capture is therefore compared with the dataset's view of
the static scene (the first frame of a recorded episode) using ORB features outside
the mat and away from the arm, and a similarity transform
(shift, small turn, scale). If the view moved, the capture is warped back onto the
dataset view before anything else reads it; the original is kept as top-raw.jpg.
Views that cannot be matched, or moved too far to trust a warp, block planning.

Check camera: the robot first moves to that dataset frame's arm pose, so the arm is
part of the comparison too (same pose, same place in the image).
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'calibration/top-camera-dataset-reference.jpg'
REFERENCE_POSE = ROOT / 'calibration/top-camera-dataset-reference.json'
POSE_TOLERANCE_DEG = 1.5
ALIGN_FROM_PX = 1.     # smaller shifts are left as they are (sub-pixel noise)
MAX_WARP_PX = 25.      # larger moves: the camera must be put back
MIN_INLIERS = 30


def base_config():
    return json.loads((ROOT / 'calibration/native-action-current.json').read_text())


def live_config(config=None):
    """Live captures use the dataset camera calibration: their images are aligned to it."""
    return config if config is not None else base_config()


def project(camera, points):
    R = np.diag([1, -1, -1]) @ Rotation.from_quat(np.array(camera['quaternion'])[[1, 2, 3, 0]]).as_matrix().T
    focal = 240 / np.tan(np.deg2rad(camera['fovy'] / 2))
    p = (R @ (np.asarray(points, float) - np.asarray(camera['position'], float)).T).T
    return np.c_[focal * p[:, 0] / p[:, 2] + 320, focal * p[:, 1] / p[:, 2] + 240]


def reference(dataset=None):
    """The dataset frame the live camera is compared with: the first top-camera frame
    of the episode whose starting arm pose is closest to the typical (median) start,
    together with that arm pose. Built once from SIMCUBE_1O1B_VF."""
    if REFERENCE.exists() and REFERENCE_POSE.exists():
        return cv2.imread(str(REFERENCE))
    from dataset_to_sim.dataset import DatasetIndex
    from episode0 import DATA
    index = DatasetIndex(Path(dataset) if dataset else DATA)
    starts = np.array([index.episode(i).state_deg[0] for i in range(index.episode_count)])
    episode = int(np.argmin(np.abs(starts - np.median(starts, axis=0)).max(axis=1)))
    part = index.episode(episode).video_slices['top']
    capture = cv2.VideoCapture(str(part.path)); capture.set(cv2.CAP_PROP_POS_FRAMES, round(part.start_time * 30))
    ok, image = capture.read(); capture.release()
    if not ok:
        raise ValueError('Could not read the recorded dataset to build the camera reference')
    REFERENCE.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(REFERENCE), image)
    REFERENCE_POSE.write_text(json.dumps({'dataset': str(index.root) if hasattr(index, 'root') else str(DATA), 'episode': episode, 'frame': 0,
                                          'state_native': starts[episode].tolist()}, indent=2))
    return image


def reference_pose():
    reference()
    return json.loads(REFERENCE_POSE.read_text())


def _static_mask(config, image, arm=False):
    """Pixels of the fixed scene: outside the mat (objects move there) and, unless the
    arm stands in the dataset frame's pose (`arm`), not the arm."""
    camera = config['base_calibration']['cameras']['top']; mat = config['base_calibration']['world']['mat']
    half = np.asarray(mat['size'][:2]) / 2 + .03; c = np.asarray(mat['position'][:2])
    corners = project(camera, [[c[0] + sx * half[0], c[1] + sy * half[1], 0.] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    mask = np.full(image.shape[:2], 255, np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 0)
    if not arm:
        orange = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (5, 120, 120), (22, 255, 255))
        mask[cv2.dilate(orange, np.ones((25, 25), np.uint8)) > 0] = 0
    return mask


def measure(image, config=None, arm=False):
    """Similarity transform from the dataset view to `image`, and how far points on
    the mat moved (pixels)."""
    config = config or base_config()
    ref = reference()
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), _static_mask(config, ref, arm))
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), _static_mask(config, image, arm))
    if d1 is None or d2 is None:
        return None, {'ok': False, 'moved_px': None, 'note': 'Too little texture around the mat to compare with the dataset view'}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < .75 * n.distance]
    if len(good) < MIN_INLIERS:
        return None, {'ok': False, 'moved_px': None, 'note': f'Only {len(good)} features match the dataset view'}
    a = np.float32([k1[m.queryIdx].pt for m in good]); b = np.float32([k2[m.trainIdx].pt for m in good])
    A, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.)
    if A is None or inliers.sum() < MIN_INLIERS:
        return None, {'ok': False, 'moved_px': None, 'note': 'The view does not match the dataset view'}
    mat = config['base_calibration']['world']['mat']
    half = np.asarray(mat['size'][:2]) / 2; c = np.asarray(mat['position'][:2]); top = mat['position'][2] + mat['size'][2] / 2
    grid = np.array([[c[0] + sx * half[0], c[1] + sy * half[1], top] for sx in (-1, 0, 1) for sy in (-1, 0, 1)])
    uv = project(config['base_calibration']['cameras']['top'], grid)
    moved = (uv @ A[:, :2].T + A[:, 2]) - uv
    shift = float(np.max(np.linalg.norm(moved, axis=1)))
    turn = float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))); scale = float(np.hypot(A[0, 0], A[1, 0]))
    return A, {'ok': True, 'moved_px': round(shift, 2), 'turn_deg': round(turn, 3), 'scale': round(scale, 4), 'inliers': int(inliers.sum())}


def align(folder, config=None, name='top.jpg'):
    """Warp a top-camera image (default the capture's top.jpg) onto the dataset view
    when the camera moved."""
    path = Path(folder) / name; image = cv2.imread(str(path))
    A, result = measure(image, config)
    result['aligned'] = False
    if A is None:
        return result
    if result['moved_px'] > MAX_WARP_PX:
        result.update(ok=False, note=f"The top camera moved {result['moved_px']:.0f} px from the dataset view; put it back where it was")
        return result
    if result['moved_px'] > ALIGN_FROM_PX:
        path.with_name(path.stem + '-raw.jpg').write_bytes(path.read_bytes())
        warped = cv2.warpAffine(image, A, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(str(path), warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result['aligned'] = True
        # Confirm: the warped image must now match the dataset view.
        _, after = measure(warped, config)
        result['residual_px'] = after.get('moved_px')
    result['checked_at'] = time.time()
    return result
