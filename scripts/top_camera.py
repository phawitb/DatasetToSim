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


# Extra dataset frames where the arm is raised and nearly still (joints within 2.3 deg
# over +-3 frames, so the recorded joints match the picture): over the bowl side and
# over the cube side. Reached from Home through the raised corridor; paths checked
# for contacts in MuJoCo. The recorded episodes hold still raised only rarely.
EXTRA_FRAMES = [(48, 118), (54, 48)]


def poses():
    """Comparison poses: 0 is the reference rest pose, then the raised arm poses.
    Each has its dataset frame (image file) and arm pose (native degrees)."""
    from dataset_to_sim.dataset import DatasetIndex
    from episode0 import DATA
    data = reference_pose()
    if len(data.get('poses', [])) == 1 + len(EXTRA_FRAMES):
        return data['poses']
    index = DatasetIndex(DATA)
    out = [{'episode': data['episode'], 'frame': 0, 'state_native': data['state_native'], 'image': REFERENCE.name, 'raised': False}]
    for episode, frame in EXTRA_FRAMES:
        ep = index.episode(episode); part = ep.video_slices['top']
        capture = cv2.VideoCapture(str(part.path)); capture.set(cv2.CAP_PROP_POS_FRAMES, round(part.start_time * 30) + frame)
        ok, image = capture.read(); capture.release()
        if not ok:
            raise ValueError(f'Could not read dataset episode {episode} frame {frame}')
        name = f'top-camera-dataset-e{episode}-f{frame}.jpg'
        cv2.imwrite(str(REFERENCE.parent / name), image)
        out.append({'episode': episode, 'frame': frame, 'state_native': ep.state_deg[frame].tolist(), 'image': name, 'raised': True})
    data['poses'] = out
    REFERENCE_POSE.write_text(json.dumps(data, indent=2))
    return out


def compare_arm(reference_image, image):
    """Where the orange arm is in both images: overlap, and the shift that best lines
    the two arm silhouettes up (phase correlation; the fixed base is cut off)."""
    def arm(im):
        mask = cv2.inRange(cv2.cvtColor(im, cv2.COLOR_BGR2HSV), (5, 120, 120), (22, 255, 255))
        mask[400:] = 0
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    a, b = arm(reference_image), arm(image)
    if a.sum() < 500 or b.sum() < 500:
        return {'arm_iou': None, 'arm_offset_px': None}
    (dx, dy), _ = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))
    return {'arm_iou': round(float((a & b).sum() / (a | b).sum()), 3), 'arm_offset_px': [round(dx, 1), round(dy, 1)]}
