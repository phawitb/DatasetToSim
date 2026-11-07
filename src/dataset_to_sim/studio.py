"""Shared contracts for the local Real / Sim Studio and hardware worker."""
from pathlib import Path
import time
import numpy as np

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
SPEED_LIMITS = np.array([35.] * 5 + [50.])


def safe_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('File is outside the media directory or does not exist')
    return path


def validate_trajectory(actions, timestamps, limits):
    q, t, limits = np.asarray(actions, float), np.asarray(timestamps, float), np.asarray(limits, float)
    if q.ndim != 2 or q.shape[1] != 6 or len(q) < 2 or t.shape != (len(q),):
        raise ValueError('Expected at least two six-joint commands and matching times')
    if not np.isfinite(q).all() or not np.isfinite(t).all() or t[0] != 0 or np.any(np.diff(t) <= 0):
        raise ValueError('Commands and increasing timestamps must be finite and start at zero')
    if limits.shape != (6, 2) or not np.isfinite(limits).all() or np.any(limits[:,0] >= limits[:,1]):
        raise ValueError('Invalid joint limits')
    overshoot = np.maximum(limits[:,0] - q, 0) + np.maximum(q - limits[:,1], 0)
    if np.any(overshoot[0] > np.array([2.]*5+[0.])) or np.any(np.diff(overshoot,axis=0) > 1e-6) or np.any((q[:,5] < 0) | (q[:,5] > 100)):
        raise ValueError('Trajectory exceeds native joint limits')
    if np.any(np.abs(np.diff(q, axis=0)) / np.diff(t)[:,None] > SPEED_LIMITS + 1e-5):
        raise ValueError('Trajectory exceeds speed limits')
    if t[-1] > 120:
        raise ValueError('Trajectory is longer than 120 seconds')


def execution_gate(plan, current, calibration_hash, now=None):
    now = time.time() if now is None else now
    if plan.get('source') != 'live':
        raise ValueError('Dataset rehearsals cannot run on hardware; capture a live scene')
    if plan.get('outcome') != 'success' or not plan.get('transfer_checks_passed'):
        raise ValueError('Simulation and transfer checks must pass first')
    if plan.get('executed'):
        raise ValueError('This plan has already been consumed; capture again')
    if not 0 <= now - plan['captured_at'] <= 180:
        raise ValueError('Capture expired; capture and simulate again')
    if plan.get('calibration_hash') != calibration_hash:
        raise ValueError('Robot calibration changed')
    current = np.asarray(current, float)
    start = np.asarray(plan['initial_native'], float)
    if current.shape != (6,) or start.shape != (6,) or not np.isfinite([current,start]).all() or np.any(np.abs(current-start) > np.array([2.]*5+[3.])):
        raise ValueError('Robot has moved since capture; capture and simulate again')


def native_limits(config):
    return np.array([[-(config['follower_calibration'][j]['range_max']-config['follower_calibration'][j]['range_min'])*180/4095,
                      (config['follower_calibration'][j]['range_max']-config['follower_calibration'][j]['range_min'])*180/4095]
                     for j in JOINTS[:5]] + [[0,100]])


def validate_scene_match(planned, observed):
    stacking='target_cube' in planned
    if ('target_cube' in observed)!=stacking or observed.get('bowl_present',True)!=planned.get('bowl_present',True):raise ValueError('Scene objects changed; capture again')
    objects=['cube']+(['bowl'] if planned.get('bowl_present',True) else [])+(['target_cube'] if stacking else [])
    for obj in objects:
        tolerance=.003 if stacking and obj!='bowl' else .008
        if np.linalg.norm(np.array(observed[obj]['position'])-planned[obj]['position'])>tolerance:
            raise ValueError('Objects moved; capture and simulate again')
    def yaw(q): return 2*np.arctan2(q[3],q[0])
    # Square cube yaw is equivalent modulo 90 degrees.
    for obj in ['cube']+(['target_cube'] if stacking else []):
        difference=(yaw(observed[obj]['quaternion'])-yaw(planned[obj]['quaternion'])+np.pi/4)%(np.pi/2)-np.pi/4
        if abs(difference)>np.deg2rad(10):raise ValueError('Cube rotated; capture and simulate again')
