"""Strict native-action replay of an imported dataset in its auto-calibrated workcell.

The robot, wrist camera, joint/gripper adapter, motor model and contact
parameters are the shared strict-replay configuration, unchanged. The import
calibration replaces only the top camera, the mat and the initial object poses.
Recorded actions are applied exactly; nothing is retimed, clipped or repaired.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from dataset_to_sim.calibration import Calibration  # noqa: E402
from dataset_to_sim.observations import camera_sample_indices  # noqa: E402
from dataset_to_sim.strict_replay import NativeJointAdapter, integrate_actions, array_hash  # noqa: E402
import import_calibration as ic  # noqa: E402


def replay_config(base, calibration):
    """Shared configuration with only the fitted scene substituted."""
    config = copy.deepcopy(base)
    data = config['base_calibration']; world = data['world']
    camera = calibration['top_camera']
    data['cameras']['top'] = {'position': camera['position'], 'quaternion': camera['quaternion'], 'fovy': camera['fovy'],
                              'target': camera['target']}
    world['mat'] = {k: calibration['mat'][k] for k in ('position', 'size', 'quaternion')}
    # Robot height relative to the table plane, measured with the wrist camera (0 when not measured).
    world['robot_base']['position'][2] += float(calibration.get('robot_lift_m', 0.))
    # Dataset-level arm motor model, when identification improved held-out tracking.
    if calibration.get('identified_dynamics'): config['identified_dynamics'] = calibration['identified_dynamics']
    specs = calibration['objects']; manipulated = specs[calibration['manipulated']]
    if manipulated['shape'] != 'box':
        raise ValueError('Imported replay currently supports box-shaped manipulated objects (cube, eraser)')
    size = manipulated['size_m']
    world['cube'].update(size=float(max(size)), size_xyz=[float(v) for v in size], object=calibration['manipulated'])
    config['estimates']['cube_mass_kg'] = float(manipulated['mass_kg'])
    container = specs[calibration['container']]
    world['bowl'].update(outer_radius=container['top_radius_m'], height=container['height_m'])
    config['import_calibration'] = {k: calibration[k] for k in ('dataset', 'manipulated', 'container', 'created')}
    return config


def episode_calibration(config, poses, manipulated, container):
    data = copy.deepcopy(config['base_calibration']); world = data['world']
    x, y, yaw = poses[manipulated]['pose_xy_yaw']; support = poses[manipulated]['support_z_m']
    world['cube']['position'] = [x, y, support + world['cube']['size_xyz'][2] / 2 + .0002]
    world['cube']['quaternion'] = [float(np.cos(yaw / 2)), 0., 0., float(np.sin(yaw / 2))]
    bx, by, _ = poses[container]['pose_xy_yaw']
    world['bowl']['position'] = [bx, by, poses[container]['support_z_m']]
    return Calibration(ROOT / 'calibration/import-episode.json', data)


def score(trace, inside, lift=.04):
    """Same task criteria as the shared strict replay (lift 4 cm, bilateral grasp, inside the bowl, released, settled)."""
    if float(trace['cube_pose'][:, 2].max()) < float(trace['cube_pose'][0, 2] + lift): return 'not_lifted'
    if trace['bilateral'].sum() < 2: return 'no_bilateral_grasp'
    support = trace['audit'].get('grasp_support')
    if support and (support['longest_secure_grasp_s'] < .08 or support['longest_secure_lift_s'] < .03): return 'no_bilateral_grasp'
    if not inside['inside']: return 'missed_bowl'
    if trace['audit']['final_finger_contact']: return 'not_released'
    if np.linalg.norm(trace['cube_pose'][-10:, :3] - trace['cube_pose'][-1, :3], axis=1).max() > .003: return 'not_settled'
    if trace['audit']['final_cube_linear_speed_m_s'] > .03 or trace['audit']['final_cube_angular_speed_rad_s'] > .3: return 'not_settled'
    return 'success'


APPEARANCE = {'eraser': ('.95 .95 .93 1', ('eraser_sleeve', '.78 .80 .84 1', 'sleeve_length_m')), 'cube': ('.82 .07 .07 1', None)}


def scene_model(cal, config, calibration):
    """Strict replay model with the library object's look (physics unchanged)."""
    import xml.etree.ElementTree as ET
    from strict_action_scene import build_model
    root = ET.fromstring(build_model(cal, config, return_xml=True))
    key = calibration['manipulated']; rgba, sleeve = APPEARANCE.get(key, ('.9 .9 .9 1', None))
    geom = root.find(".//geom[@name='cube_geom']"); geom.set('rgba', rgba); geom.attrib.pop('material', None)
    if sleeve:
        size = [float(v) / 2 for v in cal.world['cube']['size_xyz']]
        length = json.loads((ROOT / 'assets/models/eraser/eraser.json').read_text())[sleeve[2]]
        ET.SubElement(root.find(".//body[@name='cube']"), 'geom', name=sleeve[0], type='box', rgba=sleeve[1], contype='0', conaffinity='0',
                      mass='0', group='2', size=f'{length / 2} {size[1] + .0001} {size[2] + .0001}')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def run_episode(config, dataset, index, poses, folder, calibration, render=True):
    from strict_action_scene import containment
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    ep = dataset.episode(index); adapter = NativeJointAdapter(**config['adapter'])
    cal = episode_calibration(config, poses, calibration['manipulated'], calibration['container'])
    model = scene_model(cal, config, calibration)
    trace = integrate_actions(model, ep, adapter)
    np.testing.assert_array_equal(trace['native_action_applied'], ep.action_deg)
    native_state = adapter.to_native(trace['qpos'])
    inside = containment(model, trace['qpos'][-1], trace['cube_pose'][-1], cal, config)
    row = {'episode': index, 'frames': ep.frame_count, 'outcome': score(trace, inside),
           'native_action_sha256': array_hash(ep.action_deg), 'action_equality_verified': True,
           'joint_state_rmse_native': np.sqrt(np.mean((native_state[:-1] - ep.state_deg) ** 2, axis=0)).tolist(),
           'max_object_z_m': float(trace['cube_pose'][:, 2].max()), 'initial_object_pose': trace['cube_pose'][0].tolist(),
           'final_object': trace['cube_pose'][-1, :3].tolist(), 'bowl_position': cal.world['bowl']['position'],
           'bowl_containment': inside, 'bilateral_intervals': int(trace['bilateral'].sum()),
           'initial_max_penetration_m': trace['audit']['initial_max_penetration_m'], 'audit': trace['audit'],
           'scene_fit': poses}
    (folder / 'calibration.json').write_text(json.dumps(cal.data, indent=2))
    np.savez_compressed(folder / 'trace.npz', qpos=trace['qpos'], cube_pose=trace['cube_pose'], native=native_state,
                        bilateral=trace['bilateral'], sample_times=trace['sample_times'])
    if render:
        row['top_robot_mask_iou'] = render_comparison(model, cal, config, ep, trace, folder)
    (folder / 'report.json').write_text(json.dumps(row, indent=2))
    return row


def render_comparison(model, cal, config, ep, trace, folder):
    """Real | sim video in the Studio viewer layout (1280 x 1024: top row, wrist row, 32 px labels).

    Real frames are upscaled from the source resolution; sim renders at 640 x 480
    with the same field of view. Returns per-frame orange-robot mask IoU (top,
    at source resolution).
    """
    frames = {view: ic.read_frames(ep, view, range(ep.frame_count)) for view in ('top', 'wrist')}
    height, width = frames['top'][0].shape[:2]
    delays = config.get('camera_delay_frames', {})
    sensors = {view: camera_sample_indices(ep.frame_count, delays.get(view, 0)) for view in ('top', 'wrist')}
    d = mujoco.MjData(model); addr = model.jnt_qposadr[model.joint('cube_free').id]
    option = mujoco.MjvOption(); option.geomgroup[3:] = 0; option.sitegroup[:] = 0
    silhouette = ic.RobotSilhouette(model, width, height)
    raw = folder / 'comparison-raw.mp4'
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 1024))
    if not writer.isOpened(): raise RuntimeError('Video encoder unavailable')
    ious = []
    try:
        with mujoco.Renderer(model, 480, 640) as renderer:
            for f in range(ep.frame_count):
                rows = []
                for view, camera in (('top', 'top_cam'), ('wrist', 'wrist_cam')):
                    k = int(sensors[view][f])
                    d.qpos[:6] = trace['qpos'][k]; d.qpos[addr:addr + 7] = trace['cube_pose'][k]; mujoco.mj_forward(model, d)
                    renderer.update_scene(d, camera=camera, scene_option=option); sim = renderer.render()
                    canvas = np.zeros((512, 1280, 3), np.uint8)
                    canvas[32:, :640] = cv2.resize(frames[view][f], (640, 480), interpolation=cv2.INTER_CUBIC); canvas[32:, 640:] = sim
                    delay = f' | camera delay {delays.get(view, 0)}f' if delays.get(view, 0) else ''
                    for text, pos in ((f'REAL | EP {ep.index:02d} | {view} | frame {f:03d} | {f / 30:.2f}s', (10, 23)),
                                      (f'MuJoCo | ORIGINAL actions | {view}{delay}', (650, 23))):
                        cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, .49, (255, 255, 255), 1, cv2.LINE_AA)
                    rows.append(canvas)
                    if view == 'top':
                        sim_mask = silhouette.render(trace['qpos'][k]); real_mask = ic.robot_mask(frames['top'][f])
                        ious.append(float((sim_mask & real_mask).sum() / max(1, (sim_mask | real_mask).sum())))
                combined = np.vstack(rows)
                writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                if f in (0, ep.frame_count // 2, ep.frame_count - 1):
                    cv2.imwrite(str(folder / f'comparison-{f:03d}.jpg'), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    finally:
        writer.release(); silhouette.close()
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(raw), '-c:v', 'libx264', '-crf', '22', '-preset', 'fast',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(folder / 'real-vs-sim.mp4')], check=True)
    raw.unlink(missing_ok=True)
    return {'mean': float(np.mean(ious)), 'min': float(np.min(ious)), 'per_frame': ious}
