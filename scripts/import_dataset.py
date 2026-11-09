"""Import a LeRobot SO-101 dataset: automatic workcell calibration, then strict replay of every episode.

    .venv/bin/python scripts/import_dataset.py DATASET --objects eraser green-bowl --output output/imports/NAME

Writes progress.json while running (read by the Studio), calibration.json with
the fitted camera/mat/object poses and evidence, and results.json with the
per-episode strict replay outcomes. Everything is computed by code from the
dataset itself; no manual input after choosing the library models.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from dataset_to_sim.dataset import DatasetIndex  # noqa: E402
from dataset_to_sim.strict_replay import NativeJointAdapter  # noqa: E402
import import_calibration as ic  # noqa: E402
import import_replay as ir  # noqa: E402


class Progress:
    def __init__(self, folder):
        self.path = Path(folder) / 'progress.json'; self.state = {'status': 'running', 'stage': 'starting', 'log': []}
    def __call__(self, message, **fields):
        self.state['log'] = (self.state['log'] + [message])[-200:]; self.state.update(fields)
        self.path.write_text(json.dumps(self.state, indent=2)); print(message, flush=True)


def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def calibrate(dataset, selected, folder, log, camera=None):
    config = ic.load_config(); adapter = NativeJointAdapter(**config['adapter'])
    specs = ic.library_objects()
    unknown = [key for key in selected if key not in specs]
    if unknown: raise ValueError('Unknown library models: ' + ', '.join(unknown))
    manipulated = [k for k in selected if specs[k]['role'] == 'manipulated']
    container = [k for k in selected if specs[k]['role'] == 'container']
    if len(manipulated) != 1 or len(container) != 1:
        raise ValueError('Choose exactly one object to pick and one bowl from the Model library')
    firsts = [ic.read_frames(dataset.episode(i), 'top', [0])[0] for i in range(dataset.episode_count)]
    height, width = firsts[0].shape[:2]
    sam = ic.SamRefiner()

    def candidate_blobs(rgb, exclusion, tag):
        mat_mask, _ = ic.dark_surface_mask(rgb)
        blobs, _ = ic.object_candidates(rgb, mat_mask, exclusion)
        return sam.refine(rgb, blobs, tag)[0], mat_mask

    # Camera-independent object evidence for choosing the field of view: blobs
    # that do not touch the robot's orange pixels, on a spread of episodes.
    scoring = []
    for index in np.linspace(0, dataset.episode_count - 1, min(6, dataset.episode_count)).astype(int):
        rgb = firsts[index]
        near_robot = cv2.dilate(ic.robot_mask(rgb).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        blobs, _ = candidate_blobs(rgb, near_robot, f'score-{index}')
        scoring.append((rgb, [b for b in blobs if not (b & cv2.dilate(near_robot.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).any()], near_robot))

    def scene_score(cam):
        values = []
        for rgb, blobs, near in scoring:
            plane = {'position': [0, 0, .002], 'size': [1, 1, .002]}  # assumed 2 mm mat on the table for scoring only
            try:
                fitted, _ = ic.fit_objects(rgb, selected, specs, cam, plane, None, near, blobs=blobs)
                values.append(np.mean([v['mask_iou'] for v in fitted.values()]))
            except ValueError:
                values.append(0.)
        return float(np.mean(values))

    if camera is None:
        log('Fitting the top camera to the robot silhouette…', stage='camera')
        camera, camera_report, overlays = ic.fit_top_camera(dataset, config, log=log, scene_score=scene_score)
        cv2.imwrite(str(folder / 'camera-fit.jpg'), cv2.cvtColor(np.vstack([np.hstack(overlays[i:i + 5]) for i in range(0, len(overlays) - len(overlays) % 5, 5)]), cv2.COLOR_RGB2BGR))
    else:
        camera_report = {'method': 'reused from an earlier fit'}
    log(f'Top camera fitted · robot mask IoU {camera_report.get("mean_mask_iou", float("nan")):.3f} · fov {camera["fovy"]:.1f} deg', stage='mat')
    robots, exclusions = [], []
    model = ic.robot_model(config)
    renderer = ic.RobotSilhouette(model, width, height)
    full = mujoco.Renderer(model, height, width); full.enable_segmentation_rendering()
    renderer.set_camera(camera['position'], camera['quaternion'], camera['fovy'])
    robot_bodies = set(range(model.nbody)) - {model.body('cube').id, model.body('bowl').id, 0}
    robot_geoms = np.flatnonzero(np.isin(model.geom_bodyid, list(robot_bodies)))

    def robot_pixels(rgb, native):
        renderer.data.qpos[:6] = adapter.to_model(native); mujoco.mj_forward(model, renderer.data)
        full.update_scene(renderer.data, camera=renderer.camera, scene_option=renderer.option)
        seg = full.render(); robot = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], robot_geoms)
        # Real orange pixels count as robot only where they join the rendered arm (no specks on objects).
        real = ic.robot_mask(rgb).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(real)
        near = cv2.dilate(robot.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        for k in range(1, count):
            component = labels == k
            if (component & near).any() and stats[k, cv2.CC_STAT_AREA] >= 20: robot |= component
        # The jaw cover and cables hang around the robot: exclude a margin.
        return robot, cv2.dilate(robot.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)

    try:
        for index in range(dataset.episode_count):
            robot, exclusion = robot_pixels(firsts[index], dataset.episode(index).state_deg[0])
            robots.append(robot); exclusions.append(exclusion)
        mat, mat_report, mat_mask = ic.fit_mat(firsts, robots, camera, log=log)
        log('Locating the selected models in each episode…', stage='objects')
        poses, object_reports = {}, []

        def locate(index, keys, frame, rgb):
            exclusion = exclusions[index] if frame == 0 else robot_pixels(rgb, dataset.episode(index).state_deg[frame])[1]
            blobs, _ = ic.object_candidates(rgb, mat_mask, exclusion)
            blobs = sam.refine(rgb, blobs, f'episode-{index}-{frame}')[0]
            fitted, _ = ic.fit_objects(rgb, keys, specs, camera, mat, mat_mask, exclusion, blobs=blobs)
            for item in fitted.values(): item['source_frame'] = frame
            return fitted, exclusion

        for index, rgb in enumerate(firsts):
            try:
                fitted, exclusion = locate(index, selected, 0, rgb)
            except ValueError:
                # Fit each model alone; a bowl hidden by the arm at frame 0 is
                # searched in later frames (it stays put until the object is placed).
                fitted, exclusion = {}, exclusions[index]
                for key in selected:
                    try: fitted.update(locate(index, [key], 0, rgb)[0])
                    except ValueError as error: log(f'Episode {index}: {error}')
                episode = dataset.episode(index)
                if container[0] not in fitted:
                    for frame in range(10, episode.frame_count // 2, 10):
                        try:
                            fitted.update(locate(index, [container[0]], frame, ic.read_frames(episode, 'top', [frame])[0])[0]); break
                        except ValueError: continue
            missing = [specs[k]['name'] for k in selected if k not in fitted]
            if missing:
                object_reports.append({'episode': index, 'error': ', '.join(missing) + ' not found'}); log(f'Episode {index}: {", ".join(missing)} not found'); continue
            poses[index] = fitted
            overlay = rgb.copy(); overlay[exclusion] = (overlay[exclusion] * .6).astype(np.uint8)
            for key, item in fitted.items():
                sil = ic.silhouette(specs[key], item['pose_xy_yaw'], item['support_z_m'], camera, rgb.shape[:2])
                contours, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0) if specs[key]['role'] == 'manipulated' else (0, 160, 255), 1)
            corners, _ = ic.project(ic.rectangle_corners([*mat['position'][:2], mat['yaw_rad'], *mat['size'][:2]], mat['position'][2] + mat['size'][2] / 2), camera, width, height)
            cv2.polylines(overlay, [np.round(corners).astype(np.int32)], True, (255, 0, 255), 1)
            cv2.imwrite(str(folder / f'scene-fit-{index:03d}.jpg'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            object_reports.append({'episode': index, **{k: {'mask_iou': v['mask_iou'], 'blob_area_ratio': v['blob_area_ratio'], 'source_frame': v['source_frame']} for k, v in fitted.items()}})
            log(f'Episode {index}: ' + ', '.join(f'{specs[k]["name"]} IoU {v["mask_iou"]:.2f}' + (f' (frame {v["source_frame"]})' if v['source_frame'] else '') for k, v in fitted.items()))
    finally:
        renderer.close(); full.close()
    sam.close()
    calibration = {'schema_version': 1, 'dataset': str(dataset.root), 'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                   'selected_models': selected, 'manipulated': manipulated[0], 'container': container[0],
                   'objects': {k: specs[k] for k in selected}, 'top_camera': camera, 'top_camera_report': camera_report,
                   'mat': mat, 'mat_report': mat_report, 'episode_poses': {str(k): v for k, v in poses.items()},
                   'object_reports': object_reports, 'shared_profile': 'shared-calibration-v10 (calibration/native-action-current.json)',
                   'automatic': True, 'manual_inputs': ['library model selection only']}
    write(folder / 'calibration.json', calibration)
    return calibration


def replay(dataset, calibration, folder, log, episodes=None, render=True):
    config = ir.replay_config(ic.load_config(), calibration); rows = []
    episodes = list(range(dataset.episode_count)) if episodes is None else episodes
    for n, index in enumerate(episodes):
        poses = calibration['episode_poses'].get(str(index))
        log(f'Replaying episode {index} ({n + 1}/{len(episodes)})…', stage='replay', done=n, total=len(episodes))
        if poses is None:
            rows.append({'episode': index, 'outcome': 'scene_not_fitted'}); continue
        try:
            row = ir.run_episode(config, dataset, index, poses, folder / f'episode-{index:03d}', calibration, render=render)
        except Exception as error:
            row = {'episode': index, 'outcome': 'error', 'error': f'{type(error).__name__}: {error}'}
            traceback.print_exc()
        rows.append({k: v for k, v in row.items() if k not in ('audit', 'scene_fit')} | ({'top_robot_mask_iou': {k: v for k, v in row['top_robot_mask_iou'].items() if k != 'per_frame'}} if 'top_robot_mask_iou' in row else {}))
        log(f'Episode {index}: {row["outcome"]}')
        counts = Counter(r['outcome'] for r in rows)
        write(folder / 'results.json', {'dataset': str(dataset.root), 'name': dataset.root.name, 'completed': len(rows), 'total': len(episodes),
                                        'successes': counts.get('success', 0), 'counts': dict(counts), 'results': rows,
                                        'camera_mask_iou': calibration['top_camera_report'].get('mean_mask_iou'),
                                        'note': 'Task criteria of the shared strict replay; original native actions, no repair. Not validated physical fidelity.'})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path); parser.add_argument('--objects', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--episodes', type=int, nargs='*')
    parser.add_argument('--reuse-camera', type=Path, help='calibration.json whose top camera is reused')
    parser.add_argument('--no-render', action='store_true'); parser.add_argument('--skip-replay', action='store_true')
    args = parser.parse_args()
    folder = args.output.resolve(); folder.mkdir(parents=True, exist_ok=True); log = Progress(folder)
    try:
        dataset = DatasetIndex(args.dataset)
        log(f'{dataset.episode_count} episodes · {dataset.root.name}', stage='camera', name=dataset.root.name, total=dataset.episode_count)
        camera = json.loads(args.reuse_camera.read_text())['top_camera'] if args.reuse_camera else None
        calibration = calibrate(dataset, args.objects, folder, log, camera)
        if not args.skip_replay:
            rows = replay(dataset, calibration, folder, log, args.episodes, render=not args.no_render)
            log(f'Done · {sum(r["outcome"] == "success" for r in rows)}/{len(rows)} task successes', stage='complete', status='complete', done=len(rows))
        else:
            log('Calibration complete', stage='complete', status='complete')
    except Exception as error:
        traceback.print_exc(); log(f'Import failed: {error}', status='error', error=str(error)); sys.exit(1)


if __name__ == '__main__':
    main()
