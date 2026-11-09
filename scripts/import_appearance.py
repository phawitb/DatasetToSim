"""Real background / Realistic Simulation views for an imported dataset.

The SIMCUBE workcell has a hand-built appearance reference (background plate
and colour profile). An imported recording gets its own, built automatically
from its frames and the auto-calibrated scene:

* background plate: per-pixel median of upscaled Top frames across episodes,
  with every rendered robot/object pixel (dilated) and every real orange
  robot pixel removed, holes filled from the same real surface;
* colour profile: robust Lab statistics of real vs rendered pixels per
  semantic class (mat, table, robot, dark robot parts, bowl, picked object)
  on paired frames, for both cameras.

Rendering then reuses the SIMCUBE episode renderer unchanged (display only:
archived states are rendered again; physics is never rerun).

    python scripts/import_appearance.py video IMPORT_ID EPISODE MODE
    python scripts/import_appearance.py frame IMPORT_ID EPISODE MODE FRAME TARGET
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault('MUJOCO_GL', 'cgl' if sys.platform == 'darwin' else 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from dataset_to_sim.dataset import DatasetIndex  # noqa: E402
from dataset_to_sim.observations import camera_sample_indices  # noqa: E402
from dataset_to_sim.strict_replay import NativeJointAdapter  # noqa: E402
import import_calibration as ic  # noqa: E402
import import_replay as ir  # noqa: E402
from studio_episode_appearance import EpisodeAppearance, MODES, as_640, APPEARANCE_VERSION  # noqa: E402
from studio_real_appearance import RealAppearance, complete_background, geom_classes, robust_stats, segmentation_ids  # noqa: E402

IMPORTS = ROOT / 'output/imports'
REFERENCE_VERSION = 8
FILTERS = {
    # Warm-lit recordings: the robot is strongly saturated orange, the mat is a
    # less saturated brown of similar hue, so value/saturation floors matter.
    'robot': lambda hsv: (hsv[..., 0] <= 26) & (hsv[..., 1] > 125) & (hsv[..., 2] > 150),
    'dark_robot': lambda hsv: (hsv[..., 2] < 80),
    'bowl': lambda hsv: (hsv[..., 1] < 120) & (hsv[..., 2] > 110),
    'object': lambda hsv: (hsv[..., 1] < 70) & (hsv[..., 2] > 185),
    'work_mat': lambda hsv: (hsv[..., 2] < 170),
    'table': lambda hsv: (hsv[..., 1] < 90) & (hsv[..., 2] > 170),
    # The wrist camera keeps the bowl's own (teal) hue; reject grey/white table and mat spill.
    ('wrist', 'bowl'): lambda hsv: (hsv[..., 0] > 60) & (hsv[..., 0] < 105) & (hsv[..., 1] > 35) & (hsv[..., 2] > 60),
}


def class_filter(view, kind):
    return FILTERS.get((view, kind)) or FILTERS.get(kind)


def import_folder(identifier):
    folder = (IMPORTS / identifier).resolve()
    if not folder.is_relative_to(IMPORTS.resolve()) or not (folder / 'calibration.json').is_file():
        raise ValueError('Import not found')
    return folder


class ImportScene:
    """Shared configuration, dataset and per-episode model of one import."""

    def __init__(self, identifier):
        self.folder = import_folder(identifier); self.identifier = identifier
        self.calibration = json.loads((self.folder / 'calibration.json').read_text())
        self.config = ir.replay_config(ic.load_config(), self.calibration)
        self.dataset = DatasetIndex(self.calibration['dataset'])
        self.adapter = NativeJointAdapter(**self.config['adapter'])

    def episode_folder(self, index):
        folder = self.folder / f'episode-{index:03d}'
        if not (folder / 'trace.npz').is_file(): raise ValueError('Episode not found')
        return folder

    def model(self, index):
        from dataset_to_sim.calibration import Calibration
        cal = Calibration(self.folder / 'calibration.json', json.loads((self.episode_folder(index) / 'calibration.json').read_text()))
        model = ir.scene_model(cal, self.config, self.calibration)
        camera = self.config['wrist_camera']; cid = model.camera('wrist_cam').id
        model.cam_pos[cid] = camera['position']; model.cam_quat[cid] = camera['quaternion']; model.cam_fovy[cid] = camera['fovy']
        return model

    def trace(self, index):
        with np.load(self.episode_folder(index) / 'trace.npz') as trace:
            return trace['qpos'].copy(), trace['cube_pose'].copy()


def build_reference(scene, log=print):
    """Background plate, valid mask and colour profile for one import (cached)."""
    reference = scene.folder / f'appearance-reference-v{REFERENCE_VERSION}'
    if (reference / 'profile.json').is_file(): return reference
    reference.mkdir(exist_ok=True)
    plates, real_colors, sim_colors, samples = [], {'top': {}, 'wrist': {}}, {'top': {}, 'wrist': {}}, []
    response = {'top': {'radius': [], 'log_ratio': [], 'sharp': []}, 'wrist': {'radius': [], 'log_ratio': [], 'sharp': []}}
    yy, xx = np.mgrid[:480, :640]; radius = np.hypot(xx - 320, yy - 240) / np.hypot(320, 240)
    option = mujoco.MjvOption(); option.geomgroup[3:] = 0; option.sitegroup[:] = 0
    episodes = sorted(int(p.name.split('-')[1]) for p in scene.folder.glob('episode-*') if (p / 'trace.npz').is_file())
    for index in episodes[::max(1, len(episodes) // 10)]:
        ep = scene.dataset.episode(index); model = scene.model(index); data = mujoco.MjData(model); classes = geom_classes(model)
        qpos, cube = scene.trace(index); addr = model.jnt_qposadr[model.joint('cube_free').id]
        delays = scene.config.get('camera_delay_frames', {})
        frames = sorted({int((ep.frame_count - 1) * f) for f in (0, .15, .3, .45, .6, .75, .9, 1.)})
        reals = {view: ic.read_frames(ep, view, frames) for view in ('top', 'wrist')}
        lens = scene.config['wrist_camera'].get('distortion'); maps = None
        if lens:
            fovy = scene.config['wrist_camera']['fovy']; render_fovy = ic.covering_fovy(fovy, *lens, 640, 480)
            model.cam_fovy[model.camera('wrist_cam').id] = render_fovy; maps = ic.distortion_maps(fovy, *lens, 640, 480, render_fovy)
        with mujoco.Renderer(model, 480, 640) as renderer:
            for n, frame in enumerate(frames):
                for view in ('top', 'wrist'):
                    k = int(camera_sample_indices(ep.frame_count, delays.get(view, 0))[frame])
                    data.qpos[:6] = qpos[k]; data.qpos[addr:addr + 7] = cube[k]; mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=view + '_cam', scene_option=option)
                    sim = renderer.render().copy(); ids = segmentation_ids(renderer)
                    if view == 'wrist' and maps is not None:
                        sim = ic.distort_render(sim, maps)
                        ids = cv2.remap(ids.astype(np.float32), maps[0], maps[1], cv2.INTER_NEAREST, borderValue=-1).astype(int)
                    rgb = as_640(reals[view][n]); hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                    for kind in set(classes.values()):
                        mask = np.isin(ids, [i for i, c in classes.items() if c == kind]).astype('uint8')
                        mask = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
                        if class_filter(view, kind): mask &= class_filter(view, kind)(hsv)
                        # Objects move after the grasp: only the first frame pairs them reliably in Top.
                        # Top: objects move after the grasp, so only the first frame pairs them.
                        # Wrist: the colour filter keeps real object pixels where the masks agree.
                        if kind in ('object', 'bowl') and view == 'top' and n: continue
                        if mask.sum() > 30:
                            real_colors[view].setdefault(kind, []).append(rgb[mask][::2]); sim_colors[view].setdefault(kind, []).append(sim[mask][::2])
                    # Camera response: radial fall-off (vignetting) on the mat, and sharpness.
                    mat_pixels = cv2.erode(np.isin(ids, [i for i, c in classes.items() if c == 'work_mat']).astype('uint8'), np.ones((9, 9), np.uint8)) > 0
                    mat_pixels &= FILTERS['work_mat'](hsv)
                    if mat_pixels.sum() > 2000:
                        real_l = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float) + 2; sim_l = cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY).astype(float) + 2
                        pick = np.flatnonzero(mat_pixels.ravel())[::7]
                        response[view]['radius'].append(radius.ravel()[pick]); response[view]['log_ratio'].append(np.log(real_l.ravel()[pick] / sim_l.ravel()[pick]))
                    response[view]['sharp'].append((cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY)))
                    if view == 'top':
                        surfaces = np.isin(ids, [i for i, c in classes.items() if c in ('work_mat', 'table', 'floor')])
                        foreground = cv2.dilate((~surfaces).astype('uint8'), np.ones((31, 31), np.uint8)) > 0
                        robot = cv2.dilate(ic.robot_mask(rgb).astype('uint8'), np.ones((25, 25), np.uint8)) > 0
                        static = surfaces & ~foreground & ~robot
                        plate = rgb.astype(float); plate[~static] = np.nan; plates.append(plate)
                        samples.append({'episode': index, 'frame': frame})
        log(f'appearance reference: episode {index}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning); plate = np.nanmedian(plates, axis=0)
    valid = np.isfinite(plate).all(2); clean = np.nan_to_num(plate).astype('uint8')
    camera = scene.calibration['top_camera']; mat = scene.calibration['mat']
    corners, _ = ic.project(ic.rectangle_corners([*mat['position'][:2], mat['yaw_rad'], *mat['size'][:2]], mat['position'][2] + mat['size'][2] / 2), camera, 640, 480)
    mat_mask = np.zeros((480, 640), np.uint8); cv2.fillConvexPoly(mat_mask, np.round(corners).astype(np.int32), 1)
    clean, valid, surface_rgb = complete_background(clean, valid, mat_mask > 0)
    camera_response = {}
    for view, items in response.items():
        entry = {'vignetting': [0., 0.], 'blur_sigma_px': 0.}
        if len(items['radius']) >= 3:
            r = np.concatenate(items['radius']); y = np.concatenate(items['log_ratio'])
            A = np.c_[np.ones_like(r), r ** 2, r ** 4]; coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            entry['vignetting'] = [float(np.clip(coef[1], -3, 1)), float(np.clip(coef[2], -3, 3))]
        # Blur that makes rendered sharpness (Laplacian energy) match the real camera.
        def energy(image): return float(cv2.Laplacian(image.astype(np.float32), cv2.CV_32F).var())
        real_e = np.median([energy(a) for a, _ in items['sharp']])
        best = min(np.arange(0, 3.01, .25), key=lambda sigma: abs(np.log((np.median([energy(cv2.GaussianBlur(b, (0, 0), sigma) if sigma else b) for _, b in items['sharp']]) + 1) / (real_e + 1))))
        entry['blur_sigma_px'] = float(best)
        camera_response[view] = entry
    appearance = {}
    for view in ('top', 'wrist'):
        appearance[view] = {kind: {'real': robust_stats(np.concatenate(pixels)), 'sim': robust_stats(np.concatenate(sim_colors[view][kind])), 'measured': True}
                            for kind, pixels in real_colors[view].items() if sum(len(p) for p in pixels) >= 200}
    for kind in ('object', 'bowl'):
        if kind in appearance['wrist']: appearance['wrist'][kind]['wrist_measured'] = True
    for view in ('top', 'wrist'):
        for kind in ('work_mat', 'table'):
            if kind not in appearance[view]: appearance[view][kind] = appearance['top'].get(kind) or {'real': robust_stats(np.tile(np.uint8(surface_rgb[kind]), (50, 1))), 'sim': None}
    cv2.imwrite(str(reference / 'background.png'), cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(reference / 'valid-mask.png'), valid.astype('uint8') * 255)
    profile = {'schema_version': 1, 'dataset': scene.dataset.root.name, 'import': scene.identifier, 'camera_response': camera_response,
               'camera': {k: camera[k] for k in ('position', 'quaternion', 'fovy')}, 'samples': samples,
               'background_sha256': hashlib.sha256((reference / 'background.png').read_bytes()).hexdigest(),
               'filled_fraction': float((~valid).mean()), 'surface_rgb': surface_rgb, 'appearance': appearance, 'color_sources': {},
               'method': 'Automatic: masked temporal median of upscaled Top frames (rendered robot/objects and real orange pixels removed), '
                         'holes filled from the same real surface; Lab statistics of real vs rendered pixels per semantic class on paired frames.'}
    (reference / 'profile.json').write_text(json.dumps(profile, indent=2))
    return reference


class ImportedEpisodeAppearance(EpisodeAppearance):
    object_kinds = ('robot', 'dark_robot', 'object', 'bowl')

    def __init__(self, identifier, index, mode):  # noqa: D401 - replaces the SIMCUBE archive loader
        if mode not in MODES: raise ValueError('Unknown appearance mode')
        scene = ImportScene(identifier); reference = build_reference(scene)
        self.index, self.mode = index, mode; self.label_prefix = scene.dataset.root.name + ' | '
        self.settings = scene.config; self.ep = scene.dataset.episode(index)
        self.model = scene.model(index); self.bowl_note = None
        self.model.light_castshadow[:] = 0
        self.base_light = {name: getattr(self.model, 'light_' + name).copy() for name in ('pos', 'dir', 'cutoff')}
        self.fitted_light = None
        self.data = mujoco.MjData(self.model); self.cube = self.model.joint('cube_free').qposadr[0]
        self.qpos, self.cube_pose = scene.trace(index)
        self.option = mujoco.MjvOption(); self.option.geomgroup[3:] = 0; self.option.sitegroup[:] = 0
        delays = self.settings.get('camera_delay_frames') or {}
        self.delays = {view: delays.get(view, 0) for view in ('top', 'wrist')}
        self.samples = {view: camera_sample_indices(self.ep.frame_count, self.delays[view]) for view in ('top', 'wrist')}
        self.output = scene.episode_folder(index); self.work = self.output / f'appearance-{mode}-i{REFERENCE_VERSION}'
        self.valid_plate = cv2.imread(str(reference / 'valid-mask.png'), cv2.IMREAD_GRAYSCALE) > 0
        plate = reference / 'background.png'
        asset = {'id': 'calibrated-real', 'name': 'Imported workcell (automatic)', 'mode': 'calibrated', 'path': str(plate),
                 'sha256': hashlib.sha256(plate.read_bytes()).hexdigest()}
        self.appearance = RealAppearance(self.work, self.model, MODES[mode] | {'background_id': 'calibrated-real', '_background_asset': asset}, reference=reference)
        self.appearance.min_gain = .35
        self.reference_profile = json.loads((reference / 'profile.json').read_text())
        lens = self.settings['wrist_camera'].get('distortion'); self.lens_maps = None
        if lens:
            cid = self.model.camera('wrist_cam').id; fovy = self.settings['wrist_camera']['fovy']
            render_fovy = ic.covering_fovy(fovy, *lens, 640, 480); self.model.cam_fovy[cid] = render_fovy
            self.lens_maps = ic.distortion_maps(fovy, *lens, 640, 480, render_fovy)
        self.response = self.reference_profile.get('camera_response', {})
        yy, xx = np.mgrid[:480, :640]; self.radius = np.hypot(xx - 320, yy - 240) / np.hypot(320, 240)

    def video_file(self):
        return video_name(self.mode)

    def real_mask(self, view, kind, hsv):
        selected = class_filter(view, kind)
        return selected(hsv) if selected else super().real_mask(view, kind, hsv)

    def match_colors(self, renderer):
        """Per-episode colours; a class this episode never shows clearly in Wrist keeps the
        dataset-wide Wrist measurement instead of a Top colour (different exposure)."""
        saved, measured = super().match_colors(renderer)
        profile = self.appearance.profile['appearance']
        for kind, stats in self.reference_profile['appearance'].get('wrist', {}).items():
            if kind in measured['wrist'] or kind in ('work_mat', 'table', 'floor') or not stats.get('sim'): continue
            profile['wrist'][kind] = saved['wrist'][kind] = dict(stats, wrist_measured=True)
        return saved, measured

    def sim_rgb(self, renderer, view, frame):
        """Rendered view with the real camera's fitted vignetting and blur."""
        result = super().sim_rgb(renderer, view, frame).astype(np.float32)
        response = self.response.get(view)
        if not response:
            result = result.astype(np.uint8)
            return ic.distort_render(result, self.lens_maps) if view == 'wrist' and self.lens_maps is not None else result
        a, b = response['vignetting']; gain = np.exp(a * self.radius ** 2 + b * self.radius ** 4)
        gain /= gain[self.radius < .35].mean()  # colour matching already set the central level
        result *= gain[..., None]
        if response['blur_sigma_px'] > 0: result = cv2.GaussianBlur(result, (0, 0), response['blur_sigma_px'])
        if view == 'wrist' and self.lens_maps is not None: result = ic.distort_render(result, self.lens_maps)
        return np.clip(result, 0, 255).astype(np.uint8)


def video_name(mode):
    return f'comparison-{mode}-v{APPEARANCE_VERSION}-i{REFERENCE_VERSION}.mp4'


if __name__ == '__main__':
    command, identifier, index, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    renderer = ImportedEpisodeAppearance(identifier, index, mode)
    if command == 'video': renderer.video()
    elif command == 'frame': renderer.frame(int(sys.argv[5]), Path(sys.argv[6]))
    else: raise SystemExit('usage: import_appearance.py video|frame IMPORT_ID EPISODE MODE [FRAME TARGET]')
