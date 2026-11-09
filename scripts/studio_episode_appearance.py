"""Render archived benchmark states with a selectable Simulation appearance.

Display only: archived qpos/cube poses are rendered again, physics is never
rerun and no state, action or camera delay changes. Modes:
  background  calibrated real-camera background plate on the workcell planes
  realistic   background plate, per-camera color matching of every object
              and faint soft shadows fitted to the real Top frames
Raw MuJoCo output stays in studio_video.py / studio_frame.py.
"""
from pathlib import Path
import os, sys, json, subprocess
os.environ.setdefault('MUJOCO_GL', 'cgl' if sys.platform == 'darwin' else 'egl')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
import cv2, mujoco, numpy as np
from dataset_to_sim.archive import load_saved_model
from dataset_to_sim.dataset import DatasetIndex
from dataset_to_sim.observations import camera_sample_indices
from episode0 import DATA
from strict_action_scene import current_bowl_visual, with_bowl_visual
from studio_real_appearance import REFERENCE, RealAppearance, robust_stats, segmentation_ids

# Rendered light falloff on the background is removed: the real plate already
# carries the real exposure. Realistic adds faint, blurred shadows whose light
# direction, softness and strength are fitted to the episode's real Top frames.
MODES = {'background': {'real_background': True, 'real_image_processing': False, 'rendered_shading': False},
         'realistic': {'real_background': True, 'real_image_processing': True, 'rendered_shading': False}}
# Objects whose colors are measured separately in each camera, per episode.
# Real pixels inside eroded rendered masks are kept only if their hue fits.
OBJECT_KINDS = ('robot', 'dark_robot', 'red', 'bowl')
WRIST_SURFACES = ('work_mat', 'table')
ERODE = {'top': 9, 'wrist': 15}
MIN_PIXELS = {'top': 300, 'wrist': 1500}
LIGHT_CENTER = np.array([.23, 0., 0.])
MAX_SHADOW = .35  # Strength ceiling: shadows stay faint even if the fit is strong.


def real_filter(kind, hsv):
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return {'object': (sat < 90) & (val > 140),'robot': (hue <= 25) & (sat > 100) & (val > 60),
            'red': ((hue < 8) | (hue > 170)) & (sat > 120) & (val > 50),
            'bowl': (hue > 35) & (hue < 110) & (val > 50) & (sat < 150),
            'dark_robot': (val < 80) & (sat < 110),
            'work_mat': (sat < 90) & (val < 120),
            'table': (sat < 60) & (val > 100)}[kind]


def light_direction(azimuth, elevation):
    az, el = np.deg2rad([azimuth, elevation])
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


LABELS = {'background': 'MuJoCo | real background', 'realistic': 'MuJoCo | realistic appearance'}
RUN = ROOT / 'output/shared-calibration-v10'
# Bump when rendering changes so cached videos and stills are not reused.
APPEARANCE_VERSION = 4


def video_name(mode):
    return f'comparison-{mode}-v{APPEARANCE_VERSION}.mp4'


def output_folder(index):
    return ROOT / 'output/studio' / f'replay-{index:02d}'


def as_640(rgb):
    """Studio canvases are 640 x 480; lower-resolution recordings are upscaled for display."""
    return rgb if rgb.shape[:2] == (480, 640) else cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_CUBIC)


class EpisodeAppearance:
    object_kinds = OBJECT_KINDS
    filters = {}
    label_prefix = ''

    def __init__(self, index, mode):
        if mode not in MODES: raise ValueError('Unknown appearance mode')
        self.index, self.mode = index, mode
        folder = RUN / f'episode-{index:02d}'
        if not (folder / 'report.json').is_file(): raise ValueError('Episode not found')
        self.settings = json.loads((RUN / 'settings.json').read_text())
        report = json.loads((folder / 'report.json').read_text())
        self.ep = DatasetIndex(DATA).episode(index)
        self.model = load_saved_model(folder, self.settings, report['model_archive_sha256'])
        self.bowl_note = None
        bowl = current_bowl_visual(json.loads((ROOT / 'calibration/native-action-current.json').read_text()))
        if bowl is not None:
            # Display only: the archived physics states are unchanged; the static
            # bowl is drawn with the current measured geometry.
            self.model = with_bowl_visual(folder / 'scene.xml', *bowl)
            self.bowl_note = 'Bowl drawn with current measured geometry (2 mm wall, rounded bottom); archived physics used the original bowl.'
        camera = self.settings['wrist_camera']; cid = self.model.camera('wrist_cam').id
        self.model.cam_pos[cid] = camera['position']; self.model.cam_quat[cid] = camera['quaternion']; self.model.cam_fovy[cid] = camera['fovy']
        self.model.light_castshadow[:] = 0
        # Colors are rendered under the original light; a fitted light is used
        # only to compute the faint shadow map.
        self.base_light = {name: getattr(self.model, 'light_' + name).copy() for name in ('pos', 'dir', 'cutoff')}
        self.fitted_light = None
        self.data = mujoco.MjData(self.model); self.cube = self.model.joint('cube_free').qposadr[0]
        with np.load(folder / 'strict_trace.npz') as trace:
            self.qpos = trace['qpos'].copy(); self.cube_pose = trace['cube_pose'].copy()
        self.option = mujoco.MjvOption(); self.option.geomgroup[3:] = 0; self.option.sitegroup[:] = 0
        delays = self.settings.get('camera_delay_frames') or {}
        self.delays = {view: delays.get(view, 0) for view in ('top', 'wrist')}
        self.samples = {view: camera_sample_indices(self.ep.frame_count, self.delays[view]) for view in ('top', 'wrist')}
        self.output = output_folder(index); self.work = self.output / f'appearance-{mode}'
        self.valid_plate = cv2.imread(str(REFERENCE / 'valid-mask.png'), cv2.IMREAD_GRAYSCALE) > 0
        self.appearance = RealAppearance(self.work, self.model, MODES[mode] | {'background_id': 'calibrated-real'})
        # Real objects are lit diffusely; allow flattening rendered shading
        # contrast further than generation does.
        self.appearance.min_gain = .35

    def video_file(self):
        return video_name(self.mode)

    def real_mask(self, view, kind, hsv):
        """Real pixels that plausibly belong to a semantic class."""
        return real_filter(kind, hsv)

    def pose(self, state):
        self.data.qpos[:6] = self.qpos[state]; self.data.qpos[self.cube:self.cube + 7] = self.cube_pose[state]
        mujoco.mj_forward(self.model, self.data)

    def calibrate(self, renderer):
        self.shadow = None
        if not self.appearance.processing: return
        cache = self.work / f'episode-calibration-v{APPEARANCE_VERSION}.json'
        if cache.is_file():
            saved = json.loads(cache.read_text())
            self.fitted_light = (saved['shadow']['azimuth_degrees'], saved['shadow']['elevation_degrees'])
            self.appearance.profile['appearance'] = saved['profile']
        else:
            saved = {'shadow': self.fit_shadow(renderer)}
            poses = []
            for i in np.unique(np.linspace(0, self.ep.frame_count - 1, min(self.ep.frame_count, 12), dtype=int)):
                self.pose(i); poses.append(self.data.qpos.copy())
            # The benchmark workcell has only the red cube: absent green statistics
            # are estimated for color matching and never alter geometry.
            self.appearance._calibrate_poses(renderer, self.data, poses, self.option, allow_unseen=True)
            _, saved['measured'] = self.match_colors(renderer)
            saved['profile'] = self.appearance.profile['appearance']  # complete, so a cached load needs no rendering
            cache.write_text(json.dumps(saved, indent=1))
        self.shadow = saved['shadow']
        if self.bowl_note: self.appearance.provenance['bowl_geometry'] = self.bowl_note
        self.appearance.provenance['episode_calibration'] = {'shadow': saved['shadow'], 'measured': saved.get('measured'),
            'method': 'Per-episode, per-camera Lab statistics from eroded rendered masks and hue-filtered real pixels on paired camera-delayed frames. '
                      'Shadow light direction, blur and strength fitted to real-frame/background-plate luminance on Top background pixels; strength capped at %.2f.' % MAX_SHADOW}

    def render_stats(self, renderer, view, kind, frames):
        pixels = []
        for frame in frames:
            self.pose(int(self.samples[view][frame]))
            renderer.update_scene(self.data, camera=view + '_cam', scene_option=self.option)
            sim = renderer.render().copy(); ids = segmentation_ids(renderer)
            mask = cv2.erode(np.isin(ids, [i for i, c in self.appearance.classes.items() if c == kind]).astype('uint8'), np.ones((5, 5), np.uint8)) > 0
            if mask.sum() > 40: pixels.append(sim[mask])
        return robust_stats(np.concatenate(pixels)) if pixels else None

    def real_frame(self, view, frame):
        source = self.ep.video_slices[view]; cap = cv2.VideoCapture(str(source.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(source.start_time * 30) + frame); ok, bgr = cap.read(); cap.release()
        if not ok: raise ValueError('Could not decode recorded frame')
        return as_640(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def set_light(self, azimuth, elevation):
        direction = light_direction(azimuth, elevation)
        for i in range(self.model.nlight):
            self.model.light_pos[i] = LIGHT_CENTER + 2. * direction; self.model.light_dir[i] = -direction
        self.model.light_cutoff[:] = np.maximum(self.base_light['cutoff'], 70)

    def restore_light(self):
        for name, value in self.base_light.items(): getattr(self.model, 'light_' + name)[:] = value

    def shadow_map(self, renderer, view, light=None):
        """Log luminance ratio of shadowed/unshadowed renders under the fitted light."""
        self.set_light(*(light or self.fitted_light)); mujoco.mj_forward(self.model, self.data)  # light_xpos
        try:
            renderer.update_scene(self.data, camera=view + '_cam', scene_option=self.option); plain = gray(renderer.render())
            self.model.light_castshadow[:] = 1
            renderer.update_scene(self.data, camera=view + '_cam', scene_option=self.option); shaded = gray(renderer.render())
        finally:
            self.model.light_castshadow[:] = 0; self.restore_light(); mujoco.mj_forward(self.model, self.data)
        return np.log((shaded + 4) / (plain + 4))

    def fit_shadow(self, renderer):
        plate = gray(self.appearance.plate)
        valid_plate = self.valid_plate
        surfaces = [i for i, c in self.appearance.classes.items() if c in ('work_mat', 'table', 'floor')]
        frames = np.unique(np.linspace(0, self.ep.frame_count - 1, 6, dtype=int)); observed = []
        for frame in frames:
            real = gray(self.real_frame('top', int(frame))); self.pose(int(self.samples['top'][frame]))
            renderer.update_scene(self.data, camera='top_cam', scene_option=self.option); ids = segmentation_ids(renderer)
            objects = cv2.dilate((~np.isin(ids, surfaces)).astype('uint8'), np.ones((21, 21), np.uint8)) > 0
            log_ratio = np.log((real + 4) / (plate + 4)); mask = np.isin(ids, surfaces) & valid_plate & ~objects
            mask &= np.abs(log_ratio - np.median(log_ratio[mask])) < .35  # cables, misaligned arm
            observed.append((int(frame), mask[::3, ::3], (log_ratio - np.median(log_ratio[mask]))[::3, ::3]))
        def shadow_maps(azimuth, elevation):
            maps = []
            for frame, _, _ in observed:
                self.pose(int(self.samples['top'][frame])); maps.append(self.shadow_map(renderer, 'top', (azimuth, elevation)))
            return maps
        def score(maps, sigma=0.):
            xs, ys = [], []
            for (_, mask, target), image in zip(observed, maps):
                if sigma: image = cv2.GaussianBlur(image, (0, 0), sigma)
                xs.append(image[::3, ::3][mask]); ys.append(target[mask])
            x, y = np.concatenate(xs), np.concatenate(ys)
            if x.std() < 1e-4: return -1., 0.
            return float(np.corrcoef(x, y)[0, 1]), float(np.cov(x, y)[0, 1] / x.var())
        best = (-2., 0., 90.)
        for elevation in (40., 55., 70., 85.):
            for azimuth in range(-180, 180, 30):
                corr, _ = score(shadow_maps(azimuth, elevation), 4.)
                if corr > best[0]: best = (corr, float(azimuth), elevation)
        _, azimuth, elevation = best; maps = shadow_maps(azimuth, elevation)
        fits = {sigma: score(maps, sigma) for sigma in (2., 4., 7., 11., 16., 24.)}
        sigma = max(fits, key=lambda key: fits[key][0]); corr, slope = fits[sigma]
        # A weak fit still gets a faint shadow from the best direction.
        strength = float(np.clip(slope if corr > .05 else .1, .08, MAX_SHADOW))
        self.fitted_light = (azimuth, elevation)
        return {'azimuth_degrees': azimuth, 'elevation_degrees': elevation, 'blur_sigma_px': sigma, 'strength': strength, 'correlation': corr}

    def match_colors(self, renderer):
        """Measure real and rendered colors separately for Top and Wrist."""
        classes = self.appearance.classes; profile = self.appearance.profile['appearance']
        frames = np.unique(np.linspace(0, self.ep.frame_count - 1, 10, dtype=int))
        saved = {'top': {}, 'wrist': {}}; measured = {'top': [], 'wrist': []}
        for view in ('top', 'wrist'):
            kinds = self.object_kinds + (WRIST_SURFACES if view == 'wrist' else ())
            samples = {kind: [] for kind in kinds}
            for frame in frames:
                real = self.real_frame(view, int(frame)); hsv = cv2.cvtColor(real, cv2.COLOR_RGB2HSV)
                self.pose(int(self.samples[view][frame]))
                renderer.update_scene(self.data, camera=view + '_cam', scene_option=self.option)
                sim = renderer.render().copy(); ids = segmentation_ids(renderer)
                for kind in kinds:
                    mask = np.isin(ids, [i for i, c in classes.items() if c == kind]).astype('uint8')
                    mask = cv2.erode(mask, np.ones((ERODE[view], ERODE[view]), np.uint8)) > 0
                    real_mask = mask & self.real_mask(view, kind, hsv)
                    if real_mask.sum() > 40: samples[kind].append((real[real_mask], sim[mask], float(gray(real[real_mask][None])[0].mean())))
            for kind in kinds:
                # Real frames where the arm shades an object are excluded: only
                # the brighter frames set the unshadowed object color.
                rows = samples[kind]
                if len(rows) >= 3:
                    # Wrist shading by the arm is strong: keep its brightest third.
                    cut = np.percentile([row[2] for row in rows], 67 if view == 'wrist' else 50); rows = [row for row in rows if row[2] >= cut]
                real = [row[0] for row in rows]; sim = [row[1] for row in rows]
                if sum(len(x) for x in real) < MIN_PIXELS[view]: continue
                stats = {'real': robust_stats(np.concatenate(real)), 'measured': True}
                if kind in WRIST_SURFACES:
                    # Wrist background pixels come from the Top plate: keep its source stats.
                    stats['sim'] = profile[view].get(kind, {}).get('sim') or profile['top'][kind]['real']
                else: stats['sim'] = robust_stats(np.concatenate(sim))
                if view == 'wrist': stats['wrist_measured'] = True
                profile[view][kind] = saved[view][kind] = stats; measured[view].append(kind)
        # An object never measured in Wrist keeps its Top color without the
        # mat-based exposure shift, which darkens bright objects too much.
        for kind in self.object_kinds + ('green',):
            if kind in measured['wrist']: continue
            if kind in measured['top']:
                profile['wrist'][kind] = saved['wrist'][kind] = dict(profile['top'][kind], sim=self.render_stats(renderer, 'wrist', kind, frames) or profile['top'][kind]['sim'], wrist_measured=True)
            elif kind in ('red', 'green', 'object'):
                profile['wrist'].pop(kind, None); saved['wrist'][kind] = None
        return saved, measured

    def sim_rgb(self, renderer, view, frame):
        self.pose(int(self.samples[view][frame]))
        renderer.update_scene(self.data, camera=view + '_cam', scene_option=self.option)
        result = self.appearance.render_frame(renderer, self.data, view, self.option, renderer.render().copy())
        if not self.shadow: return result
        # Soft, faint shadow: blurred shadowed/unshadowed luminance ratio.
        ratio = cv2.GaussianBlur(np.exp(np.clip(self.shadow_map(renderer, view), np.log(.2), 0)), (0, 0), self.shadow['blur_sigma_px'])
        ratio = 1 - self.shadow['strength'] * (1 - ratio)
        return np.clip(result.astype(np.float32) * ratio[:, :, None], 0, 255).astype(np.uint8)

    def canvas(self, renderer, real, frame):
        rows = []
        for view in ('top', 'wrist'):
            canvas = np.zeros((512, 1280, 3), np.uint8); canvas[32:, :640] = real[view]; canvas[32:, 640:] = self.sim_rgb(renderer, view, frame)
            note = f' | camera delay {self.delays[view]}f' if self.delays[view] else ''
            for text, pos in [(f'REAL | {self.label_prefix}EP {self.index:02d} | {view} | frame {frame:03d} | {frame / 30:.2f}s', (10, 23)), (f'{LABELS[self.mode]} | {view}{note}', (650, 23))]:
                cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, .49, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(canvas)
        return np.vstack(rows)

    def frame(self, frame, target):
        if not 0 <= frame < self.ep.frame_count: raise ValueError('Frame out of bounds')
        real = {}
        for view in ('top', 'wrist'):
            source = self.ep.video_slices[view]; cap = cv2.VideoCapture(str(source.path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(source.start_time * 30) + frame); ok, bgr = cap.read(); cap.release()
            if not ok: raise ValueError('Could not decode recorded frame')
            real[view] = as_640(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with mujoco.Renderer(self.model, height=480, width=640) as renderer:
            self.calibrate(renderer)
            image = self.canvas(renderer, real, frame)
        if not cv2.imwrite(str(target), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)): raise RuntimeError('Could not save frame')

    def video(self):
        folder = self.output; folder.mkdir(parents=True, exist_ok=True)
        target = folder / self.video_file(); raw = target.with_name(target.stem + '-raw.mp4'); partial = target.with_name(target.stem + '-partial.mp4')
        captures = {view: cv2.VideoCapture(str(self.ep.video_slices[view].path)) for view in ('top', 'wrist')}
        for view, cap in captures.items(): cap.set(cv2.CAP_PROP_POS_FRAMES, round(self.ep.video_slices[view].start_time * 30))
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 1024))
        try:
            if not writer.isOpened(): raise RuntimeError('Encoder unavailable')
            with mujoco.Renderer(self.model, height=480, width=640) as renderer:
                self.calibrate(renderer)
                for f in range(self.ep.frame_count):
                    real = {}
                    for view, cap in captures.items():
                        ok, bgr = cap.read()
                        if not ok: raise RuntimeError('Incomplete source video')
                        real[view] = as_640(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    writer.write(cv2.cvtColor(self.canvas(renderer, real, f), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
            for cap in captures.values(): cap.release()
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(raw), '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)], check=True)
        partial.replace(target); raw.unlink()
        (folder / f'manifest-{self.mode}-v{APPEARANCE_VERSION}.json').write_text(json.dumps({'episode': self.index, 'frames': self.ep.frame_count, 'fps': 30, 'mode': self.mode, 'version': APPEARANCE_VERSION,
            'camera_delay_frames': self.delays, 'physics_rerun': False, 'appearance': self.appearance.provenance}, indent=2))
        return target


if __name__ == '__main__':
    command, index, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    renderer = EpisodeAppearance(index, mode)
    if command == 'video': renderer.video()
    elif command == 'frame': renderer.frame(int(sys.argv[4]), Path(sys.argv[5]))
    else: raise SystemExit('usage: studio_episode_appearance.py video INDEX MODE | frame INDEX MODE FRAME TARGET')
