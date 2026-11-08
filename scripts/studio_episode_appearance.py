"""Render archived benchmark states with a selectable Simulation appearance.

Display only: archived qpos/cube poses are rendered again, physics is never
rerun and no state, action or camera delay changes. Modes:
  background  calibrated real-camera background plate on the workcell planes
  realistic   background plate plus per-surface real-camera color/exposure
              matching and achromatic rendered shading
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
from studio_real_appearance import RealAppearance, robust_stats, segmentation_ids

# Cast shadows and rendered light falloff are removed: the real plate already
# carries the real exposure, and MuJoCo shadows do not match the real lights.
MODES = {'background': {'real_background': True, 'real_image_processing': False, 'rendered_shading': False},
         'realistic': {'real_background': True, 'real_image_processing': True, 'rendered_shading': False}}
# Per-episode color targets measured on this episode's own real frames.
EPISODE_KINDS = ('bowl', 'robot')
LABELS = {'background': 'MuJoCo | real background', 'realistic': 'MuJoCo | realistic appearance'}
RUN = ROOT / 'output/shared-calibration-v10'
# Bump when rendering changes so cached videos and stills are not reused.
APPEARANCE_VERSION = 2


def video_name(mode):
    return f'comparison-{mode}-v{APPEARANCE_VERSION}.mp4'


def output_folder(index):
    return ROOT / 'output/studio' / f'replay-{index:02d}'


class EpisodeAppearance:
    def __init__(self, index, mode):
        if mode not in MODES: raise ValueError('Unknown appearance mode')
        self.index, self.mode = index, mode
        folder = RUN / f'episode-{index:02d}'
        if not (folder / 'report.json').is_file(): raise ValueError('Episode not found')
        self.settings = json.loads((RUN / 'settings.json').read_text())
        report = json.loads((folder / 'report.json').read_text())
        self.ep = DatasetIndex(DATA).episode(index)
        self.model = load_saved_model(folder, self.settings, report['model_archive_sha256'])
        camera = self.settings['wrist_camera']; cid = self.model.camera('wrist_cam').id
        self.model.cam_pos[cid] = camera['position']; self.model.cam_quat[cid] = camera['quaternion']; self.model.cam_fovy[cid] = camera['fovy']
        self.model.light_castshadow[:] = 0
        self.data = mujoco.MjData(self.model); self.cube = self.model.joint('cube_free').qposadr[0]
        with np.load(folder / 'strict_trace.npz') as trace:
            self.qpos = trace['qpos'].copy(); self.cube_pose = trace['cube_pose'].copy()
        self.option = mujoco.MjvOption(); self.option.geomgroup[3:] = 0; self.option.sitegroup[:] = 0
        delays = self.settings.get('camera_delay_frames') or {}
        self.delays = {view: delays.get(view, 0) for view in ('top', 'wrist')}
        self.samples = {view: camera_sample_indices(self.ep.frame_count, self.delays[view]) for view in ('top', 'wrist')}
        self.work = output_folder(index) / f'appearance-{mode}'
        self.appearance = RealAppearance(self.work, self.model, MODES[mode] | {'background_id': 'calibrated-real'})

    def pose(self, state):
        self.data.qpos[:6] = self.qpos[state]; self.data.qpos[self.cube:self.cube + 7] = self.cube_pose[state]
        mujoco.mj_forward(self.model, self.data)

    def calibrate(self, renderer):
        if not self.appearance.processing: return
        poses = []
        for i in np.unique(np.linspace(0, self.ep.frame_count - 1, min(self.ep.frame_count, 12), dtype=int)):
            self.pose(i); poses.append(self.data.qpos.copy())
        # The benchmark workcell has only the red cube: absent green statistics
        # are estimated for color matching and never alter geometry.
        self.appearance._calibrate_poses(renderer, self.data, poses, self.option, allow_unseen=True)
        self.match_episode(renderer)

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
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def match_episode(self, renderer):
        """Measure real and rendered colors of each object on paired frames.

        Masks come from the rendered geometry at the camera-delayed state and are
        eroded against calibration error; real pixels are additionally filtered
        by hue so the arm, cables or cube passing in front are not counted.
        """
        classes = self.appearance.classes; profile = self.appearance.profile['appearance']
        frames = np.unique(np.linspace(0, self.ep.frame_count - 1, 8, dtype=int))
        # Top only: the fixed camera aligns with rendered geometry. Wrist masks
        # drift with arm calibration error, so Wrist reuses the Top targets.
        view = 'top'; samples = {kind: ([], []) for kind in EPISODE_KINDS}
        for frame in frames:
            real = self.real_frame(view, int(frame)); hsv = cv2.cvtColor(real, cv2.COLOR_RGB2HSV)
            self.pose(int(self.samples[view][frame]))
            renderer.update_scene(self.data, camera=view + '_cam', scene_option=self.option)
            sim = renderer.render().copy(); ids = segmentation_ids(renderer)
            hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            for kind in EPISODE_KINDS:
                mask = np.isin(ids, [i for i, c in classes.items() if c == kind]).astype('uint8')
                mask = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
                if kind == 'bowl': real_mask = mask & (hue > 35) & (hue < 110) & (val > 60)
                elif kind == 'red': real_mask = mask & ((hue < 12) | (hue > 165)) & (sat > 90)
                else: real_mask = mask & (hue < 30) & (sat > 90) & (val > 70)
                if real_mask.sum() > 40:
                    samples[kind][0].append(real[real_mask]); samples[kind][1].append(sim[mask])
        measured = []
        for kind in EPISODE_KINDS:
            real, sim = samples[kind]
            if sum(len(x) for x in real) < 400: continue
            profile['top'][kind] = {'real': robust_stats(np.concatenate(real)), 'sim': robust_stats(np.concatenate(sim)), 'measured': True}
            measured.append(kind)
        # The mat-based Top-to-Wrist exposure shift darkens bright objects far
        # more than the real Wrist camera does. The bowl keeps its measured Top
        # target; rendered cube colors already match the real Wrist camera.
        if 'bowl' in profile['top']:
            profile['wrist']['bowl'] = {'real': profile['top']['bowl']['real'], 'sim': self.render_stats(renderer, 'wrist', 'bowl', frames) or profile['top']['bowl']['sim'], 'wrist_measured': True}
        for kind in ('red', 'green'): profile['wrist'].pop(kind, None)
        self.appearance.provenance['episode_color_matching'] = {'frames': frames.tolist(), 'measured_top': measured,
            'method': 'Per-episode Lab statistics from eroded rendered masks and hue-filtered real pixels on paired camera-delayed frames.'}

    def sim_rgb(self, renderer, view, frame):
        self.pose(int(self.samples[view][frame]))
        renderer.update_scene(self.data, camera=view + '_cam', scene_option=self.option)
        return self.appearance.render_frame(renderer, self.data, view, self.option, renderer.render().copy())

    def canvas(self, renderer, real, frame):
        rows = []
        for view in ('top', 'wrist'):
            canvas = np.zeros((512, 1280, 3), np.uint8); canvas[32:, :640] = real[view]; canvas[32:, 640:] = self.sim_rgb(renderer, view, frame)
            note = f' | camera delay {self.delays[view]}f' if self.delays[view] else ''
            for text, pos in [(f'REAL | EP {self.index:02d} | {view} | frame {frame:03d} | {frame / 30:.2f}s', (10, 23)), (f'{LABELS[self.mode]} | {view}{note}', (650, 23))]:
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
            real[view] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with mujoco.Renderer(self.model, height=480, width=640) as renderer:
            self.calibrate(renderer)
            image = self.canvas(renderer, real, frame)
        if not cv2.imwrite(str(target), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)): raise RuntimeError('Could not save frame')

    def video(self):
        folder = output_folder(self.index); folder.mkdir(parents=True, exist_ok=True)
        target = folder / video_name(self.mode); raw = target.with_name(target.stem + '-raw.mp4'); partial = target.with_name(target.stem + '-partial.mp4')
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
                        real[view] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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
