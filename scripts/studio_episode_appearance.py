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
from studio_real_appearance import RealAppearance

MODES = {'background': {'real_background': True, 'real_image_processing': False},
         'realistic': {'real_background': True, 'real_image_processing': True}}
LABELS = {'background': 'MuJoCo | real background', 'realistic': 'MuJoCo | realistic appearance'}
RUN = ROOT / 'output/shared-calibration-v10'


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
        raw = folder / f'comparison-{self.mode}-raw.mp4'; partial = folder / f'comparison-{self.mode}-partial.mp4'; target = folder / f'comparison-{self.mode}.mp4'
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
        (folder / f'manifest-{self.mode}.json').write_text(json.dumps({'episode': self.index, 'frames': self.ep.frame_count, 'fps': 30, 'mode': self.mode,
            'camera_delay_frames': self.delays, 'physics_rerun': False, 'appearance': self.appearance.provenance}, indent=2))
        return target


if __name__ == '__main__':
    command, index, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    renderer = EpisodeAppearance(index, mode)
    if command == 'video': renderer.video()
    elif command == 'frame': renderer.frame(int(sys.argv[4]), Path(sys.argv[5]))
    else: raise SystemExit('usage: studio_episode_appearance.py video INDEX MODE | frame INDEX MODE FRAME TARGET')
