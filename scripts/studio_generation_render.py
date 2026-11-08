"""Render measured simulated states with bounded, replayable visual variation."""
import hashlib
import json
import subprocess
from pathlib import Path
import cv2
import mujoco
import numpy as np
from studio_generation_scene import load_generation_scene
from studio_generation_visuals import configure_visuals, sensor_parameters


def sensor_transform(rgb, parameters, rng):
    if parameters == dict(gain=1., offset=0., gamma=1., noise=0., blur=0.):
        return rgb.copy()
    frame = np.clip(rgb.astype(np.float32) * parameters['gain'] + parameters['offset'], 0, 255)
    frame = 255 * (frame / 255) ** parameters['gamma']
    if parameters['blur']:
        frame = cv2.GaussianBlur(frame, (3, 3), parameters['blur'])
    if parameters['noise']:
        # Achromatic sensor noise cannot swap red/green language labels.
        frame += rng.normal(0, parameters['noise'], (*rgb.shape[:2], 1))
    return np.clip(frame, 0, 255).astype(np.uint8)


def render_episode(folder, seed, options, progress=lambda *_: None):
    folder = Path(folder)
    _, _, _, _, model = load_generation_scene(folder)
    draws = configure_visuals(model, seed, options)
    appearance = None
    if options.get('real_background') or options.get('real_image_processing'):
        from studio_real_appearance import RealAppearance
        appearance = RealAppearance(folder, model, options, lighting=draws['lighting'])
        draws['real_appearance'] = appearance.provenance
    (folder / 'visual-provenance.json').write_text(json.dumps(draws, indent=2))
    mujoco.mj_saveLastXML(str(folder / 'render-scene.xml'), model)
    draws['render_scene_sha256'] = hashlib.sha256((folder / 'render-scene.xml').read_bytes()).hexdigest()
    trace = np.load(folder / 'trace.npz')
    plan = np.load(folder / 'plan.npz')
    n = len(plan['actions'])
    if len(trace['qpos']) != n + 1 or len(trace['cube_pose']) != n + 1:
        raise ValueError('Rendering requires N+1 integrated states for N actions')
    data = mujoco.MjData(model)
    cube = model.joint('cube_free').qposadr[0]
    other = model.joint('target_cube_free').qposadr[0]
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    writers = {view: cv2.VideoWriter(str(folder / f'{view}-raw.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480)) for view in ('top', 'wrist')}
    randoms = {view: np.random.default_rng(np.random.SeedSequence([seed, 319, index])) for index, view in enumerate(writers)}
    try:
        if not all(writer.isOpened() for writer in writers.values()):
            raise RuntimeError('Video encoder unavailable')
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            if appearance and appearance.processing:appearance.calibrate_objects(renderer,data,trace,cube,other,option)
            for i in range(n):
                # Render observed, pre-command states; never substitute target actions.
                data.qpos[:6] = trace['qpos'][i]
                data.qpos[cube:cube+7] = trace['cube_pose'][i]
                data.qpos[other:other+7] = trace['target_pose'][i]
                mujoco.mj_forward(model, data)
                for view, writer in writers.items():
                    renderer.update_scene(data, camera=view + '_cam', scene_option=option)
                    rgb = renderer.render().copy()
                    if appearance:rgb = appearance.render_frame(renderer, data, view, option, rgb)
                    rgb = sensor_transform(rgb, draws['sensor'], randoms[view])
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    if i == 0:
                        cv2.imwrite(str(folder / f'{view}.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if i % 60 == 0:
                    progress(i, n)
    finally:
        for writer in writers.values():
            writer.release()
        trace.close(); plan.close()
    for view in writers:
        raw = folder / f'{view}-raw.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(raw), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(folder / f'{view}.mp4')], check=True)
        raw.unlink()
    draws['frames'] = n
    draws['observation_alignment'] = 'pre-action state i, image i, command i; terminal state excluded'
    (folder / 'visual-provenance.json').write_text(json.dumps(draws, indent=2))
    return draws


def render_initial_pair(folder, seed, options):
    """Render initial-layout stills through the same pipeline, without planning."""
    from dataset_to_sim.strict_replay import NativeJointAdapter
    from studio_real_appearance import RealAppearance
    folder = Path(folder)
    capture, config, _, _, model = load_generation_scene(folder)
    draws = configure_visuals(model, seed, options)
    data = mujoco.MjData(model)
    data.qpos[:6] = NativeJointAdapter(**config['adapter']).to_model(capture['initial_native'])
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    appearance = None
    if options.get('real_background') or options.get('real_image_processing'):
        appearance = RealAppearance(folder, model, options, lighting=draws['lighting'])
        draws['real_appearance'] = appearance.provenance
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        if appearance and appearance.processing:
            appearance.calibrate_frame(renderer, data, option)
        for index, view in enumerate(('top', 'wrist')):
            renderer.update_scene(data, camera=view + '_cam', scene_option=option)
            rgb = renderer.render().copy()
            if appearance:
                rgb = appearance.render_frame(renderer, data, view, option, rgb)
            rng = np.random.default_rng(np.random.SeedSequence([seed, 319, index]))
            rgb = sensor_transform(rgb, draws['sensor'], rng)
            if not cv2.imwrite(str(folder / f'{view}.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError('Could not save preview image')
    draws['preview'] = 'Approximate initial appearance/layout only; physical feasibility is not evaluated. Cube color matching uses visible initial pixels.'
    (folder / 'visual-provenance.json').write_text(json.dumps(draws, indent=2))
    return draws
