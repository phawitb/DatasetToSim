"""Render archived physics states with the same appearance controls as generation."""
import json
import subprocess
from pathlib import Path
import cv2
import mujoco
import numpy as np
from studio_generation_visuals import configure_visuals
from studio_generation_render import sensor_transform
from studio_replay_dataset_source import sha, write_json


def render_episode(folder, seed, options, *, preview=False, progress=lambda *_:None):
    folder = Path(folder)
    settings = json.loads((folder / 'generation-config.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(folder / 'scene.xml'))
    bowl_note = None
    if options.get('appearance_profile') == 'realistic':
        from strict_action_scene import ROOT, current_bowl_visual, with_bowl_visual
        bowl = current_bowl_visual(json.loads((ROOT / 'calibration/native-action-current.json').read_text()))
        if bowl is not None:
            # Display only: the static bowl is drawn with the current measured
            # geometry; archived physics states are rendered unchanged.
            model = with_bowl_visual(folder / 'scene.xml', *bowl)
            bowl_note = 'Bowl drawn with current measured geometry; archived physics used the original bowl.'
    camera = settings['wrist_camera']; cid = model.camera('wrist_cam').id
    model.cam_pos[cid] = camera['position']; model.cam_quat[cid] = camera['quaternion']; model.cam_fovy[cid] = camera['fovy']
    draws = configure_visuals(model, seed, options)
    if bowl_note: draws['bowl_geometry'] = bowl_note
    draws.update(source_type='real_replay', observation_alignment='Pre-action integrated state k; no state-following, action repair or image delay.')
    appearance = None
    if options['real_background'] or options['real_image_processing']:
        from studio_real_appearance import RealAppearance
        appearance = RealAppearance(folder, model, options, lighting=draws['lighting'])
        draws['real_appearance'] = appearance.provenance
    mujoco.mj_saveLastXML(str(folder / 'render-scene.xml'), model)
    draws['render_scene_sha256'] = sha(folder / 'render-scene.xml')
    with np.load(folder / 'trace.npz', allow_pickle=False) as source:
        trace = {key:source[key] for key in ('qpos','cube_pose')}
    with np.load(folder / 'plan.npz', allow_pickle=False) as plan:
        n = len(plan['actions'])
    if len(trace['qpos']) != n + 1 or len(trace['cube_pose']) != n + 1:
        raise ValueError('Rendering needs N+1 integrated states for N real commands')
    data = mujoco.MjData(model); cube = model.joint('cube_free').qposadr[0]
    option = mujoco.MjvOption(); option.geomgroup[3:] = 0; option.sitegroup[:] = 0
    writers = {} if preview else {view:cv2.VideoWriter(str(folder / f'{view}-raw.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480)) for view in ('top','wrist')}
    randoms = {view:np.random.default_rng(np.random.SeedSequence([seed,319,i])) for i,view in enumerate(('top','wrist'))}
    try:
        if not all(writer.isOpened() for writer in writers.values()):
            raise RuntimeError('Video encoder unavailable')
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            if appearance and appearance.processing:
                poses = []
                for i in np.unique(np.linspace(0,n-1, min(n,12),dtype=int)):
                    pose = data.qpos.copy(); pose[:6] = trace['qpos'][i]; pose[cube:cube+7] = trace['cube_pose'][i]
                    poses.append(pose)
                # This workcell contains one red cube, so absent green task cube
                # statistics must not force geometry changes or invented motion.
                appearance._calibrate_poses(renderer,data,poses,option,allow_unseen=True)
            for i in range(1 if preview else n):
                data.qpos[:6] = trace['qpos'][i]; data.qpos[cube:cube+7] = trace['cube_pose'][i]
                mujoco.mj_forward(model,data)
                for view in ('top','wrist'):
                    renderer.update_scene(data,camera=view+'_cam',scene_option=option)
                    rgb = renderer.render().copy()
                    if appearance:
                        rgb = appearance.render_frame(renderer,data,view,option,rgb)
                    rgb = sensor_transform(rgb,draws['sensor'],randoms[view])
                    bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                    if i == 0 and not cv2.imwrite(str(folder / f'{view}.jpg'),bgr):
                        raise RuntimeError('Could not save rendered image')
                    if view in writers:
                        writers[view].write(bgr)
                if i % 60 == 0:
                    progress(i,n)
    finally:
        for writer in writers.values():
            writer.release()
    for view in writers:
        raw = folder / f'{view}-raw.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(raw),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(folder / f'{view}.mp4')],check=True)
        raw.unlink()
    draws['frames'] = 1 if preview else n
    draws['preview'] = preview
    write_json(folder / 'visual-provenance.json',draws)
    return draws
