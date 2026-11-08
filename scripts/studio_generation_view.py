"""Causal generated-episode inspection and disposable, single-clock previews."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import cv2
import numpy as np
from dataset_to_sim.joints import FEATURE_JOINT_NAMES
from dataset_to_sim.strict_replay import array_hash


def episode_data(root, index):
    root=Path(root)
    episodes=json.loads((root/'job.json').read_text())['episodes']
    if index<0 or index>=len(episodes):raise ValueError('Episode not found')
    if episodes[index].get('status')!='accepted':raise ValueError('Choose an accepted episode to visualize')
    folder=root/'episodes'/f'{index:05d}'
    metadata=json.loads((folder/'episode.json').read_text())
    with np.load(folder/'plan.npz',allow_pickle=False) as plan:
        actions=plan['actions'];timestamps=plan['timestamps'];n=len(actions)
    with np.load(folder/'trace.npz',allow_pickle=False) as trace:states=trace['native']
    if actions.shape!=(n,6) or not n or states.shape!=(n+1,6):raise ValueError('Expected N actions and N+1 integrated native states')
    if not np.isfinite(actions).all() or not np.isfinite(states).all():raise ValueError('Non-finite state/action values')
    if timestamps.shape!=(n,) or not np.allclose(timestamps,np.arange(n)/30,atol=1e-5,rtol=0):raise ValueError('Episode must have synchronized 30 Hz timestamps')
    if array_hash(actions)!=metadata['action_hash']:raise ValueError('Action provenance hash mismatch')
    return dict(frames=n,fps=30,duration=n/30,joints=list(FEATURE_JOINT_NAMES),units=['degrees']*5+['0–100'],
                timestamps=timestamps.tolist(),state=states[:-1].tolist(),action=actions.tolist(),
                instruction=metadata['instruction'],task_id=episodes[index]['task_id'],
                alignment='Observed state before command at each frame; terminal state excluded.')


def video_shape(path):
    capture=cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():raise ValueError('Preview camera video is unavailable')
        return tuple(round(capture.get(key)) for key in [cv2.CAP_PROP_FRAME_COUNT,cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT]),capture.get(cv2.CAP_PROP_FPS)
    finally:capture.release()


def combine_cameras(folder,destination,n):
    folder=Path(folder);destination=Path(destination)
    for camera in ('top','wrist'):
        shape,fps=video_shape(folder/f'{camera}.mp4')
        if shape!=(n,640,480) or abs(fps-30)>.001:raise ValueError('Camera frames, size or FPS do not match state/action rows')
    temporary=destination.with_suffix('.partial.mp4')
    try:
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(folder/'top.mp4'),'-i',str(folder/'wrist.mp4'),
                        '-filter_complex','[0:v]setpts=PTS-STARTPTS[left];[1:v]setpts=PTS-STARTPTS[right];[left][right]hstack=inputs=2[v]',
                        '-map','[v]','-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=120)
        shape,fps=video_shape(temporary)
        if shape!=(n,1280,480) or abs(fps-30)>.001:raise ValueError('Combined preview frames differ from episode')
        temporary.replace(destination)
    finally:temporary.unlink(missing_ok=True)


class GenerationViewer:
    def __init__(self):
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='generation-preview')
        self.pending={};self.lock=threading.Lock()

    def get(self,root,index):
        root=Path(root);data=episode_data(root,index);folder=root/'episodes'/f'{index:05d}'
        signature=hashlib.sha256('|'.join(f'{p.stat().st_size}:{p.stat().st_mtime_ns}' for p in [folder/'top.mp4',folder/'wrist.mp4',folder/'plan.npz',folder/'trace.npz']).encode()).hexdigest()[:16]
        name=f'paired-{signature}.mp4';destination=folder/name;key=str(destination)
        with self.lock:
            if destination.is_file():return data|{'status':'ready','video_url':f'/media/generated/{root.name}/episodes/{index:05d}/{name}'}
            future=self.pending.get(key)
            if future and future.done():
                try:future.result()
                except Exception as exc:raise ValueError('Could not prepare synchronized preview: '+str(exc)) from exc
            if future is None:self.pending[key]=self.pool.submit(combine_cameras,folder,destination,data['frames'])
        return {'status':'preparing','frames':data['frames'],'instruction':data['instruction']}

    def close(self):self.pool.shutdown(wait=False,cancel_futures=True)
