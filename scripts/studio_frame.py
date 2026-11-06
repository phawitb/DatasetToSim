"""Render one archived observation with its original camera delays."""
from pathlib import Path
import sys,json,os
os.environ.setdefault('MUJOCO_GL','cgl' if sys.platform=='darwin' else 'egl')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import cv2,mujoco,numpy as np
from dataset_to_sim.dataset import DatasetIndex
from dataset_to_sim.archive import load_saved_model
from episode0 import DATA

def render(index,frame,target):
    run=ROOT/'output/shared-calibration-v10';folder=run/f'episode-{index:02d}'
    ep=DatasetIndex(DATA).episode(index)
    if not 0<=frame<ep.frame_count:raise ValueError('Frame out of bounds')
    settings=json.loads((run/'settings.json').read_text());report=json.loads((folder/'report.json').read_text())
    m=load_saved_model(folder,settings,report['model_archive_sha256']);d=mujoco.MjData(m)
    trace=np.load(folder/'strict_trace.npz');addr=m.joint('cube_free').qposadr[0]
    option=mujoco.MjvOption();option.geomgroup[3:]=0;option.sitegroup[:]=0
    rows=[]
    with mujoco.Renderer(m,height=480,width=640) as renderer:
        for view in ['top','wrist']:
            source=ep.video_slices[view];cap=cv2.VideoCapture(str(source.path));cap.set(cv2.CAP_PROP_POS_FRAMES,round(source.start_time*30)+frame);ok,real=cap.read();cap.release()
            if not ok:raise ValueError('Could not decode recorded frame')
            f=max(0,frame-settings.get('camera_delay_frames',{}).get(view,0))
            d.qpos[:6]=trace['qpos'][f];d.qpos[addr:addr+7]=trace['cube_pose'][f];mujoco.mj_forward(m,d)
            renderer.update_scene(d,camera=f'{view}_cam',scene_option=option)
            canvas=np.zeros((512,1280,3),np.uint8);canvas[32:,:640]=real;canvas[32:,640:]=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR);rows.append(canvas)
    cv2.imwrite(str(target),np.vstack(rows))

if __name__=='__main__':render(int(sys.argv[1]),int(sys.argv[2]),Path(sys.argv[3]))
