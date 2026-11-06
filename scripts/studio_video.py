"""Create missing benchmark videos from archived states without rerunning physics."""
from pathlib import Path
import os,sys,json,subprocess
os.environ.setdefault('MUJOCO_GL','cgl' if sys.platform=='darwin' else 'egl')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from dataset_to_sim.archive import load_saved_model
from dataset_to_sim.dataset import DatasetIndex
from dataset_to_sim.calibration import Calibration
from episode0 import DATA
from episode0_physics import render_video


def render(index):
    run=ROOT/'output/shared-calibration-v10';folder=run/f'episode-{index:02d}';output=ROOT/'output/studio'/f'replay-{index:02d}';output.mkdir(exist_ok=True)
    settings=json.loads((run/'settings.json').read_text());report=json.loads((folder/'report.json').read_text())
    ep=DatasetIndex(DATA).episode(index);model=load_saved_model(folder,settings,report['model_archive_sha256'])
    with np.load(folder/'strict_trace.npz') as trace:
        render_video(Calibration.load(folder/'calibration.json'),ep,model,trace['qpos'][:-1],trace['cube_pose'][:-1],output,label='MuJoCo | ORIGINAL actions',camera_config=settings['wrist_camera'],save_scene_xml=False,camera_delay_frames=settings.get('camera_delay_frames'))
    target=output/'comparison.mp4';partial=output/'comparison-partial.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(output/'comparison-raw.mp4'),'-c:v','libx264','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)],check=True)
    partial.replace(target);(output/'comparison-raw.mp4').unlink()
    (output/'manifest.json').write_text(json.dumps({'episode':index,'frames':ep.frame_count,'fps':30,'source_model_sha256':report['model_archive_sha256'],'source_trace_sha256':report['trace_archive_sha256'],'camera_delay_frames':settings.get('camera_delay_frames'),'physics_rerun':False},indent=2))

if __name__=='__main__':render(int(sys.argv[1]))
