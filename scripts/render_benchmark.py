"""Render stored physics traces beside their matching real video slices."""
import argparse
import subprocess
import json
from benchmark72 import BATCH, Calibration, DatasetIndex, DATA, mujoco, np, render_video

def render(tag, episodes):
    ds=DatasetIndex(DATA)
    settings=json.loads((BATCH/tag/'settings.json').read_text())
    for i in episodes:
        folder=BATCH/tag/f'episode-{i:02d}'
        cal=Calibration.load(folder/'calibration.json')
        model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
        trace=np.load(folder/'physics_trace.npz')
        label='MuJoCo | trajectory repaired' if settings.get('trajectory_repair') else 'MuJoCo | calibrated replay'
        render_video(cal,ds.episode(i),model,trace['qpos'],trace['cube_pose'],folder,label=label,camera_config=settings['wrist_camera'])
        output=folder/'real-vs-sim.mp4'
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(folder/'comparison-raw.mp4'),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True)
        print(output,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('tag');parser.add_argument('episodes',nargs='+',type=int)
    args=parser.parse_args();render(args.tag,args.episodes)
