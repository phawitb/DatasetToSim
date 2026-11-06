"""Compare real masks against actual integrated replay states and cameras."""
import argparse,json
from pathlib import Path
import cv2,mujoco,numpy as np
from benchmark72 import DatasetIndex,DATA,read_real
from dataset_to_sim.archive import load_saved_model
from identify_wrist_current import mask


def evaluate(folders,output):
 ds=DatasetIndex(DATA);ids=[0,3,12,21,24,38,40,44,66,71];records={}
 for i in ids:
  e=ds.episode(i);frames=sorted(set(int(v*(e.frame_count-1))for v in [0,.25,.45,.6,.75,.9]));images={view:read_real(e,view)for view in ['top','wrist']};records[i]=[(f,{v:mask(cv2.resize(images[v][f],(320,240)),'jaw')for v in images})for f in frames]
 results=[]
 for folder in map(Path,folders):
  c=json.loads((folder/'settings.json').read_text());rows=[]
  for i in ids:
   p=folder/f'episode-{i:02d}';trace=np.load(p/'strict_trace.npz');m=load_saved_model(p,c);d=mujoco.MjData(m);opt=mujoco.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0
   with mujoco.Renderer(m,240,320)as renderer:
    for frame,targets in records[i]:
     row={'episode':i,'frame':frame}
     for view in ['top','wrist']:
      sample=max(0,frame-c.get('camera_delay_frames',{}).get(view,0));d.qpos[:6]=trace['qpos'][sample];d.qpos[6:13]=trace['cube_pose'][sample];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=view+'_cam',scene_option=opt);sim=mask(renderer.render(),'jaw');target=targets[view]
      row[view+'_orange_iou']=float(np.logical_and(target,sim).sum()/max(1,np.logical_or(target,sim).sum()))
     rows.append(row)
  results.append({'folder':str(folder),'evaluation_episodes':ids,'sampled_frames':len(rows),'mean':{view:float(np.mean([r[view+'_orange_iou']for r in rows]))for view in ['top','wrist']},'rows':rows,'scope':'actual integrated q/cube, frozen cameras and integer causal image delays; orange segmentation proxy, not photorealism or physical-accuracy percentage; all72 used for geometry calibration'})
 Path(output).write_text(json.dumps(results,indent=2));print([(r['folder'],r['mean'])for r in results],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folders',nargs='+');p.add_argument('--output',required=True);a=p.parse_args();evaluate(a.folders,a.output)
