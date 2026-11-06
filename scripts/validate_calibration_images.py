"""Compare camera/arm alignment on measured states, independently of physics task success."""
import sys,json,copy
from pathlib import Path
import cv2,mujoco,numpy as np
from identify_dynamics import ROOT,OUT,BASE,DATA,DatasetIndex
from strict_action_scene import episode_scene,build_model
from dataset_to_sim.strict_replay import NativeJointAdapter
from episode0 import read_real,orange
from fit_wrist_physics import mask


def evaluate(paths):
    ds=DatasetIndex(DATA);records=[]
    for i in [0,1,3,4,6,8,12,27]:
        e=ds.episode(i);frames={v:read_real(e,v) for v in ['top','wrist']};scene=json.loads((BASE/f'episode-{i:02d}/scene-observation.json').read_text())
        for f in [0,int(.4*e.frame_count),int(.7*e.frame_count),int(.9*e.frame_count)]:
            records.append((i,f,e.state_deg[f],scene,{v:cv2.resize(frames[v][f],(240,180)) for v in frames}))
    output=[]
    for path in paths:
        c=json.loads(Path(path).read_text());adapter=NativeJointAdapter(**c['adapter']);m=build_model(episode_scene(c,records[0][3]),c);d=mujoco.MjData(m);opt=mujoco.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0;rows=[]
        with mujoco.Renderer(m,180,240) as renderer:
            for i,f,state,scene,images in records:
                cal=episode_scene(c,scene);m.body_pos[m.body('bowl').id]=cal.world['bowl']['position'];d.qpos[:6]=adapter.to_model(state);d.qpos[6:9]=cal.world['cube']['position'];d.qpos[9:13]=cal.world['cube']['quaternion'];mujoco.mj_forward(m,d)
                row={'episode':i,'frame':f,'held_out':i%3==0}
                for view in ['top','wrist']:
                    renderer.update_scene(d,camera=view+'_cam',scene_option=opt);rgb=renderer.render()
                    kinds=['arm'] if view=='top' else ['jaw','bowl']
                    for kind in kinds:
                        a=orange(images[view]) if kind=='arm' else mask(images[view],kind);b=orange(rgb) if kind=='arm' else mask(rgb,kind)
                        if a.sum()>20:row[view+'_'+kind+'_iou']=float(np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum()))
                rows.append(row)
        summary={'config':str(path),'rows':rows}
        for split in ['all','held_out']:
            selected=rows if split=='all' else [r for r in rows if r['held_out']]
            summary[split]={k:float(np.mean([r[k] for r in selected if k in r])) for k in ['top_arm_iou','wrist_jaw_iou','wrist_bowl_iou']}
        print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True);output.append(summary)
    (OUT/'image-validation.json').write_text(json.dumps(output,indent=2))

if __name__=='__main__':evaluate(sys.argv[1:])
