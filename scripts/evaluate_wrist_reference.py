"""Independent-frame jaw camera evaluation with fixed measured native joints."""
import json
from pathlib import Path
import cv2,mujoco,numpy as np
from identify_dynamics import DATA,OUT,DatasetIndex
from strict_action_scene import episode_scene,build_model
from dataset_to_sim.strict_replay import NativeJointAdapter
from identify_wrist_current import mask
from episode0 import read_real


def evaluate():
    config=json.loads(Path('calibration/native-action-current.json').read_text());ds=DatasetIndex(DATA);records=[]
    ids=[0,3,12,21,24,38,40,44,66,71]
    for i in ids:
        e=ds.episode(i);images=read_real(e,'wrist');scene=json.loads(Path(f'output/touch-zero-v1/episode-{i:02d}/scene-observation.json').read_text())
        for f in sorted(set(int(v*(e.frame_count-1))for v in [0,.25,.45,.6,.75,.9])):records.append((e,f,scene,mask(cv2.resize(images[f],(240,180)),'jaw')))
    results=[]
    for name,wrist in [('before',config['wrist_camera']),('after',json.loads((OUT/'wrist-balanced.json').read_text()))]:
        config['wrist_camera']=wrist;a=NativeJointAdapter(**config['adapter']);m=build_model(episode_scene(config,records[0][2]),config);d=mujoco.MjData(m);m.geom_group[m.geom('cube_geom').id]=3;opt=mujoco.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0;rows=[]
        with mujoco.Renderer(m,180,240) as renderer:
            for e,f,scene,target in records:
                lag=wrist['exposure_delay_frames'];t=max(0,f-lag);lo=int(t);hi=min(lo+1,e.frame_count-1);q=a.to_model(e.state_deg[lo]*(1-t+lo)+e.state_deg[hi]*(t-lo));d.qpos[:6]=q
                cal=episode_scene(config,scene);m.body_pos[m.body('bowl').id]=cal.world['bowl']['position'];mujoco.mj_forward(m,d);renderer.update_scene(d,camera='wrist_cam',scene_option=opt);sim=mask(renderer.render(),'jaw');union=np.logical_or(sim,target).sum()
                if target.sum()>20:rows.append({'episode':e.index,'frame':f,'iou':float(np.logical_and(sim,target).sum()/max(1,union))})
        results.append({'candidate':name,'camera_training_episodes':[1,4,8,28,37,45,50,64],'evaluation_episodes':ids,'evaluation_mean_iou':float(np.mean([r['iou']for r in rows])),'rows':rows,'scope':'jaw-only image calibration on measured joints; not integrated-state validation, not a blind task-generalization split'})
    (OUT/'wrist-balanced-validation.json').write_text(json.dumps(results,indent=2));print([(r['candidate'],r['evaluation_mean_iou'])for r in results])

if __name__=='__main__':evaluate()
