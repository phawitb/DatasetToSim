"""Shared robot frame and top-camera exposure lag from real arm silhouettes."""
import json,copy,sys
from pathlib import Path
import cv2,mujoco,numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from identify_dynamics import OUT,BASE,DATA,DatasetIndex
from strict_action_scene import episode_scene,build_model
from dataset_to_sim.strict_replay import NativeJointAdapter
from episode0 import read_real,orange


def fit():
    config=json.loads((BASE/'settings.json').read_text());config['gripper_pad_proxies']=False;config['hardware_joint_limits']=True;config['cube_projection_height_fraction']=1.;config['bowl_projection_height_fraction']=1.
    ds=DatasetIndex(DATA);records=[]
    for i in [1,4,8,13,28,40,61]:
        e=ds.episode(i);frames=read_real(e,'top');scene=json.loads((BASE/f'episode-{i:02d}/scene-observation.json').read_text())
        for f in [max(6,int(.25*e.frame_count)),int(.5*e.frame_count),int(.8*e.frame_count)]:
            target=orange(cv2.resize(frames[f],(240,180))).astype(bool);records.append((e,f,scene,target))
    cal=episode_scene(config,records[0][2]);m=build_model(cal,config);d=mujoco.MjData(m);base=m.body('base').id;basepos=m.body_pos[base].copy();offsets=np.array(config['adapter']['arm_offset_deg']);opt=mujoco.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0;calls=0;best=[5.]
    initial_state=ds.episode(0).state_deg[0]
    with mujoco.Renderer(m,180,240) as renderer:
        def cost(x):
            nonlocal calls
            ac=copy.deepcopy(config['adapter']);ac['arm_offset_deg']=(offsets+x[:5]).tolist();ac['gripper_closed_model_deg']=x[10];a=NativeJointAdapter(**ac)
            m.body_pos[base]=basepos+x[5:8]/1000;m.body_quat[base]=Rotation.from_euler('xy',x[8:10],degrees=True).as_quat()[[3,0,1,2]]
            scores=[]
            for e,f,scene,target in records:
                t=max(0,f-x[11]);lo=int(t);hi=min(lo+1,e.frame_count-1);state=(1-t+lo)*e.state_deg[lo]+(t-lo)*e.state_deg[hi]
                c=episode_scene(config,scene);m.body_pos[m.body('bowl').id]=c.world['bowl']['position'];d.qpos[:6]=a.to_model(state);d.qpos[6:9]=c.world['cube']['position'];d.qpos[9:13]=c.world['cube']['quaternion'];mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera='top_cam',scene_option=opt);sim=orange(renderer.render()).astype(bool);scores.append(np.logical_and(sim,target).sum()/max(1,np.logical_or(sim,target).sum()))
            d.qpos[:6]=a.to_model(initial_state);mujoco.mj_forward(m,d)
            depth=max([0.]+[-co.dist for co in d.contact if {int(m.geom_contype[co.geom1]),int(m.geom_contype[co.geom2])} in ({2,4},{8,4},{2,8},{2},{8})])
            value=1-float(np.mean(scores))+max(0,depth-.0003)*25+.0002*np.sum((x[:5]/5)**2)+.0001*np.sum((x[5:10]/5)**2)
            calls+=1
            if value<best[0]:
                best[0]=value;out=copy.deepcopy(config);out['adapter']=ac;out['base_calibration']['world']['robot_base']['position']=m.body_pos[base].tolist();out['base_calibration']['world']['robot_base']['quaternion']=m.body_quat[base].tolist();out['camera_exposure_delay_frames']={'top':float(x[11]),'wrist':0.}
                out['visual_frame_identification']={'training_episodes':[1,4,8,13,28,40,61],'frames':[[e.index,f] for e,f,_,_ in records],'mean_iou':float(np.mean(scores)),'episode0_initial_depth_m':depth,'objective':value,'parameters':x.tolist(),'evaluations':calls,'measured_state_interpolation_for_camera_exposure_calibration_only':True}
                (OUT/'top-visual-fit.json').write_text(json.dumps(out,indent=2));print('best',calls,'iou',np.mean(scores),'depth',depth,'lag',x[11],flush=True)
            return value
        x0=np.r_[np.zeros(10),-10.,1.]
        cost(x0)
        minimize(cost,x0,method='Powell',bounds=[(-10,10)]*5+[(-20,20),(-20,20),(0,35)]+[(-8,8)]*2+[(-14,-3),(0,6)],options={'maxfev':1300,'maxiter':10,'ftol':.0002,'xtol':.002})
        print('DONE',calls,'best',best[0],flush=True)

if __name__=='__main__':fit()
