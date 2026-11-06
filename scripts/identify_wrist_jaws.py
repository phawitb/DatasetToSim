"""Fit rigid wrist camera from native measured joints and image masks, not outcomes."""
import json,sys
from pathlib import Path
import numpy as np
import mujoco,cv2
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from identify_dynamics import DATA,ROOT,OUT,BASE,DatasetIndex
from strict_action_scene import episode_scene,build_model
from dataset_to_sim.strict_replay import NativeJointAdapter
from episode0 import read_real

def mask(rgb,kind):
    h=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    return (h[:,:,0]>3)&(h[:,:,0]<30)&(h[:,:,1]>100)&(h[:,:,2]>35)


def fit(config_path):
    config=json.loads(Path(config_path).read_text());adapter=NativeJointAdapter(**config['adapter']);ds=DatasetIndex(DATA)
    records=[]
    for i in [1,4,8,28]:
        e=ds.episode(i);frames=read_real(e,'wrist');cal=episode_scene(config,json.loads((BASE/f'episode-{i:02d}/scene-observation.json').read_text()))
        peak=int(np.argmax(e.state_deg[:,5]));indices=[0,peak,e.frame_count-1]
        for level in [40,25,15,9]:
            ids=np.flatnonzero(e.state_deg[peak:,5]<level)
            if len(ids):indices.append(peak+int(ids[0]))
        for f in sorted(set(indices)):
            rgb=cv2.resize(frames[f],(240,180));targets=[mask(rgb,'jaw')]
            records.append((e,f,cal.world['bowl']['position'],targets))
    model=build_model(cal,config);data=mujoco.MjData(model);cid=model.camera('wrist_cam').id
    model.geom_group[model.geom('cube_geom').id]=3
    previous=json.loads((OUT/'wrist-camera-fit.json').read_text());p0=np.array(previous['position']);R0=Rotation.from_quat(np.array(previous['quaternion'])[[1,2,3,0]]).as_matrix()
    opt=mujoco.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0;calls=0;best=[2.]
    with mujoco.Renderer(model,180,240) as renderer:
        def cost(x):
            nonlocal calls
            model.cam_pos[cid]=p0+x[:3];model.cam_quat[cid]=Rotation.from_matrix(R0@Rotation.from_rotvec(x[3:6]).as_matrix()).as_quat()[[3,0,1,2]];model.cam_fovy[cid]=x[6]
            scores=[]
            for e,f,bowl,targets in records:
                t=min(e.frame_count-1,max(0,f-x[7]));lo=int(t);hi=min(lo+1,e.frame_count-1)
                q=adapter.to_model(e.state_deg[lo]*(1-t+lo)+e.state_deg[hi]*(t-lo))
                q[5]+=np.deg2rad(x[8]-config['adapter']['gripper_closed_model_deg'])
                model.body_pos[model.body('bowl').id]=bowl;data.qpos[:6]=q;mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera='wrist_cam',scene_option=opt);rgb=renderer.render()
                for kind,target in zip(['jaw'],targets):
                    sim=mask(rgb,kind);union=np.logical_or(sim,target).sum()
                    if target.sum()>20:scores.append(np.logical_and(sim,target).sum()/max(1,union))
            calls+=1;value=1-float(np.mean(scores))
            if value<best[0]:
                best[0]=value
                result={'position':model.cam_pos[cid].tolist(),'quaternion':model.cam_quat[cid].tolist(),'fovy':float(model.cam_fovy[cid]),'exposure_delay_frames':float(x[7]),'training_mask_iou':1-value,'training_episodes':[1,4,8,28],'fit_frames':[[r[0].index,r[1]] for r in records],'evaluations':calls,'uses_measured_states_for_calibration_only':True,'cube_hidden_in_calibration_renders':True}
                result['gripper_closed_model_deg']=float(x[8]);result['objective']='distal jaw orange masks only; red cube excluded; signed lag for identification only'
                (OUT/'wrist-jaw-identification.json').write_text(json.dumps(result,indent=2));print('best',calls,1-value,'zero',x[8],'lag',x[7],flush=True)
            return value
        x0=np.r_[np.zeros(6),previous['fovy'],0.,-10.];start=cost(x0)
        minimize(cost,x0,method='Powell',bounds=[(-.015,.015)]*3+[(-.25,.25)]*3+[(43,80),(-3,4),(-16,-4)],options={'maxfev':1200,'maxiter':12,'ftol':.0001,'xtol':.0003})
    print('initial',1-start,'final',1-best[0],flush=True)

if __name__=='__main__':fit(sys.argv[1])
