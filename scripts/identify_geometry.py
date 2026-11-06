"""Calibrate shared robot-frame offsets from observed cube grasp correspondences."""
import json,copy
import numpy as np
import mujoco
from scipy.optimize import least_squares
from identify_dynamics import BASE,OUT,DATA,ROOT,DatasetIndex
from strict_action_scene import build_model,episode_scene
from episode0 import Calibration
from dataset_to_sim.strict_replay import NativeJointAdapter


def observations(config=None):
    ds=DatasetIndex(DATA);records=[]
    for i in range(72):
        e=ds.episode(i);g=e.state_deg[:,5];found=[]
        for f in range(3,len(g)-3):
            if 7<g[f]<11 and e.action_deg[f,5]<2 and np.max(g[max(0,f-25):f])>20 and np.max(np.abs(g[f:f+3]-g[f]))<.7:
                found.append(f)
        if found:
            f=found[0];cal=episode_scene(config,json.loads((BASE/f'episode-{i:02d}/scene-observation.json').read_text())) if config else Calibration.load(BASE/f'episode-{i:02d}/calibration.json')
            records.append({'episode':i,'frame':f,'state':e.state_deg[f].tolist(),'cube':cal.world['cube']['position']})
    return records


def fit(clearance=False,native_jaws=False,top_face=False,full_clearance=False,tip_anchor=False):
    config=json.loads((BASE/'settings.json').read_text());config['gripper_pad_proxies']=not native_jaws;config['hardware_joint_limits']=tip_anchor;config['cube_projection_height_fraction']=1. if top_face else .5;
    if tip_anchor:
        names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'];f=config['follower_calibration']
        config['adapter']['arm_offset_deg']=[((f[n]['range_min']+f[n]['range_max'])/2-2048)*360/4095 for n in names]
    obs=observations(config);(OUT/'grasp-observations.json').write_text(json.dumps(obs,indent=2))
    cal=Calibration.load(BASE/'episode-01/calibration.json');m=build_model(cal,config);d=mujoco.MjData(m)
    base=m.body('base').id if mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'base')>=0 else 1
    print('base',m.body(base).name,m.body_pos[base]);basepos=m.body_pos[base].copy();default=np.array(config['adapter']['arm_offset_deg'])
    train=[o for o in obs if o['episode']%3!=0]
    ds=DatasetIndex(DATA)
    initial_states=[ds.episode(i).state_deg[0] for i in range(72)] if clearance else []
    if full_clearance:
        for i in range(72):
            if i%3!=0:initial_states.extend(ds.episode(i).state_deg[::10])
    print('constraint poses',len(initial_states),flush=True)
    def residual(x,records,regularize=True):
        c=copy.deepcopy(config['adapter']);c['arm_offset_deg']=x[:5].tolist();c['gripper_closed_model_deg']=x[8];adapter=NativeJointAdapter(**c)
        m.body_pos[base]=basepos+x[5:8]
        errors=[]
        for o in records:
            d.qpos[:6]=adapter.to_model(o['state']);mujoco.mj_forward(m,d)
            center=(d.geom('fixed_jaw_sph_tip1').xpos+d.geom('moving_jaw_sph_tip1').xpos)/2 if tip_anchor else (d.geom('fixed_pad').xpos+d.geom('moving_pad').xpos)/2
            error=(center-o['cube'])/.004
            if tip_anchor:
                # Distal fingertip should reach the lower half of the 19.5 mm cube, with tabletop clearance.
                error[2]=(center[2]-o['cube'][2]+.007)/.004
            elif full_clearance:
                # Contact may occur anywhere along the jaw face; do not force its reference point to cube mid-height.
                dz=center[2]-o['cube'][2];error[2]=(min(0.,dz)+max(0.,dz-.012))/.008
            errors.extend(error)
        if regularize:
            errors.extend((x[:5]-default)/8);errors.extend(x[5:8]/.025);errors.append((x[8]+6)/2)
            for state in initial_states:
                d.qpos[:6]=adapter.to_model(state);mujoco.mj_forward(m,d)
                depth=max([0.]+[-co.dist for co in d.contact if {int(m.geom_contype[co.geom1]),int(m.geom_contype[co.geom2])} in ({2,4},{8,4},{2,8},{2},{8})])
                errors.append(max(0.,depth-.0001)/.0001)
        return np.array(errors)
    initial=np.r_[default,[0,0,0],-10. if native_jaws else -6.]
    result=least_squares(lambda x:residual(x,train),initial,bounds=(np.r_[default-12,[-.025,-.025,0 if clearance else -.025],-10.001 if native_jaws else -12],np.r_[default+12,[.025,.025,.04],-9.999 if native_jaws else 0]),loss='soft_l1',max_nfev=100)
    for name,records in [('train',train),('held_out',[o for o in obs if o['episode']%3==0])]:
        print(name,'initial mm',np.sqrt(np.mean(residual(initial,records,False)**2))*4,'fit mm',np.sqrt(np.mean(residual(result.x,records,False)**2))*4)
    print('x',result.x,'records',len(obs))
    config['adapter']['arm_offset_deg']=result.x[:5].tolist();config['adapter']['gripper_closed_model_deg']=float(result.x[8]);config['base_calibration']['world']['robot_base']['position']=(np.array(config['base_calibration']['world']['robot_base']['position'])+result.x[5:8]).tolist()
    config['geometry_identification']={'training_episodes':[o['episode'] for o in train],'method':'shared frame alignment from observed native-state grasp centers to initial image cube centers; no episode-specific parameters','caveat':'cube may move before stable contact; correspondence residual is not independently measured ground truth','parameters':result.x.tolist(),'initial_collision_constraints_all_episodes':clearance,'training_path_collision_constraints':full_clearance,'native_encoder_frame_prior_and_fingertip_anchor':tip_anchor}
    (OUT/('geometry-tip-low.json' if tip_anchor else ('geometry-full-clearance.json' if full_clearance else ('geometry-top-face.json' if top_face else ('geometry-native-jaws.json' if native_jaws else ('geometry-clearance-fit.json' if clearance else 'geometry-fit.json')))))).write_text(json.dumps(config,indent=2))

if __name__=='__main__':fit()
