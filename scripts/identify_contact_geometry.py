"""Shared-frame calibration using physical cube/jaw distances, not a hand-point proxy."""
import copy,json,sys
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from identify_dynamics import BASE,OUT,DATA,DatasetIndex
from strict_action_scene import episode_scene,build_model
from dataset_to_sim.strict_replay import NativeJointAdapter


def fit(seed=0,fixed_base=False):
    config=json.loads((OUT/'combined-v5.json').read_text());config['hardware_joint_limits']=True
    if fixed_base:config=json.loads((OUT/'confirmed-measurements.json').read_text())
    base_reference=np.array(json.loads((BASE/'settings.json').read_text())['base_calibration']['world']['robot_base']['position'])
    observations=json.loads((OUT/'grasp-observations.json').read_text());train=[o for o in observations if o['episode']%3!=0]
    refined=json.loads((OUT/'confirmed-scenes.json').read_text()) if fixed_base else None
    for o in train:
        scene=refined[o['episode']] if refined else json.loads((BASE/f"episode-{o['episode']:02d}/scene-observation.json").read_text())
        cal=episode_scene(config,scene)
        o['cube']=cal.world['cube']['position'];o['quaternion']=cal.world['cube']['quaternion']
    cal=episode_scene(config,json.loads((BASE/'episode-01/scene-observation.json').read_text()));m=build_model(cal,config);d=mujoco.MjData(m);base=m.body('base').id;cube=m.geom('cube_geom').id
    fixed=[i for i in range(m.ngeom) if m.geom_contype[i]==2 and m.geom_bodyid[i]==m.body('gripper').id and m.geom(i).name not in ['fixed_jaw_box1','fixed_jaw_box2']]
    moving=[i for i in range(m.ngeom) if m.geom_contype[i]==2 and m.geom_bodyid[i]==m.body('moving_jaw_so101_v1').id and m.geom(i).name!='moving_jaw_box1']
    ds=DatasetIndex(DATA);eps=[ds.episode(i) for i in range(72)];constraints=np.array([e.state_deg[0] for e in eps]+[s for e in eps if e.index%3!=0 for s in e.state_deg[::20]])
    names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'];f=config['follower_calibration'];prior=np.array([((f[n]['range_min']+f[n]['range_max'])/2-2048)*360/4095 for n in names])
    calls=0;best=[float('inf')]
    def apply(x):
        ac=copy.deepcopy(config['adapter']);ac['arm_offset_deg']=x[:5].tolist();ac['gripper_closed_model_deg']=float(x[10]);adapter=NativeJointAdapter(**ac)
        m.body_pos[base]=base_reference+[x[5]/1000,x[6]/1000,x[7]/1000]
        m.body_quat[base]=Rotation.from_euler('xy',x[8:10],degrees=True).as_quat()[[3,0,1,2]]
        return adapter
    def residual(x):
        nonlocal calls
        a=apply(x);errors=[];distances=[]
        for o in train:
            d.qpos[:6]=a.to_model(o['state']);d.qpos[6:9]=o['cube'];d.qpos[9:13]=o['quaternion'];mujoco.mj_forward(m,d)
            for ids in [fixed,moving]:
                distance=min(mujoco.mj_geomDistance(m,d,cube,g,.05,None) for g in ids);errors.append(distance/.003);distances.append(distance)
            local=d.body('gripper').xmat.reshape(3,3).T@(d.geom('cube_geom').xpos-d.body('gripper').xpos)
            errors.append(local[1]/.03) # Broad alignment prior; geometric contacts determine the pinch depth.
        depths=[]
        for q in a.to_model(constraints):
            d.qpos[:6]=q;mujoco.mj_forward(m,d)
            depth=max([0.]+[-co.dist for co in d.contact if {int(m.geom_contype[co.geom1]),int(m.geom_contype[co.geom2])} in ({2,4},{8,4},{2,8},{2},{8})]);depths.append(depth);errors.append(max(0,depth-.00015)/.0002)
        errors.extend((x[:5]-prior)/8);errors.extend(x[5:7]/20);errors.append(x[7]/30);errors.extend(x[8:10]/5);errors.append((x[10]+10)/4)
        calls+=1;loss=np.mean(2*(np.sqrt(1+np.array(errors)**2)-1))
        if loss<best[0]:
            best[0]=loss;out=copy.deepcopy(config);out['adapter']['arm_offset_deg']=x[:5].tolist();out['adapter']['gripper_closed_model_deg']=float(x[10]);out['base_calibration']['world']['robot_base']['position']=m.body_pos[base].tolist();out['base_calibration']['world']['robot_base']['quaternion']=m.body_quat[base].tolist()
            out['geometry_identification']={'method':'signed physical jaw/cube distance using measured-state grasp observations, shared base pose and joint references','training_episodes':[o['episode'] for o in train],'all_episode_initial_states_constrained':True,'training_path_stride':20,'parameters':x.tolist(),'contact_distance_rmse_m':float(np.sqrt(np.mean(np.array(distances)**2))),'constraint_max_depth_m':max(depths),'loss':float(loss),'calls':calls,'caveat':'Initial cube image is used as a noisy grasp-position observation; pushes or early lift cause outliers.'}
            if fixed_base:
                out['measurements']=json.loads((BASE.parent.parent/'calibration/real-measurements.json').read_text())
                out['geometry_identification']['base_support_constraint']='User confirmed standard SO-101 directly on tabletop; no riser or tilt'
                out['scene_detection']='cube-silhouette-v1'
            (OUT/f'contact-geometry-{"fixedbase-" if fixed_base else ""}seed-{seed}.json').write_text(json.dumps(out,indent=2))
            if calls%10==0:print('best',calls,'contact rmsmm',out['geometry_identification']['contact_distance_rmse_m']*1000,'depthmm',max(depths)*1000,'loss',loss,flush=True)
        return np.array(errors)
    current=config['base_calibration']['world']['robot_base']['position'];initial=np.r_[config['adapter']['arm_offset_deg'],(np.array(current)-base_reference)*1000,[0,0],config['adapter']['gripper_closed_model_deg']]
    lo=np.r_[prior-12,[-25,-25,0],[-8,-8],-14];hi=np.r_[prior+12,[25,25,45],[8,8],-3]
    if seed==1:initial[8:10]=[2,-2];initial[10]=-7.5
    if seed==2:initial[8:10]=[-2,2];initial[10]=-11.5
    if fixed_base:
        initial[7]=(config['base_calibration']['world']['robot_base']['position'][2]-base_reference[2])*1000
        initial[8:10]=0
        lo[:5]=prior-20;hi[:5]=prior+20
        if seed==1:initial[:5]=prior
        if seed==2:initial[:5]=[6.4,-4.1,-4,-8.9,2.8]
    from scipy.optimize._numdiff import approx_derivative
    def jac(x):return approx_derivative(residual,x,method='2-point',abs_step=np.r_[[.03]*5,[.05]*3,[.03]*3],bounds=(lo,hi))
    if fixed_base:
        # Eliminate fixed dimensions; near-zero bound widths corrupt numerical
        # Jacobians and cause premature trust-region termination.
        free=np.array([0,1,2,3,4,5,6,10])
        def expand(v):
            x=initial.copy();x[free]=v;return x
        def fun(v):return residual(expand(v))
        def reduced_jac(v):return approx_derivative(fun,v,method='2-point',abs_step=np.array([.03]*5+[.05]*2+[.03]),bounds=(lo[free],hi[free]))
        result=least_squares(fun,initial[free],jac=reduced_jac,bounds=(lo[free],hi[free]),loss='soft_l1',max_nfev=180,ftol=1e-6,xtol=1e-6)
    else:
        result=least_squares(residual,initial,jac=jac,bounds=(lo,hi),loss='soft_l1',max_nfev=100,ftol=1e-5,xtol=1e-5)
    print('DONE',result.message,'params',result.x,flush=True)

if __name__=='__main__':fit(int(sys.argv[1]) if len(sys.argv)>1 else 0,'--fixed-base' in sys.argv)
