"""Fit one shared robot reference from stable real grasp-state evidence.

Original action prefixes drive physical rollouts. No terminal success objective,
per-episode physical parameters, or cube pose/trajectory corrections are allowed.
"""
import argparse,copy,json
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import mujoco,numpy as np
from scipy.optimize import differential_evolution
from identify_dynamics import fast_trace,DatasetIndex,DATA
from strict_action_scene import episode_scene,build_model
from dataset_to_sim.strict_replay import NativeJointAdapter
from dataset_to_sim.identification import held_gripper_frames


def fit(config_path,source,output,iterations):
    c=json.loads(Path(config_path).read_text());ds=DatasetIndex(DATA)
    ids=[1,4,8,13,20,28,31,37,38,40,44,49,50,61,64,71];items=[]
    for i in ids:
        e=ds.episode(i);held=held_gripper_frames(e.state_deg[:,5],e.action_deg[:,5]);indices=np.flatnonzero(held)
        if not len(indices):raise ValueError(f'No stable held evidence in episode{i}')
        stop=min(e.frame_count,int(indices[-1])+5)
        e=replace(e,frame_indices=e.frame_indices[:stop],dataset_indices=e.dataset_indices[:stop],timestamps=e.timestamps[:stop],state_deg=e.state_deg[:stop],action_deg=e.action_deg[:stop])
        scene=json.loads((Path(source)/f'episode-{i:02d}/scene-observation.json').read_text());m=build_model(episode_scene(c,scene),c)
        items.append((e,m,m.jnt_range[:5].copy(),held[:stop],mujoco.MjData(m)))
    prior=np.array(c['adapter']['arm_offset_deg']);basepos=np.array(c['base_calibration']['world']['robot_base']['position']);initial=np.array([ds.episode(i).state_deg[0]for i in range(72)]);guard=items[0][1];gd=mujoco.MjData(guard);best=[np.inf];calls=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        def objective(x):
            nonlocal calls
            ac=copy.deepcopy(c['adapter']);ac['arm_offset_deg']=(prior+x[:5]).tolist();a=NativeJointAdapter(**ac)
            for _,m,ranges,_,_ in items:
                m.body_pos[m.body('base').id]=basepos+[x[5]/1000,x[6]/1000,0];m.jnt_range[:5]=ranges+np.deg2rad(x[:5])[:,None]
            depth=0.
            for q in a.to_model(initial):
                gd.qpos[:6]=q;mujoco.mj_forward(guard,gd)
                depth=max(depth,max([0.]+[-co.dist for co in gd.contact if {int(guard.geom_contype[co.geom1]),int(guard.geom_contype[co.geom2])} in ({2,4},{8,4},{2,8},{2},{8})]))
            if depth>.00005:return 30+depth*30000+float(x@x)*.01
            def one(item):
                e,m,_,held,d=item;q,cube=fast_trace(m,e,a);native=a.to_native(q);error=native[held,5]-e.state_deg[held,5]
                grip=float(np.mean(2*(np.sqrt(1+error**2)-1)));arm=float(np.mean(np.minimum((native[8:,:5]-e.state_deg[8:,:5])**2,9)))
                # Contact gap at real stable hold, after the simulator's own push.
                samples=np.flatnonzero(held);sample=int(samples[min(3,len(samples)-1)]);d.qpos[:6]=q[sample];d.qpos[6:13]=cube[sample];mujoco.mj_forward(m,d);cid=m.geom('cube_geom').id;gap=0.
                for body in ['gripper','moving_jaw_so101_v1']:
                    geoms=[k for k in range(m.ngeom)if m.geom_contype[k]==2 and m.geom_bodyid[k]==m.body(body).id]
                    distance=min(mujoco.mj_geomDistance(m,d,cid,k,.1,None)for k in geoms);gap+=min(.03,max(0,distance))/.005
                return {'episode':e.index,'held_loss':grip,'held_rmse':float(np.sqrt(np.mean(error**2))),'arm_capped_mse':arm,'stable_hold_gap_scaled':gap,'held_frames':samples.tolist()}
            rows=list(pool.map(one,items));value=float(np.mean([r['held_loss']+.1*r['arm_capped_mse']+r['stable_hold_gap_scaled'] for r in rows])+.01*np.sum(x[:5]**2)+.001*np.sum(x[5:]**2));calls+=1
            if value<best[0]:
                best[0]=value;out=copy.deepcopy(c);out['adapter']=ac;out['base_calibration']['world']['robot_base']['position']=(basepos+[x[5]/1000,x[6]/1000,0]).tolist()
                out['stable_grasp_identification']={'training_episodes':ids,'objective':'stable real blocked-gripper softL1 plus physical gap at stable hold plus0.1arm error; no final outcome reward','held_selector':'stable blocked gripper, no opening25 threshold','loss':value,'calls':calls,'shared_reference_delta_deg_and_mm':x.tolist(),'per_episode':rows,'initial_max_penetration_m':depth,'zero_and_measured_dimensions_fixed':True,'no_episode_specific_parameters':True,'no_action_modification':True}
                Path(output).write_text(json.dumps(out,indent=2));print('best',calls,'loss',value,'delta',np.round(x,4).tolist(),flush=True)
            return value
        baseline=objective(np.zeros(7));result=differential_evolution(objective,[(-1.5,1.5)]*5+[(-4,4)]*2,x0=np.zeros(7),popsize=5,maxiter=iterations,seed=31,polish=False,tol=.002)
        print('DONE',baseline,'->',best[0],result.message,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',default='output/touch-zero-v1');p.add_argument('--output',default='output/system-identification/stable-grasp-v2.json');p.add_argument('--iterations',type=int,default=10);a=p.parse_args();fit(a.config,a.source,a.output,a.iterations)
