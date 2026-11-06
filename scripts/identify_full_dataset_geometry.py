"""Fit one shared robot reference from stable real grasp-state evidence.

Original full action streams drive physical rollouts. Observed final images enter calibration,
but no per-episode physical parameters or cube pose/trajectory corrections are allowed.
"""
import argparse,copy,json
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import mujoco,numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation
from identify_dynamics import fast_trace,DatasetIndex,DATA
from strict_action_scene import episode_scene,build_model
from dataset_to_sim.strict_replay import NativeJointAdapter
from dataset_to_sim.identification import held_gripper_frames


def fit(config_path,source,output,iterations):
    c=json.loads(Path(config_path).read_text());ds=DatasetIndex(DATA)
    ids=list(range(72));items=[]
    observations=json.loads(Path('output/system-identification/real-terminal-observations.json').read_text())
    targets={r['episode']:np.asarray(r['cube_pixel']) for r in observations if r.get('near_bowl_image',False)}
    cam=c['base_calibration']['cameras']['top'];campos=np.array(cam['position']);rotation=Rotation.from_quat(np.array(cam['quaternion'])[[1,2,3,0]]).as_matrix();focal=240/np.tan(np.deg2rad(cam['fovy']/2))
    for i in ids:
        e=ds.episode(i);held=held_gripper_frames(e.state_deg[:,5],e.action_deg[:,5]);indices=np.flatnonzero(held)
        if not len(indices):raise ValueError(f'No stable held evidence in episode{i}')
        stop=e.frame_count
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
                endpoint=0.;pixel_error=None
                if e.index in targets:
                    local=(cube[-1,:3]-campos)@rotation@np.diag([1,-1,-1]);uv=local[:2]/local[2]*focal+[320,240]
                    pixel_error=float(np.linalg.norm(uv-targets[e.index]));endpoint=2*(np.sqrt(1+(max(0,pixel_error-8)/20)**2)-1)
                return {'terminal_pixel_error':pixel_error,'terminal_loss':endpoint,'episode':e.index,'held_loss':grip,'held_rmse':float(np.sqrt(np.mean(error**2))),'arm_capped_mse':arm,'stable_hold_gap_scaled':gap,'held_frames':samples.tolist()}
            rows=list(pool.map(one,items));losses=np.array([r['held_loss']+.1*r['arm_capped_mse']+.5*r['stable_hold_gap_scaled']+r['terminal_loss'] for r in rows]);value=float(np.mean(losses)+.25*np.mean(np.sort(losses)[-18:])+.01*np.sum(x[:5]**2)+.001*np.sum(x[5:]**2));calls+=1
            if value<best[0]:
                best[0]=value;out=copy.deepcopy(c);out['adapter']=ac;out['base_calibration']['world']['robot_base']['position']=(basepos+[x[5]/1000,x[6]/1000,0]).tolist()
                out['full_dataset_identification']={'training_episodes':ids,'objective':'all72 native stable grasp-state error plus physical grasp gap and observed terminal cube image error; mean plus0.25worst-quartile loss; calibration-set result, not heldout generalization','held_selector':'stable blocked gripper, no opening25 threshold','loss':value,'calls':calls,'shared_reference_delta_deg_and_mm':x.tolist(),'per_episode':rows,'initial_max_penetration_m':depth,'terminal_centroid_tolerance_px':8,'terminal_targets':{str(k):v.tolist()for k,v in targets.items()},'zero_and_measured_dimensions_fixed':True,'no_episode_specific_parameters':True,'no_action_modification':True}
                Path(output).write_text(json.dumps(out,indent=2));print('best',calls,'loss',value,'delta',np.round(x,4).tolist(),flush=True)
            return value
        baseline=objective(np.zeros(7));result=differential_evolution(objective,[(-1.5,1.5)]*5+[(-4,4)]*2,x0=np.zeros(7),popsize=4,maxiter=iterations,seed=47,polish=False,tol=.002)
        print('DONE',baseline,'->',best[0],result.message,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',default='output/touch-zero-v1');p.add_argument('--output',default='output/system-identification/full-dataset-v4.json');p.add_argument('--iterations',type=int,default=6);a=p.parse_args();fit(a.config,a.source,a.output,a.iterations)
