"""Bounded shared-reference calibration, emphasizing the worst grasp-state errors.

All 72 original action streams remain intact. Screening outcomes are diagnostic
only; an accepted candidate must pass the separate strict replay and audit.
"""
import argparse, copy, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import mujoco
import numpy as np
from scipy.optimize import differential_evolution, minimize
from identify_dynamics import fast_trace, DatasetIndex, DATA
from strict_action_scene import episode_scene, build_model, containment
from dataset_to_sim.strict_replay import NativeJointAdapter
from dataset_to_sim.identification import held_gripper_frames


def fit(args):
    c=json.loads(Path(args.config).read_text()); ds=DatasetIndex(DATA)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    items=[]
    for i in range(72):
        e=ds.episode(i)
        scene=json.loads((Path(args.source)/f'episode-{i:02d}/scene-observation.json').read_text())
        cal=episode_scene(c,scene);m=build_model(cal,c)
        items.append((e,m,m.jnt_range[:5].copy(),held_gripper_frames(e.state_deg[:,5],e.action_deg[:,5]),cal))
    prior=np.asarray(c['adapter']['arm_offset_deg']);bp=np.asarray(c['base_calibration']['world']['robot_base']['position'])
    initial=np.array([e.state_deg[0] for e,*_ in items]);guard=items[0][1];gd=mujoco.MjData(guard)
    best=float('inf');calls=0;history=[];best_screen=(-1,float('inf'));started=time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        def objective(x):
            nonlocal best,calls,best_screen
            calls+=1;ac=copy.deepcopy(c['adapter']);ac['arm_offset_deg']=(prior+x[:5]).tolist();a=NativeJointAdapter(**ac)
            for _,m,ranges,_,_ in items:
                m.body_pos[m.body('base').id]=bp+[x[5]/1000,x[6]/1000,0]
                m.jnt_range[:5]=ranges+np.deg2rad(x[:5])[:,None]
            depth=0.
            for q in a.to_model(initial):
                gd.qpos[:6]=q;mujoco.mj_forward(guard,gd)
                depth=max(depth,max([0.]+[-co.dist for co in gd.contact if {int(guard.geom_contype[co.geom1]),int(guard.geom_contype[co.geom2])} in ({2,4},{8,4},{2,8},{2},{8})]))
            if depth>.00005:return 100+depth*30000
            def one(item):
                e,m,_,held,cal=item;q,cube=fast_trace(m,e,a);native=a.to_native(q)
                error=native[held,5]-e.state_deg[held,5]
                grip=float(np.mean(2*(np.sqrt(1+error**2)-1)))
                arm=float(np.mean((native[:,:5]-e.state_deg[:,:5])**2))
                # Screening is not the strict secure-contact success criterion.
                lift=float(cube[:,2].max()-cube[0,2]);inside=containment(m,q[-1],cube[-1],cal,c)['inside']
                screen=bool(lift>=.04 and inside)
                gap=0.
                if args.gap_weight:
                    f=int(np.flatnonzero(held)[0]);d=mujoco.MjData(m);d.qpos[:6]=q[f];d.qpos[6:13]=cube[f];mujoco.mj_forward(m,d);cid=m.geom('cube_geom').id
                    for body in ['gripper','moving_jaw_so101_v1']:
                        ids=[j for j in range(m.ngeom)if m.geom_contype[j]==2 and m.geom_bodyid[j]==m.body(body).id]
                        gap+=max(0.,min(mujoco.mj_geomDistance(m,d,cid,j,.1,None)for j in ids))/.003
                return {'episode':e.index,'gap_scaled':gap,'held_loss':grip,'arm_mse':arm,'screen_lift_and_inside':screen,'lift_m':lift,'terminal_cube':cube[-1,:3].tolist()}
            rows=list(pool.map(one,items));losses=np.array([r['held_loss'] for r in rows]);arm=float(np.mean([r['arm_mse'] for r in rows]))
            value=float(losses.mean()+np.sort(losses)[-8:].mean()+.5*arm+.02*np.sum(x[:5]**2)+.002*np.sum(x[5:]**2))
            gaps=np.array([r['gap_scaled']for r in rows]);value+=args.gap_weight*(.25*gaps.mean()+np.sort(gaps)[-4:].mean())
            screen=sum(r['screen_lift_and_inside'] for r in rows)
            record={'call':calls,'loss':value,'screen_passes':screen,'arm_rmse_deg':arm**.5,'delta_deg_mm':x.tolist(),'initial_depth_m':depth,'gap_weight':args.gap_weight}
            history.append(record)
            def save(path):
                candidate=copy.deepcopy(c);candidate['adapter']=ac;candidate['base_calibration']['world']['robot_base']['position']=(bp+[x[5]/1000,x[6]/1000,0]).tolist()
                candidate['worst_grasp_identification']={'objective':'mean stable held-gripper softL1 plus worst8 mean plus0.5armMSE; optional simulated bilateral contact-gap term at real hold onset; all72 calibration set','record':record,'per_episode':rows,'native_actions_unchanged':True,'shared_parameters_only':True,'measured_dimensions_and_gripper_zero_fixed':True}
                path.write_text(json.dumps(candidate,indent=2))
            if value<best:
                best=value;save(out/'best-state.json');print('BEST',calls,'loss',round(value,4),'screen',screen,'arm',round(arm**.5,4),'delta',np.round(x,3).tolist(),flush=True)
            # Keep task candidates only when native arm error improves on v7.
            if arm**.5<=.695 and (screen>best_screen[0] or (screen==best_screen[0] and value<best_screen[1])):
                best_screen=(screen,value);save(out/'best-screen.json');save(out/f'candidate-{calls:04d}.json');print('SCREEN',calls,screen,'loss',round(value,4),flush=True)
            (out/'trials.json').write_text(json.dumps(history,indent=2))
            if calls%20==0:print('PROGRESS',calls,'elapsed',round(time.time()-started),flush=True)
            return value
        objective(np.zeros(7))
        # Small bounded calibration changes, never episode/frame dependent.
        bounds=[(-1.5,1.5)]*5+[(-5,5)]*2
        if args.method=='local':
            simplex=[np.zeros(7)]
            for j,step in enumerate([.1]*5+[.4,.4]):
                plus=np.zeros(7);plus[j]=step;minus=-plus
                simplex.append(plus if objective(plus)<objective(minus)else minus)
            result=minimize(objective,np.zeros(7),method='Nelder-Mead',bounds=bounds,options={'initial_simplex':np.array(simplex),'maxfev':args.iterations*20,'xatol':.001,'fatol':.0001})
        else:
            result=differential_evolution(objective,bounds,x0=np.zeros(7),popsize=5,maxiter=args.iterations,seed=args.seed,polish=False,tol=.001)
        print('DONE',result.message,'best',best,'screen',best_screen,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='calibration/native-action-current.json');p.add_argument('--source',default='output/measured-bowl-task-v7');p.add_argument('--output',default='output/system-identification/worst-grasp-v8');p.add_argument('--iterations',type=int,default=12);p.add_argument('--seed',type=int,default=86);p.add_argument('--gap-weight',type=float,default=0.);p.add_argument('--method',choices=['global','local'],default='global');fit(p.parse_args())
