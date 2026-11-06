"""Refit shared motor response against native states under current geometry."""
import argparse,copy,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from identify_dynamics import fast_trace,DatasetIndex,DATA
from strict_action_scene import episode_scene,build_model
from dataset_to_sim.strict_replay import NativeJointAdapter
from dataset_to_sim.identification import held_gripper_frames


def fit(config_path,source,output,budget):
    c=json.loads(Path(config_path).read_text());a=NativeJointAdapter(**c['adapter']);ds=DatasetIndex(DATA)
    ids=[1,4,8,13,20,28,31,37,38,40,44,49,50,61,64,71]
    items=[]
    for i in ids:
        e=ds.episode(i);scene=json.loads((Path(source)/f'episode-{i:02d}/scene-observation.json').read_text())
        items.append((e,build_model(episode_scene(c,scene),c),held_gripper_frames(e.state_deg[:,5],e.action_deg[:,5])))
    dyn=c['identified_dynamics'];losses=[];calls=0
    def set_motor(m,j,x):
        kp,kv,arm,friction,delay=np.exp(x[0]),np.exp(x[1]),np.exp(x[2]),x[3],x[4]
        m.actuator_gainprm[j,0]=kp;m.actuator_biasprm[j,1]=-kp;m.actuator_biasprm[j,2]=-kv
        m.dof_armature[j]=arm;m.dof_frictionloss[j]=friction;m.actuator_delay[j]=delay
    with ThreadPoolExecutor(max_workers=4) as pool:
        for j in [1,2,3,0,4,5]:
            best=[float('inf'),None]
            def one(item):
                e,m,held=item;q,_=fast_trace(m,e,a);native=a.to_native(q);mask=np.arange(len(q))>=8
                if j==5:mask &= ((e.state_deg[:,5]>18)&(e.action_deg[:,5]>18))|((e.state_deg[:,5]<2)&(e.action_deg[:,5]<2))
                error=native[:,j]-e.state_deg[:,j];loss=2*(np.sqrt(1+error**2)-1)
                return .8*np.mean(loss[mask])+.2*np.mean(loss[held]) if j<5 and held.any() else np.mean(loss[mask])
            def objective(x):
                nonlocal calls
                for _,m,_ in items:set_motor(m,j,x)
                value=float(np.mean(list(pool.map(one,items))));calls+=1
                if value<best[0]:best[:]=value,x.copy()
                return value
            x=np.array([np.log(dyn['kp'][j]),np.log(dyn['kv'][j]),np.log(dyn['armature'][j]),dyn['frictionloss'][j],dyn['delay_s'][j]])
            before=objective(x);simplex=np.tile(x,(6,1));simplex[1:]+=np.diag([.2,.25,.3,.025,.01])
            minimize(objective,x,method='Nelder-Mead',bounds=[(np.log(3),np.log(200)),(np.log(.015),np.log(8)),(np.log(.0001),np.log(.15)),(0,.3),(0,.15)],options={'maxfev':budget,'initial_simplex':simplex,'xatol':.001,'fatol':.00005})
            x=best[1]
            for key,value in zip(['kp','kv','armature','frictionloss','delay_s'],[np.exp(x[0]),np.exp(x[1]),np.exp(x[2]),x[3],x[4]]):dyn[key][j]=float(value)
            for _,m,_ in items:set_motor(m,j,x)
            losses.append({'joint':j,'before':before,'after':best[0]})
            c['native_motor_refinement']={'training_episodes':ids,'objective':'80%per-episode native softL1 plus20%stable held-state error for arms; free/empty gripper only; no object outcome reward','losses':losses,'calls':calls,'held_selector':'stable blocked observations, no minimum opening25 requirement','calibrated_geometry_unchanged':True}
            Path(output).write_text(json.dumps(c,indent=2));print('joint',j,'loss',before,'->',best[0],{k:dyn[k][j]for k in dyn},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='calibration/native-action-current.json');p.add_argument('--source',default='output/touch-zero-v1');p.add_argument('--output',default='output/system-identification/native-motor-v2.json');p.add_argument('--budget',type=int,default=90);args=p.parse_args();fit(args.config,args.source,args.output,args.budget)
