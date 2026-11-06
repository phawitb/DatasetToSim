"""Sensitivity between two identified motor parameter sets; never blend actions."""
import argparse,copy,json,itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from identify_dynamics import DatasetIndex,DATA,fast_trace
from strict_action_scene import build_model,episode_scene,containment
from dataset_to_sim.strict_replay import NativeJointAdapter
from dataset_to_sim.identification import held_gripper_frames


def main(args):
 c=json.loads(Path(args.config).read_text());other=json.loads(Path(args.other).read_text());a=NativeJointAdapter(**c['adapter']);ds=DatasetIndex(DATA);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);items=[]
 for i in range(72):
  e=ds.episode(i);s=json.loads(Path(f'{args.source}/episode-{i:02d}/scene-observation.json').read_text());cal=episode_scene(c,s);m=build_model(cal,c);items.append((e,m,held_gripper_frames(e.state_deg[:,5],e.action_deg[:,5]),cal))
 records=[];best=(-1,float('inf'))
 with ThreadPoolExecutor(max_workers=4)as pool:
  for trial,(arm,grip)in enumerate(itertools.product([0.,.25,.5,.75,1.],[0.,.5,1.])):
   weights=np.array([arm]*5+[grip]);dyn={k:((1-weights)*np.array(v)+weights*np.array(other['identified_dynamics'][k])).tolist()for k,v in c['identified_dynamics'].items()}
   for _,m,_,_ in items:
    m.actuator_gainprm[:,0]=dyn['kp'];m.actuator_biasprm[:,1]=-np.array(dyn['kp']);m.actuator_biasprm[:,2]=-np.array(dyn['kv']);m.actuator_delay[:]=dyn['delay_s'];m.dof_armature[:6]=dyn['armature'];m.dof_frictionloss[:6]=dyn['frictionloss']
   def one(item):
    e,m,held,cal=item;q,cube=fast_trace(m,e,a);native=a.to_native(q);error=native-e.state_deg
    return {'episode':e.index,'squared_error':np.sum(error**2,axis=0).tolist(),'frames':len(q),'held_squared_error':float(np.sum(error[held,5]**2)),'held_frames':int(held.sum()),'screen_pass':bool(cube[:,2].max()>=cube[0,2]+.04 and containment(m,q[-1],cube[-1],cal,c)['inside'])}
   rows=list(pool.map(one,items));rmse=np.sqrt(np.sum([r['squared_error']for r in rows],axis=0)/sum(r['frames']for r in rows));arm_rmse=float(np.sqrt(np.mean(rmse[:5]**2)));grip_rmse=float(rmse[5]);count=sum(r['screen_pass']for r in rows)
   row={'trial':trial,'arm_parameter_fraction':arm,'gripper_parameter_fraction':grip,'arm_rmse_deg':arm_rmse,'gripper_rmse_native':grip_rmse,'screen_passes':count,'screen_failures':[r['episode']for r in rows if not r['screen_pass']]};records.append(row)
   loss=arm_rmse+.1*grip_rmse
   if count>best[0]or(count==best[0]and loss<best[1]):
    best=(count,loss);candidate=copy.deepcopy(c);candidate['identified_dynamics']=dyn;candidate['motor_sensitivity']={'selection':row,'calibration_episodes':list(range(72)),'same_constants_all_episodes':True,'native_actions_not_blended_or_changed':True,'source_parameter_set':str(args.other)};(out/f'candidate-{trial:02d}.json').write_text(json.dumps(candidate,indent=2));(out/'best.json').write_text(json.dumps(candidate,indent=2))
   (out/'trials.json').write_text(json.dumps(records,indent=2));print(row,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',default='output/system-identification/shared-contacts-v8/candidate-043.json');p.add_argument('--other',default='calibration/native-action-state-fit.json');p.add_argument('--source',default='output/shared-contact-v8');p.add_argument('--output',default='output/system-identification/motor-response-v9');main(p.parse_args())
