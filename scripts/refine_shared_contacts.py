"""Shared material/solver sensitivity under frozen geometry and native actions."""
import argparse,copy,itertools,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from identify_dynamics import DatasetIndex,DATA,fast_trace
from strict_action_scene import build_model,episode_scene,containment
from dataset_to_sim.strict_replay import NativeJointAdapter
from dataset_to_sim.identification import held_gripper_frames


def run(args):
 c=json.loads(Path(args.config).read_text());a=NativeJointAdapter(**c['adapter']);ds=DatasetIndex(DATA);items=[];out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 for i in range(72):
  e=ds.episode(i);s=json.loads(Path(f'{args.source}/episode-{i:02d}/scene-observation.json').read_text());cal=episode_scene(c,s);m=build_model(cal,c);items.append((e,m,held_gripper_frames(e.state_deg[:,5],e.action_deg[:,5]),cal))
 trials=[];best=(0,np.inf);best_loss=np.inf
 values=list(itertools.product([.002,.004,.006,.01],[.5,.8,1.,1.4],[.3,.6,.9],[.001,.005]))
 with ThreadPoolExecutor(max_workers=4)as pool:
  for idx,(tc,jaw,table,torsion) in enumerate(values):
   for _,m,_,_ in items:
    m.geom_friction[m.geom_contype==2]=[jaw,torsion,.0001]
    for name in ['table','work_mat']:m.geom_friction[m.geom(name).id]=[table,.0002,.0001]
    m.pair_friction[:]=[jaw,jaw,torsion,.0001,.0001]
    m.geom_solref[m.geom_contype>0]=[tc,1];m.pair_solref[:]=[tc,1]
   def one(item):
    e,m,held,cal=item;q,cube=fast_trace(m,e,a);native=a.to_native(q);error=native[held,5]-e.state_deg[held,5]
    grip=float(np.mean(2*(np.sqrt(1+error**2)-1)));arm=float(np.mean((native[:,:5]-e.state_deg[:,:5])**2));lift=float(cube[:,2].max()-cube[0,2]);inside=containment(m,q[-1],cube[-1],cal,c)['inside']
    return {'episode':e.index,'held_loss':grip,'arm_mse':arm,'screen_lift_and_inside':bool(lift>=.04 and inside)}
   rows=list(pool.map(one,items));losses=np.array([r['held_loss']for r in rows]);arm=float(np.mean([r['arm_mse']for r in rows]));loss=float(losses.mean()+np.sort(losses)[-8:].mean()+.5*arm);count=sum(r['screen_lift_and_inside']for r in rows)
   row={'trial':idx,'time_constant_s':tc,'jaw_sliding':jaw,'table_sliding':table,'jaw_torsion_m':torsion,'loss':loss,'arm_rmse_deg':arm**.5,'screen_passes':count,'screen_failures':[r['episode']for r in rows if not r['screen_lift_and_inside']]};trials.append(row)
   def save(path):
    x=copy.deepcopy(c);x['contact_model']['solref']=[tc,1];x['contact_materials']['jaw']=[jaw,torsion,.0001]
    for name in ['table','work_mat']:x['contact_materials'][name]=[table,.0002,.0001]
    x['shared_contact_refinement']={'calibration_episodes':list(range(72)),'selection':row,'per_episode':rows,'physical_materials_are_unmeasured_estimates':True,'native_actions_unchanged':True,'shared_parameters_only':True};path.write_text(json.dumps(x,indent=2))
   if loss<best_loss:best_loss=loss;save(out/'best-state.json')
   if arm**.5<.70 and (count>best[0] or (count==best[0] and loss<best[1])):
    best=(count,loss);save(out/'best-screen.json');save(out/f'candidate-{idx:03d}.json');print('BEST',row,flush=True)
   (out/'trials.json').write_text(json.dumps(trials,indent=2))
   if idx%12==0:print('PROGRESS',idx,'/',len(values),flush=True)
 print('DONE',best,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',default='output/measured-bowl-task-v7');p.add_argument('--output',default='output/system-identification/shared-contacts-v8');run(p.parse_args())
