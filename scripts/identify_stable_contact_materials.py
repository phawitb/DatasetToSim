"""Identify shared contact materials using stable native grasp-state evidence."""
import argparse,copy,itertools,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from identify_dynamics import DatasetIndex,DATA,fast_trace
from strict_action_scene import build_model,episode_scene
from dataset_to_sim.strict_replay import NativeJointAdapter
from dataset_to_sim.identification import held_gripper_frames


def fit(path,output):
 c=json.loads(Path(path).read_text());a=NativeJointAdapter(**c['adapter']);ds=DatasetIndex(DATA);ids=[1,4,8,13,20,28,31,37,38,40,44,49,50,61,64,71];items=[]
 for i in ids:
  e=ds.episode(i);s=json.loads(Path(f'output/touch-zero-v1/episode-{i:02d}/scene-observation.json').read_text());m=build_model(episode_scene(c,s),c);items.append((e,m,held_gripper_frames(e.state_deg[:,5],e.action_deg[:,5])))
 rows=[];best=np.inf
 with ThreadPoolExecutor(max_workers=4)as pool:
  for jaw,table,torsion in itertools.product([.3,.6,1.,1.4],[.3,.6,.95,1.2],[.0001,.001,.005,.01]):
   for _,m,_ in items:
    m.geom_friction[m.geom_contype==2]=[jaw,torsion,.0001]
    for name in ['table','work_mat']:m.geom_friction[m.geom(name).id]=[table,.0002,.0001]
    m.pair_friction[:]=[jaw,jaw,torsion,.0001,.0001]
   def one(item):
    e,m,held=item;q,_=fast_trace(m,e,a);native=a.to_native(q);error=native[held,5]-e.state_deg[held,5];return float(np.mean(2*(np.sqrt(1+error**2)-1))+.1*np.mean(np.minimum((native[8:,:5]-e.state_deg[8:,:5])**2,9)))
   values=list(pool.map(one,items));loss=float(np.mean(values));row={'jaw_sliding':jaw,'table_sliding':table,'jaw_torsion_m':torsion,'loss':loss,'episode_losses':values};rows.append(row)
   if loss<best:
    best=loss;out=copy.deepcopy(c);out['contact_materials']={'jaw':[jaw,torsion,.0001],'cube':c['contact_materials']['cube'],'work_mat':[table,.0002,.0001],'table':[table,.0002,.0001]};out['stable_material_identification']={'training_episodes':ids,'objective':'native stable held-gripper softL1 plus0.1arm cappedMSE; no final outcome reward','best':row,'trials':len(rows),'measured_properties':False};Path(output).write_text(json.dumps(out,indent=2));print('best',len(rows),loss,jaw,table,torsion,flush=True)
   Path(output).with_name('stable-material-grid.json').write_text(json.dumps(rows,indent=2))
 print('DONE',len(rows),'best',best,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',default='output/system-identification/stable-material-v3.json');a=p.parse_args();fit(a.config,a.output)
