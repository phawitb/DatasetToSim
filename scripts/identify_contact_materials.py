"""Shared contact-material sweep against recorded held-gripper states."""
import json,itertools
import numpy as np,mujoco
from identify_dynamics import OUT,DATA,DatasetIndex,fast_trace
from strict_action_scene import build_model,episode_scene
from dataset_to_sim.strict_replay import NativeJointAdapter


def fit():
 c=json.loads((OUT/'supported-clearance.json').read_text());scenes=json.loads((OUT/'measured-2mm-scenes.json').read_text());ds=DatasetIndex(DATA);a=NativeJointAdapter(**c['adapter']);ids=[1,4,8,13,20,28,31,37,40,49,50,61,70];items=[(ds.episode(i),build_model(episode_scene(c,scenes[i]),c))for i in ids];rows=[]
 for jaw,table,torsion in itertools.product([.3,.6,.9,1.4],[.3,.6,.95],[.0002,.001,.01]):
  error=[];arm=[];lifted=0
  for e,m in items:
   m.geom_friction[m.geom_contype==2]=[jaw,torsion,.0001];m.geom('cube_geom').friction[:]=[.3,.0002,.0001]
   for name in ['work_mat','table']:m.geom(name).friction[:]=[table,.0002,.0001]
   m.pair_friction[:]=[jaw,jaw,torsion,.0001,.0001]
   q,cube=fast_trace(m,e,a);native=a.to_native(q);g=e.state_deg[:,5];mask=(g>3)&(g<14)&(e.action_deg[:,5]<3)&(np.maximum.accumulate(g)>25);error.extend(native[mask,5]-g[mask]);arm.extend((native[8:,:5]-e.state_deg[8:,:5]).ravel());lifted+=bool(cube[:,2].max()>cube[0,2]+.04)
  err=np.array(error);joint=np.array(arm);loss=float(np.mean(2*(np.sqrt(1+err**2)-1))+.1*np.mean(np.minimum(joint**2,9)));row={'jaw_sliding':jaw,'table_sliding':table,'jaw_torsion_m':torsion,'loss':loss,'held_gripper_rmse_native':float(np.sqrt(np.mean(err**2))),'diagnostic_training_lifts':lifted};rows.append(row);print(row,flush=True);(OUT/'contact-material-grid.json').write_text(json.dumps(sorted(rows,key=lambda r:r['loss']),indent=2))
 best=min(rows,key=lambda r:r['loss']);c['contact_materials']={'jaw':[best['jaw_sliding'],best['jaw_torsion_m'],.0001],'cube':[.3,.0002,.0001],'work_mat':[best['table_sliding'],.0002,.0001],'table':[best['table_sliding'],.0002,.0001]};c['material_identification']={'training_episodes':ids,'objective':'recorded held-gripper state error plus0.1weighted native arm error; not final bowl outcome','best':best,'caveat':'Estimated contact parameters; not independent material measurements'};(OUT/'contact-material-fit.json').write_text(json.dumps(c,indent=2))

if __name__=='__main__':fit()
