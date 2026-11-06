"""Identify the gripper mechanical zero, preserving the encoder-derived scale."""
import copy,json
from dataclasses import replace
import numpy as np
from identify_dynamics import OUT,DATA,DatasetIndex,fast_trace
from strict_action_scene import build_model,episode_scene
from dataset_to_sim.strict_replay import NativeJointAdapter


def fit():
 c=json.loads((OUT/'contact-response-global-fit.json').read_text());ds=DatasetIndex(DATA);scenes=json.loads((OUT/'measured-2mm-scenes.json').read_text());ids=[i for i in range(72)if i%3!=0];items=[];original=c['adapter']['gripper_closed_model_deg']
 for i in ids:
  e=ds.episode(i);g=e.action_deg[:,5];close=np.flatnonzero((g<3)&(np.maximum.accumulate(g)>25));start=int(close[0])if len(close)else 0;opens=np.flatnonzero(g[start:]>20);stop=min(e.frame_count,start+int(opens[0])+1)if len(opens)else e.frame_count;e=replace(e,timestamps=e.timestamps[:stop],state_deg=e.state_deg[:stop],action_deg=e.action_deg[:stop],frame_indices=e.frame_indices[:stop],dataset_indices=e.dataset_indices[:stop]);m=build_model(episode_scene(c,scenes[i]),c);items.append((e,m,m.jnt_range[5].copy()))
 rows=[]
 for zero in [-13.,-12.,-11.,-10.,-9.,-8.,-7.]:
  ac=copy.deepcopy(c['adapter']);ac['gripper_closed_model_deg']=zero;a=NativeJointAdapter(**ac);held=[];free=[];lifts=0
  for e,m,limits in items:
   m.jnt_range[5]=limits+np.deg2rad(zero-original);q,cube=fast_trace(m,e,a);native=a.to_native(q);g=e.state_deg[:,5];mask=(g>3)&(g<14)&(e.action_deg[:,5]<3)&(np.maximum.accumulate(g)>25);err=native[mask,5]-g[mask]
   if len(err):held.append(float(np.mean(2*(np.sqrt(1+err**2)-1))))
   mask=(g>15)|((g<2)&(e.action_deg[:,5]<2));err=native[mask,5]-g[mask];free.append(float(np.mean(2*(np.sqrt(1+err**2)-1))));lifts+=bool(cube[:,2].max()>cube[0,2]+.04)
  loss=float(np.mean(held)+.5*np.mean(free));row={'closed_model_deg':zero,'loss':loss,'held_state_loss':float(np.mean(held)),'free_state_loss':float(np.mean(free)),'diagnostic_training_lifts':lifts};rows.append(row);(OUT/'gripper-reference-grid.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
 best=min(rows,key=lambda r:r['loss']);c['adapter']['gripper_closed_model_deg']=best['closed_model_deg'];c['gripper_reference_identification']={'training_episodes':ids,'objective':'held native gripper state softL1 +0.5free-state softL1; no object success reward','fixed_encoder_scale_deg_per_native':c['gripper_degrees_per_native_unit'],'best':best,'caveat':'Mechanical zero identified jointly with approximate contact geometry; image-only zero was not uniquely established'};(OUT/'gripper-reference-fit.json').write_text(json.dumps(c,indent=2))

if __name__=='__main__':fit()
