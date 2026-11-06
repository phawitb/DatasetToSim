"""Identify one fixed robot reference from physical grasp response.

Loss is measured gripper/arm state error during the original action prefix.
The simulator performs all pushing/grasping; no object pose tracking or task
success reward. Camera-derived initial cube poses and measured support stay fixed.
"""
import copy,json
from dataclasses import replace
import numpy as np,mujoco
from scipy.optimize import differential_evolution
from identify_dynamics import OUT,DATA,DatasetIndex,fast_trace
from strict_action_scene import episode_scene,build_model
from dataset_to_sim.strict_replay import NativeJointAdapter


def fit():
 c=json.loads((OUT/'contact-response-fit.json').read_text());ds=DatasetIndex(DATA);scenes=json.loads((OUT/'measured-2mm-scenes.json').read_text());ids=[1,8,20,31,37,49,50,70];items=[]
 for i in ids:
  e=ds.episode(i);g=e.action_deg[:,5];close=np.flatnonzero((g<3)&(np.maximum.accumulate(g)>25));start=int(close[0]) if len(close) else 0;opens=np.flatnonzero(g[start:]>20);stop=min(e.frame_count,start+int(opens[0])+1)if len(opens)else e.frame_count
  e=replace(e,frame_indices=e.frame_indices[:stop],dataset_indices=e.dataset_indices[:stop],timestamps=e.timestamps[:stop],state_deg=e.state_deg[:stop],action_deg=e.action_deg[:stop]);m=build_model(episode_scene(c,scenes[i]),c);items.append((e,m,m.jnt_range[:6].copy()))
 prior=np.array(c['adapter']['arm_offset_deg']);basepos=np.array(c['base_calibration']['world']['robot_base']['position']);initial_states=np.array([ds.episode(i).state_deg[0]for i in range(72)]);guard=items[0][1];gd=mujoco.MjData(guard);best=[np.inf];calls=0
 def objective(x):
  nonlocal calls
  ac=copy.deepcopy(c['adapter']);ac['arm_offset_deg']=(prior+x[:5]).tolist();a=NativeJointAdapter(**ac);losses=[];arm=[];per=[];gaps=[]
  for e,m,ranges in items:
   m.body_pos[m.body('base').id]=basepos+[x[5]/1000,x[6]/1000,0];m.jnt_range[:5]=ranges[:5]+np.deg2rad(x[:5])[:,None]
  depth=0.
  for q in a.to_model(initial_states):
   gd.qpos[:6]=q;mujoco.mj_forward(guard,gd);depth=max(depth,max([0.]+[-co.dist for co in gd.contact if {int(guard.geom_contype[co.geom1]),int(guard.geom_contype[co.geom2])} in ({2,4},{8,4},{2,8},{2},{8})]))
  if depth>.00025:return 20+depth*30000+float(x@x)*.01
  for e,m,_ in items:
   q,cube=fast_trace(m,e,a);native=a.to_native(q);g=e.state_deg[:,5];mask=(g>3)&(g<14)&(e.action_deg[:,5]<3)&(np.maximum.accumulate(g)>25);error=native[mask,5]-g[mask];loss=float(np.mean(2*(np.sqrt(1+error**2)-1)));losses.append(loss);arm.append(float(np.mean(np.minimum((native[8:,:5]-e.state_deg[8:,:5])**2,9))));per.append({'episode':e.index,'held_gripper_loss':loss,'held_gripper_rmse':float(np.sqrt(np.mean(error**2)))})
   # Dense calibration evidence: the recorded blocked gripper indicates contact.
   # Inspect the simulated cube at that moment, including its physical push.
   sample=int(np.flatnonzero(mask)[0]);probe=mujoco.MjData(m);probe.qpos[:6]=q[sample];probe.qpos[6:13]=cube[sample];mujoco.mj_forward(m,probe);cid=m.geom('cube_geom').id;gap=0.
   for body in ['gripper','moving_jaw_so101_v1']:
    ids_geom=[k for k in range(m.ngeom)if m.geom_contype[k]==2 and m.geom_bodyid[k]==m.body(body).id]
    distance=min(mujoco.mj_geomDistance(m,probe,cid,k,.1,None)for k in ids_geom);gap+=min(.03,max(0,distance))/.005
   gaps.append(gap);per[-1]['first_recorded_grasp_surface_gap_scaled']=gap
  value=float(np.mean(losses)+np.mean(gaps)+.1*np.mean(arm)+max(0,depth-.00005)*30000+.01*np.sum(x[:5]**2)+.001*np.sum(x[5:]**2));calls+=1
  if value<best[0]:
   best[0]=value;out=copy.deepcopy(c);out['adapter']=ac;out['base_calibration']['world']['robot_base']['position']=(basepos+[x[5]/1000,x[6]/1000,0]).tolist();out['contact_response_identification']={'training_episodes':ids,'source':'original action prefixes through release; observed held-gripper state and arm state errors','objective':'native gripper softL1 +simulated first-grasp jaw/cube surface gap +0.1arm capped MSE +reference priors +initial collision penalty; no final object outcome reward','loss':value,'calls':calls,'shared_reference_delta':x.tolist(),'per_episode_state_losses':per,'initial_max_depth_m':depth,'no_episode_specific_parameters':True,'no_action_modification':True};(OUT/'contact-response-global-fit.json').write_text(json.dumps(out,indent=2));print('best',calls,'loss',value,'delta',x.tolist(),'depth',depth,flush=True)
  return value
 x=np.zeros(7);start=objective(x);result=differential_evolution(objective,bounds=[(-.8,.8)]*4+[(-1.5,1.5),(-3,3),(-3,3)],x0=x,popsize=6,maxiter=8,seed=17,polish=False,tol=.002);print('DONE',result.message,'loss',start,'->',best[0],flush=True)

if __name__=='__main__':fit()
