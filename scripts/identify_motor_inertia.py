"""Fit motor response including reflected inertia against native recorded states.

Shared parameters only. No action edits, object outcome reward, or trajectory targets.
"""
import json,copy,sys,time
import numpy as np,mujoco
from scipy.optimize import minimize
from identify_dynamics import OUT,BASE,DATA,DatasetIndex,fast_trace
from strict_action_scene import episode_scene,build_model
from dataset_to_sim.strict_replay import NativeJointAdapter


def fit():
 config=json.loads((OUT/'confirmed-visual-fit.json').read_text());adapter=NativeJointAdapter(**config['adapter']);ds=DatasetIndex(DATA);scenes=json.loads((OUT/'confirmed-scenes.json').read_text());ids=[1,4,8,13,20,28,40,61]
 items=[(ds.episode(i),build_model(episode_scene(config,scenes[i]),config))for i in ids];dyn=config['identified_dynamics'];dyn['armature']=items[0][1].dof_armature[:6].tolist();dyn['frictionloss']=items[0][1].dof_frictionloss[:6].tolist();calls=0;losses=[]
 for j in range(6):
  best=[float('inf')];chosen=[None]
  def set_motor(m,x):
   kp,kv,arm,fric,delay=np.exp(x[0]),np.exp(x[1]),np.exp(x[2]),x[3],x[4]
   m.actuator_gainprm[j,0]=kp;m.actuator_biasprm[j,1]=-kp;m.actuator_biasprm[j,2]=-kv;m.dof_armature[j]=arm;m.dof_frictionloss[j]=fric;m.actuator_delay[j]=delay
  def objective(x):
   nonlocal calls
   errors=[]
   for e,m in items:
    set_motor(m,x);states,_=fast_trace(m,e,adapter);native=adapter.to_native(states);mask=np.arange(e.frame_count)>=8
    if j==5:mask &= ((e.state_deg[:,5]>15)&(e.action_deg[:,5]>15))|((e.state_deg[:,5]<2)&(e.action_deg[:,5]<2))
    errors.extend(native[mask,j]-e.state_deg[mask,j])
   errors=np.array(errors);loss=float(np.mean(2*(np.sqrt(1+errors**2)-1)));calls+=1
   if loss<best[0]:best[0]=loss;chosen[0]=x.copy()
   return loss
  x=np.array([np.log(dyn['kp'][j]),np.log(dyn['kv'][j]),np.log(dyn['armature'][j]),dyn['frictionloss'][j],max(.005,dyn['delay_s'][j])]);start=objective(x)
  simplex=np.tile(x,(6,1));simplex[1:]+=np.diag([.3,.3,.4,.015,.015])
  result=minimize(objective,x,method='Nelder-Mead',bounds=[(np.log(3),np.log(200)),(np.log(.015),np.log(8)),(np.log(.0002),np.log(.15)),(0,.15),(0,.16)],options={'maxfev':160,'initial_simplex':simplex,'xatol':.002,'fatol':.0001})
  x=chosen[0]
  for key,value in zip(['kp','kv','armature','frictionloss','delay_s'],[np.exp(x[0]),np.exp(x[1]),np.exp(x[2]),x[3],x[4]]):dyn[key][j]=float(value)
  for e,m in items:set_motor(m,x)
  losses.append({'joint':j,'before':start,'after':best[0]});config['motor_identification']={'training_episodes':ids,'objective':'native joint-state soft-L1 error; free/empty gripper only; no task outcome reward','losses':losses,'calls':calls,'reflected_inertia_and_friction_estimated_not_measured':True};(OUT/'motor-inertia-fit.json').write_text(json.dumps(config,indent=2));print('joint',j,'loss',start,'->',best[0], 'params',{k:dyn[k][j] for k in dyn},flush=True)

if __name__=='__main__':fit()
