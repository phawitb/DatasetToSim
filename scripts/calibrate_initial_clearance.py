"""Small shared reference correction to avoid impossible initial link overlap."""
import json,copy
import numpy as np,mujoco
from scipy.optimize import minimize
from identify_dynamics import OUT,DATA,DatasetIndex
from strict_action_scene import build_model,episode_scene
from dataset_to_sim.strict_replay import NativeJointAdapter


def fit():
 c=json.loads((OUT/'supported-2mm-v1.json').read_text());ds=DatasetIndex(DATA);scenes=json.loads((OUT/'measured-2mm-scenes.json').read_text());m=build_model(episode_scene(c,scenes[0]),c);d=mujoco.MjData(m);states=np.array([ds.episode(i).state_deg[0]for i in range(72)]);prior=np.array(c['adapter']['arm_offset_deg'])
 def depths(delta):
  ac=copy.deepcopy(c['adapter']);offset=prior.copy();offset[1:4]+=delta;ac['arm_offset_deg']=offset.tolist();a=NativeJointAdapter(**ac);values=[]
  for q in a.to_model(states):
   d.qpos[:6]=q;mujoco.mj_forward(m,d);values.append(max([0.]+[-co.dist for co in d.contact if {int(m.geom_contype[co.geom1]),int(m.geom_contype[co.geom2])} in ({2,4},{8,4},{2,8},{2},{8})]))
  return np.array(values)
 before=depths(np.zeros(3));r=minimize(lambda x:float(x@x),np.array([-.16,-.21,-.3]),method='COBYLA',bounds=[(-1,1)]*3,constraints=[{'type':'ineq','fun':lambda x:1000*(.00001-depths(x))}],options={'tol':1e-5,'catol':1e-7,'maxiter':500,'rhobeg':.05})
 if max(depths(r.x))>.000011:raise RuntimeError(str(r))
 offset=prior.copy();offset[1:4]+=r.x;c['adapter']['arm_offset_deg']=offset.tolist();c['initial_reference_clearance_identification']={'method':'minimum-norm shared joint reference correction under all72 initial-pose collision constraints; no action or per-frame corrections','correction_deg':r.x.tolist(),'before_max_penetration_m':float(max(before)),'after_max_penetration_m':float(max(depths(r.x))),'bound_deg':1.,'caveat':'Model reference adjustment within calibration uncertainty; source collision approximations are not measured geometry'};(OUT/'supported-clearance.json').write_text(json.dumps(c,indent=2));print(c['initial_reference_clearance_identification'])

if __name__=='__main__':fit()
