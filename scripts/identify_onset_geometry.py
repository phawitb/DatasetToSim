"""Calibrate shared references at reviewed first unilateral contact.

The cube remains at its initial RGB pose only at contact onset. Later closure
pushes are left to MuJoCo; no stationary-cube bilateral-stall assumption.
"""
import copy,json,sys
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from identify_dynamics import OUT,BASE,DATA,DatasetIndex
from strict_action_scene import episode_scene,build_model
from dataset_to_sim.strict_replay import NativeJointAdapter


def fit(seed=0):
 c=json.loads((OUT/'confirmed-visual-fit.json').read_text());scenes=json.loads((OUT/'confirmed-scenes.json').read_text());review=json.loads((OUT/'first-motion-review.json').read_text());ds=DatasetIndex(DATA)
 if seed>=3:
  c=json.loads((OUT/'measured-2mm-base.json').read_text());scenes=json.loads((OUT/'measured-2mm-scenes.json').read_text())
 lag=2.
 if seed>=2:
  wrist=json.loads((OUT/'wrist-jaw-identification.json').read_text());lag=wrist['exposure_delay_frames'];c['adapter']['gripper_closed_model_deg']=wrist['gripper_closed_model_deg'];c['wrist_camera']=wrist
 observations=[]
 for o in review['episodes']:
  i=o['episode'];f=o.get('first_contact_image_frame_estimate')
  if f is None:continue
  e=ds.episode(i);interval=o.get('first_contact_image_frame_interval_inclusive',[f,f]) if seed>=2 else [f,f];states=[]
  for image_frame in np.arange(interval[0],interval[1]+.01,.5):
   t=max(0,image_frame-lag);lo=int(t);hi=min(lo+1,e.frame_count-1);states.append(e.state_deg[lo]*(1-t+lo)+e.state_deg[hi]*(t-lo))
  cal=episode_scene(c,scenes[i]);observations.append({'episode':i,'states':states,'cube':cal.world['cube']['position'],'quaternion':cal.world['cube']['quaternion'],'review_frame':f})
 train=[o for o in observations if o['episode']%3!=0];test=[o for o in observations if o['episode']%3==0]
 m=build_model(episode_scene(c,scenes[0]),c);d=mujoco.MjData(m);base=m.body('base').id;cube=m.geom('cube_geom').id;bp=m.body_pos[base].copy();prior=np.r_[c['adapter']['arm_offset_deg'],[0,0],c['adapter']['gripper_closed_model_deg']]
 fixed=[i for i in range(m.ngeom)if m.geom_contype[i]==2 and m.geom_bodyid[i]==m.body('gripper').id and m.geom(i).name not in ['fixed_jaw_box1','fixed_jaw_box2']]
 moving=[i for i in range(m.ngeom)if m.geom_contype[i]==2 and m.geom_bodyid[i]==m.body('moving_jaw_so101_v1').id and m.geom(i).name!='moving_jaw_box1']
 constraints=np.array([ds.episode(i).state_deg[0]for i in range(72)]+[q for i in [1,4,8,13,20,28,40,61]for q in ds.episode(i).state_deg[::20]])
 best=[np.inf];calls=0
 def apply(x):
  ac=copy.deepcopy(c['adapter']);ac['arm_offset_deg']=x[:5].tolist();ac['gripper_closed_model_deg']=float(x[7]);m.body_pos[base]=bp+[x[5]/1000,x[6]/1000,0];return NativeJointAdapter(**ac)
 def contact(o,a):
  distances=[]
  for state in o['states']:
   d.qpos[:6]=a.to_model(state);d.qpos[6:9]=o['cube'];d.qpos[9:13]=o['quaternion'];mujoco.mj_forward(m,d)
   distances.append([min(mujoco.mj_geomDistance(m,d,cube,g,.1,None) for g in ids) for ids in [fixed,moving]])
  distances=np.array(distances);closest=distances[np.argmin(np.abs(distances[:,1]))].copy()
  if distances[:,1].min()<=0<=distances[:,1].max():closest[1]=0.
  return closest
 def residual(x):
  nonlocal calls
  a=apply(x);errors=[];distances=[]
  for o in train:
   left,right=contact(o,a);distances.append([o['episode'],left,right]);errors.extend([right/.0025,min(0,left+.0003)/.001])
  depths=[]
  for q in a.to_model(constraints):
   d.qpos[:6]=q;mujoco.mj_forward(m,d);depth=max([0.]+[-co.dist for co in d.contact if {int(m.geom_contype[co.geom1]),int(m.geom_contype[co.geom2])} in ({2,4},{8,4},{2,8},{2},{8})]);depths.append(depth);errors.append(max(0,depth-.0003)/.0004)
  errors.extend((x[:5]-prior[:5])/5);errors.extend(x[5:7]/15);errors.append((x[7]-prior[7])/2)
  errors=np.array(errors);loss=float(np.mean(2*(np.sqrt(1+errors**2)-1)));calls+=1
  if loss<best[0]:
   best[0]=loss;out=copy.deepcopy(c);out['adapter']['arm_offset_deg']=x[:5].tolist();out['adapter']['gripper_closed_model_deg']=float(x[7]);out['base_calibration']['world']['robot_base']['position']=m.body_pos[base].tolist();out['onset_geometry_identification']={'source':'independently reviewed first moving-jaw contact; original initial RGB cube pose; no bilateral stall fit','training_episodes':[o['episode']for o in train],'held_out_episodes':[o['episode']for o in test],'contact_distances_fixed_moving_m':distances,'held_out_contact_distances':[[o['episode'],*contact(o,a)]for o in test],'max_recorded_pose_overlap_m':max(depths),'parameters':x.tolist(),'loss':loss,'calls':calls,'camera_state_delay_frames_assumed':2,'limitations':['First-contact image brackets uncertain by 1-2 frames','Prior approach pushes cannot be completely excluded','Camera delay is an estimate','No episode-specific simulator parameters']};(OUT/f'onset-geometry-{seed}.json').write_text(json.dumps(out,indent=2));
   if calls%20==0:print('best',calls,'rms',np.sqrt(np.mean(np.array(distances)[:,2]**2))*1000,'depthmm',max(depths)*1000,flush=True)
  return errors
 lo=prior+np.array([-10]*5+[-20,-20,-1.5]);hi=prior+np.array([10]*5+[20,20,2.5]);x=prior.copy()
 if seed==1:x[:5]=[5.58,-5.05,-3.91,.176,-.044]
 def jac(x):return approx_derivative(residual,x,abs_step=np.array([.02]*5+[.03,.03,.02]),bounds=(lo,hi))
 r=least_squares(residual,x,jac=jac,bounds=(lo,hi),loss='soft_l1',max_nfev=120,ftol=1e-6,xtol=1e-6);print('DONE',r.message,r.x,flush=True)

if __name__=='__main__':fit(int(sys.argv[1])if len(sys.argv)>1 else 0)
