"""Robot reference calibration from stationary-cube wrist silhouettes.

Wrist extrinsics/zero are fixed from independent jaw-only image fitting. Only
pre-closure images and initial RGB cube poses are used; no success objective.
"""
import copy,json,itertools,sys
import cv2,mujoco,numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from identify_dynamics import OUT,DATA,DatasetIndex
from strict_action_scene import episode_scene,build_model
from dataset_to_sim.strict_replay import NativeJointAdapter
from dataset_to_sim.scene_estimation import _sample,_distance
from episode0 import read_real


def fit():
 c=json.loads((OUT/'measured-2mm-base.json').read_text());wrist=json.loads((OUT/'wrist-jaw-identification.json').read_text());c['wrist_camera']=wrist;c['adapter']['gripper_closed_model_deg']=wrist['gripper_closed_model_deg'];scenes=json.loads((OUT/'measured-2mm-scenes.json').read_text());ds=DatasetIndex(DATA)
 m=build_model(episode_scene(c,scenes[0]),c);d=mujoco.MjData(m);base=m.body('base').id;bp=m.body_pos[base].copy();cid=m.camera('wrist_cam').id;bid=m.cam_bodyid[cid];cr=Rotation.from_quat(m.cam_quat[cid][[1,2,3,0]]).as_matrix();focal=240/np.tan(np.deg2rad(wrist['fovy']/2));prior=np.r_[c['adapter']['arm_offset_deg'],0,0];lag=wrist['exposure_delay_frames'];records=[];rejected=0
 def apply(x):
  ac=copy.deepcopy(c['adapter']);ac['arm_offset_deg']=x[:5].tolist();m.body_pos[base]=bp+[x[5]/1000,x[6]/1000,0];return NativeJointAdapter(**ac)
 def project(record,adapter):
  d.qpos[:6]=adapter.to_model(record['state']);mujoco.mj_kinematics(m,d);br=d.xmat[bid].reshape(3,3);pos=d.xpos[bid]+br@m.cam_pos[cid];rot=br@cr;pts=(record['corners']-pos)@rot@np.diag([1,-1,-1]);uv=pts[:,:2]/pts[:,2:]*focal+[320,240]
  return uv[ConvexHull(uv).vertices]
 adapter=apply(prior)
 for i in range(72):
  e=ds.episode(i);g=e.action_deg[:,5];closed=np.flatnonzero((g<20)&(np.maximum.accumulate(g)>30));end=int(closed[0])-4 if len(closed)else e.frame_count//3
  if end<5:continue
  cube=scenes[i]['cube_pose_from_initial_rgb'];rotation=Rotation.from_quat(np.array(cube['quaternion'])[[1,2,3,0]]).as_matrix();corners=np.array(list(itertools.product([-.00975,.00975],repeat=3)))@rotation.T+np.array(cube['position'])-[0,0,.0002];images=read_real(e,'wrist')
  for frame in np.unique(np.linspace(3,end,8).astype(int)):
   t=max(0,frame-lag);lo=int(t);hi=min(lo+1,e.frame_count-1);state=e.state_deg[lo]*(1-t+lo)+e.state_deg[hi]*(t-lo);record={'episode':i,'frame':int(frame),'state':state,'corners':corners};pred=project(record,adapter);center=pred.mean(axis=0)
   hsv=cv2.cvtColor(images[frame],cv2.COLOR_RGB2HSV);red=(((hsv[:,:,0]<4)|(hsv[:,:,0]>162))&(hsv[:,:,1]>95)&(hsv[:,:,2]>70)).astype('uint8');contours,_=cv2.findContours(red,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);candidates=[]
   for contour in contours:
    area=cv2.contourArea(contour)
    if not 60<area<45000:continue
    hull=cv2.convexHull(contour)[:,0,:].astype(float);x,y,w,h=cv2.boundingRect(contour);dist=np.linalg.norm(hull.mean(axis=0)-center)
    if x<2 or y<2 or x+w>638 or y+h>478 or dist>150 or area/cv2.contourArea(hull.astype('float32'))<.6:continue
    candidates.append((dist,hull))
   if not candidates:rejected+=1;continue
   _,hull=min(candidates,key=lambda r:r[0]);record['hull']=hull;record['samples']=_sample(hull,16);records.append(record)
 train=[r for r in records if r['episode']%3!=0];test=[r for r in records if r['episode']%3==0];print('records',len(train),len(test),'rejected',rejected,flush=True)
 (OUT/'stationary-cube-observations.json').write_text(json.dumps([{k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in r.items()} for r in records],indent=2))
 constraints=np.array([ds.episode(i).state_deg[0]for i in range(72)]+[q for i in [1,4,8,13,20,28,40,61]for q in ds.episode(i).state_deg[::25]])
 calls=0;best=[np.inf]
 def image_error(records,a):
  errors=[]
  for r in records:
   p=project(r,a);errors.extend(_distance(_sample(p,16),r['hull']));errors.extend(_distance(r['samples'],p))
  return np.array(errors)
 def residual(x):
  nonlocal calls
  a=apply(x);pixels=image_error(train,a);errors=list(pixels);depths=[]
  for q in a.to_model(constraints):
   d.qpos[:6]=q;mujoco.mj_forward(m,d);depth=max([0.]+[-co.dist for co in d.contact if {int(m.geom_contype[co.geom1]),int(m.geom_contype[co.geom2])} in ({2,4},{8,4},{2,8},{2},{8})]);depths.append(depth);errors.append(max(0,depth-.0002)*100000)
  errors.extend((x[:5]-prior[:5])/3);errors.extend(x[5:]/10);errors=np.array(errors);loss=float(np.mean(2*(np.sqrt(1+(errors/3)**2)-1)));calls+=1
  if loss<best[0]:
   best[0]=loss;out=copy.deepcopy(c);out['adapter']['arm_offset_deg']=x[:5].tolist();out['base_calibration']['world']['robot_base']['position']=m.body_pos[base].tolist();held=image_error(test,a);out['stationary_cube_geometry_identification']={'method':'shared references from pre-closure wrist cube silhouettes; wrist camera fixed from independent jaw-only fit; initial cube poses from top RGB','training_records':len(train),'held_out_records':len(test),'training_median_boundary_error_px':float(np.median(pixels)),'held_out_median_boundary_error_px':float(np.median(held)),'training_rmse_px':float(np.sqrt(np.mean(pixels**2))),'held_out_rmse_px':float(np.sqrt(np.mean(held**2))),'max_recorded_pose_penetration_m':max(depths),'loss':loss,'calls':calls,'parameters':x.tolist(),'limits':['Pre-closure approach pushes or occlusion may still contaminate some observations','Detection gated within150px of initial model; not independent validation','Nominal camera latency1.3frames approximate']};(OUT/'stationary-cube-geometry.json').write_text(json.dumps(out,indent=2));
   if calls%10==0:print('best',calls,'median',np.median(pixels),'held',np.median(held),'depth',max(depths),flush=True)
  return errors
 lo=prior+[-10,-10,-10,-10,-10,-25,-25];hi=prior+[10,10,10,10,10,25,25]
 def jac(x):return approx_derivative(residual,x,abs_step=np.array([.02]*5+[.03,.03]),bounds=(lo,hi))
 r=least_squares(residual,prior,jac=jac,bounds=(lo,hi),loss='soft_l1',f_scale=3,max_nfev=120,ftol=1e-5,xtol=1e-5);print('DONE',r.message,r.x,flush=True)

if __name__=='__main__':fit()
