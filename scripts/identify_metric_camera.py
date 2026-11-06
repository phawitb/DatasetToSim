"""Metric top-camera calibration from known 19.5 mm cube silhouettes.

Only first-frame RGB and cube dimensions are used; no robot actions or outcomes.
World XY camera origin is a gauge held at the previous reference.
"""
import json,sys,copy
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from identify_dynamics import OUT,BASE,DATA,DatasetIndex
from strict_action_scene import episode_scene
from episode0 import read_real


def samples(poly,n=24):
 p=np.asarray(poly);v=np.roll(p,-1,axis=0)-p;length=np.linalg.norm(v,axis=1);cum=np.r_[0,np.cumsum(length)];t=np.arange(n)*cum[-1]/n;idx=np.searchsorted(cum,t,side='right')-1
 return p[idx]+((t-cum[idx])/length[idx])[:,None]*v[idx]


def distances(points,polygon):
 a=np.asarray(polygon);v=np.roll(a,-1,axis=0)-a;delta=points[:,None]-a[None];t=np.clip(np.sum(delta*v,axis=2)/np.sum(v*v,axis=1),0,1)
 return np.sqrt(np.min(np.sum((delta-t[:,:,None]*v)**2,axis=2),axis=1)+1e-10)


def fit():
 c=json.loads((BASE/'settings.json').read_text());ds=DatasetIndex(DATA);observations=[];cam=c['base_calibration']['cameras']['top'];oldR=Rotation.from_quat(np.array(cam['quaternion'])[[1,2,3,0]]).as_matrix();center=np.array(cam['position']);tiles=[]
 # Exclude severe first-frame occlusion identified by mask/fixture overlap.
 for i in range(72):
  scene=json.loads((BASE/f'episode-{i:02}/scene-observation.json').read_text());rgb=read_real(ds.episode(i),'top')[0];hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);x,y,w,h,_=scene['cube_bbox'];margin=4;x-=margin;y-=margin;w+=2*margin;h+=2*margin
  cut=hsv[y:y+h,x:x+w];mask=(((cut[:,:,0]<4)|(cut[:,:,0]>162))&(cut[:,:,1]>80)&(cut[:,:,2]>90)).astype('uint8');contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);hull=cv2.convexHull(max(contours,key=cv2.contourArea))[:,0,:]+[x,y]
  cal=episode_scene(c,scene);q=cal.world['cube']['quaternion'];yaw=2*np.arctan2(q[3],q[0]);observations.append({'episode':i,'hull':hull.tolist(),'initial':[*cal.world['cube']['position'][:2],yaw]})
  crop=rgb[y:y+h,x:x+w].copy();cv2.polylines(crop,[np.int32(hull)-[x,y]],True,(0,255,0),1);tile=cv2.resize(crop,(120,120));cv2.putText(tile,str(i),(3,15),0,.5,(0,255,0),1);tiles.append(tile)
 for page in range(3):cv2.imwrite(str(OUT/f'metric-masks-{page}.jpg'),cv2.cvtColor(np.vstack([np.hstack(tiles[k:k+8]) for k in range(page*24,page*24+24,8)]),cv2.COLOR_RGB2BGR))
 train=[o for o in observations if o['episode']%3!=0 and o['episode'] not in [8,9,18,19,22,53,54,55,56,65]]
 offsets=np.array([[a,b,z] for z in [0,.0195] for a in [-.00975,.00975] for b in [-.00975,.00975]])
 def prediction(x,local):
  R=Rotation.from_rotvec(np.deg2rad(x[:3])).as_matrix()@oldR;pos=center.copy();pos[2]=x[3];f=240/np.tan(np.deg2rad(x[4]/2));world=offsets@Rotation.from_euler('z',local[2]).as_matrix().T+[local[0],local[1],.005];p=(world-pos)@R@np.diag([1,-1,-1]);uv=p[:,:2]/p[:,2:]*f+[320,240];return cv2.convexHull(uv.astype('float32'))[:,0,:].astype(float)
 def residual(x):
  err=[]
  for j,o in enumerate(train):
   predicted=prediction(x,x[5+3*j:8+3*j]);observed=np.array(o['hull']);err.extend(distances(samples(predicted),observed));err.extend(distances(samples(observed),predicted))
  err.extend(x[:3]/30);err.extend([(x[3]-center[2])/.5,(x[4]-cam['fovy'])/40]);return np.array(err)
 initial=np.r_[[0,0,0,center[2],cam['fovy']],np.array([o['initial'] for o in train]).ravel()];lo=np.r_[[-15,-15,-15,.3,20],np.tile([-.2,-.5,-10],len(train))];hi=np.r_[[15,15,15,1.2,90],np.tile([.7,.5,10],len(train))]
 sparse=lil_matrix((48*len(train)+5,len(initial)));sparse[:,:5]=1
 for j in range(len(train)):sparse[j*48:(j+1)*48,5+3*j:8+3*j]=1
 print('initial meanpx',np.mean(residual(initial)),len(train),'training silhouettes',flush=True)
 result=least_squares(residual,initial,bounds=(lo,hi),jac_sparsity=sparse.tocsr(),diff_step=1e-3,loss='soft_l1',f_scale=1.,max_nfev=200,verbose=1,ftol=1e-5,xtol=1e-5)
 x=result.x;out=copy.deepcopy(c);camera=out['base_calibration']['cameras']['top'];camera['position'][2]=float(x[3]);camera['quaternion']=Rotation.from_matrix(Rotation.from_rotvec(np.deg2rad(x[:3])).as_matrix()@oldR).as_quat()[[3,0,1,2]].tolist();camera['fovy']=float(x[4]);out['metric_camera_identification']={'training_episodes':[o['episode']for o in train],'source':'19.5mm cube initial RGB silhouettes only; no actions/states/outcomes','mean_boundary_error_px':float(np.mean(residual(x)[:-5])),'camera_parameters':x[:5].tolist(),'message':result.message,'local_object_poses_are_nuisance_fit_only':True};(OUT/'metric-camera-fit.json').write_text(json.dumps(out,indent=2));(OUT/'metric-camera-observations.json').write_text(json.dumps(observations,indent=2));print(out['metric_camera_identification'],flush=True)

if __name__=='__main__':fit()
