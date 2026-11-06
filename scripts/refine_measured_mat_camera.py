"""Fit the shared top camera to straight mat edges in initial RGB frames only.

Uses the measured405x405x2mm mat. No actions, robot states or task outcomes
enter the camera fit. This is a candidate, not an accepted calibration.
"""
import json,copy
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from benchmark72 import DatasetIndex,DATA,read_real
from dataset_to_sim.scene_estimation import project_cube


def robust_line(points):
 p=np.asarray(points,np.float32);vx,vy,x,y=cv2.fitLine(p,cv2.DIST_HUBER,0,.01,.01).ravel();return np.array([-vy,vx,vy*x-vx*y],float)

def intersection(a,b):
 p=np.cross(a,b);return p[:2]/p[2]

def edges(rgb):
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float);gray=cv2.GaussianBlur(gray,(5,5),.8);gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
 groups=[]
 for kind in ['top','right','bottom','left']:
  pts=[]
  if kind=='top':
   for x in range(205,505,3):
    y=int(np.argmax(-gy[20:55,x]))+20;pts.append([x,y])
  elif kind=='bottom':
   for x in list(range(145,235,2))+list(range(450,540,2)):
    bright=np.flatnonzero(gray[375:425,x]>160)
    if not len(bright):continue
    crossing=int(bright[0])+375;lo=max(375,crossing-4);hi=min(425,crossing+3)
    y=int(np.argmax(gy[lo:hi,x]))+lo;pts.append([x,y])
  else:
   for y in range(70,365,3):
    guess=(177+(y-32)*(118-177)/(392-32)) if kind=='left' else (528+(y-37)*(563-528)/(406-37))
    lo=int(guess)-10;hi=int(guess)+11
    x=int(np.argmax((-1 if kind=='left'else 1)*gx[y,lo:hi]))+lo;pts.append([x,y])
  groups.append(robust_line(pts))
 top,right,bottom,left=groups
 return np.array([intersection(top,left),intersection(top,right),intersection(bottom,right),intersection(bottom,left)])


def main():
 c=json.load(open('output/system-identification/shared-contacts-v8/candidate-043.json'));ds=DatasetIndex(DATA);ids=[0,12,24,36,44,48,51,60,71];observed=[];images=[]
 for i in ids:
  rgb=read_real(ds.episode(i),'top')[0];corners=edges(rgb);observed.append(corners);images.append(rgb)
 corners=np.median(observed,axis=0);cam=c['base_calibration']['cameras']['top'];mat=c['base_calibration']['world']['mat'];p=np.array(mat['position']);s=np.array(mat['size'])/2
 world=np.array([p+s*[1,1,1],p+s*[1,-1,1],p+s*[-1,-1,1],p+s*[-1,1,1]])
 R=Rotation.from_quat(np.array(cam['quaternion'])[[1,2,3,0]]).as_matrix();Rcv=np.diag([1,-1,-1])@R.T;rv=Rotation.from_matrix(Rcv).as_rotvec();t=-Rcv@cam['position'];focal=240/np.tan(np.deg2rad(cam['fovy']/2));K=np.array([[focal,0,320],[0,focal,240],[0,0,1.]])
 def project(x):return cv2.projectPoints(world,x[:3],x[3:6],K,None)[0][:,0,:]
 def residual(x):return (project(x)-corners).ravel()
 x0=np.r_[rv,t];r=least_squares(residual,x0,loss='soft_l1',f_scale=1,max_nfev=100);rot=Rotation.from_rotvec(r.x[:3]).as_matrix();cam['position']=(-rot.T@r.x[3:6]).tolist();cam['quaternion']=Rotation.from_matrix(rot.T@np.diag([1,-1,-1])).as_quat()[[3,0,1,2]].tolist()
 c['mat_camera_identification']={'initial_rgb_episodes':ids,'observed_corners_px':corners.tolist(),'corner_std_px':np.std(observed,axis=0).tolist(),'before_corner_rmse_px':float(np.sqrt(np.mean(residual(x0)**2))),'after_corner_rmse_px':float(np.sqrt(np.mean(residual(r.x)**2))),'fovy_fixed':cam['fovy'],'no_action_or_state_inputs':True,'mat_dimensions_m':mat['size'],'limitation':'Rounded corner locations extrapolated from straight edge lines; no lens-distortion calibration'}
 out=Path('output/system-identification');(out/'mat-camera-v9.json').write_text(json.dumps(c,indent=2));tiles=[]
 for i,im,obs in zip(ids,images,observed):
  for poly,color in [(obs,(0,255,0)),(project(r.x),(0,180,255))]:cv2.polylines(im,[np.int32(poly)],True,color,1)
  cv2.putText(im,f'EP{i}',(10,25),0,.7,(0,255,0),2);tiles.append(cv2.resize(im,(320,240)))
 cv2.imwrite(str(out/'mat-camera-v9-review.jpg'),cv2.cvtColor(np.vstack([np.hstack(tiles[k:k+3])for k in range(0,9,3)]),cv2.COLOR_RGB2BGR));print(json.dumps(c['mat_camera_identification'],indent=2),flush=True)

if __name__=='__main__':main()
