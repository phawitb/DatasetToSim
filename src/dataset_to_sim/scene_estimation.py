"""Initial-image cube pose using its measured dimensions and a fixed camera.

No actions, later observations, grasp events, or simulation outcomes enter this
estimator. The fit uses the complete visible box silhouette, including sides.
"""
import itertools
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def project_cube(camera,xy_yaw,edge,support):
    corners=np.array(list(itertools.product([-edge/2,edge/2],[-edge/2,edge/2],[0,edge])))
    world=corners@Rotation.from_euler('z',xy_yaw[2]).as_matrix().T+[xy_yaw[0],xy_yaw[1],support]
    rotation=Rotation.from_quat(np.array(camera['quaternion'])[[1,2,3,0]]).as_matrix()
    points=(world-np.array(camera['position']))@rotation@np.diag([1,-1,-1])
    if np.any(points[:,2]<=0):raise ValueError('Cube is behind the camera')
    focal=240/np.tan(np.deg2rad(camera['fovy']/2));pixels=points[:,:2]/points[:,2:]*focal+[320,240]
    return pixels[ConvexHull(pixels).vertices]


def _sample(polygon,n=32):
    p=np.asarray(polygon);v=np.roll(p,-1,axis=0)-p;length=np.linalg.norm(v,axis=1);cum=np.r_[0,np.cumsum(length)];t=np.arange(n)*cum[-1]/n;idx=np.searchsorted(cum,t,side='right')-1
    return p[idx]+((t-cum[idx])/length[idx])[:,None]*v[idx]


def _distance(points,polygon):
    a=np.asarray(polygon);v=np.roll(a,-1,axis=0)-a;delta=points[:,None]-a[None];t=np.clip(np.sum(delta*v,axis=2)/np.sum(v*v,axis=1),0,1)
    return np.sqrt(np.min(np.sum((delta-t[:,:,None]*v)**2,axis=2),axis=1)+1e-12)


def _top_face_orientation(cut, origin, camera, height):
    mask=(((cut[:,:,0]<8)|(cut[:,:,0]>160))&(cut[:,:,1]>65)&(cut[:,:,2]>175)).astype('uint8')
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    report={'used':False}
    if not contours:return report
    hull=cv2.convexHull(max(contours,key=cv2.contourArea))
    if cv2.contourArea(hull)<25:return report
    quad=None
    for fraction in np.linspace(.01,.12,23):
        approximation=cv2.approxPolyDP(hull,fraction*cv2.arcLength(hull,True),True)
        if len(approximation)==4:
            quad=approximation[:,0,:].astype(float)+origin
            break
    if quad is None:return report
    focal=240/np.tan(np.deg2rad(camera['fovy']/2))
    rotation=Rotation.from_quat(np.array(camera['quaternion'])[[1,2,3,0]]).as_matrix()
    rays=np.c_[(quad-[320,240])/focal,np.ones(4)]@np.diag([1,-1,-1])@rotation.T
    position=np.asarray(camera['position'])
    world=position+rays*((height-position[2])/rays[:,2])[:,None]
    edges=np.roll(world[:,:2],-1,axis=0)-world[:,:2]
    orientation=np.mean(np.exp(4j*np.arctan2(edges[:,1],edges[:,0])))
    report.update(used=bool(abs(orientation)>.85),confidence=float(abs(orientation)),yaw_rad=float(np.angle(orientation)/4),quad_px=quad.tolist())
    return report


def fit_cube_silhouette(rgb,camera,scene,initial,edge,support,use_top_face=False):
    x,y,w,h=map(int,scene['cube_bbox'][:4]);margin=4;x=max(0,x-margin);y=max(0,y-margin);w+=2*margin;h+=2*margin
    cut=cv2.cvtColor(rgb[y:y+h,x:x+w],cv2.COLOR_RGB2HSV)
    mask=(((cut[:,:,0]<4)|(cut[:,:,0]>162))&(cut[:,:,1]>80)&(cut[:,:,2]>90)).astype('uint8')
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:raise ValueError('No cube silhouette in initial RGB region')
    observed=cv2.convexHull(max(contours,key=cv2.contourArea))[:,0,:].astype(float)+[x,y]
    points=_sample(observed)
    def residual(pose):
        predicted=project_cube(camera,pose,edge,support)
        return np.r_[_distance(_sample(predicted),observed),_distance(points,predicted)]
    initial=np.asarray(initial,dtype=float).copy();shifts=[0,-np.pi/4,np.pi/4]
    top=_top_face_orientation(cut,[x,y],camera,support+edge) if use_top_face else {'used':False}
    yaw_range=np.pi/2
    if top['used']:
        initial[2]+=np.angle(np.exp(4j*(top['yaw_rad']-initial[2])))/4
        yaw_range=np.deg2rad(3);shifts=[0]
    lo=initial+[-.02,-.02,-yaw_range];hi=initial+[.02,.02,yaw_range]
    candidates=[]
    for shift in shifts:
        result=least_squares(residual,initial+[0,0,shift],bounds=(lo,hi),diff_step=1e-4,loss='soft_l1',f_scale=1.,max_nfev=80)
        candidates.append(result)
    result=min(candidates,key=lambda r:r.cost)
    return result.x,{'method':'measured-size 3D cube silhouette fit to first RGB only','top_face_orientation':top,'boundary_rmse_px':float(np.sqrt(np.mean(residual(result.x)**2))),'observed_hull':observed.tolist(),'projected_hull':project_cube(camera,result.x,edge,support).tolist(),'optimizer_success':bool(result.success),'pose_xy_yaw':result.x.tolist()}
