"""Contracts and calibrated geometry from SAM masks, without color segmentation."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np


def validate_point(point):
    p=np.asarray(point,dtype=float)
    if p.shape!=(2,) or not np.isfinite(p).all() or not (0<=p[0]<640 and 0<=p[1]<480):raise ValueError('Select a point inside the 640 × 480 captured image')
    return p.tolist()


def prompted_component(mask,point):
    point=validate_point(point);m=np.asarray(mask,dtype=np.uint8)
    if m.shape!=(480,640):raise ValueError('SAM mask has incorrect image dimensions')
    _,labels=cv2.connectedComponents((m>0).astype(np.uint8),connectivity=8)
    label=labels[int(point[1]),int(point[0])]
    if not label:raise ValueError('SAM mask does not contain the selected point')
    return labels==label


def bowl_from_mask(mask):
    contours,_=cv2.findContours(np.asarray(mask,dtype=np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    hull=cv2.convexHull(np.concatenate(contours));bx,by,bw,bh=cv2.boundingRect(hull)
    if len(hull)<5 or not (65<=bw<=190 and 55<=bh<=190):raise ValueError('To must match the calibrated 118 mm bowl profile')
    return list(cv2.fitEllipse(hull)[0])


def validate_vision_age(captured_at,now):
    if not 0<=now-captured_at<=5.:raise ValueError('SAM scene check took too long; scene image expired, retry execution')


def mask_record(mask,point,score):
    point=validate_point(point);m=np.asarray(mask).astype(bool)
    if m.shape!=(480,640) or not 30<=m.sum()<=276480:raise ValueError('SAM did not find a usable object mask')
    if not np.isfinite(score) or score<.6:raise ValueError('SAM confidence is too low; click a clearer part of the object')
    if not m[int(point[1]),int(point[0])]:raise ValueError('SAM mask does not contain the selected point')
    if not np.array_equal(prompted_component(m,point),m):raise ValueError('SAM mask contains disconnected objects; select again')
    contours,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour=max(contours,key=cv2.contourArea);x,y,w,h=cv2.boundingRect(contour)
    outline=cv2.approxPolyDP(contour,1.,True)[:,0,:].tolist()
    return {'point':point,'score':float(score),'area':int(m.sum()),'bbox':[x,y,x+w,y+h],'outline':outline,'mask_sha256':hashlib.sha256(m.astype(np.uint8).tobytes()).hexdigest()}


def validate_pair(a,b):
    a,b=np.asarray(a).astype(bool),np.asarray(b).astype(bool)
    if a.shape!=(480,640) or b.shape!=a.shape or min(a.sum(),b.sum())==0:raise ValueError('Both From and To masks are required')
    if np.count_nonzero(a&b)/min(a.sum(),b.sum())>.1:raise ValueError('Choose different objects for From and To')


def fit_mask_pose(mask,camera,initial,edge,support):
    from scipy.optimize import least_squares
    from dataset_to_sim.scene_estimation import project_cube,_sample,_distance
    contours,_=cv2.findContours(np.asarray(mask,dtype=np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    observed=cv2.convexHull(max(contours,key=cv2.contourArea))[:,0,:].astype(float);points=_sample(observed)
    def residual(pose):
        predicted=project_cube(camera,pose,edge,support)
        return np.r_[_distance(_sample(predicted),observed),_distance(points,predicted)]
    initial=np.asarray(initial,float);lo=initial+[-.02,-.02,-np.pi/2];hi=initial+[.02,.02,np.pi/2]
    fits=[least_squares(residual,initial+[0,0,shift],bounds=(lo,hi),diff_step=1e-4,loss='soft_l1',f_scale=1.,max_nfev=100) for shift in [0,-np.pi/4,np.pi/4]]
    best=min(fits,key=lambda r:r.cost);error=float(np.sqrt(np.mean(residual(best.x)**2)))
    if not best.success or error>3:raise ValueError('From mask does not match the calibrated cube profile; reselect the cube')
    return best.x,{'method':'measured cube fit to SAM mask only','boundary_rmse_px':error,'observed_hull':observed.tolist(),'projected_hull':project_cube(camera,best.x,edge,support).tolist()}


def scene_from_masks(config,source,target):
    from strict_action_scene import episode_scene
    validate_pair(source,target)
    contours,_=cv2.findContours(np.asarray(source,dtype=np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour=max(contours,key=cv2.contourArea);x,y,w,h=cv2.boundingRect(contour)
    if not (8<=w<=65 and 8<=h<=65):raise ValueError('From must match the calibrated 19.5 mm cube profile')
    corners=cv2.boxPoints(cv2.minAreaRect(contour));center=corners.mean(axis=0)
    bowl=bowl_from_mask(target)
    scene={'cube_pixel':center.tolist(),'cube_pixel_corners':corners.tolist(),'cube_bbox':[x,y,w,h,int(np.count_nonzero(source))],'bowl_pixel':list(bowl),'method':'SAM 2.1 user-selected From/To masks; calibrated metric projection'}
    cal=episode_scene(config,scene);c=cal.world['cube'];q=c['quaternion'];support=cal.data['scene_estimation']['cube_support_z_m']
    pose,fit=fit_mask_pose(source,cal.camera('top'),[*c['position'][:2],2*np.arctan2(q[3],q[0])],c['size'],support)
    scene['cube_pose_from_initial_rgb']={'position':[float(pose[0]),float(pose[1]),support+c['size']/2+.0002],'quaternion':[float(np.cos(pose[2]/2)),0.,0.,float(np.sin(pose[2]/2))]}
    scene['cube_pose_fit']=fit
    return scene


def load_selection(folder):
    folder=Path(folder);path=folder/'selection.json'
    if not path.exists():raise ValueError('Select From and To with SAM before generating a plan')
    selection=json.loads(path.read_text())
    if selection.get('image_sha256')!=hashlib.sha256((folder/'top.jpg').read_bytes()).hexdigest():raise ValueError('Capture changed; select From and To again')
    if not all(role in selection for role in ['from','to']):raise ValueError('Select both From and To with SAM')
    masks=[]
    for role in ['from','to']:
        path=(folder/selection[role]['mask_file']).resolve()
        if path.parent!=folder.resolve():raise ValueError('Invalid mask path')
        m=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if m is None:raise ValueError('Saved SAM mask is missing')
        m=m>0
        if hashlib.sha256(m.astype(np.uint8).tobytes()).hexdigest()!=selection[role]['mask_sha256']:raise ValueError('Saved SAM mask changed; select again')
        masks.append(m)
    validate_pair(*masks)
    return selection,masks
