"""Automatic SAM selection after Capture: find every modelled object on the mat.

Candidate prompt points come from the image only (objects differ from the dark
mat); SAM segments each candidate and the measured object models decide what it
is. Nothing here moves the robot. The user can still change any role by clicking,
and plan, simulation and execution keep their own checks.
"""
import cv2,numpy as np
from studio_sam_geometry import (cube_pose_from_mask,tape_pose_from_mask,bowl_pose_from_mask,pixel_to_mat,
                                 validate_placement,PLACE_CLEARANCE_M)

MAX_CANDIDATES=40


def mat_threshold(hsv):
    """Brightness separating the dark mat from the white table, from this image's
    typical mat brightness, so daylight or lamp changes do not cut the mat apart."""
    v=hsv[:,:,2].ravel();dark=np.sort(v)[:int(.4*len(v))]  # the mat covers well over 40% of the view
    return float(np.clip(np.median(dark)+35,110,170))


def mat_saturation(hsv):
    """Typical saturation of the mat (warm light tints it), from its dark pixels."""
    v=hsv[:,:,2].ravel();order=np.argsort(v)[:int(.4*len(v))]
    return float(np.median(hsv[:,:,1].ravel()[order]))


def robot_pixels(rgb):
    """Orange printed arm parts, dilated to cover cables and the gripper."""
    hsv=cv2.cvtColor(np.asarray(rgb,np.uint8),cv2.COLOR_RGB2HSV);h,s,v=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    orange=((h>=8)&(h<=25)&(s>110)&(v>90)).astype(np.uint8)
    # Only large orange parts are the arm; a red cube's shaded side is also orange-ish.
    n,labels,stats,_=cv2.connectedComponentsWithStats(orange)
    arm=np.isin(labels,[i for i in range(1,n) if stats[i,4]>=1500]).astype(np.uint8)
    return cv2.dilate(arm,np.ones((31,31),np.uint8)).astype(bool)


def candidate_points(rgb,mat_region=None):
    """Prompt points for blobs that stand out from the mat; robot pixels excluded."""
    hsv=cv2.cvtColor(np.asarray(rgb,np.uint8),cv2.COLOR_RGB2HSV);h,s=hsv[:,:,0],hsv[:,:,1];v=hsv[:,:,2].astype(np.int16)
    mat_s=mat_saturation(hsv)
    dark=cv2.morphologyEx(((v<mat_threshold(hsv))&(s<mat_s+40)).astype(np.uint8),cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(dark)
    if n<2:return []
    # The arm and its cables can cut the mat into pieces: keep every large dark piece.
    largest=int(stats[1:,4].max());mat=np.isin(labels,[i for i in range(1,n) if stats[i,4]>=max(5000,.15*largest)]).astype(np.uint8)
    region=np.zeros_like(mat);cv2.fillConvexPoly(region,cv2.convexHull(cv2.findNonZero(mat)),1)
    region=cv2.erode(region,np.ones((11,11),np.uint8))
    # The calibrated mat outline (when given) also covers mat corners hidden behind tall
    # objects, which the dark-pixel hull can cut off (a cup in a corner had no prompt).
    if mat_region is not None:region=np.maximum(region,cv2.erode(np.asarray(mat_region,np.uint8),np.ones((7,7),np.uint8)))
    background=cv2.medianBlur(np.where(mat>0,hsv[:,:,2],0).astype(np.uint8),31).astype(np.int16)
    robot=robot_pixels(rgb).astype(np.uint8)
    blobs=(((np.abs(v-background)>35)|(s>90)).astype(np.uint8)&region&(1-robot))
    blobs=cv2.morphologyEx(cv2.morphologyEx(blobs,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)),cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(blobs);points=[]
    for i in sorted(range(1,n),key=lambda i:-stats[i,4]):
        if stats[i,4]<80:continue
        component=(labels==i).astype(np.uint8);distance=cv2.distanceTransform(component,cv2.DIST_L2,3)
        y,x=np.unravel_index(np.argmax(distance),distance.shape);points.append([int(x),int(y)])
        if stats[i,4]>600:
            # A blob may join several objects (a roll beside the gripper, a cube in a bowl):
            # also prompt a sparse grid inside it.
            yy,xx=np.nonzero((distance>4)&((np.indices(distance.shape)[0]%24==0)&(np.indices(distance.shape)[1]%24==0)))
            extra=[[int(a),int(b)] for a,b in zip(xx,yy) if np.hypot(a-x,b-y)>15]
            # One prompt per ~1500 px: a blob that joins a box, a roll and a bowl still
            # gets a prompt on each of them.
            count=int(np.clip(stats[i,4]//1500,6,24))
            points.extend(extra[::max(1,len(extra)//count)][:count])
    # Painted cubes touching another object join its blob: prompt every vivid
    # coloured patch as well, before the blob grid points.
    vivid=((s>max(65,mat_s+45))&(v>70)&(((h<8)|(h>165))|((h>=14)&(h<=135)))).astype(np.uint8)&region&(1-robot)
    n,labels,stats,centres=cv2.connectedComponentsWithStats(cv2.morphologyEx(vivid,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)))
    colour=[[int(x),int(y)] for i,(x,y) in enumerate(centres) if i and 60<=stats[i,4]<=2500]
    return (colour+[p for p in points if all(np.hypot(p[0]-c[0],p[1]-c[1])>10 for c in colour)])[:MAX_CANDIDATES]


def CUBE_HUES(hue):
    """Painted cube colours by OpenCV hue (0-179)."""
    return {'red':((hue<10)|(hue>165)),'yellow':((hue>=14)&(hue<=23)),'green':((hue>=24)&(hue<=95)),'blue':((hue>=96)&(hue<=135))}


def cube_color(rgb,mask):
    hsv=cv2.cvtColor(np.asarray(rgb,np.uint8),cv2.COLOR_RGB2HSV)[np.asarray(mask,bool)]
    vivid=hsv[hsv[:,1]>45]
    if len(vivid)<max(10,.15*len(hsv)):return None
    hue=vivid[:,0];counts={name:np.count_nonzero(m) for name,m in CUBE_HUES(hue).items()}
    best=max(counts,key=counts.get)
    return best if counts[best]>.45*len(vivid) else None


def classify(config,rgb,mask):
    """Kind of a modelled object that matches the mask, or None."""
    m=np.asarray(mask,bool);contours,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return None
    contour=max(contours,key=cv2.contourArea);_,_,w,h=cv2.boundingRect(contour)
    (_,_),(rw,rh),_=cv2.minAreaRect(contour);fill=m.sum()/max(1.,rw*rh)
    if max(w,h)<=65 and fill>.6:
        try:
            world,fit,_=cube_pose_from_mask(config,m,label='Object')
            return {'kind':'cube','color':cube_color(rgb,m),'position':world['position'],'fit_rmse_px':fit['boundary_rmse_px']}
        except ValueError:pass
    if max(w,h)<=75 and fill>.55:
        try:
            # Two stacked cubes: SAM often returns the top face and the lower cube's sides together.
            world,fit,_=cube_pose_from_mask(config,m,elevated=True,label='Object')
            if fit.get('silhouette')=='stack':return {'kind':'stack','color':cube_color(rgb,m),'position':world['position'],'fit_rmse_px':fit['boundary_rmse_px']}
        except ValueError:pass
    if max(w,h)>=90:
        try:
            brightness=float(cv2.cvtColor(np.asarray(rgb,np.uint8),cv2.COLOR_RGB2HSV)[m][:,2].mean())
            if brightness>140:return {'kind':'bowl','position':bowl_pose_from_mask(config,m)['position']}
        except ValueError:pass
    hull=cv2.convexHull(contour);circularity=4*np.pi*cv2.contourArea(hull)/max(1.,cv2.arcLength(hull,True)**2)
    if 20<=min(w,h) and max(w,h)<90 and circularity>.86 and min(rw,rh)/max(rw,rh,1)>.75:
        try:
            world,fit,_=tape_pose_from_mask(config,m,label='Object')
            return {'kind':'tape','position':world['position'],'fit_rmse_px':fit['boundary_rmse_px']}
        except ValueError:pass
    return None


def deduplicate(objects):
    """Keep the higher-scoring of overlapping masks and drop objects inside a bowl."""
    kept=[]
    for item in sorted(objects,key=lambda o:-o['result']['score']):
        m=item['mask']
        if any(np.count_nonzero(m&k['mask'])/max(1,min(m.sum(),k['mask'].sum()))>.3 for k in kept):continue
        kept.append(item)
    bowls=[]
    for item in kept:
        if item['info']['kind']=='bowl':
            hull=np.zeros((480,640),np.uint8);cv2.fillConvexPoly(hull,cv2.convexHull(cv2.findNonZero(item['mask'].astype(np.uint8))),1);bowls.append(hull>0)
    # A cube or roll already in a bowl is reached only through that bowl: the bowl is its obstacle.
    return [item for item in kept if item['info']['kind']=='bowl' or not any(np.count_nonzero(item['mask']&b)>.6*item['mask'].sum() for b in bowls)]


def detect_objects(client,image,config,rgb,decode):
    """SAM on every candidate point; only masks that fit a measured object model survive."""
    found=[];rejected=[];robot=robot_pixels(rgb)
    for point in candidate_points(rgb):
        if any(m[point[1],point[0]] for m in [item['mask'] for item in found]+rejected):continue
        try:result=client.predict(image,point)
        except ValueError:continue
        mask=decode(result)
        info=None if np.count_nonzero(mask&robot)>.05*mask.sum() else classify(config,rgb,mask)  # arm-attached masks are not objects
        if info:found.append({'point':point,'result':result,'mask':mask,'info':info})
        else:rejected.append(mask)
    return deduplicate(found)


def _stacked_score(rgb,item):
    """Other-colour cube pixels just around a cube: the sides of a lower cube."""
    color=item['info'].get('color')
    ring=cv2.dilate(item['mask'].astype(np.uint8),np.ones((15,15),np.uint8)).astype(bool)
    hsv=cv2.cvtColor(np.asarray(rgb,np.uint8),cv2.COLOR_RGB2HSV)[ring&~item['mask']]
    vivid=hsv[hsv[:,1]>45];hue=vivid[:,0]
    red=np.count_nonzero((hue<10)|(hue>165));green=np.count_nonzero((hue>=20)&(hue<=95))
    return green if color=='red' else red if color=='green' else max(red,green)


def free_mat(rgb,masks=()):
    """Pixels of plain mat: not the arm, not any object (with a margin)."""
    hsv=cv2.cvtColor(np.asarray(rgb,np.uint8),cv2.COLOR_RGB2HSV)
    free=((hsv[:,:,2]<mat_threshold(hsv))&(hsv[:,:,1]<mat_saturation(hsv)+40)).astype(np.uint8)&~robot_pixels(rgb).astype(np.uint8)&1
    for m in masks:free&=~cv2.dilate(np.asarray(m,np.uint8),np.ones((21,21),np.uint8))&1
    return cv2.erode(free,np.ones((25,25),np.uint8)).astype(bool)


def auto_place(config,source_xy,obstacles,free=None):
    """Nearest reachable empty-mat spot about 100 mm from the stack; `free` is an
    image mask the spot must lie in (plain mat seen by the top camera)."""
    from studio_sam_geometry import mat_to_pixel
    edge=config['base_calibration']['world']['cube']['size']
    mat=config['base_calibration']['world']['mat'];base=np.asarray(config['base_calibration']['world']['robot_base']['position'][:2])
    best=None
    for x in np.arange(mat['position'][0]-.17,mat['position'][0]+.171,.01):
        for y in np.arange(mat['position'][1]-.17,mat['position'][1]+.171,.01):
            xy=np.array([x,y]);reach=np.linalg.norm(xy-base)
            if not .15<=reach<=.30:continue
            try:validate_placement(config,xy,source_xy,obstacles)
            except ValueError:continue
            if free is not None:
                u,v=np.round(mat_to_pixel(config,xy,edge/2)).astype(int)
                if not (0<=u<640 and 0<=v<480 and free[v,u]):continue
            cost=abs(np.linalg.norm(xy-np.asarray(source_xy[:2]))-.10)+.2*abs(reach-.22)
            if best is None or cost<best[0]:best=(cost,xy)
    return None if best is None else best[1]


def assign_roles(task,objects,rgb=None):
    """Indices for From and To (None for unstack's place) and notes for missing roles."""
    cubes=[i for i,o in enumerate(objects) if o['info']['kind']=='cube']
    bowls=sorted([i for i,o in enumerate(objects) if o['info']['kind']=='bowl'],key=lambda i:-objects[i]['mask'].sum())
    tapes=[i for i,o in enumerate(objects) if o['info']['kind']=='tape']
    def by_color(indices,color):
        return sorted(indices,key=lambda i:(objects[i]['info'].get('color')!=color,-objects[i]['result']['score']))
    source=target=None;notes=[]
    if task=='bowl':
        source=(by_color(cubes,'red') or [None])[0];target=(bowls or [None])[0]
    elif task=='tape':
        source=(sorted(tapes,key=lambda i:-objects[i]['result']['score']) or [None])[0];target=(bowls or [None])[0]
    elif task=='stack':
        ordered=by_color(cubes,'red');source=ordered[0] if ordered else None
        rest=by_color([i for i in cubes if i!=source],'green');target=rest[0] if rest else None
    elif task=='unstack':
        stacks=[i for i,o in enumerate(objects) if o['info']['kind']=='stack']
        if stacks:source=max(stacks,key=lambda i:objects[i]['result']['score'])
        elif cubes:source=max(cubes,key=lambda i:(_stacked_score(rgb,objects[i]) if rgb is not None else 0,objects[i]['result']['score']))
    names={'bowl':('cube','bowl'),'tape':('tape roll','bowl'),'stack':('cube to pick','cube to stack on'),'unstack':('top cube of the stack',None)}[task]
    if source is None:notes.append(f'No {names[0]} found; click it as From')
    if names[1] and target is None:notes.append(f'No {names[1]} found; click it as To')
    return source,target,notes
