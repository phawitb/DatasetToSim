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


def project_box(camera,xy_yaw,edge,support,height,width=None):
    """Image hull of an edge × (width or edge) × height box on `support`."""
    import itertools
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    width=edge if width is None else width
    corners=np.array(list(itertools.product([-edge/2,edge/2],[-width/2,width/2],[0,height])))
    world=corners@Rotation.from_euler('z',xy_yaw[2]).as_matrix().T+[xy_yaw[0],xy_yaw[1],support]
    rotation=Rotation.from_quat(np.array(camera['quaternion'])[[1,2,3,0]]).as_matrix()
    points=(world-np.array(camera['position']))@rotation@np.diag([1,-1,-1])
    focal=240/np.tan(np.deg2rad(camera['fovy']/2));pixels=points[:,:2]/points[:,2:]*focal+[320,240]
    return pixels[ConvexHull(pixels).vertices]


def fit_mask_pose(mask,camera,initial,edge,support,height=None,width=None,tolerance_px=3.):
    """Cube (or, with `height`/`width`, any upright box) pose fit to the SAM silhouette."""
    from scipy.optimize import least_squares
    from dataset_to_sim.scene_estimation import project_cube as _cube,_sample,_distance
    project_cube=(lambda camera,pose,edge,support:project_box(camera,pose,edge,support,height or edge,width)) if (height or width) else _cube
    contours,_=cv2.findContours(np.asarray(mask,dtype=np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    observed=cv2.convexHull(max(contours,key=cv2.contourArea))[:,0,:].astype(float);points=_sample(observed)
    def residual(pose):
        predicted=project_cube(camera,pose,edge,support)
        return np.r_[_distance(_sample(predicted),observed),_distance(points,predicted)]
    initial=np.asarray(initial,float);lo=initial+[-.02,-.02,-np.pi/2];hi=initial+[.02,.02,np.pi/2]
    fits=[least_squares(residual,initial+[0,0,shift],bounds=(lo,hi),diff_step=1e-4,loss='soft_l1',f_scale=1.,max_nfev=100) for shift in [0,-np.pi/4,np.pi/4]]
    best=min(fits,key=lambda r:r.cost);error=float(np.sqrt(np.mean(residual(best.x)**2)))
    if not best.success or error>tolerance_px:raise ValueError('From mask does not match the calibrated cube profile; reselect the cube')
    return best.x,{'method':'measured cube fit to SAM mask only','boundary_rmse_px':error,'observed_hull':observed.tolist(),'projected_hull':project_cube(camera,best.x,edge,support).tolist()}


OBJECT_KINDS=('cube','stack','tape','bowl')
# Clearance of an empty-mat unstack release from every other object center (m).
PLACE_CLEARANCE_M={'cube':.06,'stack':.06,'tape':.075,'bowl':.11,'from':.07}


def _contour(mask):
    contours,_=cv2.findContours(np.asarray(mask,dtype=np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:raise ValueError('SAM mask is empty')
    return max(contours,key=cv2.contourArea)


def _yaw_quaternion(yaw):return [float(np.cos(yaw/2)),0.,0.,float(np.sin(yaw/2))]


def mat_top(config):
    mat=config['base_calibration']['world']['mat'];return mat['position'][2]+mat['size'][2]/2


def _top_camera(config):
    from scipy.spatial.transform import Rotation
    camera=config['base_calibration']['cameras']['top'];position=np.array(camera['position'])
    R=np.diag([1,-1,-1])@Rotation.from_quat(np.array(camera['quaternion'])[[1,2,3,0]]).as_matrix().T
    focal=240/np.tan(np.deg2rad(camera['fovy']/2));K=np.array([[focal,0,320],[0,focal,240],[0,0,1.]])
    return K,R,position


def pixel_to_mat(config,pixel,height=0.):
    """Metric XY of an image pixel on a horizontal plane `height` above the mat top."""
    from strict_action_scene import ground_point
    K,R,position=_top_camera(config)
    return np.asarray(ground_point(pixel,K,R,position,mat_top(config)+height),float)[:2]


def mat_to_pixel(config,xy,height=0.):
    K,R,position=_top_camera(config)
    camera=R@(np.r_[np.asarray(xy,float)[:2],mat_top(config)+height]-position);uv=K@camera
    return (uv[:2]/uv[2]).tolist()


def cube_pose_from_mask(config,mask,*,elevated=False,label='From'):
    """Measured 19.5 mm cube fit to one SAM mask; `elevated` cubes rest on another cube."""
    from strict_action_scene import episode_scene
    contour=_contour(mask);x,y,w,h=cv2.boundingRect(contour)
    if not (8<=w<=65 and 8<=h<=65):raise ValueError(f'{label} must match the calibrated 19.5 mm cube profile')
    corners=cv2.boxPoints(cv2.minAreaRect(contour));center=corners.mean(axis=0)
    cal=episode_scene(config,{'cube_pixel':center.tolist(),'cube_pixel_corners':corners.tolist(),'bowl_pixel':[320.,240.]})
    c=cal.world['cube'];q=c['quaternion'];support=cal.data['scene_estimation']['cube_support_z_m']
    initial=[*c['position'][:2],2*np.arctan2(q[3],q[0])]
    if elevated:
        # SAM may return the top cube alone or the whole stack (top face plus the
        # lower cube's sides): fit both measured shapes and keep the better one.
        fits=[]
        for height,base in [(None,support+c['size']+.0002),(2*c['size']+.0002,support)]:
            try:fits.append((fit_mask_pose(mask,cal.camera('top'),initial,c['size'],base,height),height))
            except ValueError:pass
        if not fits:raise ValueError(f'{label} must match the calibrated 19.5 mm cube on top of another cube')
        (pose,fit),height=min(fits,key=lambda f:f[0][1]['boundary_rmse_px']);fit['silhouette']='stack' if height else 'top cube'
        support+=c['size']+.0002
    else:pose,fit=fit_mask_pose(mask,cal.camera('top'),initial,c['size'],support)
    world={'position':[float(pose[0]),float(pose[1]),support+c['size']/2+.0002],'quaternion':_yaw_quaternion(pose[2]),'size':c['size']}
    return world,fit,{'cube_pixel':center.tolist(),'cube_pixel_corners':corners.tolist(),'cube_bbox':[x,y,w,h,int(np.count_nonzero(mask))]}


def _color_pixels(rgb,mask):
    hsv=cv2.cvtColor(np.asarray(rgb,np.uint8),cv2.COLOR_RGB2HSV);m=np.asarray(mask,bool);vivid=m&(hsv[:,:,1]>45)
    hue=hsv[:,:,0]
    from studio_sam_auto import CUBE_HUES
    return {name:vivid&m for name,m in CUBE_HUES(hue).items()}


def stack_poses_from_mask(config,mask,rgb,*,label='From'):
    """Top and lower cube of a hand-made stack, each with its own position and yaw.

    The top cube is fitted alone to the pixels of its own colour (its top face and
    visible sides). The lower cube is then fitted so the two cubes together explain
    the whole SAM silhouette. Stacked by hand, the top cube is often turned and
    shifted against the lower one.
    """
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from strict_action_scene import episode_scene
    from dataset_to_sim.scene_estimation import _sample,_distance
    colors=_color_pixels(rgb,mask);name=max(colors,key=lambda k:colors[k].sum());top_mask=colors[name]
    if top_mask.sum()<.2*np.asarray(mask,bool).sum():raise ValueError(f'{label}: the top cube colour is unclear; click the top cube')
    top_mask=cv2.morphologyEx(top_mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(top_mask);top_mask=labels==1+int(np.argmax(stats[1:,4]))
    contour=_contour(top_mask);corners=cv2.boxPoints(cv2.minAreaRect(contour));center=corners.mean(axis=0)
    cal=episode_scene(config,{'cube_pixel':center.tolist(),'cube_pixel_corners':corners.tolist(),'bowl_pixel':[320.,240.]})
    c=cal.world['cube'];q=c['quaternion'];edge=c['size'];support=cal.data['scene_estimation']['cube_support_z_m'];camera=cal.camera('top')
    initial=[*c['position'][:2],2*np.arctan2(q[3],q[0])]
    top,fit=fit_mask_pose(top_mask,camera,initial,edge,support+edge+.0002)
    observed=cv2.convexHull(_contour(mask))[:,0,:].astype(float);points=_sample(observed)
    top_hull=project_box(camera,top,edge,support+edge+.0002,edge)
    def residual(pose):
        both=np.vstack([top_hull,project_box(camera,pose,edge,support,edge)]);hull=both[ConvexHull(both).vertices]
        return np.r_[_distance(_sample(hull),observed),_distance(points,hull)]
    lo=np.r_[top[:2]-.012,top[2]-np.pi/4];hi=np.r_[top[:2]+.012,top[2]+np.pi/4]
    fits=[least_squares(residual,np.clip(top+[0,0,shift],lo,hi),bounds=(lo,hi),diff_step=1e-4,loss='soft_l1',f_scale=1.,max_nfev=100) for shift in [0,-np.pi/8,np.pi/8]]
    best=min(fits,key=lambda r:r.cost);error=float(np.sqrt(np.mean(residual(best.x)**2)))
    if error>3.5:raise ValueError(f'{label} does not match two stacked 19.5 mm cubes; click the top cube')
    top_world={'position':[float(top[0]),float(top[1]),support+edge+.0002+edge/2+.0002],'quaternion':_yaw_quaternion(top[2]),'size':edge}
    bottom_world={'position':[float(best.x[0]),float(best.x[1]),support+edge/2+.0002],'quaternion':_yaw_quaternion(best.x[2]),'size':edge}
    fit.update(silhouette='top cube by colour + lower cube',top_color=name,stack_boundary_rmse_px=error,
               top_offset_from_lower_mm=float(np.linalg.norm(top[:2]-best.x[:2])*1000),
               top_yaw_from_lower_deg=float(np.rad2deg((top[2]-best.x[2]+np.pi/4)%(np.pi/2)-np.pi/4)))
    x,y,w,h=cv2.boundingRect(_contour(mask))
    return top_world,bottom_world,fit,{'cube_pixel':center.tolist(),'cube_pixel_corners':corners.tolist(),'cube_bbox':[x,y,w,h,int(np.count_nonzero(mask))]}


def tape_pose_from_mask(config,mask,*,label='From'):
    from strict_action_scene import episode_scene
    from tape_object import fit_tape_mask,dimensions
    contour=_contour(mask);x,y,w,h=cv2.boundingRect(contour)
    if not (20<=w<=170 and 20<=h<=170):raise ValueError(f'{label} must match the 53 mm tape roll profile')
    corners=cv2.boxPoints(cv2.minAreaRect(contour));center=corners.mean(axis=0)
    cal=episode_scene(config,{'cube_pixel':center.tolist(),'cube_pixel_corners':corners.tolist(),'bowl_pixel':[320.,240.]})
    support=cal.data['scene_estimation']['cube_support_z_m']
    xy,fit=fit_tape_mask(mask,cal.camera('top'),cal.world['cube']['position'][:2],support)
    size=dimensions()
    world={'position':[float(xy[0]),float(xy[1]),support+size['width_m']/2+.0002],'quaternion':[1.,0.,0.,0.],
           'width':size['width_m'],'outer_diameter':size['outer_diameter_m']}
    return world,fit,{'cube_pixel':center.tolist(),'cube_pixel_corners':corners.tolist(),'cube_bbox':[x,y,w,h,int(np.count_nonzero(mask))]}


def bowl_pose_from_mask(config,mask):
    bowl=config['base_calibration']['world']['bowl']
    xy=pixel_to_mat(config,bowl_from_mask(mask),bowl['height']*config.get('bowl_projection_height_fraction',.5))
    return {'position':[float(xy[0]),float(xy[1]),mat_top(config)],'outer_radius':bowl['outer_radius'],'height':bowl['height']}


def obstacle_world(config,kind,mask):
    """World pose of a modelled workspace object that the arm must avoid."""
    if kind=='cube':world,fit,_=cube_pose_from_mask(config,mask,label='Obstacle cube')
    elif kind=='stack':
        # Two stacked cubes: one static box, centred half way up.
        world,fit,_=cube_pose_from_mask(config,mask,elevated=True,label='Obstacle stack')
        edge=world['size'];world['position'][2]-=edge/2+.0001;world['height']=2*edge+.0002
    elif kind=='tape':world,fit,_=tape_pose_from_mask(config,mask,label='Obstacle tape roll')
    elif kind=='bowl':world,fit=bowl_pose_from_mask(config,mask),None
    else:raise ValueError('Obstacles must be a cube, tape roll or bowl')
    return {'kind':kind,**world}|({'fit_rmse_px':fit['boundary_rmse_px']} if fit else {})


def _obstacle_list(obstacles):
    if obstacles is None:return []
    if isinstance(obstacles,np.ndarray):return [('bowl',obstacles)]  # legacy single bowl obstacle
    return [(item['kind'],item['mask']) if isinstance(item,dict) else tuple(item) for item in obstacles]


def is_place_point(target):
    return target is not None and not isinstance(target,np.ndarray) and np.asarray(target).shape==(2,)


def validate_placement(config,placement,source_xy,obstacles):
    """An unstack release spot must be on the mat and clear of every object."""
    mat=config['base_calibration']['world']['mat'];xy=np.asarray(placement,float)
    if np.any(np.abs(xy-mat['position'][:2])>np.asarray(mat['size'][:2])/2-.03):raise ValueError('To must be an empty spot on the mat, away from its edge')
    if np.linalg.norm(xy-np.asarray(source_xy[:2]))<PLACE_CLEARANCE_M['from']:raise ValueError('To must be at least 70 mm from the stack')
    for item in obstacles:
        if np.linalg.norm(xy-np.asarray(item['position'][:2]))<PLACE_CLEARANCE_M[item['kind']]:raise ValueError(f'To is too close to a {item["kind"]}; choose an empty spot')


def scene_from_masks(config,source,target,obstacles=None,*,task='bowl',bowl_present=None,rgb=None):
    """Metric scene from SAM masks.

    `obstacles` lists every other modelled object as (kind, mask). For stack and
    unstack the first bowl becomes the calibrated bowl body; all other objects
    become static collision obstacles. For unstack, `target` is the clicked
    empty-mat pixel where the top cube is released.
    """
    if task not in ['bowl','stack','tape','unstack']:raise ValueError('Choose Place cube in bowl, Stack on cube, Unstack cube or Place tape in bowl')
    obstacles=_obstacle_list(obstacles)
    for _,mask in obstacles:validate_pair(source,mask)
    if task=='unstack':
        if not is_place_point(target):raise ValueError('Select To: an empty spot on the mat for the top cube')
    else:
        validate_pair(source,target)
        for _,mask in obstacles:validate_pair(target,mask)
    bowls=[mask for kind,mask in obstacles if kind=='bowl']
    if task in ['stack','unstack']:
        if bowl_present is None:bowl_present=bool(bowls)
        if bowl_present and not bowls:raise ValueError('Select the bowl obstacle, or mark that no bowl is present before Capture')
        model_bowl=bowls[0] if bowl_present else None
    else:model_bowl=target
    extra=[(kind,mask) for kind,mask in obstacles if not (model_bowl is not None and mask is model_bowl)]
    if task=='tape':scene=tape_scene_from_masks(config,source,target)
    else:
        lower=None
        if task=='unstack' and rgb is not None:world,lower,fit,pixels=stack_poses_from_mask(config,source,rgb)
        else:world,fit,pixels=cube_pose_from_mask(config,source,elevated=task=='unstack')
        bowl=bowl_from_mask(model_bowl) if model_bowl is not None else [320.,240.]
        scene={**pixels,'bowl_pixel':list(bowl),'method':'SAM 2.1 From/To masks; calibrated metric projection',
               'cube_pose_from_initial_rgb':{'position':world['position'],'quaternion':world['quaternion']},'cube_pose_fit':fit}
    if task=='stack':
        # Fit To with exactly the same measured cube geometry as From.
        target_world,fit,_=cube_pose_from_mask(config,target,label='To')
        scene.update(task='stack',bowl_present=bool(bowl_present),target_cube=target_world,target_cube_fit=fit)
        if np.linalg.norm(np.array(target_world['position'][:2])-scene['cube_pose_from_initial_rgb']['position'][:2])<.045:raise ValueError('Separate the two cubes by at least 45 mm for a clear grasp')
    scene['obstacles']=[obstacle_world(config,kind,mask) for kind,mask in extra]
    if task=='unstack':
        # The lower cube is hidden under the top one: same XY and yaw, on the mat.
        top=scene['cube_pose_from_initial_rgb'];edge=config['base_calibration']['world']['cube']['size']
        bottom=lower or {'position':[*top['position'][:2],top['position'][2]-edge-.0002],'quaternion':list(top['quaternion']),'size':edge}
        xy=pixel_to_mat(config,target,edge/2)
        validate_placement(config,xy,top['position'],scene['obstacles'])
        scene.update(task='unstack',bowl_present=bool(bowl_present),target_cube=bottom,place_pixel=[float(v) for v in target],
                     placement={'position':[float(xy[0]),float(xy[1]),mat_top(config)+edge/2+.0002],'quaternion':[1.,0.,0.,0.],'size':edge})
    return scene


def tape_scene_from_masks(config,source,target):
    """From = the 53 mm tape roll, To = the bowl."""
    from strict_action_scene import episode_scene
    from tape_object import fit_tape_mask,dimensions
    contours,_=cv2.findContours(np.asarray(source,dtype=np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour=max(contours,key=cv2.contourArea);x,y,w,h=cv2.boundingRect(contour)
    if not (20<=w<=170 and 20<=h<=170):raise ValueError('From must match the 53 mm tape roll profile')
    corners=cv2.boxPoints(cv2.minAreaRect(contour));center=corners.mean(axis=0)
    scene={'object':'tape','cube_pixel':center.tolist(),'cube_pixel_corners':corners.tolist(),'cube_bbox':[x,y,w,h,int(np.count_nonzero(source))],
           'bowl_pixel':list(bowl_from_mask(target)),'method':'SAM 2.1 user-selected From/To masks; calibrated metric projection'}
    # The cube projection only seeds the support surface and an initial XY.
    cal=episode_scene(config,scene);support=cal.data['scene_estimation']['cube_support_z_m']
    xy,fit=fit_tape_mask(source,cal.camera('top'),cal.world['cube']['position'][:2],support)
    scene['cube_pose_from_initial_rgb']={'position':[float(xy[0]),float(xy[1]),support+dimensions()['width_m']/2+.0002],'quaternion':[1.,0.,0.,0.]}
    scene['tape_pose_fit']=fit
    return scene


def _saved_mask(folder,record):
    path=(folder/record['mask_file']).resolve()
    if path.parent!=folder.resolve():raise ValueError('Invalid mask path')
    m=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if m is None:raise ValueError('Saved SAM mask is missing')
    m=m>0
    if hashlib.sha256(m.astype(np.uint8).tobytes()).hexdigest()!=record['mask_sha256']:raise ValueError('Saved SAM mask changed; select again')
    return m


def selection_obstacles(selection,capture=None):
    """Obstacle records; older stack selections stored one bowl under 'obstacle'."""
    if 'obstacles' in selection:return list(selection['obstacles'])
    if 'obstacle' in selection and (capture is None or (capture.get('task')=='stack' and capture.get('bowl_present',True))):return [{**selection['obstacle'],'kind':'bowl'}]
    return []


def load_selection(folder):
    """Selection and [From mask, To mask (or unstack place pixel), [(kind, mask) obstacles]]."""
    folder=Path(folder);path=folder/'selection.json'
    if not path.exists():raise ValueError('Select From and To with SAM before generating a plan')
    selection=json.loads(path.read_text())
    if selection.get('image_sha256')!=hashlib.sha256((folder/'top.jpg').read_bytes()).hexdigest():raise ValueError('Capture changed; select From and To again')
    if not all(role in selection for role in ['from','to']):raise ValueError('Select both From and To with SAM')
    capture=json.loads((folder/'capture.json').read_text()) if (folder/'capture.json').exists() else {}
    if capture.get('task')=='stack' and capture.get('bowl_present',True) and 'obstacles' not in selection and 'obstacle' not in selection:
        raise ValueError('Select the bowl obstacle before generating a stack plan')
    source=_saved_mask(folder,selection['from'])
    target=list(selection['to']['point']) if selection['to'].get('kind')=='place' else _saved_mask(folder,selection['to'])
    obstacles=[(item['kind'],_saved_mask(folder,item)) for item in selection_obstacles(selection,capture)]
    masks=[source]+([] if is_place_point(target) else [target])+[m for _,m in obstacles]
    for i,a in enumerate(masks):
        for b in masks[i+1:]:validate_pair(a,b)
    return selection,[source,target,obstacles]
