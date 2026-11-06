"""Live workcell detection across the mat; archived benchmark detector is unchanged."""
import cv2,numpy as np
from strict_action_scene import episode_scene
from dataset_to_sim.scene_estimation import fit_cube_silhouette


def detect_bowl(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask=((hsv[:,:,0]>=35)&(hsv[:,:,0]<=100)&(hsv[:,:,1]>=8)&(hsv[:,:,2]>110)).astype(np.uint8)
    roi=np.zeros(mask.shape,np.uint8);roi[35:400,120:565]=1;mask*=roi
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour);x,y,w,h=cv2.boundingRect(contour)
        if 2000<area<30000 and .6<w/h<1.6 and len(contour)>=5:candidates.append((area,contour))
    if not candidates:raise ValueError('No clear green bowl detected on the mat')
    contour=max(candidates,key=lambda c:c[0])[1]
    return list(cv2.fitEllipse(cv2.convexHull(contour))[0])


def estimate_live_scene(config,rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    roi=np.zeros(rgb.shape[:2],np.uint8);roi[30:405,90:620]=1
    red=(((hsv[:,:,0]<3)|(hsv[:,:,0]>160))&(hsv[:,:,1]>85)&(hsv[:,:,2]>130)&roi.astype(bool)).astype(np.uint8)
    n,labels,stats,centers=cv2.connectedComponentsWithStats(red);candidates=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if 100<=area<=1600 and .5<w/h<1.8 and w<=40 and h<=40:candidates.append((float(area/(w*h)*np.sqrt(area)),centers[i].tolist(),stats[i].tolist()))
    if not candidates:raise ValueError('No unoccluded red cube detected; return Home and place the cube on the mat')
    _,cube,box=max(candidates);x,y,w,h,_=box
    corners=cv2.boxPoints(cv2.minAreaRect(cv2.findNonZero(red[y:y+h,x:x+w])))+[x,y]
    scene={'cube_pixel':cube,'cube_pixel_corners':corners.tolist(),'cube_bbox':box,'bowl_pixel':detect_bowl(rgb),'method':'live color detection across calibrated mat'}
    cal=episode_scene(config,scene);c=cal.world['cube'];q=c['quaternion'];support=cal.data['scene_estimation']['cube_support_z_m']
    pose,report=fit_cube_silhouette(rgb,cal.camera('top'),scene,[*c['position'][:2],2*np.arctan2(q[3],q[0])],c['size'],support,use_top_face=config.get('scene_detection')=='cube-silhouette-v2')
    scene['cube_pose_from_initial_rgb']={'position':[float(pose[0]),float(pose[1]),support+c['size']/2+.0002],'quaternion':[float(np.cos(pose[2]/2)),0.,0.,float(np.sin(pose[2]/2))]}
    scene['cube_pose_fit']=report
    return scene


def verify_bowl_occlusion(rgb,expected_pixel):
    """Require visible bowl remnants AND the arm at its expected image region."""
    if rgb is None or rgb.shape!=(480,640,3):raise ValueError('Camera image unavailable for Home')
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);yy,xx=np.indices((480,640))
    x,y=expected_pixel;region=((xx-x)/66)**2+((yy-y)/62)**2<=1
    orange=(hsv[:,:,0]>=4)&(hsv[:,:,0]<=30)&(hsv[:,:,1]>85)&(hsv[:,:,2]>80)
    green=(hsv[:,:,0]>=35)&(hsv[:,:,0]<=100)&(hsv[:,:,1]>=8)&(hsv[:,:,2]>110)
    arm_pixels=int(np.count_nonzero(orange&region));bowl_pixels=int(np.count_nonzero(green&region))
    if arm_pixels<1800 or bowl_pixels<450:raise ValueError('Cannot verify that the arm hides the bowl; support and reposition with Torque off')
    return {'visible_bowl_pixels':bowl_pixels,'occluding_arm_pixels':arm_pixels,'expected_bowl_pixel':list(expected_pixel)}
