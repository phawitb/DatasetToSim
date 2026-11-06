"""All-episode audit with fixed robot/camera/physics calibration and observed scenes."""
import copy
import hashlib
import traceback
from episode0_physics import *

BATCH=ROOT/'output/benchmark72'
BATCH.mkdir(parents=True,exist_ok=True)

def detect_scene(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    roi=np.zeros(rgb.shape[:2],np.uint8);roi[30:405,90:620]=1
    red=(((hsv[:,:,0]<3)|(hsv[:,:,0]>160))&(hsv[:,:,1]>85)&(hsv[:,:,2]>130)&roi.astype(bool)).astype(np.uint8)
    n,labels,stats,centers=cv2.connectedComponentsWithStats(red)
    candidates=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if 100<=area<=1600 and .5<w/h<1.8 and w<=40 and h<=40:
            candidates.append((float(area/(w*h)*np.sqrt(area)),centers[i].tolist(),stats[i].tolist()))
    if not candidates:raise ValueError('No unoccluded red cube detected in first frame')
    _,cube,box=max(candidates)
    x,y,w,h,_=box
    points=cv2.findNonZero(red[y:y+h,x:x+w])
    corners=cv2.boxPoints(cv2.minAreaRect(points))+[x,y]
    r,g,b=[rgb[:,:,i].astype(float) for i in range(3)]
    green=((g>r*1.01)&(b>=r)&(g>125)).astype(np.uint8)
    green[:110]=0;green[250:]=0;green[:,:410]=0;green[:,550:]=0
    contours,_=cv2.findContours(green,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:raise ValueError('No green bowl detected in first frame')
    contour=max(contours,key=cv2.contourArea)
    if cv2.contourArea(contour)<2000:raise ValueError('Bowl mask too small')
    bowl=list(cv2.fitEllipse(cv2.convexHull(contour))[0])
    return {'cube_pixel':cube,'cube_pixel_corners':corners.tolist(),'bowl_pixel':bowl,'cube_bbox':box,'method':'first-frame color connected components and minimum rectangle; estimated dimensions; no per-episode tuning'}

def episode_calibration(base,scene):
    c=copy.deepcopy(base);K,R,p,_,_=camera_fit()
    # Same fixed metric conversion for every episode. Centroid is near visible top face.
    cube=ground_point(scene['cube_pixel'],K,R,p,.022)
    bowl=ground_point(scene['bowl_pixel'],K,R,p,.025)
    c.data['world']['cube']['position']=[*cube[:2],.0162]
    c.data['world']['bowl']['position']=[*bowl[:2],.005]
    return c

def prepare():
    ds=DatasetIndex(DATA);base=Calibration.load(ROOT/'calibration/episode0.json');scenes=[];tiles=[]
    for i in range(ds.episode_count):
        ep=ds.episode(i);cap=cv2.VideoCapture(str(ep.video_slices['top'].path));cap.set(cv2.CAP_PROP_POS_FRAMES,round(ep.video_slices['top'].start_time*30));ok,bgr=cap.read();cap.release()
        if not ok:raise RuntimeError(f'Video decoding failed episode {i}')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        row={'episode':i,'frames':ep.frame_count}
        try:
            row.update(detect_scene(rgb));c=episode_calibration(base,row)
            row['cube_xy']=c.world['cube']['position'][:2];row['bowl_xy']=c.world['bowl']['position'][:2]
            for key,col in [('cube_pixel',(0,255,255)),('bowl_pixel',(255,0,255))]:cv2.circle(bgr,tuple(map(round,row[key])),7,col,2)
        except Exception as ex:row['error']=str(ex)
        scenes.append(row);tile=cv2.resize(bgr,(320,240));cv2.putText(tile,f'EP {i:02d}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),3);cv2.putText(tile,f'EP {i:02d}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);tiles.append(tile)
    for page in range(3):cv2.imwrite(str(BATCH/f'scenes-{page}.jpg'),np.vstack([np.hstack(tiles[j:j+6]) for j in range(page*24,page*24+24,6)]))
    (BATCH/'scenes-v2.json').write_text(json.dumps(scenes,indent=2));print('prepared',len(scenes),'scene errors',sum('error'in s for s in scenes))

def fit_shared_plane():
    ds=DatasetIndex(DATA);base=Calibration.load(ROOT/'calibration/episode0.json');m=physical_model(base);d=mujoco.MjData(m)
    scenes=json.loads((BATCH/'scenes-v2.json').read_text());uv=[];xy=[];anchors=[]
    for s in scenes:
        if 'error'in s:continue
        ep=ds.episode(s['episode']);g=ep.action_deg[:,5];events=np.flatnonzero((g[1:]<5)&(g[:-1]>=5))+1
        if not len(events):continue
        f=min(int(events[0])+3,ep.frame_count-1)
        d.qpos[:6]=base.joint_convention.to_radians(ep.state_deg[f]);d.qpos[5]=np.deg2rad(ep.state_deg[f,5]-8);mujoco.mj_forward(m,d)
        center=(d.geom('fixed_pad').xpos+d.geom('moving_pad').xpos)/2
        uv.append(s['cube_pixel']);xy.append(center[:2].copy());anchors.append({'episode':s['episode'],'grasp_frame':f,'pad_center':center.tolist()})
    H,inliers=cv2.findHomography(np.array(uv),np.array(xy),cv2.RANSAC,.015)
    predicted=cv2.perspectiveTransform(np.array(uv,dtype=float)[None],H)[0]
    residual=np.linalg.norm(predicted-xy,axis=1)
    result={'method':'one global robust image-to-grasp-plane homography fitted from first closing-event pad midpoints','H':H.tolist(),'anchors':anchors,'inliers':int(inliers.sum()),'median_xy_error_m':float(np.median(residual)),'max_xy_error_m':float(residual.max()),'evaluation_note':'Fitted on these same demonstrations; not an unseen-data success rate.'}
    (ROOT/'calibration/shared72.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items()if k not in ['H','anchors']})

def verdict(s):
    if s['max_cube_z']<.055:return 'not_lifted'
    if s['bilateral_frames']<2:return 'no_bilateral_grasp'
    if not s['final_inside_bowl']:return 'missed_bowl'
    if s.get('final_finger_contact',s['final_bilateral_contact']):return 'not_released'
    if s.get('final_cube_motion_last_10_frames_m',0)>.003:return 'not_settled'
    return 'success'

def run(tag='baseline'):
    if tag not in ['baseline','shared-calibration','corrected-detection','vertical-clearance','responsive-gripper','oriented-gripper','repaired']:raise ValueError(f'Unknown run mode: {tag}')
    ds=DatasetIndex(DATA);base=Calibration.load(ROOT/'calibration/episode0.json');scenes=json.loads((BATCH/('scenes.json'if tag=='baseline'else 'scenes-v2.json')).read_text());results=[]
    folder=BATCH/tag;folder.mkdir(exist_ok=True)
    shared=json.loads((ROOT/'calibration/shared72.json').read_text()) if tag!='baseline' else None
    if tag in ['responsive-gripper','oriented-gripper','repaired']:base.data['physics']={'gripper_joint_damping':.1,'gripper_armature':.003,'gripper_servo_kv':.1}
    settings={'robot':base.data,'shared_plane':shared,'wrist_camera':json.loads((ROOT/'calibration/wrist-full-contact.json').read_text()),'clearance_mode':'vertical' if tag in ['vertical-clearance','responsive-gripper','oriented-gripper','repaired'] else 'wrist','grip_offset':-8,'observed_cube_yaw':tag in ['oriented-gripper','repaired'],'trajectory_repair':tag=='repaired'}
    settings['code_sha256']={name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in ['benchmark72.py','episode0_physics.py','trajectory_repair.py','episode0.py']}
    settings['repair_settings']={'grasp_center_z_m':.024,'lift_center_z_m':.105,'approach_side_clearance_m':.006,'gripper_open_deg':35,'gripper_closed_deg':-8,'first_close_threshold_deg':5,'release_threshold_deg':20,'uses_simulation_success_feedback':False}
    (folder/'settings.json').write_text(json.dumps(settings,indent=2))
    (folder/'scenes.json').write_text(json.dumps(scenes,indent=2))
    digest=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    for scene in scenes:
        i=scene['episode'];row={'episode':i,'frames':scene['frames']}
        try:
            if 'error' in scene:raise ValueError(scene['error'])
            c=episode_calibration(base,scene);ep=ds.episode(i)
            if tag!='baseline':
                xy=cv2.perspectiveTransform(np.array([[scene['cube_pixel']]],dtype=float),np.array(shared['H']))[0,0]
                c.data['world']['cube']['position'][:2]=xy.tolist()
                if settings['observed_cube_yaw']:
                    corners=cv2.perspectiveTransform(np.array([scene['cube_pixel_corners']],dtype=float),np.array(shared['H']))[0]
                    edge=corners[1]-corners[0];yaw=np.arctan2(edge[1],edge[0])
                    c.data['world']['cube']['quaternion']=[float(np.cos(yaw/2)),0.,0.,float(np.sin(yaw/2))]
            epdir=folder/f'episode-{i:02d}';epdir.mkdir(exist_ok=True)
            (epdir/'calibration.json').write_text(json.dumps(c.data,indent=2))
            summary,m,joints,poses=simulate(c,ep,c.world['cube']['position'][:2],clearance_mode=settings['clearance_mode'],trace_dir=epdir,repair=settings['trajectory_repair'])
            row.update(summary);row['outcome']=verdict(summary)
            np.savez_compressed(folder/f'episode-{i:02d}.npz',qpos=joints,cube_pose=poses,timestamps=ep.timestamps)
        except Exception as ex:row.update(outcome='error',error=f'{type(ex).__name__}: {ex}')
        results.append(row);print(i,row['outcome'],round(row.get('max_cube_z',0),3),row.get('error',''),flush=True)
        report={'tag':tag,'total':72,'completed':len(results),'successes':sum(r['outcome']=='success'for r in results),'runtime_errors':sum(r['outcome']=='error'for r in results),'calibration_sha256':digest,'same_calibration_all_episodes':True,'scene_source':'initial RGB and shared mapping; no scene pose tuning by outcome','evaluation_note':'Calibration fitted on the same 72 demonstrations; not a held-out generalization test. Arm targets are adjusted for tabletop clearance.','results':results}
        (folder/'results.json').write_text(json.dumps(report,indent=2))
    print('TOTAL',report['successes'],'/',report['total'],flush=True)

if __name__=='__main__':
    import sys
    if '--prepare' in sys.argv:prepare()
    elif '--fit' in sys.argv:fit_shared_plane()
    else:run(sys.argv[1]if len(sys.argv)>1 else 'baseline')
