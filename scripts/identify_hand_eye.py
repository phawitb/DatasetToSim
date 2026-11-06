"""Calibrate shared wrist/base frames from stationary cube image observations."""
import copy,json,sys
from pathlib import Path
import cv2,mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from identify_dynamics import ROOT,OUT,BASE,DATA,DatasetIndex
from strict_action_scene import episode_scene,build_model
from dataset_to_sim.strict_replay import NativeJointAdapter
from episode0 import read_real


def red_landmark(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask=(((hsv[:,:,0]<4)|(hsv[:,:,0]>165))&(hsv[:,:,1]>100)&(hsv[:,:,2]>35)).astype(np.uint8)
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    candidates=[]
    for j in range(1,n):
        x,y,w,h,area=stats[j]
        if area>150 and area<15000 and .45<w/h<2.2 and x>2 and y>2 and x+w<638 and y+h<478:
            candidates.append((area,j))
    if not candidates:return None
    area,j=max(candidates);points=cv2.findNonZero((labels==j).astype(np.uint8));box=cv2.minAreaRect(points)
    return {'uv':list(box[0]),'area':int(area),'bbox':stats[j].tolist()}


def prepare():
    ds=DatasetIndex(DATA);config=json.loads((OUT/'combined-v5.json').read_text());records=[];tiles=[]
    for i in range(72):
        e=ds.episode(i);opened=np.flatnonzero(e.action_deg[:,5]>20)
        if not len(opened):continue
        closed=np.flatnonzero((np.arange(e.frame_count)>opened[0])&(e.action_deg[:,5]<5))
        if not len(closed):continue
        end=int(closed[0])-12
        if end<3:continue
        frames=read_real(e,'wrist');scene=json.loads((BASE/f'episode-{i:02d}/scene-observation.json').read_text());cal=episode_scene(config,scene)
        cube=np.array(cal.world['cube']['position'])
        for f in np.unique(np.linspace(3,end,8).astype(int)):
            found=red_landmark(frames[f])
            if not found:continue
            record={'episode':i,'frame':int(f),'uv':found['uv'],'area':found['area'],'world_cube_center':cube.tolist()};records.append(record)
            if i in [0,1,4,8,28,55]:
                rgb=frames[f].copy();cv2.circle(rgb,tuple(map(round,found['uv'])),8,(0,255,0),2);cv2.putText(rgb,f'{i}:{f}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2);tiles.append(cv2.resize(rgb,(200,150)))
    (OUT/'hand-eye-observations.json').write_text(json.dumps(records,indent=2));print('observations',len(records),flush=True)
    for p in range((len(tiles)+17)//18):
        part=tiles[p*18:(p+1)*18];part+= [np.zeros_like(tiles[0])]*(18-len(part));cv2.imwrite(str(OUT/f'hand-eye-detections-{p}.jpg'),cv2.cvtColor(np.vstack([np.hstack(part[j:j+6]) for j in range(0,18,6)]),cv2.COLOR_RGB2BGR))


def fit():
    records=json.loads((OUT/'hand-eye-observations.json').read_text());train=[o for o in records if o['episode']%3!=0];ds=DatasetIndex(DATA);eps={i:ds.episode(i) for i in range(72)}
    config=json.loads((OUT/'combined-v5.json').read_text());m=build_model(episode_scene(config,json.loads((BASE/'episode-01/scene-observation.json').read_text())),config);d=mujoco.MjData(m);base=m.body('base').id;cid=m.camera('wrist_cam').id
    offsets=np.array(config['adapter']['arm_offset_deg']);basepos=m.body_pos[base].copy();campos=m.cam_pos[cid].copy();camR=Rotation.from_quat(m.cam_quat[cid][[1,2,3,0]]).as_matrix();fovy=m.cam_fovy[cid];calls=0;best=[1e10]
    def apply(x):
        ac=copy.deepcopy(config['adapter']);ac['arm_offset_deg']=(offsets+x[:5]).tolist();adapter=NativeJointAdapter(**ac)
        m.body_pos[base]=basepos+x[5:8]/1000;m.body_quat[base]=Rotation.from_euler('xy',x[8:10],degrees=True).as_quat()[[3,0,1,2]]
        m.cam_pos[cid]=campos+x[10:13]/1000;m.cam_quat[cid]=Rotation.from_matrix(camR@Rotation.from_rotvec(np.deg2rad(x[13:16])).as_matrix()).as_quat()[[3,0,1,2]];m.cam_fovy[cid]=x[16]
        return adapter
    def residual(x,observations,regularize=True):
        nonlocal calls
        adapter=apply(x);errors=[]
        for o in observations:
            e=eps[o['episode']];time=np.clip(o['frame']-x[17],0,e.frame_count-1);lo=int(time);hi=min(lo+1,e.frame_count-1);state=(1-time+lo)*e.state_deg[lo]+(time-lo)*e.state_deg[hi]
            d.qpos[:6]=adapter.to_model(state);mujoco.mj_kinematics(m,d);mujoco.mj_camlight(m,d)
            xyz=d.cam_xmat[cid].reshape(3,3).T@(np.array(o['world_cube_center'])-d.cam_xpos[cid]);depth=max(.03,-xyz[2]);f=240/np.tan(np.deg2rad(x[16]/2));uv=[320+f*xyz[0]/depth,240-f*xyz[1]/depth]
            errors.extend((np.array(uv)-o['uv'])/5)
        if regularize:
            errors.extend(x[:5]/5);errors.extend(x[5:8]/15);errors.extend(x[8:10]/5);errors.extend(x[10:13]/10);errors.extend(x[13:16]/10);errors.extend([(x[16]-fovy)/10,x[17]/3])
            calls+=1
            if calls%50==0:print('eval',calls,'median reprojpx',np.median(np.abs(np.array(errors[:len(observations)*2])))*5,flush=True)
        return np.array(errors)
    initial=np.r_[np.zeros(16),fovy,1.]
    lo=np.r_[[-8]*5,[-20,-20,-basepos[2]*1000],[-8]*2,[-20]*3,[-25]*3,40,0];hi=np.r_[[8]*5,[20,20,30],[8]*2,[20]*3,[25]*3,85,6]
    initial_res=residual(initial,train,False)
    result=least_squares(lambda x:residual(x,train),initial,bounds=(lo,hi),loss='soft_l1',max_nfev=100,ftol=1e-6,xtol=1e-6)
    x=result.x;adapter=apply(x);config['adapter']['arm_offset_deg']=(offsets+x[:5]).tolist();config['base_calibration']['world']['robot_base']['position']=m.body_pos[base].tolist();config['base_calibration']['world']['robot_base']['quaternion']=m.body_quat[base].tolist();config['wrist_camera'].update(position=m.cam_pos[cid].tolist(),quaternion=m.cam_quat[cid].tolist(),fovy=float(m.cam_fovy[cid]))
    # Evaluate using the original base/offset reference captured above.
    metrics={}
    for label,obs in [('train',train),('held_out',[o for o in records if o['episode']%3==0])]:
        # residual() uses config only for gripper scale, with captured arm offsets.
        rr=residual(x,obs,False).reshape(-1,2)*5;metrics[label]={'count':len(obs),'median_error_px':float(np.median(np.linalg.norm(rr,axis=1))),'rmse_px':float(np.sqrt(np.mean(rr**2)))}
    config['hand_eye_identification']={'method':'stationary pre-grasp full cube silhouette center reprojection; measured states used only for calibration','training_episodes':sorted(set(o['episode'] for o in train)),'parameters':x.tolist(),'wrist_exposure_delay_frames':float(x[17]),'metrics':metrics,'initial_train_rmse_px':float(np.sqrt(np.mean(initial_res**2))*5),'assumptions':'Cube remains static until 12 frames before close command; low-value red mask includes dark cube faces; silhouette rectangle center approximates projected 3D center.'}
    (OUT/'hand-eye-fit.json').write_text(json.dumps(config,indent=2));print(json.dumps(config['hand_eye_identification'],indent=2),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:fit()
