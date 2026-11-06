"""Find image-only cube-motion onset for contact calibration, not replay control."""
import json,sys,numpy as np,cv2
from identify_dynamics import OUT,BASE,DATA,DatasetIndex
from episode0 import read_real


def prepare():
 ds=DatasetIndex(DATA);scenes=json.loads((OUT/'confirmed-scenes.json').read_text());rows=[]
 for i in range(72):
  e=ds.episode(i);frames=read_real(e,'top');scene=scenes[i];cx,cy=map(round,scene['cube_pixel']);x0=max(0,cx-40);y0=max(0,cy-40);tracked=[];prev=np.array([cx,cy],float)
  g=e.state_deg[:,5];closed=np.flatnonzero((g<12)&(np.maximum.accumulate(g)>25));end=min(e.frame_count,int(closed[0])+12) if len(closed) else e.frame_count
  for f in range(end):
   hsv=cv2.cvtColor(frames[f][y0:y0+80,x0:x0+80],cv2.COLOR_RGB2HSV);mask=(((hsv[:,:,0]<4)|(hsv[:,:,0]>162))&(hsv[:,:,1]>85)&(hsv[:,:,2]>130)).astype('uint8');n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
   candidates=[j for j in range(1,n) if 70<=stats[j,cv2.CC_STAT_AREA]<=1600 and np.linalg.norm(centers[j]+[x0,y0]-prev)<25]
   if candidates:
    j=min(candidates,key=lambda j:np.linalg.norm(centers[j]+[x0,y0]-prev));prev=centers[j]+[x0,y0];tracked.append([f,*prev,stats[j,cv2.CC_STAT_AREA]])
   else:tracked.append([f,np.nan,np.nan,0])
  track=np.array(tracked);initial=np.nanmedian(track[:5,1:3],axis=0);area=np.median(track[:5,3]);shift=np.linalg.norm(track[:,1:3]-initial,axis=1);start=max(5,int(closed[0])-25) if len(closed) else 5;onset=None
  for f in range(start,end-3):
   if np.all(shift[f:f+3]>2) and np.all((track[f:f+3,3]>.65*area)&(track[f:f+3,3]<1.4*area)):
    onset=f;break
  row={'episode':i,'motion_frame':onset,'initial_rgb_centroid':initial.tolist(),'track':[[int(v[0]),*[float(x) if np.isfinite(x) else None for x in v[1:]]]for v in track],'method':'first 3 frames with red-cube centroid displacement >2px; initial-area gate; first-closure window','automatic_onset_requires_visual_review':True}
  if onset is not None:
   state_frame=max(0,onset-2);row['state_frame_assuming_two_frame_camera_lag']=state_frame;row['state']=e.state_deg[state_frame].tolist();row['shift_px']=float(shift[onset]);row['cube']=scene['cube_pose_from_initial_rgb']['position'];row['quaternion']=scene['cube_pose_from_initial_rgb']['quaternion']
  rows.append(row)
 (OUT/'first-contact-observations.json').write_text(json.dumps(rows,indent=2));print('onsets',sum(r['motion_frame']is not None for r in rows));print([(r['episode'],r['motion_frame']) for r in rows])

if __name__=='__main__':prepare()
