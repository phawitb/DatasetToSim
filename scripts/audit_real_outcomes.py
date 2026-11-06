"""Image-only terminal real-scene diagnostic; not a physical success oracle."""
import json,cv2,numpy as np
from strict_action_replay import DATA,ROOT,DatasetIndex,detect_scene,read_real
OUT=ROOT/'output/system-identification'
ds=DatasetIndex(DATA);rows=[];tiles=[]
for i in range(72):
 e=ds.episode(i);rgb=read_real(e,'top')[-1];row={'episode':i,'frame':e.frame_count-1}
 try:
  s=detect_scene(rgb);distance=float(np.linalg.norm(np.array(s['cube_pixel'])-s['bowl_pixel']));row.update(cube_pixel=s['cube_pixel'],bowl_pixel=s['bowl_pixel'],distance_px=distance,near_bowl_image=distance<55)
  cv2.circle(rgb,tuple(map(round,s['cube_pixel'])),12,(0,255,0),2)
 except ValueError as err:row['error']=str(err)
 cv2.putText(rgb,f'EP {i:02d}',(15,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2);tiles.append(cv2.resize(rgb,(200,150)));rows.append(row)
for page in range(3):cv2.imwrite(str(OUT/f'real-final-{page}.jpg'),cv2.cvtColor(np.vstack([np.hstack(tiles[j:j+6]) for j in range(page*24,(page+1)*24,6)]),cv2.COLOR_RGB2BGR))
(OUT/'real-terminal-observations.json').write_text(json.dumps(rows,indent=2));print([r for r in rows if not r.get('near_bowl_image')],flush=True)
