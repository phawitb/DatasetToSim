"""Home checks the bowl independently: the picked cube need not stay at From."""
import json,time,uuid
from pathlib import Path
import cv2,numpy as np
from studio_sam_client import decode_mask,save_role
from studio_sam_geometry import load_selection,scene_from_masks,bowl_from_mask


def home_scene(client,folder,reference,image,config,tag='home-selection'):
    folder,reference=Path(folder),Path(reference)
    capture=json.loads((reference/'capture.json').read_text())
    if capture.get('calibration_hash')!=config['follower_calibration_sha256']:raise ValueError('Home reference calibration changed')
    if not 0<=time.time()-capture.get('captured_at',0)<=600:raise ValueError('Home reference expired; capture and select again, or support the arm and use Torque off')
    selection,masks=load_selection(reference)
    scene=scene_from_masks(config,*masks)
    # Probe interior points of the previously selected bowl; a jaw may hide the
    # original click after transfer. Never reuse the original cube click.
    target=masks[1];distance=cv2.distanceTransform(target.astype(np.uint8),cv2.DIST_L2,3)
    yy,xx=np.nonzero(distance>8)
    points=[selection['to']['point']]
    if len(xx):
        for index in np.linspace(0,len(xx)-1,8).astype(int):points.append([int(xx[index]),int(yy[index])])
    before=np.array(scene['bowl_pixel']);accepted=None
    for point in points:
        try:
            result=client.predict(image,point)
            if result['model']!=selection['model']:raise ValueError('SAM model changed')
            mask=decode_mask(result);center=bowl_from_mask(mask)
            intersection=np.count_nonzero(mask&target)
            # Require most of the bowl and its outside edge to remain visible;
            # do not interpret a robot/desk mask as a hidden bowl.
            if intersection/max(1,np.count_nonzero(mask|target))<.72:continue
            if np.count_nonzero(mask)<np.count_nonzero(target)*.8:continue
            if np.linalg.norm(np.array(center)-before)>5:continue
            accepted=(result,center);break
        except ValueError:continue
    if accepted is None:raise ValueError('SAM cannot verify the bowl for Home; support the arm and use Torque off to clear the camera')
    result,center=accepted;scene['bowl_pixel']=center
    revision=uuid.uuid4().hex
    report={'reference':reference.name,'image_sha256':result['image_sha256'],'model':result['model'],'to':save_role(folder,'to',result,revision),'scene':scene,'reference_selection_revision':selection['revision'],'cube_note':'Prior SAM cube is retained simulation geometry only; Home does not target or re-prompt From'}
    (folder/f'{tag}.json').write_text(json.dumps(report,indent=2))
    return report
