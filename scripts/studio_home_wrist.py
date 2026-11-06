"""Current Wrist SAM evidence for returning from a completed placement."""
import copy,hashlib,json,uuid
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import least_squares
from studio_sam_client import decode_mask,save_role
from studio_sam_geometry import load_selection


def released_reference(folder,q,calibration_hash):
    from dataset_to_sim.strict_replay import array_hash
    folder=Path(folder)
    try:
        report=json.loads((folder/'plan.json').read_text());trace=json.loads((folder/'hardware-trace.json').read_text())
        with np.load(folder/'plan.npz') as saved:actions=saved['actions']
    except (OSError,KeyError,ValueError) as exc:raise ValueError('Wrist Home requires a recorded completed placement') from exc
    if not (report.get('source')=='live' and report.get('purpose')!='home' and report.get('executed') and report.get('outcome')=='success' and report.get('transfer_checks_passed') and trace.get('completed') and not trace.get('error')):raise ValueError('Wrist Home requires a completed placement')
    if report.get('calibration_hash')!=calibration_hash:raise ValueError('Placement calibration changed')
    if array_hash(actions)!=report.get('action_hash') or report.get('action_hash')!=report.get('simulated_action_hash'):raise ValueError('Placement commands changed')
    samples=trace.get('samples',[])
    if len(samples)!=len(actions) or not samples:raise ValueError('Incomplete placement trace')
    targets=np.asarray([s['target'] for s in samples],float)
    if targets.shape!=actions.shape or not np.allclose(targets,actions,atol=1e-6,rtol=0):raise ValueError('Recorded commands differ from the placement plan')
    q=np.asarray(q,float);end=actions[-1];measured=np.asarray(samples[-1]['measured'],float)
    if measured.shape!=(6,) or not np.isfinite(measured).all() or np.any(np.abs(measured-end)>[10,10,10,10,10,25]):raise ValueError('Placement end tracking was not valid')
    # Association only: current Wrist evidence and physics still use the measured
    # pose. Bound both comparisons so settling and endpoint error cannot add up.
    if end[5]<25 or q.shape!=(6,) or not np.isfinite(q).all() or q[5]<25 or np.any(np.abs(q-measured)>[10,10,10,10,10,3]) or np.any(np.abs(q-end)>[10,10,10,10,10,3]):raise ValueError('Arm has drifted outside the verified placement pose; Home needs a new current-scene recovery plan')
    return report


def wrist_camera(config,scene,q):
    import mujoco
    from strict_action_scene import episode_scene,build_model
    from dataset_to_sim.strict_replay import NativeJointAdapter
    cal=episode_scene(config,scene);m=build_model(cal,config);d=mujoco.MjData(m)
    d.qpos[:6]=NativeJointAdapter(**config['adapter']).to_model(np.asarray(q,float));mujoco.mj_forward(m,d)
    i=m.camera('wrist_cam').id
    return {'position':d.cam_xpos[i].tolist(),'rotation':d.cam_xmat[i].reshape(3,3).tolist(),'fovy':float(m.cam_fovy[i])},cal.world['bowl']


def project_rim(camera,bowl,xy):
    a=np.linspace(0,2*np.pi,128,endpoint=False);r=bowl['outer_radius'];z=bowl['position'][2]+bowl['height']
    world=np.c_[xy[0]+r*np.cos(a),xy[1]+r*np.sin(a),np.full(len(a),z)]
    points=(world-np.asarray(camera['position']))@np.asarray(camera['rotation'])@np.diag([1,-1,-1])
    if np.any(points[:,2]<=.015):raise ValueError('Bowl rim is behind or too close to Wrist camera')
    focal=240/np.tan(np.deg2rad(camera['fovy']/2));uv=points[:,:2]/points[:,2:]*focal+[320,240]
    _,clip=cv2.intersectConvexConvex(cv2.convexHull(uv.astype(np.float32)),np.array([[0,0],[639,0],[639,479],[0,479]],np.float32))
    if clip is None or len(clip)<5:raise ValueError('Bowl rim is outside Wrist view')
    return clip[:,0,:].astype(float)


def fit_bowl(mask,camera,bowl):
    from dataset_to_sim.scene_estimation import _sample,_distance
    mask=np.asarray(mask,bool)
    if mask.shape!=(480,640) or not 4000<mask.sum()<276480:raise ValueError('Insufficient Wrist bowl mask')
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    observed=cv2.convexHull(max(contours,key=cv2.contourArea))[:,0,:].astype(float)
    points=_sample(observed);initial=np.array(bowl['position'][:2])
    def residual(xy):
        predicted=project_rim(camera,bowl,xy)
        return np.r_[_distance(_sample(predicted),observed),_distance(points,predicted)]
    fit=least_squares(residual,initial,bounds=(initial-.025,initial+.025),diff_step=1e-4,loss='soft_l1',f_scale=3.,max_nfev=80)
    predicted=project_rim(camera,bowl,fit.x);model_mask=np.zeros((480,640),np.uint8);cv2.fillConvexPoly(model_mask,predicted.astype(np.int32),1)
    iou=float(np.count_nonzero(mask&(model_mask>0))/np.count_nonzero(mask|(model_mask>0)))
    rmse=float(np.sqrt(np.mean(residual(fit.x)**2)));delta=float(np.linalg.norm(fit.x-initial))
    if not fit.success or rmse>10 or iou<.88 or delta>.020:raise ValueError('Current Wrist mask does not match the calibrated bowl near the placement')
    return fit.x.tolist(),{'boundary_rmse_px':rmse,'iou':iou,'reference_shift_m':delta,'projected_rim':predicted.tolist(),'clearance_margin_m':.020}


def wrist_home_scene(client,folder,reference,image,config,q,tag='home-selection'):
    folder,reference=Path(folder),Path(reference)
    report=released_reference(reference,q,config['follower_calibration_sha256'])
    selection,_=load_selection(reference);scene=copy.deepcopy(report['scene'])
    camera,bowl=wrist_camera(config,scene,q);rim=project_rim(camera,bowl,bowl['position'][:2])
    center=rim.mean(axis=0);left=0.5*center+0.5*rim[np.argmin(rim[:,0])];right=0.5*center+0.5*rim[np.argmax(rim[:,0])]
    # Prefer the side of the bowl so the released cube at its center is not selected.
    for point in [left,right,center]:
        try:
            result=client.predict(image,np.clip(point,[1,1],[638,478]).tolist())
            if result['model']['weights_sha256']!=selection['model']['weights_sha256']:raise ValueError('SAM model changed')
            xy,fit=fit_bowl(decode_mask(result),camera,bowl)
            break
        except ValueError as exc:last_error=exc
    else:raise ValueError(f'Cannot verify Home from current Wrist camera: {last_error}')
    scene['bowl_position_from_wrist']=[*xy,bowl['position'][2]]
    revision=uuid.uuid4().hex
    record={'reference':reference.name,'image_sha256':result['image_sha256'],'model':result['model'],'to':save_role(folder,'to-wrist',result,revision),'scene':scene,'reference_selection_revision':selection['revision'],'view':'wrist','method':'Current Wrist SAM + measured-joint camera projection','camera':camera,'native_pose':list(q),'fit':fit,'reference_plan_sha256':hashlib.sha256((reference/'plan.npz').read_bytes()).hexdigest()}
    (folder/f'{tag}.json').write_text(json.dumps(record,indent=2))
    return record
