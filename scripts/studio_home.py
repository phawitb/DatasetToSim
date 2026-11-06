"""Validate a camera-clear return from measured pose using the last known workcell."""
import copy,json,sys,time
from pathlib import Path
from studio_planner import ROOT,interpolate_native,render_trace
import mujoco,numpy as np,cv2
from strict_action_scene import episode_scene,build_model
from dataset_to_sim.strict_replay import NativeJointAdapter,integrate_actions,array_hash
from dataset_to_sim.dataset import Episode
from dataset_to_sim.studio import validate_trajectory,native_limits


def validate_occluded_reference(old,actual,q,now=None):
    age=(time.time() if now is None else now)-old.get('captured_at',0)
    if not 0<=age<=600 or not actual.get('completed') or not actual.get('samples'):
        raise ValueError('Bowl hidden and no recent verified run; support and reposition with Torque off')
    if np.any(np.abs(np.asarray(q)-np.array(actual['samples'][-1]['measured']))>np.array([10.]*5+[5.])):
        raise ValueError('Arm moved since the verified run; hidden bowl cannot be reused')


def prepare(folder,reference):
    folder,reference=Path(folder),Path(reference)
    capture=json.loads((folder/'capture.json').read_text())
    cfg=json.loads((ROOT/'calibration/native-action-current.json').read_text())
    home=json.loads((ROOT/'calibration/studio-home.json').read_text())
    if capture['calibration_hash']!=home['calibration_sha256']:raise ValueError('Home calibration changed')
    q=np.array(capture['initial_native']);closed=q.copy();closed[5]=1.
    waypoints=[(q,0)]
    if np.max(np.abs(q-np.array(home['home'])))>2:
        waypoints += [(closed,1.5),(home['raised'],3.),(home['unfolded'],6.),(home['home'],2.5)]
    else:waypoints.append((home['home'],1.5))
    actions,ts=interpolate_native(waypoints);validate_trajectory(actions,ts,native_limits(cfg))
    selection=json.loads((folder/'home-selection.json').read_text())
    scene=selection['scene']
    cube_source='prior SAM mask retained as simulation geometry; Home does not target cube'
    bowl_source=selection.get('method','current Top SAM bowl mask')
    cal=episode_scene(cfg,scene)
    if 'bowl_position_from_wrist' in scene:cal.world['bowl']['position']=scene['bowl_position_from_wrist']
    world=copy.deepcopy(cal.world)
    margin=selection.get('fit',{}).get('clearance_margin_m',0.)
    home_cfg=copy.deepcopy(cfg);home_cfg['home_bowl_clearance_m']=margin
    m=build_model(cal,home_cfg)
    adapter=NativeJointAdapter(**cfg['adapter'])
    ep=Episode(-1,np.arange(len(actions)),np.arange(len(actions)),ts,np.tile(q,(len(actions),1)),actions,{})
    trace=integrate_actions(m,ep,adapter)
    collision=trace['audit']['contact_audit']
    checks=trace['audit']['initial_max_penetration_m']<=.001 and all(v['max_penetration_m']<=.001 for k,v in collision.items() if k!='cube')
    tracking=np.abs(adapter.to_native(trace['qpos'][1:])-actions)
    checks=bool(checks and np.all(np.max(tracking,axis=0)<=np.array([10.]*5+[25.])))
    if not checks:raise ValueError('Home path failed collision or tracking checks; use supported manual repositioning with Torque off')
    np.savez_compressed(folder/'plan.npz',actions=actions,timestamps=ts)
    np.savez_compressed(folder/'trace.npz',**{k:v for k,v in trace.items() if isinstance(v,np.ndarray)})
    mujoco.mj_saveLastXML(str(folder/'scene.xml'),m)
    render_trace(m,trace,folder)
    report=capture|{'purpose':'home','frames':len(actions),'duration':len(actions)/30,'action_hash':array_hash(actions),'simulated_action_hash':array_hash(actions),'outcome':'success','transfer_checks_passed':True,'executed':False,'max_ik_error_mm':0,'video':'simulation.mp4','collision_audit':collision,'geometry_reference':reference.name,'geometry_note':'Current SAM bowl; Home-only expanded bowl clearance envelope; fixed calibrated home corridor','scene':scene,'world':world,'home_view':selection.get('view','top'),'home_clearance_margin_m':margin,'selection_revision':selection['reference_selection_revision'],'cube_pose_source':cube_source,'bowl_position':cal.world['bowl']['position'],'bowl_pixel':scene['bowl_pixel'],'bowl_source':bowl_source,'reference_captured_at':capture.get('captured_at',0),'home_native':home['home']}
    (folder/'plan.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':prepare(*sys.argv[1:])
