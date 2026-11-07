"""Image-conditioned pick/place planning, separate from strict replay benchmarks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import os
os.environ.setdefault('MUJOCO_GL', 'cgl' if sys.platform == 'darwin' else 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares, brentq
from strict_action_scene import estimate_scene, containment
from studio_stack import scene_calibration as episode_scene,build_scene_model as build_model,stack_verdict
from strict_action_replay import score
from dataset_to_sim.strict_replay import NativeJointAdapter, integrate_actions, array_hash
from dataset_to_sim.dataset import Episode
from dataset_to_sim.studio import native_limits, validate_trajectory, SPEED_LIMITS
from studio_motion import interpolate_flow, interpolate_ruckig, validate_technique, TECHNIQUES


def interpolate_native(waypoints):
    result=[np.asarray(waypoints[0][0],float)]
    for target,minimum_duration in waypoints[1:]:
        start=result[-1]; target=np.asarray(target,float)
        duration=max(minimum_duration,float(np.max(1.51*np.abs(target-start)/(SPEED_LIMITS*.8))))
        n=max(1,int(np.ceil(duration*30)))
        for f in range(1,n+1):
            a=f/n; a=a*a*(3-2*a)
            result.append(target.copy() if f == n else (1-a)*start+a*target)
    return np.array(result),np.arange(len(result))/30


def load_scene(folder):
    capture=json.loads((folder/'capture.json').read_text())
    config=json.loads((ROOT/'calibration/native-action-current.json').read_text())
    rgb=cv2.cvtColor(cv2.imread(str(folder/'top.jpg')),cv2.COLOR_BGR2RGB)
    if rgb.shape != (480,640,3): raise ValueError('Calibrated top camera requires 640 × 480 images')
    if capture.get('source')=='live':
        from studio_sam_geometry import load_selection,scene_from_masks
        selection,masks=load_selection(folder)
        capture.update(selection_revision=selection['revision'],selection=selection)
        scene=scene_from_masks(config,*masks,task=capture.get('task','bowl'),bowl_present=capture.get('bowl_present',True))
    else:scene=estimate_scene(config,rgb)
    cal=episode_scene(config,scene)
    return capture,config,scene,cal,build_model(cal,config)


def plan(folder,stack_offset=None,preview=True,technique="classic"):
    validate_technique(technique)
    capture,config,scene,cal,m=load_scene(folder)
    adapter=NativeJointAdapter(**config['adapter']); limits=native_limits(config)
    initial=np.array(capture['initial_native'],float); current=adapter.to_model(initial)
    d=mujoco.MjData(m)
    bounds=adapter.to_model(np.array([limits[:,0],limits[:,1]]))[:,:5]
    lo,hi=np.min(bounds,axis=0)+1e-6,np.max(bounds,axis=0)-1e-6
    cube=np.array(cal.world['cube']['position']); bowl=np.array(cal.world['bowl']['position'])
    stacking=capture.get('task')=='stack'
    destination=np.array(cal.world['target_cube']['position']) if stacking else bowl
    placement=destination.copy()
    if stack_offset is not None:placement+=np.asarray(stack_offset,float)
    # Center the OPEN mouth over the cube during descent. Only one jaw moves:
    # using the closed-grasp center while open leaves the fixed jaw on the cube's
    # edge, so small real tracking errors make it land on top instead of straddle.
    # For closing/holding, retain the nominal occupied aperture, not empty closure.
    aperture=.15
    def center(q):
        d.qpos[:6]=q;d.qpos[5]=max(aperture,q[5]);mujoco.mj_forward(m,d)
        return (d.geom('fixed_pad').xpos+d.geom('moving_pad').xpos)/2
    start=center(current).copy()
    reference=np.array([0.,-20.,65.,45.,0.,0.])
    # Use a recorded successful grasp posture only as the IK branch seed.
    from dataset_to_sim.dataset import DatasetIndex
    from episode0 import DATA
    ep=DatasetIndex(DATA).episode(0)
    close=np.flatnonzero((ep.action_deg[1:,5]<5)&(ep.action_deg[:-1,5]>=5))[0]+1
    reference=adapter.to_model(ep.action_deg[min(close+3,len(ep.action_deg)-1)])
    roll_offset=reference[4]-reference[0]
    grasp_z=max(cube[2]+.010, .024)
    hover=max(.12,grasp_z+.09,bowl[2]+cal.world['bowl']['height']+.065)
    errors=[]; waypoints=[(initial,0)]; names=['Start']; stages=[]
    def waypoint(name,q,seconds):
        waypoints.append((q,seconds));names.append(name)
    def add(name,xyz,grip,seed=None,seconds=1.):
        nonlocal current
        prior=current.copy(); seed=prior if seed is None else seed
        opening=adapter.to_model(np.array([0,0,0,0,0,grip]))[5]
        def residual(x):
            q=np.r_[x,opening]
            return np.r_[center(q)-xyz,.012*(x[1:4].sum()-np.pi/2),.008*(x[4]-x[0]-roll_offset),.00003*(x-prior[:5])]
        fit=least_squares(residual,np.clip(seed[:5],lo,hi),bounds=(lo,hi),max_nfev=180,gtol=1e-10)
        current=np.r_[fit.x,opening]
        err=float(np.linalg.norm(center(current)-xyz)); errors.append(err)
        if err>.004: raise ValueError(f'{name}: IK target unreachable ({err*1000:.1f} mm error)')
        q=adapter.to_native(current);q[5]=grip
        waypoint(name,q,seconds);stages.append({'name':name,'target_m':np.asarray(xyz).tolist(),'ik_error_mm':err*1000})
    # Unfold the distal joint before a vertical lift from the compact rest pose.
    # Otherwise an IK solution can drive the jaws through the shoulder housing.
    if start[2] < .08 and start[0] < cube[0]:
        unfolded=initial.copy();unfolded[3]=max(limits[3,0],initial[3]-45.)
        waypoint('Unfold from rest',unfolded,2.5);current=adapter.to_model(unfolded)
        start=center(current).copy()
        stages.append({'name':'Unfold from rest','target_m':start.tolist(),'ik_error_mm':0.})
    add('Raise',np.r_[start[:2],max(hover,start[2])],initial[5],seconds=1.5)
    add('Open above cube',np.r_[cube[:2],hover],35.,reference,1.5)
    add('Approach',np.r_[cube[:2],grasp_z],35.,seconds=1.5)
    add('Grasp',np.r_[cube[:2],grasp_z],0.,seconds=1.)
    add('Lift',np.r_[cube[:2],hover],0.,seconds=1.5)
    add('Transfer',np.r_[placement[:2],hover],0.,seconds=2.)
    if stacking:
        place_z=grasp_z+(placement[2]-cube[2])+cal.world['target_cube']['size']+.005
        add('Lower onto cube',np.r_[placement[:2],place_z],0.,seconds=2.)
        waypoint('Placement settle',waypoints[-1][0].copy(),.7)
        placement_knot=len(waypoints)-1
        add('Release on cube',np.r_[placement[:2],place_z],40.,seconds=1.5)
        add('Retreat',np.r_[placement[:2],hover],40.,seconds=2.)
        home=json.loads((ROOT/'calibration/studio-home.json').read_text())
        if capture['calibration_hash']!=home['calibration_sha256']:raise ValueError('Home calibration changed')
        closed=waypoints[-1][0].copy();closed[5]=1.
        for name,q,seconds in [('Close after retreat',closed,1.5),('Return raised',home['raised'],3.),('Return unfolded',home['unfolded'],6.),('Home',home['home'],2.5)]:waypoint(name,q,seconds)
        stages.append({'name':'Return Home','target_m':None,'ik_error_mm':0.})
        waypoint('Final settle',waypoints[-1][0].copy(),2.)
    else:
        add('Release',np.r_[bowl[:2],hover],40.,seconds=1.)
        waypoint('Final settle',waypoints[-1][0].copy(),1.5)
    classic_actions,classic_times=interpolate_native(waypoints)
    motion={'technique':technique,'label':TECHNIQUES[technique],'classic_duration':float(classic_times[-1]+1/30)}
    if technique!='classic':
        travel={'Unfold from rest','Raise','Open above cube','Lift','Transfer','Retreat','Return raised','Return unfolded','Home'}
        flowing=[(q,seconds*.55 if name in travel else seconds) for (q,seconds),name in zip(waypoints,names)]
        blend={i for i,name in enumerate(names) if name in {'Raise','Open above cube','Lift','Transfer','Retreat','Return raised','Return unfolded'}}
        if not stacking:blend.discard(names.index('Transfer'))
        actions,timestamps,detail=(interpolate_ruckig if technique=='flow' else interpolate_flow)(flowing,blend)
        motion.update(detail)
        if stacking:placement_index=detail['knot_indices'][placement_knot]
    else:
        actions,timestamps=classic_actions,classic_times
        if stacking:placement_index=len(interpolate_native(waypoints[:placement_knot+1])[0])-1
    motion['waypoints']=names
    motion['duration']=float(timestamps[-1]+1/30)
    motion['time_saved_seconds']=motion['classic_duration']-motion['duration']
    validate_trajectory(actions,timestamps,limits)
    np.savez_compressed(folder/'plan.npz',actions=actions,timestamps=timestamps)
    report={**capture,'scene':scene,'world':cal.world,'stages':stages,'frames':len(actions),'duration':float(timestamps[-1]+1/30),
            'action_hash':array_hash(actions),'outcome':'not_simulated','transfer_checks_passed':False,'executed':False,
            'method':'Aperture-aware IK centers the open jaws on descent; smooth closing waypoints; contact-driven MuJoCo validation required',
            'max_ik_error_mm':max(errors)*1000,'limits':limits.tolist(),'motion_technique':technique,'motion':motion}
    if stacking:report.update(placement_index=placement_index,placement_offset_m=np.asarray(stack_offset if stack_offset is not None else [0,0,0]).tolist())
    (folder/'plan.json').write_text(json.dumps(report,indent=2))
    (folder/'scene.json').write_text(json.dumps(cal.data,indent=2))
    mujoco.mj_saveLastXML(str(folder/'scene.xml'),m)
    if preview:render_preview(m,adapter.to_model(initial),folder/'preview.jpg')
    return report


def render_preview(model,q,path):
    d=mujoco.MjData(model);d.qpos[:6]=q;mujoco.mj_forward(model,d)
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        renderer.update_scene(d,camera='top_cam')
        cv2.imwrite(str(path),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))


def prepare_plan(folder,technique=None):
    """Plan stack placement against the predicted held object, not pad center alone."""
    from studio_stack import predicted_held_position
    options=folder/'planning-options.json'
    technique=validate_technique(technique if technique is not None else json.loads(options.read_text()).get('motion_technique','classic') if options.exists() else 'classic')
    report=plan(folder,preview=False,technique=technique)
    if report.get('task')=='stack':
        offset=np.zeros(3);iterations=[]
        goal=np.asarray(report['world']['target_cube']['position'],float).copy()
        goal[2]+=report['world']['target_cube']['size']+.004
        for attempt in range(5):
            capture,config,scene,cal,model=load_scene(folder);adapter=NativeJointAdapter(**config['adapter']);saved=np.load(folder/'plan.npz')
            held=predicted_held_position(model,adapter,capture['initial_native'],saved['actions'],saved['timestamps'],report['placement_index'])
            error=goal-held;iterations.append({'offset_m':offset.tolist(),'held_center_m':held.tolist(),'goal_m':goal.tolist(),'error_m':error.tolist()})
            if np.linalg.norm(error[:2])<.0015 and abs(error[2])<.002:break
            candidate=offset+error*.85
            if np.any(np.abs(candidate)>.020):raise ValueError('Stack placement correction exceeds 20 mm; reposition the cubes and capture again')
            if attempt==4:raise ValueError('Stack placement did not converge; reposition the cubes and capture again')
            offset=candidate;report=plan(folder,stack_offset=offset,preview=False,technique=technique)
        report['stack_planning']={'method':'Bounded contact-physics prediction of held cube center; scene and physical parameters unchanged','iterations':iterations,'maximum_offset_m':.020,'release_gap_m':.004}
        (folder/'plan.json').write_text(json.dumps(report,indent=2))
    capture,config,scene,cal,model=load_scene(folder)
    render_preview(model,NativeJointAdapter(**config['adapter']).to_model(capture['initial_native']),folder/'preview.jpg')
    return report


def simulate(folder):
    report=json.loads((folder/'plan.json').read_text()); capture,config,scene,cal,m=load_scene(folder)
    adapter=NativeJointAdapter(**config['adapter']); saved=np.load(folder/'plan.npz');actions=saved['actions'];ts=saved['timestamps']
    if array_hash(actions)!=report['action_hash']: raise ValueError('Plan changed since generation')
    validate_trajectory(actions,ts,native_limits(config))
    states=np.tile(report['initial_native'],(len(actions),1))
    ep=Episode(-1,np.arange(len(actions)),np.arange(len(actions)),ts,states,actions,{})
    trace=integrate_actions(m,ep,adapter)
    if report.get('task')=='stack':
        inside=None;stack=stack_verdict(trace,cal.world['cube']['size']);outcome=stack['outcome'];report['stack_assessment']=stack
    else:
        inside=containment(m,trace['qpos'][-1],trace['cube_pose'][-1],cal,config);outcome=score(trace,inside)
    collision={k:v for k,v in trace['audit']['contact_audit'].items() if k!='cube'}
    checks=trace['audit']['initial_max_penetration_m']<=.001 and all(v['max_penetration_m']<=.001 for v in collision.values())
    tracking=np.abs(adapter.to_native(trace['qpos'][1:])-actions)
    checks=bool(checks and np.all(np.max(tracking,axis=0)<=np.array([10.]*5+[25.])))
    report.update(outcome=outcome,transfer_checks_passed=checks,collision_audit=collision,audit=trace['audit'],bowl_containment=inside,
                  max_cube_height_m=float(trace['cube_pose'][:,2].max()),max_tracking_error=np.max(tracking,axis=0).tolist(),
                  simulated_action_hash=array_hash(trace['native_action_applied']))
    extra={k:trace[k] for k in ['target_pose','target_velocity'] if k in trace}
    np.savez_compressed(folder/'trace.npz',qpos=trace['qpos'],cube_pose=trace['cube_pose'],native=adapter.to_native(trace['qpos']),timestamps=trace['sample_times'],**extra)
    # Save the physical verdict even if the graphics environment fails.
    (folder/'plan.json').write_text(json.dumps(report,indent=2))
    try:
        render_trace(m,trace,folder)
        report['video']='simulation.mp4'
    except Exception as exc:
        report['render_error']=str(exc)
    (folder/'plan.json').write_text(json.dumps(report,indent=2))
    return report


def render_trace(m,trace,folder):
    d=mujoco.MjData(m);cube_addr=m.joint('cube_free').qposadr[0]
    raw=folder/'simulation-raw.mp4'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,480))
    if not writer.isOpened():raise RuntimeError('Video writer unavailable')
    option=mujoco.MjvOption();option.geomgroup[3]=0
    try:
        with mujoco.Renderer(m,height=480,width=640) as renderer:
            for index,(q,cube) in enumerate(zip(trace['qpos'],trace['cube_pose'])):
                d.qpos[:6]=q;d.qpos[cube_addr:cube_addr+7]=cube;mujoco.mj_forward(m,d)
                if 'target_pose' in trace:
                    address=m.joint('target_cube_free').qposadr[0];d.qpos[address:address+7]=trace['target_pose'][index];mujoco.mj_forward(m,d)
                frames=[]
                for view in ['top_cam','wrist_cam']:
                    renderer.update_scene(d,camera=view,scene_option=option);frames.append(renderer.render().copy())
                writer.write(cv2.cvtColor(np.hstack(frames),cv2.COLOR_RGB2BGR))
    finally:writer.release()
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'simulation.mp4')],check=True)
    raw.unlink()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['plan','simulate']);parser.add_argument('folder',type=Path);args=parser.parse_args()
    result=prepare_plan(args.folder) if args.mode=='plan' else simulate(args.folder)
    print(json.dumps({'outcome':result['outcome'],'frames':result['frames']}))
