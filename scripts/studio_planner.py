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
from dataset_to_sim.studio import native_limits, validate_trajectory, motion_speed_limits, validate_speed_scale
from studio_motion import interpolate_flow, interpolate_ruckig, validate_technique, TECHNIQUES


JOINT_LABELS=['Shoulder pan','Shoulder lift','Elbow','Wrist flex','Wrist roll','Gripper']
HARDWARE_SHOULDER_LIFT_MARGIN_DEG=8.
# Gap left between a box and the fixed jaw while the jaws come down around it.
FIXED_JAW_GAP_M=.004


def interpolate_native(waypoints,speed_scale=1.,derivative_limits=None):
    speed_limits=motion_speed_limits(speed_scale) if derivative_limits is None else np.asarray(derivative_limits)[0]
    result=[np.asarray(waypoints[0][0],float)]
    for target,minimum_duration in waypoints[1:]:
        start=result[-1]; target=np.asarray(target,float)
        duration=max(minimum_duration,float(np.max(1.51*np.abs(target-start)/(speed_limits*.8))))
        n=max(1,int(np.ceil(duration*30)))
        for f in range(1,n+1):
            a=f/n; a=a*a*(3-2*a)
            result.append(target.copy() if f == n else (1-a)*start+a*target)
    return np.array(result),np.arange(len(result))/30


def load_pickplace_scene(folder,capture,config,candidate=None):
    """Pick-and-place: every object is a free body; the spec comes from the contact registry."""
    import copy
    from episode0 import Calibration
    from pickplace import load_registry,settle,build_model as pickplace_model,plan_spec,height,object_quaternion
    registry=load_registry();scene=json.loads((folder/'pickplace-scene.json').read_text())
    capture.update(selection_revision=scene.get('selection_revision'))
    settle(scene,registry,config)
    # Measured arm placement error over the mat (workspace calibration). A Wrist-checked
    # object was measured in the arm's own frame already, so it is not shifted again.
    from pickplace import load_correction,apply_correction
    apply_correction(scene,load_correction(),keep={scene['pick']} if capture.get('purpose')=='refine' else ())
    if candidate is None:candidate=int(scene.get('grasp_candidate',0))
    cal=Calibration(ROOT/'calibration/native-action-current.json',copy.deepcopy(config['base_calibration']))
    bowl=next((o for o in scene['objects'] if o['kind']=='bowl'),None)
    if bowl:cal.world['bowl']['position']=list(bowl['position']);cal.world['bowl']['quaternion']=[float(np.cos(bowl.get('yaw',0)/2)),0.,0.,float(np.sin(bowl.get('yaw',0)/2))]
    picked=next(o for o in scene['objects'] if o['id']==scene.get('pick'))
    shape=registry['objects'][picked['kind']]['shape']
    cal.world['cube']=dict(position=list(picked['position']),quaternion=object_quaternion(picked),
                           size=float(height(shape)),kind=picked['kind'])
    options=json.loads((folder/'planning-options.json').read_text()) if (folder/'planning-options.json').exists() else {}
    spec=plan_spec(scene,registry,config,candidate,options.get('grasp_raise_m',0.))
    model=pickplace_model(cal,config,scene,registry)
    from pickplace_initialization import initialize_scene
    initialize_scene(model,scene,registry,NativeJointAdapter(**config['adapter']).to_model(capture['initial_native']),folder/'initial-state.json')
    if scene['initialization']['accepted_poses']:
        from pickplace import object_quaternion
        cal.world['cube'].update(position=list(picked['position']),quaternion=object_quaternion(picked))
        spec=plan_spec(scene,registry,config,candidate,options.get('grasp_raise_m',0.))
        model=pickplace_model(cal,config,scene,registry)
    _check_home_clear(model,config,scene,spec,registry,shape)
    cal.world['pickplace']=spec;cal.world['objects']=scene['objects']
    return capture,config,scene,cal,model


def _check_home_clear(model,config,scene,spec,registry,shape):
    """Every episode ends at Home: the placed object must not end up under the arm's
    Home pose (a cup set near the base was pinned by the gripper and never released)."""
    from pickplace import footprint_radius
    home=json.loads((ROOT/'calibration/studio-home.json').read_text())['home']
    d=mujoco.MjData(model);d.qpos[:6]=NativeJointAdapter(**config['adapter']).to_model(np.asarray(home,float));mujoco.mj_kinematics(model,d)
    objects={model.body(o['body']).id for o in scene['objects'] if o.get('body')}|{model.body('cube').id}
    bottom=spec['place']['bottom_z'];top=bottom+(shape['size_m'][2] if shape['type']=='box' else shape.get('height_m',shape.get('diameter_m',.02)))
    xy=np.asarray(spec['place']['xy'],float);r=min(footprint_radius(shape),.06)
    for g in range(model.ngeom):
        b=model.geom_bodyid[g]
        if b in objects or model.body_weldid[b]==0 or not model.geom_contype[g]:continue
        half=np.abs(d.geom_xmat[g].reshape(3,3))@model.geom_aabb[g][3:];centre=d.geom_xpos[g]+d.geom_xmat[g].reshape(3,3)@model.geom_aabb[g][:3]
        if centre[2]-half[2]>top+.008:continue
        gap=np.maximum(np.abs(centre[:2]-xy)-half[:2],0.)
        if np.linalg.norm(gap)<r+.008:
            raise ValueError(f"To is under the arm's Home pose: the {spec['kind']} would be pinned by the arm when it returns Home; choose a spot further from the robot")


def load_scene(folder,candidate=None):
    capture=json.loads((folder/'capture.json').read_text())
    if capture.get('task')=='pickplace':
        from top_camera import live_config
        config=live_config()  # the live top camera, when it has been calibrated
        return load_pickplace_scene(folder,capture,config,candidate)
    if capture.get('source')=='synthetic':
        from studio_generation_scene import load_generation_scene
        return load_generation_scene(folder)
    config=json.loads((ROOT/'calibration/native-action-current.json').read_text())
    rgb=cv2.cvtColor(cv2.imread(str(folder/'top.jpg')),cv2.COLOR_BGR2RGB)
    if rgb.shape != (480,640,3): raise ValueError('Calibrated top camera requires 640 × 480 images')
    if capture.get('source')=='live':
        from studio_sam_geometry import load_selection,scene_from_masks
        selection,masks=load_selection(folder)
        capture.update(selection_revision=selection['revision'],selection=selection)
        scene=scene_from_masks(config,*masks,task=capture.get('task','bowl'),bowl_present=capture.get('bowl_present'),rgb=rgb)
    else:scene=estimate_scene(config,rgb)
    cal=episode_scene(config,scene)
    return capture,config,scene,cal,build_model(cal,config)


def plan(folder,stack_offset=None,preview=True,technique="classic",speed_scale=1.,timing_profile=None,hover_extra=0.,grasp_turn=None,grasp_candidate=None):
    """`hover_extra` raises the travel height and `grasp_turn` (0 or 1) closes the jaws
    across a chosen cube face pair; both only change the route around obstacles."""
    validate_technique(technique)
    speed_scale=validate_speed_scale(speed_scale)
    capture,config,scene,cal,m=load_scene(folder) if grasp_candidate is None else load_scene(folder,grasp_candidate)
    adapter=NativeJointAdapter(**config['adapter']); limits=native_limits(config)
    initial=np.array(capture['initial_native'],float); current=adapter.to_model(initial)
    d=mujoco.MjData(m)
    bounds=adapter.to_model(np.array([limits[:,0],limits[:,1]]))[:,:5]
    lo,hi=np.min(bounds,axis=0)+1e-6,np.max(bounds,axis=0)-1e-6
    cube=np.array(cal.world['cube']['position']); bowl=np.array(cal.world['bowl']['position'])
    stacking=capture.get('task')=='stack'
    unstacking=capture.get('task')=='unstack'
    placing=stacking or unstacking
    destination=np.array(cal.world['placement']['position']) if unstacking else np.array(cal.world['target_cube']['position']) if stacking else bowl
    placement=destination.copy()
    if stack_offset is not None:placement+=np.asarray(stack_offset,float)
    # Center the OPEN mouth over the cube during descent. Only one jaw moves:
    # using the closed-grasp center while open leaves the fixed jaw on the cube's
    # edge, so small real tracking errors make it land on top instead of straddle.
    # For closing/holding, retain the nominal occupied aperture, not empty closure.
    aperture=.15
    # Native gripper commands: open around the object, and open to release.
    open_grip,release_grip,close_grip=35.,40.,0.
    from tape_object import is_tape,grasp_width,jaw_openings,FIXED_PAD_RADIUS_M
    tape=is_tape(cal.world)
    pp=cal.world.get('pickplace')
    if pp:
        # One algorithm for every object: grasp, carry and place from the contact registry.
        g=pp['grasp'];tape=g['pads']=='fixed_in_hole'
        cube=np.array(g['xyz'],float)
        placing=pp['place']['release']=='lower' and not tape
        placement=np.r_[pp['place']['pad_xy'],0.]
        if stack_offset is not None:placement+=np.asarray(stack_offset,float)
        if not tape and g['native']!='from_width':open_grip,close_grip,release_grip=float(g['native']['open']),float(g['native']['close']),float(g['native']['release'])
    if tape:
        # The roll is grasped through its hole: the jaws pinch its measured wall.
        aperture,open_grip,close_grip,release_grip=jaw_openings(m,d,adapter,current,grasp_width())
    elif pp and pp['grasp']['native']=='from_width':
        aperture,open_grip,close_grip,release_grip=jaw_openings(m,d,adapter,current,pp['grasp']['width_m'])
    fixed_ref=None
    tape_radius=FIXED_PAD_RADIUS_M
    if pp and not tape and pp['grasp']['pads']=='between' and pp.get('shape') in ('box','cylinder'):
        fixed_ref=float(m.geom('fixed_pad').size[0])+FIXED_JAW_GAP_M+pp['grasp']['width_m']/2
    def center(q):
        if tape:
            # Fixed jaw in the core hole, moving jaw outside: the reference point is the
            # roll axis implied by the fixed pad. Enter centrally; carry from the
            # inner-wall contact after the moving jaw has slid the roll onto it.
            d.qpos[:6]=q;d.qpos[5]=aperture;mujoco.mj_forward(m,d)
            fixed=d.geom('fixed_pad').xpos;toward=(d.geom('moving_pad').xpos-fixed)[:2]
            return np.r_[fixed[:2]-tape_radius*toward/np.linalg.norm(toward),fixed[2]]
        d.qpos[:6]=q;d.qpos[5]=max(aperture,q[5]);mujoco.mj_forward(m,d)
        if fixed_ref is not None:
            # A box is set beside the fixed jaw, not midway between the pads: the real
            # moving jaw opens less than the model, and midway it came down on the cube.
            # The moving jaw then presses the box against the fixed one; the reference
            # does not depend on the opening, so the arm holds still while closing.
            fixed=d.geom('fixed_pad').xpos;toward=(d.geom('moving_pad').xpos-fixed)[:2]
            return np.r_[fixed[:2]+fixed_ref*toward/np.linalg.norm(toward),fixed[2]]
        return (d.geom('fixed_pad').xpos+d.geom('moving_pad').xpos)/2
    def jaw_angle():
        # Horizontal direction the jaws close along, after center(q) ran forward kinematics.
        v=d.geom('moving_pad').xpos-d.geom('fixed_pad').xpos;return np.arctan2(v[1],v[0])
    def yaw_of(item):
        q=np.asarray(item['quaternion'],float);return 2*np.arctan2(q[3],q[0])
    # Generated scenes may rotate cubes freely: close the jaws across a face pair
    # (any multiple of 90 deg). Real New Run scenes keep the original roll rule.
    # A hand-stacked top cube is often turned: close the jaws across its own faces.
    align=bool(cal.world.get('align_grasp_yaw',False)) or (grasp_turn is not None and not tape) or (unstacking and capture.get('source')=='live') or bool(pp)
    wrap90=lambda a:(a+np.pi/4)%(np.pi/2)-np.pi/4
    if pp:wrap90=lambda a:(a+np.pi/2)%np.pi-np.pi/2  # registry jaw axes are exact (period 180°: two symmetric pads)
    start=center(current).copy()
    reference=np.array([0.,-20.,65.,45.,0.,0.])
    # Use a recorded successful grasp posture only as the IK branch seed.
    from dataset_to_sim.dataset import DatasetIndex
    from episode0 import DATA
    ep=DatasetIndex(DATA).episode(0)
    close=np.flatnonzero((ep.action_deg[1:,5]<5)&(ep.action_deg[:-1,5]>=5))[0]+1
    reference=adapter.to_model(ep.action_deg[min(close+3,len(ep.action_deg)-1)])
    roll_offset=reference[4]-reference[0]
    # A stack needs a shallower bite: the standard jaw extends below the
    # aperture center and otherwise closes around both stacked cubes.
    grasp_z=max(cube[2]+(.022 if unstacking else .006 if tape else .010), .024)
    hover=max(.12,grasp_z+.09,bowl[2]+cal.world['bowl']['height']+.065)
    if pp:grasp_z=max(pp['grasp']['z_cmd'],.024);hover=pp['hover_m']
    # Travel high enough to carry the object over every obstacle top as well.
    for item in cal.world.get('obstacles',[]):
        top=item['position'][2]+(item['height'] if item['kind']=='bowl' else item.get('height',item.get('width',item.get('size',0.)))/2)
        hover=max(hover,top+.065)
    hover+=hover_extra
    errors=[]; waypoints=[(initial,0)]; names=['Start']; stages=[]
    def waypoint(name,q,seconds):
        waypoints.append((q,seconds));names.append(name)
    def add(name,xyz,grip,seed=None,seconds=1.,yaw=None,pitch=np.pi/2,exact=False,dry=False,flat=False):
        """IK to a pad-centre target and append it; `exact` matches the jaw direction
        over a full turn (not modulo 180°); `dry` only returns the solution."""
        nonlocal current
        prior=current.copy(); seed=prior if seed is None else seed
        opening=adapter.to_model(np.array([0,0,0,0,0,grip]))[5]
        yaw=yaw if align else None
        turn=(lambda a:(a+np.pi)%(2*np.pi)-np.pi) if exact else wrap90
        def residual(x):
            q=np.r_[x,opening];c=center(q)
            roll=.05*turn(jaw_angle()-yaw) if yaw is not None else .008*(x[4]-x[0]-roll_offset)
            level=[]
            if tape:
                # The moving jaw swings in an arc; keep both pads level at the held
                # width so it meets the outer wall, not the top face. center(q) left
                # the model at that held aperture.
                level=[d.geom('fixed_pad').xpos[2]-d.geom('moving_pad').xpos[2]]
            if flat:
                # Jaws close horizontally (across the arm plane), so a held pen hangs
                # upright; otherwise the jaw-direction term is ill-defined.
                level=[d.geom('fixed_pad').xpos[2]-d.geom('moving_pad').xpos[2]]
            # pitch: lift+elbow+wrist flex sum; pi/2 points the jaws down, 0 away from the robot.
            return np.r_[c-xyz,.012*(x[1:4].sum()-pitch),roll,level,.00003*(x-prior[:5])]
        fit=least_squares(residual,np.clip(seed[:5],lo,hi),bounds=(lo,hi),max_nfev=180,gtol=1e-10)
        if dry:
            q=np.r_[fit.x,opening];return q,float(np.linalg.norm(center(q)-xyz))
        current=np.r_[fit.x,opening]
        err=float(np.linalg.norm(center(current)-xyz)); errors.append(err)
        if err>.004: raise ValueError(f'{name}: IK target unreachable ({err*1000:.1f} mm error)')
        q=adapter.to_native(current);q[5]=grip
        waypoint(name,q,seconds);stages.append({'name':name,'target_m':np.asarray(xyz).tolist(),'ik_error_mm':err*1000})
    def return_home():
        # Every episode ends at Home through the calibrated raised/unfolded corridor.
        home=json.loads((ROOT/'calibration/studio-home.json').read_text())
        if capture['calibration_hash']!=home['calibration_sha256']:raise ValueError('Home calibration changed')
        closed=waypoints[-1][0].copy();closed[5]=1.
        for name,q,seconds in [('Close after retreat',closed,1.5),('Return raised',home['raised'],3.),('Return unfolded',home['unfolded'],6.),('Home',home['home'],2.5)]:waypoint(name,q,seconds)
        stages.append({'name':'Return Home','target_m':None,'ik_error_mm':0.})
        waypoint('Final settle',waypoints[-1][0].copy(),2.)
    # After a Wrist check the arm already holds still just above the object: only a
    # short correction to the measured pre-grasp remains, so no raise or travel.
    refining=capture.get('purpose')=='refine'
    source_yaw=yaw_of(cal.world['cube'])+(np.pi/2*grasp_turn if grasp_turn and not tape else 0.)
    # Stacking turns the held cube to the lower cube's faces; other placements are free.
    place_yaw=yaw_of(cal.world['target_cube']) if stacking else None
    if pp:source_yaw=pp['grasp']['jaw_yaw'];place_yaw=pp['place']['yaw']
    if not refining:
        # Unfold the distal joint before a vertical lift from the compact rest pose.
        # Otherwise an IK solution can drive the jaws through the shoulder housing.
        # From the compact rest pose the jaws sit on the mat next to the shoulder: unfold
        # first whenever starting low (objects close to the base included).
        if start[2] < .08 and (start[0] < cube[0] or pp):
            unfolded=initial.copy();unfolded[3]=max(limits[3,0],initial[3]-45.)
            waypoint('Unfold from rest',unfolded,2.5);current=adapter.to_model(unfolded)
            start=center(current).copy()
            stages.append({'name':'Unfold from rest','target_m':start.tolist(),'ik_error_mm':0.})
        add('Raise',np.r_[start[:2],max(hover,start[2])],initial[5],seconds=1.5)
        add('Open above cube',np.r_[cube[:2],hover],open_grip,reference,1.5,yaw=source_yaw)
    # Live grasps hold still at the pre-grasp, 45 mm above the grasp with the grasp's
    # wrist turn, so the Wrist camera measures the object right before the short
    # final descent; the rest is then planned and simulated again from that measure.
    options=json.loads((folder/'planning-options.json').read_text()) if (folder/'planning-options.json').exists() else {}
    # Experimental: at present the Wrist fit relies on the arm model's absolute camera pose.
    look=(options.get('wrist_check',False) and technique=='natural' and capture.get('source')=='live' and not refining and not tape
          and capture.get('task') in ('bowl','stack','unstack','pickplace') and not (pp and pp['kind']=='bowl'))
    if technique=='natural' or refining:
        # End the coordinated travel above contact height. The short final
        # descent keeps the yaw aligned before the jaws surround the object.
        add('Pregrasp',np.r_[cube[:2],grasp_z+.045],open_grip,seconds=1. if refining else .6,yaw=source_yaw)
    if look:waypoint('Look',waypoints[-1][0].copy(),.8)
    add('Approach',np.r_[cube[:2],grasp_z],open_grip,seconds=1.5,yaw=source_yaw)
    if technique=='natural':waypoint('Grasp settle',waypoints[-1][0].copy(),.4)
    add('Grasp',np.r_[cube[:2],grasp_z],close_grip,seconds=1.,yaw=source_yaw);grasp_q=current.copy()
    if tape:
        grasp_check_native=waypoints[-1][0].copy()
        waypoint('Grip check',grasp_check_native,.4)
        from tape_object import dimensions
        tape_radius=dimensions()['core_inner_diameter_m']/2-float(m.geom('fixed_pad').size[0])
    add('Lift',np.r_[cube[:2],hover],close_grip,seconds=1.5,yaw=source_yaw)
    add('Transfer',np.r_[placement[:2],hover],close_grip,seconds=2.,yaw=place_yaw)
    if placing:
        place_z=pp['place']['pad_z']+(stack_offset[2] if stack_offset is not None else 0.) if pp else grasp_z+(placement[2]-cube[2])+(cal.world['target_cube']['size'] if stacking else 0)+.005
        add('Lower onto cube' if stacking else 'Lower onto table',np.r_[placement[:2],place_z],close_grip,seconds=2.,yaw=place_yaw)
        waypoint('Placement settle',waypoints[-1][0].copy(),.7)
        placement_knot=len(waypoints)-1
        add('Release on cube' if stacking else 'Release on table',np.r_[placement[:2],place_z],release_grip if pp else 40.,seconds=1.5,yaw=place_yaw)
        add('Retreat',np.r_[placement[:2],hover],40.,seconds=2.)
        return_home()
    elif pp and pp['place']['release']=='tilt':
        # A long object into a container (a pen into a cup): above the container, turn
        # the wrist so the gripper points away from the robot and the object hangs
        # upright, lower its end into the container, let go, and back out upwards.
        t=pp['place'];xy=placement[:2];base_xy=np.asarray(config['base_calibration']['world']['robot_base']['position'][:2])
        # Which end hangs down depends on which way the jaws face (a full turn, not
        # modulo 180°): carry the held pose to each choice and keep the one that puts
        # the registry's down end (the far end from the grasp) into the container.
        gripper=m.geom_bodyid[m.geom('fixed_pad').id]
        d.qpos[:6]=grasp_q;mujoco.mj_forward(m,d);held=d.xmat[gripper].reshape(3,3).T@d.xmat[m.body('cube').id].reshape(3,3)
        def stand(z,pitch,yaw):
            # Pad XY such that the pen crosses the rim height over the container centre
            # (a tilted gripper tilts the pen as well).
            target=np.asarray(xy,float)
            # The wrist roll may need half a turn from the carry pose: seed IK from both.
            seeds=[current.copy()]
            for turn in (np.pi,-np.pi):
                seed=current.copy();seed[4]=np.clip(seed[4]+turn,lo[4],hi[4]);seeds.append(seed)
            def solve(target):
                return min((add('',np.r_[target,z],close_grip,seed=seed,yaw=yaw,pitch=pitch,exact=True,dry=True,flat=True) for seed in seeds),key=lambda r:r[1])
            for _ in range(3):
                q,err=solve(target)
                ref=center(q);axis=(d.xmat[gripper].reshape(3,3)@held)[:,0]
                if abs(axis[2])<.3:return q,1.,target,axis
                target=np.asarray(xy,float)-(t['rim_z']-ref[2])/axis[2]*axis[:2]
            return q,err,target,axis
        # Arm geometry (wrist camera mount included) must stay clear of the container while
        # the pen is lowered: near the base the camera mount under the turned wrist hit the cup.
        dest=next(o for o in cal.world['objects'] if o['id']==t['object'])
        cont=m.body(dest['body']).id
        objects={m.body(o['body']).id for o in cal.world['objects'] if o.get('body')}|{m.body('cube').id}
        cont_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==cont and m.geom_contype[g]]
        arm_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g] not in objects and m.body_weldid[m.geom_bodyid[g]]!=0 and m.geom_contype[g]]
        def clearance(q):
            d.qpos[:]=m.qpos0;d.qpos[:6]=q;mujoco.mj_forward(m,d)
            return min(mujoco.mj_geomDistance(m,d,a,b,.03,None) for a in arm_geoms for b in cont_geoms)
        # Pointing straight ahead first; far containers need the gripper tipped down a little,
        # near ones tipped up (the wrist camera mount hangs below the turned wrist).
        camera=[g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ('camera_mount',) or m.geom(g).name.startswith('wrist_camera')]
        def camera_up(q):
            # The wrist camera side faces up (the user's suggestion): it then cannot hit the rim.
            d.qpos[:6]=q[:6];mujoco.mj_forward(m,d)
            return not camera or min(d.geom_xpos[g][2] for g in camera)>d.geom('fixed_pad').xpos[2]
        # Preference: camera up with the far end down; far end down; camera up; anything upright.
        best=None
        for tier,pitch,yaw,raise_ in [(k_,p_,y_,r_) for k_ in range(4) for r_ in (0.,.015,.03) for p_ in (0.,-.15,.15,-.3,.3,.45,.6) for y_ in (place_yaw,place_yaw+np.pi)]:
            if best:break
            if True:
                q_hi,err_hi,xy_hi,axis=stand(t['tilt_z'],pitch,yaw)
                far_down=axis[2]*t['down_end_sign']<-.7
                if err_hi>.004 or abs(axis[2])<.7 or (tier in (0,1) and not far_down) or (tier in (0,2) and not camera_up(q_hi)):continue
                q_lo,err_lo,xy_lo,_=stand(t['pad_z_tilt']+raise_,pitch,yaw)
                if err_lo>.004:continue
                # Clear of the container at the bottom and half way down, by 18 mm: the loaded arm
                # sags and trails its command by about that much here.
                q_mid,_,_,_=stand((t['pad_z_tilt']+raise_+t['tilt_z'])/2,pitch,yaw)
                if min(clearance(q_lo[:6]),clearance(q_mid[:6]))<.018:continue
                best=(pitch,yaw,xy_hi,xy_lo,raise_)
                if tier>getattr(plan,'_stand_tier',3):
                    raise ValueError('Standing the pen up: another grasp keeps the wrist camera up with the far end down')
        if best is None:raise ValueError('Standing the pen up: the container is out of reach; move it closer to the robot')
        pitch,yaw,xy_hi,xy_lo,raise_=best;pad_z=t['pad_z_tilt']+raise_
        add('Stand upright above container',np.r_[xy_hi,t['tilt_z']],close_grip,seconds=2.,yaw=yaw,pitch=pitch,exact=True,flat=True)
        add('Lower into container',np.r_[xy_lo,pad_z],close_grip,seconds=2.,yaw=yaw,pitch=pitch,exact=True,flat=True)
        waypoint('Placement settle',waypoints[-1][0].copy(),.7)
        # Open fully (a small opening dragged the pen back out), then rise.
        add('Release in container',np.r_[xy_lo,pad_z],open_grip,seconds=1.,yaw=yaw,pitch=pitch,exact=True,flat=True)
        back=xy_lo
        # Short straight steps: a single joint-space move bowed back over the rim.
        add('Rise clear of rim',np.r_[back,pad_z+.04],open_grip,seconds=1.,yaw=yaw,pitch=pitch,exact=True,flat=True)
        add('Rise out of container',np.r_[back,t['tilt_z']],open_grip,seconds=1.5,yaw=yaw,pitch=pitch,exact=True,flat=True)
        # Point the jaws down again up here, so the way Home does not sweep the container.
        add('Turn jaws down',np.r_[back,t['tilt_z']],open_grip,seconds=1.5)
        return_home()
    elif tape:
        # The roll hangs on the fixed jaw through its hole, so it cannot be dropped:
        # lower it onto the bowl floor, open a little, then lift the jaw out.
        place_z=grasp_z+config['estimates']['bowl_floor_thickness_m']+.006
        target=bowl[:2]
        if pp:place_z=pp['place']['pad_z']+.001;target=placement[:2]
        add('Lower into bowl',np.r_[target,place_z],close_grip,seconds=2.)
        waypoint('Placement settle',waypoints[-1][0].copy(),.5)
        add('Release in bowl',np.r_[target,place_z],release_grip,seconds=1.)
        # Not a travel stage: speed scaling never hurries the jaw out of the core.
        add('Lift out of roll',np.r_[target,hover],release_grip,seconds=2.)
        return_home()
    else:
        add('Release',np.r_[(placement if pp else bowl)[:2],hover],release_grip,seconds=1.)
        return_home()
    # Compare against the original Classic route, not Natural's added grasp guards.
    classic_actions,classic_times=interpolate_native([w for w,name in zip(waypoints,names) if name not in {'Pregrasp','Grasp settle'}])
    motion={'technique':technique,'label':TECHNIQUES[technique],'classic_duration':float(classic_times[-1]+1/30)}
    from studio_generation_timing import recorded_timing, generation_durations
    from studio_timing import validate_timing_profile, timing_contract_hash
    timing_profile=validate_timing_profile(timing_profile if timing_profile is not None else capture.get('generation_options',{}).get('timing_profile','conservative') if capture.get('source')=='synthetic' else 'conservative')
    timing=recorded_timing(speed_scale) if timing_profile=='recorded' else None
    derivative_limits=None if timing is None else timing['limits']
    if timing is not None:
        waypoints=generation_durations(waypoints,names)
        if pp:
            # Still holds long enough for the hardware to close the servos' remaining
            # error before the jaws close and before the release.
            waypoints=[(q,max(sec,.6) if name in ('Grasp settle','Placement settle') else sec) for (q,sec),name in zip(waypoints,names)]
        motion.update(timing_profile='recorded',timing_reference=timing,
                      enforced_derivatives=['velocity'] if technique=='classic' else ['velocity','acceleration','jerk'])
    else:motion['timing_profile']='conservative'
    travel={'Unfold from rest','Raise','Open above cube','Lift','Transfer','Retreat','Return raised','Return unfolded','Home'}
    scaled=[(q,seconds/speed_scale if name in travel else seconds) for (q,seconds),name in zip(waypoints,names)]
    if technique=='natural':
        from studio_natural import interpolate_natural
        # A roll can slide within its grip. Clear the source vertically before
        # translating; a blended low lift grazed a cup after a 3 mm source shift.
        actions,timestamps,detail=interpolate_natural(waypoints,names,speed_scale=speed_scale,derivative_limits=derivative_limits,lift_stop=tape or getattr(plan,'_lift_stop',False),return_stop=tape)
        motion.update(detail);names=detail['waypoints']
        if placing:placement_index=detail['task_knot_indices']['Placement settle']
        look_index=detail['task_knot_indices']['Look'] if look else None
    elif technique!='classic':
        flowing=[(q,seconds*.55 if name in travel else seconds) for (q,seconds),name in zip(scaled,names)]
        blend={i for i,name in enumerate(names) if name in {'Raise','Open above cube','Lift','Transfer','Retreat','Return raised','Return unfolded'}}
        if tape:
            blend.discard(names.index('Lift'))
            blend.discard(names.index('Return raised'))
        if not placing:blend.discard(names.index('Transfer'))
        actions,timestamps,detail=(interpolate_ruckig if technique=='flow' else interpolate_flow)(flowing,blend,speed_scale=speed_scale,derivative_limits=derivative_limits)
        motion.update(detail)
        if placing:placement_index=detail['knot_indices'][placement_knot]
    else:
        actions,timestamps=interpolate_native(scaled,speed_scale=speed_scale,derivative_limits=derivative_limits)
        if placing:placement_index=len(interpolate_native(scaled[:placement_knot+1],speed_scale=speed_scale,derivative_limits=derivative_limits)[0])-1
    motion.update(speed_scale=speed_scale,velocity_limits=((motion_speed_limits(speed_scale) if derivative_limits is None else np.asarray(derivative_limits)[0])*(.8 if technique=='classic' else .92)).tolist())
    motion['waypoints']=names
    motion['duration']=float(timestamps[-1]+1/30)
    motion['time_saved_seconds']=motion['classic_duration']-motion['duration']
    validate_trajectory(actions,timestamps,limits,speed_scale=speed_scale,velocity_limits=None if derivative_limits is None else derivative_limits[0])
    np.savez_compressed(folder/'plan.npz',actions=actions,timestamps=timestamps)
    report={**capture,'scene':scene,'world':cal.world,'stages':stages,'frames':len(actions),'duration':float(timestamps[-1]+1/30),
            'action_hash':array_hash(actions),'timestamp_hash':array_hash(timestamps),'speed_scale':speed_scale,'outcome':'not_simulated','transfer_checks_passed':False,'executed':False,
            'method':'Aperture-aware IK centers the open jaws on descent; smooth closing waypoints; contact-driven MuJoCo validation required',
            'max_ik_error_mm':max(errors)*1000,'limits':limits.tolist(),'motion_technique':technique,'motion':motion,'timing_profile':timing_profile,
            'route':{'hover_m':float(hover),'hover_extra_m':float(hover_extra),'grasp_turn':grasp_turn}}
    if capture.get('task')=='pickplace':
        from pickplace_initialization import scene_fingerprint
        report['scene_contract_hash']=scene_fingerprint(m,config,scene)
    if tape:
        from studio_grasp import checkpoint_index
        report['grasp_check_index']=checkpoint_index(actions,timestamps,grasp_check_native)
    if timing is not None:report['timing_contract_hash']=timing_contract_hash(report)
    if technique=='natural' and look:report.update(look_index=int(look_index),look_target_m=np.r_[cube[:2],hover].tolist())
    if placing:report.update(placement_index=placement_index,placement_offset_m=np.asarray(stack_offset if stack_offset is not None else [0,0,0]).tolist())
    (folder/'plan.json').write_text(json.dumps(report,indent=2))
    (folder/'scene.json').write_text(json.dumps(cal.data,indent=2))
    mujoco.mj_saveLastXML(str(folder/'scene.xml'),m)
    if preview:render_preview(m,adapter.to_model(initial),folder/'preview.jpg')
    return report


def rotation_z(a):
    c,s=np.cos(a),np.sin(a);return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def render_preview(model,q,path):
    d=mujoco.MjData(model);d.qpos[:6]=q;mujoco.mj_forward(model,d)
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        renderer.update_scene(d,camera='top_cam')
        cv2.imwrite(str(path),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))


def _prepare_route(folder,technique,speed_scale,timing_profile,hover_extra=0.,grasp_turn=None):
    """One route: plan, and for stack/unstack correct the release against the held object."""
    from studio_stack import predicted_held_position
    pickplace=json.loads((folder/'capture.json').read_text()).get('task')=='pickplace'
    # Pick-and-place routes choose a registry grasp candidate instead of a cube face turn.
    route=dict(hover_extra=hover_extra,grasp_candidate=grasp_turn or 0) if pickplace else dict(hover_extra=hover_extra,grasp_turn=grasp_turn)
    grasp_index=route.get('grasp_candidate')
    report=plan(folder,preview=False,technique=technique,speed_scale=speed_scale,timing_profile=timing_profile,**route)
    if report.get('task') in {'stack','unstack'} or (pickplace and 'placement_index' in report):
        offset=np.zeros(3);iterations=[]
        stacking=report['task']=='stack'
        if pickplace:
            spec=report['world']['pickplace']
            # Set onto another object: release only once the held object rests on it (no
            # gap), so it cannot tip or roll off; elsewhere a 4 mm drop is harmless.
            gap=0. if spec['place']['type']=='top' else .004
            goal=np.r_[spec['place']['xy'],spec['place']['bottom_z']+spec['picked_bottom_offset_m']+gap]
        else:
            goal=np.asarray(report['world']['target_cube' if stacking else 'placement']['position'],float).copy()
            goal[2]+=(report['world']['target_cube']['size'] if stacking else 0)+.004
        for attempt in range(5):
            capture,config,scene,cal,model=load_scene(folder,grasp_index);adapter=NativeJointAdapter(**config['adapter']);saved=np.load(folder/'plan.npz')
            held=predicted_held_position(model,adapter,capture['initial_native'],saved['actions'],saved['timestamps'],report['placement_index'])
            error=goal-held
            if pickplace and error[2]>0 and held[2]-spec['picked_bottom_offset_m']<=spec['place']['bottom_z']+.0015:
                # Already resting on the target surface at release (a thin object held with
                # the jaw tips at the mat): only the XY correction matters.
                error[2]=0.
            iterations.append({'offset_m':offset.tolist(),'held_center_m':held.tolist(),'goal_m':goal.tolist(),'error_m':error.tolist()})
            if np.linalg.norm(error[:2])<.0015 and abs(error[2])<.002:break
            candidate=offset+error*.85
            stuck=np.any(np.abs(candidate)>.020) or attempt==4
            if stuck and not pickplace:
                if np.any(np.abs(candidate)>.020):raise ValueError('Stack placement correction exceeds 20 mm; reposition the cubes and capture again')
                raise ValueError('Stack placement did not converge; reposition the cubes and capture again')
            if stuck:
                # A swinging held object (a cup pinched at its rim) never settles to 1.5 mm:
                # keep the best correction found and let the contact simulation judge it.
                best=min(iterations,key=lambda it:np.linalg.norm(np.asarray(it['error_m'])[:2])+abs(it['error_m'][2]))
                offset=np.asarray(best['offset_m'],float)
                report=plan(folder,stack_offset=offset if np.any(offset) else None,preview=False,technique=technique,speed_scale=speed_scale,timing_profile=timing_profile,**route)
                iterations.append({'kept_best_offset_m':offset.tolist()})
                break
            offset=candidate;report=plan(folder,stack_offset=offset,preview=False,technique=technique,speed_scale=speed_scale,timing_profile=timing_profile,**route)
        report['stack_planning']={'method':'Bounded contact-physics prediction of held cube center; scene and physical parameters unchanged','iterations':iterations,'maximum_offset_m':.020,'release_gap_m':gap if pickplace else .004}
        (folder/'plan.json').write_text(json.dumps(report,indent=2))
    if pickplace:
        report['grasp_candidate']=grasp_index;(folder/'plan.json').write_text(json.dumps(report,indent=2))
    return report


def pickplace_obstacles(model,world):
    """Geoms the arm and the carried object must clear: every object that stays put,
    except the destination and the support the picked object starts on."""
    spec=world['pickplace'];objects={o['id']:o for o in world['objects']}
    picked=objects[spec['pick']]
    keep_destination=spec['place']['type']=='inside' and spec['grasp']['pads']!='fixed_in_hole'
    skip={picked.get('support'),*spec['carried']}|(set() if keep_destination else {spec['place'].get('object')})
    bodies={o['body'] for o in world['objects'] if o['id']!=spec['pick'] and o['id'] not in skip and o.get('body')}
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,b) for b in bodies]
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] in ids and (model.geom_contype[g] or model.geom_conaffinity[g])]


HELD_CLEARANCE_M=.008


def planned_obstacle_clearance(folder,candidate=None):
    """Clearance of the commanded route (object carried rigidly) from every obstacle."""
    from studio_stack import obstacle_geoms,obstacle_clearance,planned_frames,OBSTACLE_PLAN_MARGIN_M
    capture,config,scene,cal,model=load_scene(folder,candidate);adapter=NativeJointAdapter(**config['adapter'])
    obstacles=pickplace_obstacles(model,cal.world) if 'pickplace' in cal.world else obstacle_geoms(model,cal.world,capture.get('task'))
    if not obstacles:return None
    actions=np.load(folder/'plan.npz')['actions']
    if 'pickplace' in cal.world:
        # Same rule as the simulation: robot geometry must not touch other objects.
        object_bodies={mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,o['body']) for o in cal.world['objects'] if o.get('body')}|{model.body('cube').id}
        robot=[g for g in range(model.ngeom) if model.geom_bodyid[g] not in object_bodies and model.body_weldid[model.geom_bodyid[g]]!=0 and model.geom_contype[g]&(2|8)]
        frames=planned_frames(model,adapter,actions)
        result=obstacle_clearance(model,frames,obstacles,.002,movers=robot)
        if result:result['cleared']=result['arm_minimum_m'] is None or result['arm_minimum_m']>=.002;result['slack_m']=(result['arm_minimum_m'] if result['arm_minimum_m'] is not None else .036)-.002
        # The carried object too (carried rigidly from grasp to release): a bowl held by its
        # rim swept through a standing cup. It must pass every other object by 8 mm.
        held=[g for g in range(model.ngeom) if model.geom_bodyid[g]==model.body('cube').id and (model.geom_contype[g] or model.geom_conaffinity[g])]
        # An object held off its centre (a bowl or cup by the rim) swings in the jaws: the
        # margin grows with the lever arm from the jaws to its centre.
        spec=cal.world['pickplace'];picked=next(o for o in cal.world['objects'] if o['id']==spec['pick'])
        lever=float(np.linalg.norm(np.asarray(spec['grasp']['xyz'][:2])-np.asarray(picked['position'][:2])))
        carried=obstacle_clearance(model,frames,obstacles,HELD_CLEARANCE_M+.3*lever,movers=held)
        if result and carried and carried.get('object_minimum_m') is not None:
            need=carried['object_required_m'];gap=carried['object_minimum_m']
            result.update(object_minimum_m=gap,object_required_m=need,object_closest=carried['closest'])
            if gap<need:
                result['cleared']=False;result['closest']=carried['closest'];result['minimum_m']=gap
            result['slack_m']=min(result['slack_m'],gap-need)
        return result
    return obstacle_clearance(model,planned_frames(model,adapter,actions),obstacles,OBSTACLE_PLAN_MARGIN_M)


def _physics_checked(folder,technique,speed_scale,timing_profile,report):
    """Run the contact simulation (no video) on the plan; if a joint only trails its
    command, plan again slower (down to 50%), so Generate plan hands over a plan that
    passes. Any other failure is left for the full simulation to report."""
    tries=[]
    speed=speed_scale
    while True:
        checked=simulate(folder,render=False)
        failures=checked.get('transfer_check_failures',[])
        tries.append({'speed_scale':speed,'outcome':checked['outcome'],'failures':failures})
        lag=checked['outcome']=='success' and failures and all('trails its command' in f for f in failures)
        near=[f for f in failures if f.startswith('Arm passes')]
        if near and not getattr(_physics_checked,'_routing',False):
            # Too close to an object in contact physics: try the other routes (higher
            # travel, other grasp candidates) with the same physics check.
            count=load_scene(folder)[3].world['pickplace']['candidate_count'];best=None
            for extra,turn in [(e,c) for c in range(min(count,7)) for e in (0.,.04)][1:]:
                try:candidate=_prepare_route(folder,technique,speed,timing_profile,extra,turn)
                except ValueError:continue
                result=simulate(folder,render=False);bad=[f for f in result.get('transfer_check_failures',[]) if f.startswith('Arm passes')]
                tries.append({'speed_scale':speed,'route':[extra,turn],'outcome':result['outcome'],'failures':result.get('transfer_check_failures',[])})
                if result['outcome']=='success' and not bad:best=(extra,turn);break
            if best is None:
                _prepare_route(folder,technique,speed,timing_profile)  # keep the default route; simulation reports why
            checked=simulate(folder,render=False);failures=checked.get('transfer_check_failures',[])
            lag=checked['outcome']=='success' and failures and all('trails its command' in f for f in failures)
        from pickplace import GRASP_RAISE_MAX_M
        # Slowing down also reduces the arm's sag, so jaw-tip shortfalls left after the
        # highest grasp do not stop the slower tries.
        lag=checked['outcome']=='success' and failures and all('trails its command' in f or f.startswith('Jaw tips') for f in failures) and any('trails its command' in f for f in failures)
        tips=checked.get('jaw_tip_clearance') or {}
        from pickplace import tip_margin
        low=[tips[k]-tip_margin(tips,k) for k in ('grasp_m','place_m') if tips.get(k) is not None and tips[k]<tip_margin(tips,k)]
        options_file=folder/'planning-options.json'
        options=json.loads(options_file.read_text()) if options_file.exists() else {}
        raised=float(options.get('grasp_raise_m',0.))
        if low and raised<GRASP_RAISE_MAX_M-1e-9:
            # The loaded arm sank below its command: grasp higher by the shortfall and
            # plan again at the same speed (the place height follows the grasp).
            raised=min(GRASP_RAISE_MAX_M,raised-min(low)+.001)
            options_file.write_text(json.dumps({**options,'grasp_raise_m':round(raised,4)}))
            prepare_plan._inner=True
            try:report=prepare_plan(folder,technique=technique,speed_scale=speed,preview=False,timing_profile=timing_profile)
            finally:prepare_plan._inner=False
            continue
        if not lag or speed<=.5+1e-9:break
        speed=max(.5,round(speed-.25,2))
        prepare_plan._inner=True
        try:report=prepare_plan(folder,technique=technique,speed_scale=speed,preview=False,timing_profile=timing_profile)
        finally:prepare_plan._inner=False
    report=json.loads((folder/'plan.json').read_text())
    # Hand over an unsimulated plan: the Simulation step must run on exactly this plan.
    report.update(outcome='not_simulated',transfer_checks_passed=False,simulated_action_hash=None,simulated_timestamp_hash=None,
                  simulated_speed_scale=None,simulated_timing_profile=None,simulated_timing_contract_hash=None,speed_check=tries)
    for key in ['pickplace_assessment','transfer_check_failures','arm_clearance','audit','collision_audit']:report.pop(key,None)
    (folder/'plan.json').write_text(json.dumps(report,indent=2))
    (folder/'planning-options.json').write_text(json.dumps({**(json.loads((folder/'planning-options.json').read_text()) if (folder/'planning-options.json').exists() else {}),'speed_scale':report['speed_scale']}))
    return report


def prepare_plan(folder,technique=None,speed_scale=None,preview=True,timing_profile=None):
    """Plan around obstacles: try higher travel and other cube face pairs until the
    commanded route keeps its margin; stack/unstack placement uses the held object."""
    from studio_stack import OBSTACLE_PLAN_MARGIN_M
    options=folder/'planning-options.json'
    speed_scale=validate_speed_scale(speed_scale if speed_scale is not None else json.loads(options.read_text()).get('speed_scale',1.) if options.exists() else 1.)
    technique=validate_technique(technique if technique is not None else json.loads(options.read_text()).get('motion_technique','classic') if options.exists() else 'classic')
    if timing_profile is None and options.exists():timing_profile=json.loads(options.read_text()).get('timing_profile','conservative')
    capture=json.loads((folder/'capture.json').read_text())
    if not getattr(prepare_plan,'_inner',False) and options.exists():
        # Each new plan starts from the registry grasp height.
        saved=json.loads(options.read_text())
        if saved.pop('grasp_raise_m',None) is not None:options.write_text(json.dumps(saved))
    if capture.get('source')!='live':report=_prepare_route(folder,technique,speed_scale,timing_profile)
    else:
        load_scene(folder)  # selection/scene errors are reported as they are, not as routing failures
        def clear(c):return c['cleared']
        def distance(c):return c['slack_m']
        attempts=[];chosen=None;routes=[(0.,None),(.04,None),(0.,0),(0.,1),(.04,0),(.04,1)]
        if capture.get('task')=='pickplace':
            count=load_scene(folder)[3].world['pickplace']['candidate_count']
            routes=[(extra,c) for c in range(min(count,7)) for extra in (0.,.04,.08)]
        # Standing a pen up: first only routes with the wrist camera up and the far end
        # down (any grasp); if none exists, any upright stand.
        tiers=[0,3] if capture.get('task')=='pickplace' else [3]
        # Each route first with Natural's blended lift; if the carried object would pass
        # too close to something, the same route rising straight up first.
        for extra,turn,tier,lift in [(e,t_,k,l) for k in tiers for e,t_ in routes for l in (False,True)]:
            if tier==3 and len(tiers)>1 and (chosen is not None or any('error' not in a for a in attempts)):break
            if lift and not (attempts and 'error' not in attempts[-1] and not attempts[-1].get('cleared')):continue
            plan._stand_tier=tier;plan._lift_stop=lift
            try:
                report=_prepare_route(folder,technique,speed_scale,timing_profile,extra,turn)
                c=planned_obstacle_clearance(folder,turn if capture.get('task')=='pickplace' else None)
            except ValueError as exc:
                # Another route may avoid what blocked this one (e.g. a jaw on an obstacle).
                attempts.append({'hover_extra_m':extra,'grasp_turn':turn,'lift_stop':lift,'error':str(exc)});continue
            if c is None:chosen=None;attempts=[];break  # nothing to avoid
            attempts.append({'hover_extra_m':extra,'grasp_turn':turn,'lift_stop':lift,**c})
            if chosen is None or distance(c)>distance(attempts[chosen]):chosen=len(attempts)-1
            if clear(c) or report['world']['cube'].get('object')=='tape' and extra:break  # a round roll has no face pairs
        if attempts:
            if chosen is None:
                (folder/'obstacle-planning.json').write_text(json.dumps(attempts,indent=2))
                raise ValueError(f"No route around the obstacles ({attempts[0]['error']}); move obstacles further from the object and capture again")
            best=attempts[chosen]
            if capture.get('task')=='pickplace' and not best.get('cleared') and best.get('object_minimum_m') is not None and best['object_minimum_m']<best.get('object_required_m',0.):
                # Every route (grasps, heights, straight lift) carries the object into another
                # one: say so now instead of failing every simulation try.
                _,_,scene_now,cal_now,model_now=load_scene(folder,best['grasp_turn'])
                bodies={o.get('body'):o['kind'] for o in cal_now.world['objects']}
                hit=(best.get('object_closest') or best.get('closest') or '').split(' near ')[-1]
                picked=cal_now.world['pickplace']['kind']
                raise ValueError(f"No route keeps the carried {picked} clear of the {bodies.get(hit,'object')} (closest {best['object_minimum_m']*1000:.0f} mm on the best route, needs {best['object_required_m']*1000:.0f} mm because it can swing in the jaws); move the {bodies.get(hit,'object')} away or choose another destination")
            plan._lift_stop=bool(best.get('lift_stop'))
            if chosen!=len(attempts)-1:report=_prepare_route(folder,technique,speed_scale,timing_profile,best['hover_extra_m'],best['grasp_turn'])
            report['obstacle_planning']={'method':'Commanded joints with the object carried rigidly; static obstacle collision geometry; contact physics still decides',
                                         'margin_m':OBSTACLE_PLAN_MARGIN_M,'chosen':chosen,'cleared':clear(best),'attempts':attempts}
            (folder/'plan.json').write_text(json.dumps(report,indent=2))
    if capture.get('task')=='pickplace' and capture.get('source')=='live' and not getattr(prepare_plan,'_inner',False):
        report=_physics_checked(folder,technique,speed_scale,timing_profile,report)
    capture,config,scene,cal,model=load_scene(folder,report.get('grasp_candidate'))
    if preview:render_preview(model,NativeJointAdapter(**config['adapter']).to_model(capture['initial_native']),folder/'preview.jpg')
    if not getattr(prepare_plan,'_inner',False):plan._stand_tier=3;plan._lift_stop=False
    return report


def _model_key(m,report):
    """Identity of a contact simulation: the exact actions, timing and compiled model."""
    from pickplace_initialization import model_fingerprint
    h=hashlib.sha256(model_fingerprint(m).encode())
    h.update((ROOT/'src/dataset_to_sim/strict_replay.py').read_bytes())
    h.update(str(report.get('scene_contract_hash')).encode())
    h.update(str((report['action_hash'],report.get('timestamp_hash'),report.get('initial_native'))).encode())
    return h.hexdigest()


def _cached_physics(folder,m,report,run):
    """The plan step already simulates the plan it hands over (physics check); the
    Simulation step then reuses that deterministic result instead of running the same
    contact simulation again. Any change to actions, timing or model runs it afresh."""
    import pickle
    path=folder/'physics-cache.pkl';key=_model_key(m,report)
    try:
        cached=pickle.loads(path.read_bytes())
        if cached.get('key')==key:return cached['trace']
    except Exception:pass
    trace=run()
    try:path.write_bytes(pickle.dumps({'key':key,'trace':trace}))
    except Exception:path.unlink(missing_ok=True)
    return trace


def validate_plan_scene(folder,report,loaded=None):
    """A validation cannot authorize a different model, calibration or initial scene."""
    if report.get('task')=='pickplace':
        from pickplace_initialization import scene_fingerprint
        capture,config,scene,cal,m=loaded or load_scene(folder,report.get('grasp_candidate'))
        if report.get('scene_contract_hash')!=scene_fingerprint(m,config,scene):
            raise ValueError('Scene or object dimensions changed since planning; generate a new plan')


def simulate(folder,render=True):
    report=json.loads((folder/'plan.json').read_text()); loaded=load_scene(folder,report.get('grasp_candidate'))
    capture,config,scene,cal,m=loaded
    validate_plan_scene(folder,report,loaded)
    adapter=NativeJointAdapter(**config['adapter']); saved=np.load(folder/'plan.npz');actions=saved['actions'];ts=saved['timestamps']
    if array_hash(actions)!=report['action_hash']: raise ValueError('Plan changed since generation')
    if 'timestamp_hash' in report and array_hash(ts)!=report['timestamp_hash']:raise ValueError('Plan timing changed since generation')
    from studio_timing import validated_motion_limits
    validate_trajectory(actions,ts,native_limits(config),speed_scale=report.get('speed_scale',1.),velocity_limits=validated_motion_limits(report))
    states=np.tile(report['initial_native'],(len(actions),1))
    ep=Episode(-1,np.arange(len(actions)),np.arange(len(actions)),ts,states,actions,{})
    trace=_cached_physics(folder,m,report,lambda:integrate_actions(m,ep,adapter))
    if report.get('task')=='pickplace':
        from pickplace import verdict,load_registry
        inside=None;outcome,assessment=verdict(m,trace,scene,cal.world['pickplace'],load_registry(),config);report['pickplace_assessment']=assessment
    elif report.get('task')=='stack':
        inside=None;stack=stack_verdict(trace,cal.world['cube']['size']);outcome=stack['outcome'];report['stack_assessment']=stack
    elif report.get('task')=='unstack':
        from studio_generation_scene import unstack_verdict,_table_support
        mat=cal.world['mat'];inside=None
        assessment=unstack_verdict(trace,cal.world['cube']['size'],cal.world['placement']['position'],mat['position'][2]+mat['size'][2]/2,_table_support(m,trace))
        outcome=assessment['outcome'];report['unstack_assessment']=assessment
    else:
        inside=containment(m,trace['qpos'][-1],trace['cube_pose'][-1],cal,config);outcome=score(trace,inside)
    # Objects touching objects (resting, stacking) are judged by the verdict, not as collisions.
    collision={k:v for k,v in trace['audit']['contact_audit'].items() if k not in ('cube','objects')}
    tracking=np.abs(adapter.to_native(trace['qpos'][1:])-actions);peak=np.max(tracking,axis=0)
    failures=[]
    if trace['audit']['initial_max_penetration_m']>.001:failures.append(f"Arm touches the table at the start ({trace['audit']['initial_max_penetration_m']*1000:.1f} mm): return Home and capture again")
    for name,value in collision.items():
        if value['max_penetration_m']>.001:failures.append(f"Collision {name.replace('_',' ')} ({value['max_penetration_m']*1000:.1f} mm)")
    # Hardware stops when any arm joint trails its command by 10 deg. On the real arm the
    # loaded shoulder lift trails the simulation by up to ~3 deg while lifting, so it keeps
    # a 2 deg margin here (every observed hardware stop had >9 deg simulated shoulder lag).
    limits=np.array([10.,HARDWARE_SHOULDER_LIFT_MARGIN_DEG,10.,10.,10.,25.])
    for joint,(value,limit) in enumerate(zip(peak,limits)):
        if value>limit:failures.append(f"{JOINT_LABELS[joint]} trails its command by {value:.1f}° (limit {limit:.0f}°): lower Travel speed")
    from studio_stack import obstacle_geoms,obstacle_clearance,trace_frames,OBSTACLE_SIM_MARGIN_M
    blocked=[] if report.get('task')=='pickplace' else obstacle_geoms(m,cal.world,report.get('task'))
    if blocked:
        # Arm and carried object must stay clear of every obstacle in the contact trace.
        clearance=obstacle_clearance(m,trace_frames(m,trace),blocked,OBSTACLE_SIM_MARGIN_M)
        report['obstacle_clearance']=clearance
        if not clearance['cleared']:
            need=clearance['object_required_m'] if clearance['closest'].startswith('cube ') else clearance['required_m']
            failures.append(f"Obstacle clearance {clearance['minimum_m']*1000:.1f} mm ({clearance['closest'].replace('_',' ')}; need {need*1000:.1f} mm): move the obstacle or the object")
    if report.get('task')=='pickplace':
        # Only touching counts (the collision audit above); the jaw tips must also stay
        # above the held object's base and above the surface it is set on.
        from pickplace import jaw_tip_clearance,load_registry,tip_margin
        spec=cal.world['pickplace'];picked=next(o for o in cal.world['objects'] if o['id']==spec['pick'])
        if spec['grasp']['pads']=='between' and load_registry()['objects'][picked['kind']]['shape']['type'] in ('box','cylinder'):
            tips=jaw_tip_clearance(m,trace,scene,spec,load_registry());report['jaw_tip_clearance']=tips
            for key,label in [('grasp_m','base of the held object'),('place_m','surface it is set on')]:
                need=tip_margin(tips,key)
                if tips[key] is not None and tips[key]<need:
                    failures.append(f"Jaw tips reach {tips[key]*1000:.1f} mm above the {label} (need {need*1000:.0f} mm): grasp higher")
    checks=not failures
    report.update(outcome=outcome,transfer_checks_passed=checks,transfer_check_failures=failures,collision_audit=collision,audit=trace['audit'],bowl_containment=inside,
                  max_cube_height_m=float(trace['cube_pose'][:,2].max()),max_tracking_error=np.max(tracking,axis=0).tolist(),
                  simulated_action_hash=array_hash(trace['native_action_applied']),simulated_timestamp_hash=array_hash(ts),simulated_speed_scale=report.get('speed_scale',1.),
                  simulated_timing_profile=report.get('timing_profile','conservative'),simulated_timing_contract_hash=report.get('timing_contract_hash'))
    if report.get('task')=='pickplace':
        report['simulated_scene_contract_hash']=report['scene_contract_hash']
    extra={k:trace[k] for k in ['target_pose','target_velocity'] if k in trace}
    for joint,values in trace.get('object_poses',{}).items():extra['object_'+joint]=values
    np.savez_compressed(folder/'trace.npz',qpos=trace['qpos'],cube_pose=trace['cube_pose'],native=adapter.to_native(trace['qpos']),timestamps=trace['sample_times'],**extra)
    # Save the physical verdict even if the graphics environment fails.
    (folder/'plan.json').write_text(json.dumps(report,indent=2))
    if not render:return report
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
                for joint,values in trace.get('object_poses',{}).items():
                    address=m.joint(joint).qposadr[0];d.qpos[address:address+7]=values[index];mujoco.mj_forward(m,d)
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
