"""Live stacking geometry and physical verdict; archived benchmark stays unchanged."""
import copy,itertools,xml.etree.ElementTree as ET
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from strict_action_scene import episode_scene,build_model


# Arm and held object must keep this distance from every obstacle: planned commands
# (before contact physics) and the simulated contact trace, respectively.
OBSTACLE_PLAN_MARGIN_M=.008
OBSTACLE_SIM_MARGIN_M=.003


def scene_calibration(config,scene):
    cal=episode_scene(config,scene)
    if scene.get('object')=='tape':
        from tape_object import TAPE,dimensions
        size=dimensions();cal.world['cube'].update(object=TAPE,width=size['width_m'],outer_diameter=size['outer_diameter_m'])
    if scene.get('task') in ('stack','unstack'):
        cal.world['target_cube']=copy.deepcopy(scene['target_cube'])
        cal.world['bowl_present']=bool(scene['bowl_present'])
    if scene.get('task')=='unstack':cal.world['placement']=copy.deepcopy(scene['placement'])
    if scene.get('obstacles'):cal.world['obstacles']=copy.deepcopy(scene['obstacles'])
    return cal


def add_obstacles(root,world):
    """Static collision bodies for every other modelled object in the workspace."""
    from strict_action_scene import COLLISION
    body_root=root.find('worldbody');names=[]
    for index,item in enumerate(world.get('obstacles',[])):
        name=f'obstacle_{index}';kind=item['kind']
        body=ET.SubElement(body_root,'body',name=name,pos=' '.join(map(str,item['position'])),quat=' '.join(map(str,item.get('quaternion',[1,0,0,0]))))
        if kind=='cube':shape=dict(type='box',size=' '.join([str(item['size']/2)]*3))
        elif kind=='stack':shape=dict(type='box',size=f"{item['size']/2} {item['size']/2} {item['height']/2}")
        elif kind=='tape':shape=dict(type='cylinder',size=f"{item['outer_diameter']/2} {item['width']/2}")
        elif kind=='bowl':shape=dict(type='cylinder',size=f"{item['outer_radius']} {item['height']/2}",pos=f"0 0 {item['height']/2}")
        else:raise ValueError('Unknown obstacle')
        rgba={'cube':'.45 .45 .5 1','stack':'.45 .45 .5 1','tape':'.8 .78 .7 .7','bowl':'.5 .65 .6 .6'}[kind]
        ET.SubElement(body,'geom',name=name+'_geom',**shape,**COLLISION)
        ET.SubElement(body,'geom',name=name+'_visual',**shape,rgba=rgba,contype='0',conaffinity='0',group='1',mass='0')
        names.append(name)
    return names


def build_scene_model(cal,config):
    from tape_object import apply_tape,is_tape
    if 'target_cube' not in cal.world and not is_tape(cal.world) and not cal.world.get('obstacles'):return build_model(cal,config)
    root=ET.fromstring(build_model(cal,config,return_xml=True));world=root.find('worldbody')
    if 'target_cube' in cal.world:
        if not cal.world.get('bowl_present',True):world.remove(root.find(".//body[@name='bowl']"))
        target=cal.world['target_cube'];body=ET.SubElement(world,'body',name='target_cube',pos=' '.join(map(str,target['position'])),quat=' '.join(map(str,target['quaternion'])))
        ET.SubElement(body,'freejoint',name='target_cube_free')
        geom=copy.deepcopy(root.find(".//geom[@name='cube_geom']"));geom.set('name','target_cube_geom');geom.set('rgba','.65 .48 .27 1');body.append(geom)
    add_obstacles(root,cal.world)
    return mujoco.MjModel.from_xml_string(ET.tostring(apply_tape(root,cal.world),encoding='unicode'))


def obstacle_geoms(model,world,task):
    """Geoms the arm and held object must avoid: obstacles, and a bowl that is not the goal."""
    names=[model.geom(i).name for i in range(model.ngeom)]
    ids=[i for i,n in enumerate(names) if n.startswith('obstacle_') and n.endswith('_geom')]
    if task in ('stack','unstack') and world.get('bowl_present',False):
        bowl=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'bowl')
        ids+=[i for i in range(model.ngeom) if model.geom_bodyid[i]==bowl and (model.geom_contype[i] or model.geom_conaffinity[i])]
    return ids


def obstacle_clearance(model,frames,obstacles,required,movers=None):
    """Distances from the obstacles to the arm and to the manipulated object.

    `frames` are full qpos vectors. The arm must keep `required`. The object may
    start closer than that beside an obstacle, so it must keep the smaller of
    `required` and its starting distance less 1 mm. Contacts are detected with the
    obstacle margin widened, so every pair closer than 3 × `required` is seen.
    """
    obstacles=list(obstacles)
    if not obstacles:return None
    cube=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'cube')
    moving=set(int(i) for i in range(model.ngeom) if model.body_weldid[model.geom_bodyid[i]]!=0 and (model.geom_contype[i] or model.geom_conaffinity[i]))
    if movers is not None:moving=set(int(i) for i in movers)-set(int(o) for o in obstacles)
    saved=model.geom_margin[obstacles].copy();model.geom_margin[obstacles]=3*required
    data=mujoco.MjData(model);blocked=set(obstacles);detect=3*required
    arm=[detect,None,None];held=[detect,None,None];initial=detect
    try:
        for index,q in enumerate(frames):
            data.qpos[:]=q;mujoco.mj_fwdPosition(model,data)
            for contact in data.contact[:data.ncon]:
                pair={int(contact.geom1),int(contact.geom2)}
                if not (pair&blocked and pair&moving) or not pair-blocked:continue
                other=(pair-blocked).pop();hit=(pair&blocked).pop();dist=float(contact.dist)
                is_object=model.geom_bodyid[other]==cube
                if is_object and index==0:initial=min(initial,dist)
                item=held if is_object else arm
                if dist<item[0]:item[:]=[dist,f'{model.body(model.geom_bodyid[other]).name} near {model.body(model.geom_bodyid[hit]).name}',index]
    finally:model.geom_margin[obstacles]=saved
    object_required=max(0.,min(required,initial-.001))
    slack=min(arm[0]-required,held[0]-object_required)
    worst=arm if arm[0]-required<=held[0]-object_required else held
    return {'minimum_m':None if worst[0]>=detect else worst[0],'closest':worst[1],'frame':worst[2],'required_m':required,
            'arm_minimum_m':None if arm[0]>=detect else arm[0],'object_minimum_m':None if held[0]>=detect else held[0],
            'object_initial_m':None if initial>=detect else initial,'object_required_m':object_required,'slack_m':slack,'cleared':bool(slack>=0)}


def planned_frames(model,adapter,actions):
    """Commanded arm poses with the object carried rigidly between grasp and release."""
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    cube=model.joint('cube_free').qposadr[0];gripper=model.body('gripper').id
    grip=np.asarray(actions)[:,5];opened=np.flatnonzero(grip>=grip.max()-10)
    closed=[i for i in range(len(grip)) if len(opened) and i>opened[0] and grip[i]<=grip.min()+3]
    grasp=closed[0] if closed else None
    release=next((i for i in range(grasp,len(grip)) if grip[i]>=grip[grasp]+15),len(grip)) if grasp is not None else None
    targets=adapter.to_model(np.asarray(actions));relative=None;frames=[]
    for i,q in enumerate(targets):
        data.qpos[:6]=q
        if grasp is not None and grasp<=i<release:
            mujoco.mj_kinematics(model,data)
            R=data.xmat[gripper].reshape(3,3);p=data.xpos[gripper]
            if relative is None:
                pose=data.qpos[cube:cube+7];Rc=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
                relative=(R.T@(pose[:3]-p),R.T@Rc)
            position=R@relative[0]+p;q_obj=Rotation.from_matrix(R@relative[1]).as_quat()[[3,0,1,2]]
            data.qpos[cube:cube+3]=position;data.qpos[cube+3:cube+7]=q_obj
        frames.append(data.qpos.copy())
    return frames


def trace_frames(model,trace):
    """Simulated contact trace as full qpos vectors (static obstacles have no joints)."""
    data=mujoco.MjData(model);frames=[]
    cube=model.joint('cube_free').qposadr[0]
    target=model.joint('target_cube_free').qposadr[0] if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'target_cube_free')>=0 else None
    for i,q in enumerate(trace['qpos']):
        data.qpos[:6]=q;data.qpos[cube:cube+7]=trace['cube_pose'][i]
        if target is not None and 'target_pose' in trace:data.qpos[target:target+7]=trace['target_pose'][i]
        frames.append(data.qpos.copy())
    return frames


def predicted_held_position(model,adapter,initial,actions,times,index):
    """Bounded planning probe, full contact physics; no object/state corrections."""
    data=mujoco.MjData(model);data.qpos[:6]=adapter.to_model(initial)
    for j,count in enumerate(model.actuator_history[:,0]):
        if count:mujoco.mj_initCtrlHistory(model,data,j,None,np.full((int(count),1),data.qpos[j]))
    mujoco.mj_forward(model,data);dt=float(model.opt.timestep)
    try:
        for i,command in enumerate(adapter.to_model(actions[:index])):
            data.ctrl[:]=command;end=float(times[i+1]);steps=int((end-data.time+1e-10)/dt)
            if steps:mujoco.mj_step(model,data,nstep=steps)
            if end-data.time>1e-10:
                model.opt.timestep=end-data.time;mujoco.mj_step(model,data);model.opt.timestep=dt
            if any(w.number for w in data.warning) or not np.isfinite(data.qpos).all():raise ValueError('Stack planning physics became unstable')
        address=model.joint('cube_free').qposadr[0]
        return data.qpos[address:address+3].copy()
    finally:model.opt.timestep=dt


def stack_verdict(trace,edge):
    a=trace['audit'];source=np.asarray(trace['cube_pose']);target=np.asarray(trace['target_pose'])
    top,bottom=source[-1],target[-1]
    def rotation(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
    rs,rt=rotation(top[3:]),rotation(bottom[3:])
    local=(top[:3]-bottom[:3])@rt
    corners=np.array(list(itertools.product([-1,1],repeat=3)))*edge/2
    source_vertices=(corners@rs.T+top[:3]-bottom[:3])@rt
    gap=float(source_vertices[:,2].min()-edge/2)
    # Entire center of mass must be well inside support; require substantial
    # footprint overlap, upright cubes, actual support force and settled motion.
    center_supported=bool(np.max(np.abs(local[:2]))<=edge/2-.003)
    upright=bool(min(rs[2,2],rt[2,2])>np.cos(np.deg2rad(8)))
    target_shift=float(np.linalg.norm(bottom[:3]-target[0,:3]))
    stable=bool(np.max(np.linalg.norm(source[-30:,:3]-top[:3],axis=1))<.0015 and np.max(np.linalg.norm(target[-30:,:3]-bottom[:3],axis=1))<.0015)
    velocity=np.asarray(trace['target_velocity'])[-1]
    settled=stable and np.linalg.norm(velocity[:3])<.01 and np.linalg.norm(velocity[3:])<.2 and a['final_cube_linear_speed_m_s']<.01 and a['final_cube_angular_speed_rad_s']<.2
    support=a.get('grasp_support',{})
    outcome='success'
    if source[:,2].max()<source[0,2]+.04:outcome='not_lifted'
    elif support.get('longest_secure_grasp_s',0)<.08 or support.get('longest_secure_lift_s',0)<.03:outcome='no_bilateral_grasp'
    elif a['final_finger_contact']:outcome='not_released'
    elif target_shift>.004 or not upright:outcome='target_moved_or_tipped'
    elif not center_supported or not -.001<=gap<=.001 or a.get('final_stack_support_N',0)<.02:outcome='not_stacked'
    elif not settled:outcome='not_settled'
    return {'outcome':outcome,'center_supported':center_supported,'upright':upright,'support_gap_m':gap,'target_shift_m':target_shift,'settled':bool(settled),'support_force_N':a.get('final_stack_support_N',0)}
