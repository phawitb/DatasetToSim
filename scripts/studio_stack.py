"""Live stacking geometry and physical verdict; archived benchmark stays unchanged."""
import copy,itertools,xml.etree.ElementTree as ET
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from strict_action_scene import episode_scene,build_model


def scene_calibration(config,scene):
    cal=episode_scene(config,scene)
    if scene.get('task')=='stack':
        cal.world['target_cube']=copy.deepcopy(scene['target_cube'])
        cal.world['bowl_present']=bool(scene['bowl_present'])
    return cal


def build_scene_model(cal,config):
    if 'target_cube' not in cal.world:return build_model(cal,config)
    root=ET.fromstring(build_model(cal,config,return_xml=True));world=root.find('worldbody')
    if not cal.world.get('bowl_present',True):world.remove(root.find(".//body[@name='bowl']"))
    target=cal.world['target_cube'];body=ET.SubElement(world,'body',name='target_cube',pos=' '.join(map(str,target['position'])),quat=' '.join(map(str,target['quaternion'])))
    ET.SubElement(body,'freejoint',name='target_cube_free')
    geom=copy.deepcopy(root.find(".//geom[@name='cube_geom']"));geom.set('name','target_cube_geom');geom.set('rgba','.65 .48 .27 1');body.append(geom)
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))


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
