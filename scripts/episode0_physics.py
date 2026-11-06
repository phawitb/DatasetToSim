"""Actuated replay: cube is initialized once, then moved only by mj_step contacts."""
import argparse
import xml.etree.ElementTree as ET
from episode0 import *

DEST=ROOT/'output/episode0-full-contact'
DEST.mkdir(parents=True,exist_ok=True)

def physical_model(cal, grip_offset=-8, return_xml=False):
    root=ET.fromstring(model(cal,return_xml=True))
    physics=cal.data.get('physics',{})
    if 'quaternion' in cal.world['cube']:root.find(".//body[@name='cube']").set('quat',_str(cal.world['cube']['quaternion']))
    joint=root.find(".//joint[@name='gripper']")
    for key,attribute in [('gripper_joint_damping','damping'),('gripper_armature','armature')]:
        if key in physics:joint.set(attribute,str(physics[key]))
    root.find('option').set('timestep','.001')
    root.find('option').set('iterations','80')
    # All physical collision geometries participate. Visual meshes remain visual.
    for geom in root.findall('.//worldbody//geom'):
        name=geom.get('name','');cls=geom.get('class','')
        if cls in ['collision_gripper','collision_gripper_mesh']:
            geom.set('contype','2');geom.set('conaffinity','15')
            geom.set('friction','1.4 .01 .001');geom.set('condim','6');geom.set('solref','.004 1')
        elif name=='cube_geom':
            geom.set('contype','1');geom.set('conaffinity','15');geom.set('mass','.012')
            geom.set('friction','1.4 .01 .001');geom.set('condim','6');geom.set('solref','.004 1')
        elif name in ['table','work_mat','floor']:
            geom.set('contype','4');geom.set('conaffinity','15')
        elif cls=='collision':
            geom.set('contype','8');geom.set('conaffinity','15')
        else:geom.set('contype','0');geom.set('conaffinity','0')
    # Real jaws have finite compliant gripping surfaces; thin pad proxies.
    for body,name,pos in [('gripper','fixed_pad','-.009 0 -.090'),('moving_jaw_so101_v1','moving_pad','-.011 -.067 .019')]:
        ET.SubElement(root.find(f".//body[@name='{body}']"),'geom',name=name,type='box',pos=pos,size='.003 .008 .012' if body=='gripper' else '.003 .012 .008',rgba='.24 .21 .18 1',group='3',mass='.001',contype='2',conaffinity='15',friction='1.4 .01 .001',condim='6',solref='.004 1')
    bowl=root.find(".//body[@name='bowl']");bowl.set('pos',_str([*cal.world['bowl']['position'][:2],.005]))
    # Convex polygon disk avoids cylinder/box contact degeneracies during release.
    ring=np.arange(48)*2*np.pi/48
    vertices=[[.04*np.cos(a),.04*np.sin(a),z] for z in [.001,.007] for a in ring]
    ET.SubElement(root.find('asset'),'mesh',name='bowl_floor_mesh',vertex=_str(np.ravel(vertices)))
    ET.SubElement(bowl,'geom',name='bowl_floor_collision',type='mesh',mesh='bowl_floor_mesh',group='3',contype='4',conaffinity='15',solref='.004 1')
    for i,a in enumerate(np.arange(32)*2*np.pi/32):
        ET.SubElement(bowl,'geom',name=f'bowl_contact_{i}',type='box',size='.003 .006 .027',pos=_str([.05*np.cos(a),.05*np.sin(a),.027]),euler=_str([0,.34,a]),group='3',contype='4',conaffinity='15')
    for body in root.findall('.//body'):
        if body.get('name') not in ['cube','bowl']:body.set('gravcomp','1')
    for actuator in root.findall('actuator/position'):
        actuator.set('kp','150');actuator.set('kv','4')
        if actuator.get('name')=='gripper':actuator.set('kp','35');actuator.set('kv',str(physics.get('gripper_servo_kv',1)));actuator.set('forcerange','-.35 .35')
    for geom in root.findall('.//worldbody//geom'):
        if geom.get('contype') in {'1','2','4','8'}:
            geom.set('solimp','.99 .999 .0001');geom.set('solref','.002 1')
    xml=ET.tostring(root,encoding='unicode')
    return xml if return_xml else mujoco.MjModel.from_xml_string(xml)

def _str(x):return ' '.join(map(str,x))

def bowl_containment(m,d,tolerance=.0005):
    """Check all cube corners against the actual 32 inner wall planes."""
    import itertools
    cube=d.geom('cube_geom');size=m.geom_size[m.geom('cube_geom').id]
    vertices=np.array(list(itertools.product([-1,1],repeat=3)))*size
    vertices=vertices@cube.xmat.reshape(3,3).T+cube.xpos
    violations=[];wall_tops=[]
    for k in range(32):
        wall=d.geom(f'bowl_contact_{k}');normal=wall.xmat.reshape(3,3)[:,0]
        violations.append(float(np.max((vertices-wall.xpos)@normal)+m.geom_size[wall.id,0]))
        wall_tops.append(float(wall.xpos[2]+np.abs(wall.xmat.reshape(3,3)[2])@m.geom_size[wall.id]))
    floor_z=float(d.body('bowl').xpos[2]+.007)
    rim_z=min(wall_tops)
    inside=max(violations)<=tolerance and vertices[:,2].min()>=floor_z-tolerance and vertices[:,2].max()<=rim_z+tolerance
    return {'inside':bool(inside),'max_wall_violation_m':max(violations),'lowest_corner_z_m':float(vertices[:,2].min()),'highest_corner_z_m':float(vertices[:,2].max()),'floor_z_m':floor_z,'rim_z_m':rim_z,'tolerance_m':tolerance}

def clear_table_targets(m,targets):
    """Small wrist lift avoids commanding a finger mesh through the work surface."""
    d=mujoco.MjData(m);result=targets.copy();corrections=[]
    for f,q in enumerate(targets):
        max_lift=min(60.,np.rad2deg(q[3]-m.actuator_ctrlrange[3,0]))
        for degrees in np.arange(0,max_lift+.001,.25):
            d.qpos[:6]=q;d.qpos[3]=q[3]-np.deg2rad(degrees);mujoco.mj_forward(m,d)
            collisions=[co for co in d.contact if {int(m.geom_contype[co.geom1]),int(m.geom_contype[co.geom2])}=={2,4} and {m.geom(co.geom1).name,m.geom(co.geom2).name}&{'table','work_mat','floor'} and co.dist<0]
            if not collisions:break
        if collisions:raise ValueError(f'Cannot clear tabletop at frame {f}')
        result[f,3]=d.qpos[3];corrections.append(float(degrees))
    return result,np.array(corrections)

def clear_table_targets_vertical(m,targets):
    """Raise the hand while retaining planar position and rotation, using arm IK."""
    from scipy.optimize import least_squares
    d=mujoco.MjData(m);result=targets.copy();body=m.body('gripper').id
    def penetration():
        return max([0.]+[-c.dist for c in d.contact if {int(m.geom_contype[c.geom1]),int(m.geom_contype[c.geom2])}=={2,4} and {m.geom(c.geom1).name,m.geom(c.geom2).name}&{'table','work_mat','floor'}])
    for f,q in enumerate(targets):
        d.qpos[:6]=q;mujoco.mj_forward(m,d)
        depth=penetration()
        if depth<1e-7:continue
        position=d.xpos[body].copy();rotation=d.xmat[body].copy();candidate=q[:4].copy()
        for _ in range(8):
            position[2]+=depth+.0003
            def residual(x):
                d.qpos[:6]=q;d.qpos[:4]=x;mujoco.mj_forward(m,d)
                return np.r_[d.xpos[body]-position,.05*(d.xmat[body]-rotation),.00001*(x-q[:4])]
            fit=least_squares(residual,candidate,bounds=(m.actuator_ctrlrange[:4,0],m.actuator_ctrlrange[:4,1]),max_nfev=40,gtol=1e-10,ftol=1e-10,xtol=1e-10)
            candidate=fit.x;residual(candidate);depth=penetration()
            if depth<1e-7:break
        if depth>1e-6:raise ValueError(f'Cannot vertically clear tabletop at frame {f}: {depth:.6f} m')
        if np.linalg.norm(d.xpos[body][:2]-position[:2])>.001:raise ValueError(f'Clearance IK displaced grasp XY at frame {f}')
        result[f,:4]=candidate
    return result,np.rad2deg(result-targets)

def simulate(cal, ep, xy=None, grip_offset=-8, save=False, disable_fingers=False, clearance_mode='wrist', trace_dir=None, repair=False):
    m=physical_model(cal,grip_offset);d=mujoco.MjData(m)
    if disable_fingers:
        ids=np.flatnonzero(m.geom_contype==2);m.geom_contype[ids]=0;m.geom_conaffinity[ids]=0
    targets=cal.joint_convention.to_radians(ep.state_deg)
    targets[:,5]=np.deg2rad(ep.action_deg[:,5]+grip_offset)
    targets=np.clip(targets,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
    repair_info={'enabled':False}
    if repair:
        from trajectory_repair import repair_targets
        targets,repair_info=repair_targets(m,targets,ep,cal.world['cube'],cal.world['bowl'])
    if clearance_mode=='vertical':targets,clearance_correction=clear_table_targets_vertical(m,targets)
    else:targets,clearance_correction=clear_table_targets(m,targets)
    d.qpos[:6]=targets[0];d.ctrl[:]=targets[0]
    addr=m.jnt_qposadr[m.joint('cube_free').id];cubeid=m.body('cube').id
    d.qpos[addr:addr+3]=[* (xy if xy is not None else cal.world['cube']['position'][:2]),.0162]
    mujoco.mj_forward(m,d)
    poses=[];joints=[];forces=[];bilateral=[];times=[]
    contact_audit={name:{'solver_contacts':0,'max_penetration_m':0.,'max_normal_force_N':0.} for name in ['robot_self','finger_table','robot_workcell','cube']}
    for f in range(ep.frame_count):
        until=f/30
        both=False;peak=0
        while d.time+1e-9<until:
            t=d.time*30;lo=min(int(t),len(targets)-2);a=t-lo
            d.ctrl[:]=(1-a)*targets[lo]+a*targets[lo+1]
            mujoco.mj_step(m,d)
            sides=set()
            for contact_index,contact in enumerate(d.contact):
                g1,g2=int(contact.geom1),int(contact.geom2)
                bits={int(m.geom_contype[g1]),int(m.geom_contype[g2])}
                names={m.geom(g1).name or '',m.geom(g2).name or ''}
                category='cube' if 1 in bits else ('robot_self' if bits <= {2,8} else ('finger_table' if 2 in bits and names & {'table','work_mat','floor'} else 'robot_workcell'))
                audit=contact_audit[category];audit['solver_contacts']+=1
                if -contact.dist>audit['max_penetration_m']:
                    audit['deepest_pair']=[m.geom(g1).name or f'geom_{g1}',m.geom(g2).name or f'geom_{g2}'];audit['deepest_frame']=f
                audit['max_penetration_m']=max(audit['max_penetration_m'],float(max(0,-contact.dist)))
                cf=np.zeros(6);mujoco.mj_contactForce(m,d,contact_index,cf)
                audit['max_normal_force_N']=max(audit['max_normal_force_N'],float(abs(cf[0])))
                if cubeid not in [m.geom_bodyid[g1],m.geom_bodyid[g2]]:continue
                other=g2 if m.geom_bodyid[g1]==cubeid else g1
                name=m.geom(other).name or ''
                if name.startswith('fixed') or name=='fixed_pad':sides.add('fixed')
                if name.startswith('moving') or name=='moving_pad':sides.add('moving')
                force=np.zeros(6);mujoco.mj_contactForce(m,d,contact_index,force);peak=max(peak,float(abs(force[0])))
            both |= len(sides)==2
        mujoco.mj_forward(m,d)
        poses.append(d.qpos[addr:addr+7].copy());joints.append(d.qpos[:6].copy());forces.append(peak);bilateral.append(both);times.append(d.time)
    poses=np.array(poses);joints=np.array(joints)
    summary={'max_cube_z':float(poses[:,2].max()),'final_cube':poses[-1,:3].tolist(),'bilateral_frames':int(sum(bilateral)),'arm_tracking_rmse_rad':np.sqrt(np.mean((joints[:,:5]-targets[:,:5])**2,axis=0)).tolist(),'cube_pose_writes_after_initialization':0,'external_object_forces':False,'weld_constraints':False,'xy':xy,'grip_offset':grip_offset}
    summary.update(final_inside_bowl=bool(np.linalg.norm(poses[-1,:2]-np.array(cal.world['bowl']['position'][:2]))<.028 and .015<poses[-1,2]<.04),final_bilateral_contact=bool(bilateral[-1]),max_contact_normal_N=float(max(forces)),finger_contacts_disabled=disable_fingers,contact_audit=contact_audit,collision_configuration={'arm_self':True,'finger_table':True,'robot_bowl':True,'explicit_exclusions':int(m.nexclude),'parent_filter':'MuJoCo default: direct parent/child and same rigid body pairs filtered'},limitations=['Arm follows measured joint state using position servos with gravity compensation; gripper follows recorded action with -8 degree offset.','All source physical robot colliders and jaw pads participate; visual meshes are not collision geometry. Standard MuJoCo parent-body filtering remains enabled.','12g cube mass, pad geometry and friction are estimates, not identified real parameters.'])
    summary['table_clearance_correction']={'mode':clearance_mode,'max_degrees':float(np.abs(clearance_correction).max()),'frames_adjusted':int(np.count_nonzero(np.any(clearance_correction!=0,axis=1) if clearance_correction.ndim==2 else clearance_correction)),'note':'Vertical arm IK preserves hand XY and rotation.' if clearance_mode=='vertical' else 'Subtract smallest 0.25-degree wrist increment clearing finger/table overlap.','changes_recorded_arm_targets':True}
    summary['trajectory_repair']=repair_info
    summary['bowl_containment']=bowl_containment(m,d)
    summary['final_inside_bowl']=summary['bowl_containment']['inside']
    final_fingers=any(m.body('cube').id in [m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]] and 2 in [m.geom_contype[c.geom1],m.geom_contype[c.geom2]] for c in d.contact)
    summary['final_finger_contact']=bool(final_fingers)
    summary['final_cube_motion_last_10_frames_m']=float(np.linalg.norm(poses[-min(10,len(poses)):,:3]-poses[-1,:3],axis=1).max())
    if save or trace_dir is not None:
        destination=Path(trace_dir) if trace_dir is not None else DEST
        destination.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(destination/'physics_trace.npz',qpos=joints,cube_pose=poses,timestamps=times,contact_peak=forces,bilateral=bilateral,targets=targets,clearance_degrees=clearance_correction)
        (destination/'physics_report.json').write_text(json.dumps(summary,indent=2))
        mujoco.mj_saveLastXML(str(destination/'scene.xml'),m)
    return summary,m,joints,poses

def render_video(cal,ep,m,joints,poses,output_dir=None,label='MuJoCo contact physics',camera_config=None,save_scene_xml=True,camera_delay_frames=None):
    from dataset_to_sim.observations import camera_sample_indices
    delays=camera_delay_frames or {}
    sensor_indices={view:camera_sample_indices(ep.frame_count,delays.get(view,0)) for view in ['top','wrist']}
    destination=Path(output_dir) if output_dir is not None else DEST
    destination.mkdir(parents=True,exist_ok=True)
    camera=camera_config if camera_config is not None else json.loads((ROOT/'calibration/wrist-full-contact.json').read_text());cid=m.camera('wrist_cam').id
    m.cam_pos[cid]=camera['position'];m.cam_quat[cid]=camera['quaternion'];m.cam_fovy[cid]=camera['fovy']
    if save_scene_xml:mujoco.mj_saveLastXML(str(destination/'scene.xml'),m)
    d=mujoco.MjData(m);addr=m.jnt_qposadr[m.joint('cube_free').id]
    real={v:read_real(ep,v) for v in ['top','wrist']};opt=mujoco.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0
    writer=cv2.VideoWriter(str(destination/'comparison-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,1024))
    if not writer.isOpened():raise RuntimeError('Encoder unavailable')
    with mujoco.Renderer(m,480,640) as renderer:
        for f in range(ep.frame_count):
            # Render a previously integrated physical state; never used to drive simulation.
            rows=[]
            for view,cam in [('top','top_cam'),('wrist','wrist_cam')]:
                observed=int(sensor_indices[view][f])
                d.qpos[:6]=joints[observed];d.qpos[addr:addr+7]=poses[observed];mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera=cam,scene_option=opt);rgb=renderer.render()
                canvas=np.zeros((512,1280,3),np.uint8);canvas[32:,:640]=real[view][f];canvas[32:,640:]=rgb
                sensor_note=f' | camera delay {delays.get(view,0)}f' if delays.get(view,0) else ''
                for text,pos in [(f'REAL | EP {ep.index:02d} | {view} | frame {f:03d} | {f/30:.2f}s',(10,23)),(f'{label} | {view}{sensor_note}',(650,23))]:cv2.putText(canvas,text,pos,cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1,cv2.LINE_AA)
                rows.append(canvas)
            combined=np.vstack(rows);writer.write(cv2.cvtColor(combined,cv2.COLOR_RGB2BGR))
            if f in [0,ep.frame_count//2,ep.frame_count-1,130,165,180,190,210,234]:cv2.imwrite(str(destination/f'comparison-{f:03d}.jpg'),cv2.cvtColor(combined,cv2.COLOR_RGB2BGR))
    writer.release()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--xy',nargs=2,type=float,default=[.175,-.045]);parser.add_argument('--grip-offset',type=float,default=-8);parser.add_argument('--video',action='store_true');args=parser.parse_args()
    cal=Calibration.load(ROOT/'calibration/episode0.json');ep=DatasetIndex(DATA).episode(0)
    summary,m,joints,poses=simulate(cal,ep,args.xy,args.grip_offset,save=True)
    print(json.dumps(summary,indent=2),flush=True)
    if args.video:render_video(cal,ep,m,joints,poses)
