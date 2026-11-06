"""Measured geometry and frozen calibration for strict native-action replay."""
import copy
from dataclasses import asdict
import hashlib
import itertools
import json
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from episode0 import ROOT, Calibration, ground_point
from episode0_physics import physical_model
from dataset_to_sim.strict_replay import NativeJointAdapter

FOLLOWER=ROOT.home()/'.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json'


def make_configuration():
    measurements=json.loads((ROOT/'calibration/real-measurements.json').read_text())
    follower=json.loads(FOLLOWER.read_text());grip=follower['gripper']
    if any(motor['drive_mode']!=0 for motor in follower.values()):
        raise ValueError('This audited adapter expects the current drive_mode=0 calibration')
    base=Calibration.load(ROOT/'calibration/episode0.json')
    adapter=NativeJointAdapter(base.data['joint_convention']['offset_deg'][:5],base.data['joint_convention']['sign'][:5],grip['range_min'],grip['range_max'],4095,-10.)
    data=copy.deepcopy(base.data)
    data['world']['cube']['size']=measurements['cube']['size_xyz'][0]
    data['world']['bowl'].update(outer_radius=measurements['bowl']['top_rim_diameter']/2,height=measurements['bowl']['height'])
    if 'mat' in measurements:
        data['world']['mat']['size'][:2]=measurements['mat']['size_xy']
        if measurements['mat'].get('thickness') is not None:
            thickness=measurements['mat']['thickness'];data['world']['mat']['size'][2]=thickness
            data['world']['mat']['position'][2]=data['world']['table']['height']+thickness/2
    if measurements.get('robot',{}).get('additional_base_riser_height_m')==0:
        # The original base_so101_v2 visual mesh reaches -2.4mm below its body frame.
        data['world']['robot_base']['position'][2]=data['world']['table']['height']+.0024
        data['world']['robot_base']['quaternion']=[1,0,0,0]
    data['physics']={'gripper_joint_damping':.1,'gripper_armature':.003,'gripper_servo_kv':.1}
    return {'schema_version':1,'mode':'strict_native_action','base_calibration':data,
            'gripper_pad_proxies':False,'jaw_self_contact':True,'cube_projection_height_fraction':1.,'bowl_projection_height_fraction':1.,
            'adapter':asdict(adapter),'follower_calibration':follower,
            'follower_calibration_sha256':hashlib.sha256(FOLLOWER.read_bytes()).hexdigest(),
            'follower_calibration_source':str(FOLLOWER),'measurements':measurements,
            'gripper_degrees_per_native_unit':adapter.gripper_degrees_per_native_unit,
            'wrist_camera':json.loads((ROOT/'calibration/wrist-full-contact.json').read_text()),
            'estimates':{'wood_density_kg_m3':600.,'cube_mass_kg':600.*data['world']['cube']['size']**3,
                         'bowl_bottom_outer_radius_m':.0413,'bowl_wall_radial_thickness_m':.003,'bowl_floor_thickness_m':.004},
            'physical_control':{'arm_kp':150.,'arm_kv':4.,'arm_torque_limit_Nm':2.94,'gripper_kp':35.,'gripper_kv':.1,'gripper_torque_limit_Nm':.35,'gravity_compensation':False,'input_clipping':False},
            'limitations':['Current follower calibration is snapshotted; identity with the historical recording calibration is unverified.',
                           'Gripper encoder scale is known from current calibration; model closed reference of -10 degrees is the source joint lower limit, still an estimated mechanical zero.',
                           'Arm reference offsets, camera calibration, motor response and friction remain inherited estimates; no system identification performed in this run.',
                           'Wood density, bowl bottom radius and thickness are assumptions, not user measurements.',
                           'Dataset stores nominal observation timestamps, not exact serial-write timestamps. The final command is held for one nominal 30 Hz period.',
                           'Unit/frame conversion is continuous; raw encoder integer quantization is not emulated. The native recorded action values are preserved exactly.']}


def episode_scene(config,scene):
    data=copy.deepcopy(config['base_calibration']);world=data['world'];camera=data['cameras']['top']
    position=np.array(camera['position']);camera_R=Rotation.from_quat(np.array(camera['quaternion'])[[1,2,3,0]]).as_matrix()
    R=np.diag([1,-1,-1])@camera_R.T
    focal=240/np.tan(np.deg2rad(camera['fovy']/2));K=np.array([[focal,0,320],[0,focal,240],[0,0,1.]])
    mat=world['mat'];mat_top=mat['position'][2]+mat['size'][2]/2;edge=world['cube']['size']
    fraction=config.get('cube_projection_height_fraction',.5)
    support=mat_top
    for _ in range(2):
        p=ground_point(scene['cube_pixel'],K,R,position,support+edge*fraction)
        overlaps=np.all(np.abs(p[:2]-np.array(mat['position'][:2]))<np.array(mat['size'][:2])/2+edge/np.sqrt(2))
        support=mat_top if overlaps else world['table']['height']
    p=ground_point(scene['cube_pixel'],K,R,position,support+edge*fraction)
    world['cube']['position']=[*p[:2],support+edge/2+.0002]
    corners=np.array([ground_point(uv,K,R,position,support+edge*fraction) for uv in scene['cube_pixel_corners']])
    direction=corners[1]-corners[0];yaw=np.arctan2(direction[1],direction[0])
    world['cube']['quaternion']=[float(np.cos(yaw/2)),0.,0.,float(np.sin(yaw/2))]
    if 'cube_pose_from_initial_rgb' in scene:
        observed=scene['cube_pose_from_initial_rgb']
        world['cube']['position']=list(observed['position'])
        world['cube']['quaternion']=list(observed['quaternion'])
    p=ground_point(scene['bowl_pixel'],K,R,position,mat_top+world['bowl']['height']*config.get('bowl_projection_height_fraction',.5))
    world['bowl']['position']=[*p[:2],mat_top]
    data['scene_estimation']={'source':'initial RGB projected with the shared top camera; no grasp-state homography or episode outcome optimization','cube_support_z_m':support,'cube_detection_height_fraction':fraction}
    return Calibration(ROOT/'calibration/strict-native-action.json',data)


def estimate_scene(config,rgb):
    from benchmark72 import detect_scene
    from dataset_to_sim.scene_estimation import fit_cube_silhouette
    scene=detect_scene(rgb)
    if config.get('scene_detection') in ('cube-silhouette-v1','cube-silhouette-v2'):
        cal=episode_scene(config,scene);cube=cal.world['cube'];q=cube['quaternion']
        support=cal.data['scene_estimation']['cube_support_z_m']
        initial=[*cube['position'][:2],2*np.arctan2(q[3],q[0])]
        pose,report=fit_cube_silhouette(rgb,cal.camera('top'),scene,initial,cube['size'],support,use_top_face=config['scene_detection']=='cube-silhouette-v2')
        scene['cube_pose_from_initial_rgb']={'position':[float(pose[0]),float(pose[1]),support+cube['size']/2+.0002],'quaternion':[float(np.cos(pose[2]/2)),0.,0.,float(np.sin(pose[2]/2))]}
        scene['cube_pose_fit']=report
        scene['method']='initial RGB only; measured 19.5mm cube silhouette; shared camera; no trajectory/outcome inputs'
    return scene


def _numbers(values):return ' '.join(str(float(x)) for x in np.ravel(values))


def build_model(cal,config):
    root=ET.fromstring(physical_model(cal,return_xml=True))
    for body in root.findall('.//body'):body.set('gravcomp','0')
    for actuator in root.findall('actuator/position'):actuator.set('ctrllimited','false')
    if config.get('hardware_joint_limits',False):
        adapter=NativeJointAdapter(**config['adapter'])
        for i,actuator in enumerate(root.findall('actuator/position')):
            name=actuator.get('joint');native=np.zeros((2,6))
            if i==5:native[:,i]=[0,100]
            else:
                follower=config['follower_calibration'][name]
                half=(follower['range_max']-follower['range_min'])*360/4095/2
                native[:,i]=[-half,half]
            limits=np.sort(adapter.to_model(native)[:,i])
            root.find(f".//joint[@name='{name}']").set('range',_numbers(limits))
            actuator.set('ctrlrange',_numbers(limits))
    dynamics=config.get('identified_dynamics')
    if dynamics:
        for i,actuator in enumerate(root.findall('actuator/position')):
            for key in ['kp','kv']:
                actuator.set(key,str(dynamics[key][i]))
            actuator.set('nsample','512');actuator.set('delay',str(dynamics['delay_s'][i]));actuator.set('interp','zoh')
            joint=root.find(f".//joint[@name='{actuator.get('joint')}']")
            for key in ['damping','armature','frictionloss']:
                if key in dynamics:joint.set(key,str(dynamics[key][i]))
    root.find(".//geom[@name='cube_geom']").set('mass',str(config['estimates']['cube_mass_kg']))
    if not config.get('gripper_pad_proxies',True):
        for name in ['fixed_pad','moving_pad']:
            geom=root.find(f".//geom[@name='{name}']");geom.set('contype','0');geom.set('conaffinity','0');geom.set('mass','0')
    if config.get('jaw_self_contact',False):
        # MuJoCo normally filters parent/child body collisions. These distal
        # surfaces must still meet when the gripper closes. The proximal hinge
        # housing stays under the standard connected-link filtering.
        contact=root.find('contact')
        if contact is None:contact=ET.SubElement(root,'contact')
        fixed=[g.get('name') for g in root.findall('.//geom') if g.get('name','').startswith('fixed_jaw_') and g.get('name') not in ['fixed_jaw_box1','fixed_jaw_box2']]
        moving=[g.get('name') for g in root.findall('.//geom') if g.get('name','').startswith('moving_jaw_') and g.get('name')!='moving_jaw_box1']
        for a in fixed:
            for b in moving:
                ET.SubElement(contact,'pair',geom1=a,geom2=b,condim='6',friction='1.4 1.4 .01 .001 .001',solref=_numbers(config.get('contact_model',{}).get('solref',[.006,1])),solimp=_numbers(config.get('contact_model',{}).get('solimp',[.95,.99,.001])))
    camera=root.find(".//camera[@name='wrist_cam']");wrist=config['wrist_camera']
    for key,field in [('pos','position'),('quat','quaternion')]:camera.set(key,_numbers(wrist[field]))
    camera.set('fovy',str(wrist['fovy']))
    bowl=root.find(".//body[@name='bowl']");bowl.set('pos',_numbers(cal.world['bowl']['position']))
    for child in list(bowl):bowl.remove(child)
    asset=root.find('asset')
    for child in list(asset):
        if child.get('name') in ['bowl_visual','bowl_floor_mesh']:asset.remove(child)
    estimates=config['estimates'];r0=estimates['bowl_bottom_outer_radius_m'];r1=cal.world['bowl']['outer_radius']
    h=cal.world['bowl']['height'];wall=estimates['bowl_wall_radial_thickness_m'];floor=estimates['bowl_floor_thickness_m']
    if not (0<wall<r0<r1 and 0<floor<h):raise ValueError('Inconsistent bowl dimensions')
    # Rendered frustum: exactly 118 mm outer rim and 50 mm height.
    n=96;angles=np.arange(n)*2*np.pi/n;vertices=[];faces=[]
    for radius,z in [(r0,0),(r1,h),(r1-wall,h),(r0-wall,floor)]:
        vertices.extend([[radius*np.cos(a),radius*np.sin(a),z]for a in angles])
    for ring in range(3):
        for j in range(n):
            a=ring*n+j;b=ring*n+(j+1)%n;c=(ring+1)*n+j;d=(ring+1)*n+(j+1)%n
            faces.extend([[a,b,c],[b,d,c]])
    vertices.append([0,0,floor]);faces.extend([[4*n,3*n+j,3*n+(j+1)%n]for j in range(n)])
    ET.SubElement(asset,'mesh',name='bowl_visual',vertex=_numbers(vertices),face=' '.join(map(str,np.ravel(faces))))
    ET.SubElement(bowl,'geom',type='mesh',mesh='bowl_visual',rgba='.5 .65 .6 1',contype='0',conaffinity='0')
    # Convex floor and 32 convex wall wedges share that same measured envelope.
    vertices=[[ (r0-wall)*np.cos(a),(r0-wall)*np.sin(a),z] for z in [0,floor] for a in angles]
    ET.SubElement(asset,'mesh',name='bowl_floor_mesh',vertex=_numbers(vertices))
    ET.SubElement(bowl,'geom',name='bowl_floor_collision',type='mesh',mesh='bowl_floor_mesh',group='3',contype='4',conaffinity='15',solref='.002 1',solimp='.99 .999 .0001')
    for j in range(32):
        edges=[j*2*np.pi/32,(j+1)*2*np.pi/32]
        vertices=[[radius*np.cos(a),radius*np.sin(a),z]for z,outer in [(0,r0),(h,r1)] for radius in [outer,outer-wall] for a in edges]
        name=f'bowl_contact_{j}';ET.SubElement(asset,'mesh',name=name+'_mesh',vertex=_numbers(vertices))
        ET.SubElement(bowl,'geom',name=name,type='mesh',mesh=name+'_mesh',group='3',contype='4',conaffinity='15',solref='.002 1',solimp='.99 .999 .0001')
    if 'contact_model' in config:
        for geom in root.findall('.//worldbody//geom'):
            if int(geom.get('contype','0'))>0:
                geom.set('solref',_numbers(config['contact_model']['solref']))
                geom.set('solimp',_numbers(config['contact_model']['solimp']))
    materials=config.get('contact_materials',{})
    for geom in root.findall('.//worldbody//geom'):
        key='jaw' if geom.get('contype')=='2' else ('cube' if geom.get('name')=='cube_geom' else geom.get('name'))
        if key in materials:geom.set('friction',_numbers(materials[key]))
    if 'jaw' in materials:
        slide,torsion,rolling=materials['jaw']
        for pair in root.findall('contact/pair'):
            pair.set('friction',_numbers([slide,slide,torsion,rolling,rolling]))
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))


def containment(model,qpos,cube_pose,cal,config):
    d=mujoco.MjData(model);d.qpos[:6]=qpos;addr=model.jnt_qposadr[model.joint('cube_free').id]
    d.qpos[addr:addr+7]=cube_pose;mujoco.mj_forward(model,d)
    geom=d.geom('cube_geom');v=np.array(list(itertools.product([-1,1],repeat=3)))*model.geom_size[geom.id]
    vertices=v@geom.xmat.reshape(3,3).T+geom.xpos-d.body('bowl').xpos
    e=config['estimates'];h=cal.world['bowl']['height'];r0=e['bowl_bottom_outer_radius_m'];r1=cal.world['bowl']['outer_radius'];wall=e['bowl_wall_radial_thickness_m'];tol=.0005
    normals=np.c_[np.cos((np.arange(32)+.5)*2*np.pi/32),np.sin((np.arange(32)+.5)*2*np.pi/32)]
    radii=(r0-wall+(r1-r0)*vertices[:,2]/h)*np.cos(np.pi/32)
    violation=float(np.max(vertices[:,:2]@normals.T-radii[:,None]))
    inside=violation<=tol and vertices[:,2].min()>=e['bowl_floor_thickness_m']-tol and vertices[:,2].max()<=h+tol
    return {'inside':bool(inside),'max_wall_violation_m':violation,'lowest_corner_local_z_m':float(vertices[:,2].min()),'highest_corner_local_z_m':float(vertices[:,2].max()),'tolerance_m':tol}
