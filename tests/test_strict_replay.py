from dataclasses import replace
import numpy as np
import mujoco
from dataset_to_sim.strict_replay import NativeJointAdapter, integrate_actions
from dataset_to_sim.dataset import Episode


def adapter():
    return NativeJointAdapter([0]*5, [1]*5, 2037, 3514, 4095, -10.)


def simple_model():
    bodies=''.join(f'<body pos="{i} 0 1"><joint name="j{i}" type="hinge" axis="0 1 0" damping=".1"/><geom type="sphere" size=".05" mass=".1" contype="0" conaffinity="0"/></body>' for i in range(6))
    actuators=''.join(f'<position joint="j{i}" kp="10" kv="1" ctrllimited="false" ctrlrange="-.01 .01"/>' for i in range(6))
    return mujoco.MjModel.from_xml_string(f'<mujoco><option timestep=".007" integrator="implicitfast" gravity="0 0 0"/><worldbody>{bodies}</worldbody><actuator>{actuators}</actuator></mujoco>')


def episode():
    action=np.zeros((4,6));action[:,0]=[0,10,20,30];action[:,5]=[0,20,40,60]
    state=np.zeros_like(action)
    return Episode(0,np.arange(4),np.arange(4),np.array([0,.03,.065,.10]),state,action,{})


def test_gripper_is_encoder_calibrated_percent_and_mapping_roundtrips():
    a=adapter();values=np.array([[1,2,3,4,5,0],[1,2,3,4,5,100]],float)
    result=a.to_model(values)
    np.testing.assert_allclose(np.rad2deg(result[:,5]),[-10,-10+1477*360/4095])
    np.testing.assert_allclose(a.to_native(result),values,atol=1e-12)
    np.testing.assert_array_equal(values[:,5],[0,100])


def test_zero_order_hold_boundaries_final_action_and_no_hidden_clipping():
    e=episode();m=simple_model();a=adapter();before=e.action_deg.copy()
    trace=integrate_actions(m,e,a)
    np.testing.assert_array_equal(trace['native_action_applied'],before)
    np.testing.assert_array_equal(e.action_deg,before)
    np.testing.assert_allclose(trace['sample_times'],np.r_[e.timestamps,.10+1/30],atol=1e-12)
    np.testing.assert_allclose(trace['step_controls'],a.to_model(before)[trace['step_action_index']],atol=0,rtol=0)
    assert set(trace['step_action_index'])=={0,1,2,3}
    assert trace['step_controls'][:,0].max()>.01
    for time,index in zip(trace['step_start_times'],trace['step_action_index']):
        assert time+1e-10>=e.timestamps[index]
        if index+1<len(e.timestamps):assert time<e.timestamps[index+1]+1e-10


def test_future_measured_states_cannot_drive_simulation():
    e=episode();modified=e.state_deg.copy();modified[1:]=999
    first=integrate_actions(simple_model(),e,adapter())
    second=integrate_actions(simple_model(),replace(e,state_deg=modified),adapter())
    np.testing.assert_array_equal(first['qpos'],second['qpos'])


def test_initial_gripper_uses_measured_opening_not_zero_or_first_command():
    e=episode();state=e.state_deg.copy();action=e.action_deg.copy()
    state[0,5]=.8124576807022095;action[0,5]=.24232633411884308
    e=replace(e,state_deg=state,action_deg=action)
    trace=integrate_actions(simple_model(),e,adapter())
    np.testing.assert_allclose(adapter().to_native(trace['qpos'][0])[5],state[0,5],atol=1e-12)
    assert trace['native_action_applied'][0,5]==action[0,5]
    assert trace['qpos'][0,5]!=trace['model_targets'][0,5]


def test_future_actions_cannot_affect_earlier_samples():
    e=episode();changed=e.action_deg.copy();changed[2:,0]=-50
    first=integrate_actions(simple_model(),e,adapter())
    second=integrate_actions(simple_model(),replace(e,action_deg=changed),adapter())
    np.testing.assert_array_equal(first['qpos'][:3],second['qpos'][:3])
    assert not np.array_equal(first['qpos'][3:],second['qpos'][3:])


def test_measured_cube_and_bowl_dimensions_include_collision_geometry():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from strict_action_scene import make_configuration, episode_scene, build_model
    config=make_configuration()
    cal=episode_scene(config,{'cube_pixel':[401,240],'bowl_pixel':[480,176],'cube_pixel_corners':[[390,230],[412,230],[412,252],[390,252]]})
    m=build_model(cal,config)
    np.testing.assert_allclose(m.geom_size[m.geom('cube_geom').id],[.00975]*3)
    assert m.body('cube').mass<.005
    assert not m.body_gravcomp.any()
    assert not m.actuator_ctrllimited.any()
    assert m.neq==0 and m.nexclude==0
    assert set(m.geom_contype)>={0,1,2,4,8}
    assert cal.world['bowl']['outer_radius']==.059
    assert cal.world['bowl']['height']==.05
    # Reconstructed mesh vertices must obey the measured outer envelope.
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    vertices=[]
    for i in range(32):
        geom=d.geom(f'bowl_contact_{i}');mid=m.geom_dataid[geom.id]
        v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
        vertices.extend(v@geom.xmat.reshape(3,3).T+geom.xpos-d.body('bowl').xpos)
    vertices=np.array(vertices)
    np.testing.assert_allclose(vertices[:,2].max(),.05,atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(vertices[:,:2],axis=1).max(),.059,atol=1e-7)


def test_motor_latency_keeps_native_commands_and_initial_history():
    # A missing history initialization would pull a nonzero joint toward zero.
    import tempfile
    from pathlib import Path
    model=simple_model()
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'toy.xml';mujoco.mj_saveLastXML(str(path),model)
        text=path.read_text().replace('<general ', '<general nsample="32" delay="0.08" interp="zoh" ')
        model=mujoco.MjModel.from_xml_string(text)
    assert np.all(model.actuator_delay==.08)
    e=episode();e.action_deg[:,0]=[30,50,50,50];e.state_deg[:,0]=30
    trace=integrate_actions(model,e,adapter())
    np.testing.assert_array_equal(trace['native_action_applied'],e.action_deg)
    np.testing.assert_allclose(trace['qpos'][:4,0],np.deg2rad(30),atol=1e-10)
    assert trace['qpos'][-1,0]>np.deg2rad(30)


def test_cube_top_face_detection_projects_at_measured_top_height():
    import sys
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from strict_action_scene import make_configuration,episode_scene
    c=make_configuration();c['cube_projection_height_fraction']=1.
    scene={'cube_pixel':[400,240],'bowl_pixel':[480,176],'cube_pixel_corners':[[390,230],[410,230],[410,250],[390,250]]}
    cal=episode_scene(c,scene);cam=c['base_calibration']['cameras']['top']
    cube=np.array(cal.world['cube']['position']);cube[2]+=.0195/2-.0002
    R=Rotation.from_quat(np.array(cam['quaternion'])[[1,2,3,0]]).as_matrix()
    camera=np.diag([1,-1,-1])@R.T@(cube-cam['position']);f=240/np.tan(np.deg2rad(cam['fovy']/2))
    np.testing.assert_allclose(camera[:2]/camera[2]*f+[320,240],scene['cube_pixel'],atol=1e-8)


def test_hardware_calibration_limits_are_model_frame_limits_not_command_clipping():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from strict_action_scene import make_configuration,episode_scene,build_model
    c=make_configuration();c['hardware_joint_limits']=True
    s={'cube_pixel':[400,240],'bowl_pixel':[480,176],'cube_pixel_corners':[[390,230],[410,230],[410,250],[390,250]]}
    m=build_model(episode_scene(c,s),c)
    f=c['follower_calibration']['wrist_flex'];half=(f['range_max']-f['range_min'])*360/4095/2
    offset=c['adapter']['arm_offset_deg'][3]
    np.testing.assert_allclose(np.rad2deg(m.joint('wrist_flex').range),[-half+offset,half+offset])
    assert not m.actuator_ctrllimited.any()


def test_parent_finger_contacts_are_explicit_and_block_interpenetration():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from strict_action_scene import make_configuration,episode_scene,build_model
    c=make_configuration();c['jaw_self_contact']=True;c['gripper_pad_proxies']=False
    s={'cube_pixel':[400,240],'bowl_pixel':[480,176],'cube_pixel_corners':[[390,230],[410,230],[410,250],[390,250]]}
    m=build_model(episode_scene(c,s),c);d=mujoco.MjData(m)
    d.qpos[5]=np.deg2rad(-12);mujoco.mj_forward(m,d)
    jaws={m.body('gripper').id,m.body('moving_jaw_so101_v1').id}
    touching=[co for co in d.contact if {m.geom_bodyid[co.geom1],m.geom_bodyid[co.geom2]}==jaws]
    assert touching, 'Parent filtering must not let the two distal finger surfaces pass through each other'
    assert any(co.dist<0 for co in touching)
    assert m.nexclude==0 and m.neq==0


def test_material_calibration_changes_contacts_without_actuator_targets():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from strict_action_scene import make_configuration,episode_scene,build_model
    c=make_configuration();c['contact_materials']={'jaw':[.6,.001,.0001],'cube':[.3,.0002,.0001],'work_mat':[.4,.0002,.0001]}
    s={'cube_pixel':[400,240],'bowl_pixel':[480,176],'cube_pixel_corners':[[390,230],[410,230],[410,250],[390,250]]}
    m=build_model(episode_scene(c,s),c)
    np.testing.assert_allclose(m.geom('cube_geom').friction,[.3,.0002,.0001])
    np.testing.assert_allclose(m.geom('fixed_jaw_box4').friction,[.6,.001,.0001])
    np.testing.assert_allclose(m.geom('work_mat').friction,[.4,.0002,.0001])
    assert not m.actuator_ctrllimited.any()


def test_task_score_rejects_brief_or_unloaded_contact_events():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from strict_action_replay import score
    pose=np.zeros((15,7));pose[:,2]=.1;pose[0,2]=.02
    trace={'cube_pose':pose,'bilateral':np.ones(14,dtype=bool),'audit':{'final_finger_contact':False,'grasp_support':{'longest_secure_grasp_s':.002,'longest_secure_lift_s':.001},'final_cube_linear_speed_m_s':0.,'final_cube_angular_speed_rad_s':0.}}
    assert score(trace,{'inside':True})=='no_bilateral_grasp'
