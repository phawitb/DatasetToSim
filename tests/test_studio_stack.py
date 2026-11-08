import copy,json,sys
from pathlib import Path
import cv2,mujoco,numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_sam_geometry import scene_from_masks
from strict_action_scene import ROOT


def config_and_masks():
    from episode0 import Calibration
    from dataset_to_sim.scene_estimation import project_cube
    cfg=json.loads((ROOT/'calibration/native-action-current.json').read_text())
    camera=Calibration(ROOT/'calibration/native-action-current.json',cfg['base_calibration']).camera('top')
    masks=[]
    for pose in [[.16,.10,0],[.27,.10,.05]]:
        mask=np.zeros((480,640),np.uint8)
        hull=project_cube(camera,pose,.0195,.003)
        cv2.fillConvexPoly(mask,np.round(hull).astype(np.int32),1);masks.append(mask>0)
    return cfg,masks


def test_cube_target_uses_metric_cube_fit_and_dynamic_body():
    from studio_stack import scene_calibration,build_scene_model
    cfg,masks=config_and_masks()
    scene=scene_from_masks(cfg,*masks,task='stack',bowl_present=False)
    cal=scene_calibration(cfg,scene);model=build_scene_model(cal,cfg)
    np.testing.assert_allclose(cal.world['target_cube']['position'][:2],[.27,.10],atol=.002)
    assert model.joint('target_cube_free').type==mujoco.mjtJoint.mjJNT_FREE
    assert model.body('target_cube').mass>0
    assert mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'bowl')==-1
    assert model.geom('target_cube_geom').contype>0


def test_stack_requires_explicit_bowl_obstacle_if_present():
    cfg,masks=config_and_masks()
    with pytest.raises(ValueError,match='bowl obstacle'):
        scene_from_masks(cfg,*masks,task='stack',bowl_present=True)


def test_stack_verdict_requires_stable_supported_released_pair():
    from studio_stack import stack_verdict
    target=np.tile([.27,.10,.01275,1,0,0,0],(60,1))
    source=target.copy();source[:,2]+=.0195
    source[0,2]=.01275;source[1,2]=.15
    trace={'cube_pose':source,'target_pose':target,'bilateral':np.ones(59,bool),
           'target_velocity':np.zeros((60,6)),
           'audit':{'final_finger_contact':False,'final_stack_support_N':.044,
                    'final_cube_linear_speed_m_s':0.,'final_cube_angular_speed_rad_s':0.,
                    'grasp_support':{'longest_secure_grasp_s':1.,'longest_secure_lift_s':1.}}}
    assert stack_verdict(trace,.0195)['outcome']=='success'
    for change in ['no_support','target_moved','hovering','tipped','held','off_center','moving']:
        t=copy.deepcopy(trace)
        if change=='no_support':t['audit']['final_stack_support_N']=0
        elif change=='target_moved':t['target_pose'][1:,0]+=.01;t['cube_pose'][2:,0]+=.01
        elif change=='hovering':t['cube_pose'][2:,2]+=.005
        elif change=='tipped':t['target_pose'][-1,3:]=[.707, .707,0,0]
        elif change=='held':t['audit']['final_finger_contact']=True
        elif change=='off_center':t['cube_pose'][2:,0]+=.012
        elif change=='moving':t['target_velocity'][-1,0]=.1
        assert stack_verdict(t,.0195)['outcome']!='success',change


def test_preflight_rejects_moved_or_rotated_stack_target():
    from dataset_to_sim.studio import validate_scene_match
    a={'cube':{'position':[.1,0,.01],'quaternion':[1,0,0,0]},'bowl_present':False,
       'bowl':{'position':[.4,0,0]},'target_cube':{'position':[.2,0,.01],'quaternion':[1,0,0,0]}}
    b=copy.deepcopy(a);b['target_cube']['position'][0]+=.004
    with pytest.raises(ValueError,match='moved'):validate_scene_match(a,b)
    b=copy.deepcopy(a);b['target_cube']['quaternion']=[np.cos(.15),0,0,np.sin(.15)]
    with pytest.raises(ValueError,match='rotated'):validate_scene_match(a,b)


def test_stack_capture_freezes_task_and_obstacle_requirement(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    import studio_server as server
    monkeypatch.setattr(server,'WORK',tmp_path);monkeypatch.setattr(server,'current',None);monkeypatch.setattr(server,'jobs',{})
    monkeypatch.setattr(server,'hardware',SimpleNamespace(call=lambda command,**kw: {'running':False} if command=='status' else {'source':'live','images':{},'initial_native':[0]*6}))
    client=TestClient(server.app);headers={'X-Studio-Token':server.TOKEN}
    r=client.post('/api/capture',json={'source':'live','task':'stack','bowl_present':True},headers=headers)
    assert r.status_code==200
    captured=r.json()['capture'];saved=json.loads((tmp_path/captured['id']/'capture.json').read_text())
    assert saved['task']=='stack' and saved['bowl_present'] is True
    assert client.post('/api/capture',json={'task':'unknown'},headers=headers).status_code==409
    assert client.post('/api/capture',json={'task':'stack','bowl_present':'false'},headers=headers).status_code==409
    assert client.post('/api/capture',json={'task':'stack','source':'dataset'},headers=headers).status_code==409


def test_already_home_is_no_motion_and_clears_old_job(monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    import studio_server as server
    pose=json.loads((ROOT/'calibration/studio-home.json').read_text())['home'];calls=[]
    def call(command,**kw):
        calls.append(command)
        assert command=='status','Already Home must not dispatch or re-segment'
        return {'running':False,'positions':pose}
    monkeypatch.setattr(server,'hardware',SimpleNamespace(call=call));monkeypatch.setattr(server,'current','previous');monkeypatch.setattr(server,'jobs',{})
    r=TestClient(server.app).post('/api/robot/home',json={},headers={'X-Studio-Token':server.TOKEN,'X-Studio-Client':'owner'})
    assert r.json()=={'already_home':True} and server.current is None


def test_stack_trace_records_falling_target_without_pinning_it():
    from studio_stack import scene_calibration,build_scene_model
    from dataset_to_sim.strict_replay import integrate_actions,NativeJointAdapter
    from dataset_to_sim.dataset import Episode
    cfg,masks=config_and_masks();scene=scene_from_masks(cfg,*masks,task='stack',bowl_present=False)
    scene['target_cube']['position'][2]+=.01
    model=build_scene_model(scene_calibration(cfg,scene),cfg)
    initial=np.asarray(json.loads((ROOT/'calibration/studio-home.json').read_text())['home'])
    commands=np.tile(initial,(16,1));times=np.arange(16)/30
    episode=Episode(-1,np.arange(16),np.arange(16),times,commands,commands,{})
    trace=integrate_actions(model,episode,NativeJointAdapter(**cfg['adapter']))
    assert trace['target_pose'].shape==(17,7) and trace['target_velocity'].shape==(17,6)
    assert trace['target_pose'][-1,2]<trace['target_pose'][0,2]-.008
    assert trace['audit']['post_initial_state_writes']==0


def test_interrupted_stack_home_moves_straight_back_as_requested(tmp_path,monkeypatch):
    # Home no longer inspects the scene: after any run, interrupted or not, it moves straight back.
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    import studio_server as server
    for i,task in enumerate(['bowl','stack']):
        p=tmp_path/str(i);p.mkdir();(p/'capture.json').write_text(json.dumps({'source':'live','task':task,'captured_at':100+i}))
    calls=[]
    def call(command,**kw):
        calls.append(command)
        return {'homing':True,'duration_s':2.} if command=='go_home' else {'running':False,'positions':[0]*6}
    monkeypatch.setattr(server,'hardware',SimpleNamespace(call=call));monkeypatch.setattr(server,'WORK',tmp_path);monkeypatch.setattr(server,'current',None)
    r=TestClient(server.app).post('/api/robot/home',json={},headers={'X-Studio-Token':server.TOKEN,'X-Studio-Client':'owner'})
    assert r.status_code==200 and r.json()['homing'] and 'snapshot' not in calls and calls[-1]=='go_home'
