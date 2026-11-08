import sys
from pathlib import Path
from fastapi.testclient import TestClient
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import studio_server as server

client=TestClient(server.app)


def test_no_cross_origin_or_tokenless_robot_commands():
    assert client.post('/api/robot/stop',json={}).status_code==403
    assert client.post('/api/robot/stop',json={},headers={'X-Studio-Token':server.TOKEN,'Origin':'https://other.example'}).status_code==403


def test_catalog_and_trace_are_real_and_rehearsal_cannot_execute():
    catalog=client.get('/api/bootstrap').json()
    assert len(catalog['episodes'])==72
    ep=client.get('/api/episodes/0').json()
    assert ep['frames']==235 and len(ep['real'])==len(ep['sim'])==len(ep['actions'])==235
    assert ep['video'].endswith('real-vs-sim-strict.mp4')
    assert client.get('/api/episodes/999').status_code==409
    headers={'X-Studio-Token':server.TOKEN}
    capture=client.post('/api/capture',json={'source':'dataset','episode':0},headers=headers).json()
    assert capture['capture']['source']=='dataset'
    assert client.get('/api/plans/'+capture['id']).json()['initial_native']==ep['real'][0]
    # Unsupported arbitrary command and traversal cannot reach the hardware or filesystem.
    assert client.post('/api/robot/move',json={'positions':[90]*6},headers=headers).status_code==409
    assert client.get('/media/studio/%2e%2e/%2e%2e/README.md').status_code in [404,409]


def test_scene_rotation_requires_new_plan():
    from dataset_to_sim.studio import validate_scene_match
    old={'cube':{'position':[.2,0,.01],'quaternion':[1,0,0,0]},'bowl':{'position':[.3,0,.003]}}
    new={'cube':{'position':[.2,0,.01],'quaternion':[.9238795,0,0,.3826834]},'bowl':old['bowl']}
    import pytest
    with pytest.raises(ValueError,match='rotated'):validate_scene_match(old,new)


def test_stop_during_preflight_cancels_pending_dispatch(tmp_path,monkeypatch):
    import base64,json,threading,time,numpy as np,cv2
    from types import SimpleNamespace
    import strict_action_scene
    identifier='a'*32;folder=tmp_path/identifier;folder.mkdir()
    world={'cube':{'position':[.2,0,.01],'quaternion':[1,0,0,0]},'bowl':{'position':[.3,0,.003]}}
    report={'id':identifier,'source':'live','outcome':'success','transfer_checks_passed':True,'captured_at':time.time(),'initial_native':[0]*6,'calibration_hash':'abc','executed':False,'world':world}
    (folder/'capture.json').write_text(json.dumps(report));(folder/'plan.json').write_text(json.dumps(report));np.savez(folder/'plan.npz',actions=np.zeros((2,6)),timestamps=[0,.1])
    entered=threading.Event();release=threading.Event();sent=[]
    class FakeHardware:
        def call(self,command,**kwargs):
            if command=='status':return {'running':False}
            if command=='stop':return {'stopped':True}
            if command=='snapshot':
                entered.set();assert release.wait(3)
                return {'captured_at':time.time(),'initial_native':[0]*6,'calibration_hash':'abc','stop_generation':0,'images':{'top':base64.b64encode(cv2.imencode('.jpg',np.zeros((2,2,3),np.uint8))[1]).decode()}}
            if command=='execute':sent.append(kwargs);return {'running':True}
    monkeypatch.setattr(server,'hardware',FakeHardware());monkeypatch.setattr(server,'WORK',tmp_path);monkeypatch.setattr(server,'current',identifier);monkeypatch.setattr(server,'jobs',{})
    monkeypatch.setattr(server,'load_selection',lambda *a: ({'revision':None},[]))
    monkeypatch.setattr(server,'verify_selected_scene',lambda *a: None)
    monkeypatch.setattr(strict_action_scene,'episode_scene',lambda *a:SimpleNamespace(world=world))
    headers={'X-Studio-Token':server.TOKEN,'X-Studio-Client':'owner'};result=[]
    thread=threading.Thread(target=lambda:result.append(client.post(f'/api/plans/{identifier}/execute',json={},headers=headers)))
    thread.start();assert entered.wait(3)
    assert client.post('/api/robot/stop',json={},headers=headers).status_code==200
    release.set();thread.join(3)
    assert result[0].status_code==409 and 'Stop requested' in result[0].json()['error']
    assert not sent


def test_failed_resimulation_invalidates_old_success(tmp_path,monkeypatch):
    import json,time,numpy as np
    from types import SimpleNamespace
    identifier='b'*32;folder=tmp_path/identifier;folder.mkdir()
    report={'id':identifier,'outcome':'success','transfer_checks_passed':True,'executed':False,'simulated_action_hash':'old','video':'simulation.mp4'}
    (folder/'capture.json').write_text('{}');(folder/'plan.json').write_text(json.dumps(report));np.savez(folder/'plan.npz',actions=np.zeros((2,6)),timestamps=[0,.1])
    monkeypatch.setattr(server,'WORK',tmp_path);monkeypatch.setattr(server,'current',identifier);monkeypatch.setattr(server,'jobs',{})
    monkeypatch.setattr(server,'hardware',SimpleNamespace(call=lambda *a,**k:{'running':False}))
    monkeypatch.setattr(server.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1))
    response=client.post(f'/api/plans/{identifier}/simulate',json={},headers={'X-Studio-Token':server.TOKEN})
    assert response.status_code==200
    stored=json.loads((folder/'plan.json').read_text())
    assert stored['outcome']=='not_simulated' and not stored['transfer_checks_passed'] and stored['simulated_action_hash'] is None
    assert 'video' not in stored
    deadline=time.monotonic()+2
    while server.jobs[identifier]['status']=='running' and time.monotonic()<deadline:time.sleep(.01)
    assert server.jobs[identifier]['status']=='error'


def test_every_episode_can_request_a_playable_comparison(monkeypatch):
    import time
    assert client.post('/api/episodes/0/video',json={},headers={'X-Studio-Token':server.TOKEN}).json()['status']=='ready'
    result=client.get('/api/episodes/1/video').json()
    assert result['status'] in ['idle','rendering','ready']
    assert client.post('/api/episodes/999/video',json={},headers={'X-Studio-Token':server.TOKEN}).status_code==409


def test_completed_run_finalizes_even_after_another_capture_is_selected(tmp_path,monkeypatch):
    import json,time
    from types import SimpleNamespace
    old='c'*32;new='d'*32;folder=tmp_path/old;folder.mkdir()
    (folder/'capture.json').write_text('{}');(folder/'plan.json').write_text(json.dumps({'id':old,'executed':True}))
    monkeypatch.setattr(server,'WORK',tmp_path);monkeypatch.setattr(server,'current',new);monkeypatch.setattr(server,'jobs',{old:{'status':'executing'},new:{'status':'captured'}})
    monkeypatch.setattr(server,'hardware',SimpleNamespace(call=lambda *a,**k:{'running':False,'execution':old}))
    def finalize(*a,**k):
        (folder/'execution.json').write_text(json.dumps({'video':'hardware-comparison.mp4','completed':True}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(server.subprocess,'run',finalize)
    assert client.get('/api/status').json()['current']==new
    deadline=time.monotonic()+2
    while server.jobs[old]['status']!='complete' and time.monotonic()<deadline:time.sleep(.01)
    assert server.jobs[old]['status']=='complete'
    assert server.current==new and server.jobs[new]['status']=='captured'


def test_torque_change_invalidates_previous_capture(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(server,'current','old-capture')
    monkeypatch.setattr(server,'jobs',{})
    monkeypatch.setattr(server,'hardware',SimpleNamespace(call=lambda *a,**k:{'connected':True,'running':False,'torque_enabled':False}))
    response=client.post('/api/robot/torque_off',json={},headers={'X-Studio-Token':server.TOKEN})
    assert response.status_code==200 and server.current is None


def test_home_requires_control_page_owner(monkeypatch):
    response=client.post('/api/robot/home',json={},headers={'X-Studio-Token':server.TOKEN})
    assert response.status_code==409 and 'page' in response.json()['error'].lower()


def test_stop_during_home_snapshot_prevents_dispatch(tmp_path,monkeypatch):
    import json,threading,time,base64,numpy as np
    from types import SimpleNamespace
    reference=tmp_path/('e'*32);reference.mkdir();(reference/'scene.xml').write_text('model')
    (reference/'capture.json').write_text(json.dumps({'captured_at':time.time()}))
    (reference/'plan.json').write_text(json.dumps({'source':'live'}))
    entered=threading.Event();release=threading.Event();sent=[];snapshots=[]
    def hardware(command,**kwargs):
        if command=='status':return {'running':False}
        if command=='snapshot':
            if not snapshots:entered.set();release.wait(2)
            snapshots.append(1)
            return {'source':'live','captured_at':time.time(),'initial_native':[0]*6,'calibration_hash':'x','stop_generation':0,'images':{'top':base64.b64encode(__import__('cv2').imencode('.jpg',np.zeros((2,2,3),np.uint8))[1]).decode()}}
        if command=='execute':sent.append(kwargs)
        return {}
    def simulation(args,**kwargs):
        folder=Path(args[2]);report=json.loads((folder/'capture.json').read_text())|{'outcome':'success','transfer_checks_passed':True,'executed':False,'bowl_pixel':[0,0]}
        (folder/'plan.json').write_text(json.dumps(report));np.savez(folder/'plan.npz',actions=np.zeros((2,6)),timestamps=[0,.1])
        return SimpleNamespace(returncode=0)
    (reference/'selection.json').write_text('{}')
    monkeypatch.setattr(server,'load_selection',lambda *a: ({'revision':None},[]))
    monkeypatch.setattr(server,'home_scene',lambda *a: None)
    monkeypatch.setattr(server,'verify_home_scene',lambda *a: None)
    monkeypatch.setattr(server,'verify_selected_scene',lambda *a: None)
    monkeypatch.setattr(server,'WORK',tmp_path);monkeypatch.setattr(server,'current',None);monkeypatch.setattr(server,'jobs',{});monkeypatch.setattr(server,'stop_generation',0)
    monkeypatch.setattr(server,'hardware',SimpleNamespace(call=hardware));monkeypatch.setattr(server.subprocess,'run',simulation)
    headers={'X-Studio-Token':server.TOKEN,'X-Studio-Client':'owner'}
    result=[];thread=threading.Thread(target=lambda:result.append(client.post('/api/robot/home',json={},headers=headers)));thread.start();assert entered.wait(2)
    client.post('/api/robot/stop',json={},headers=headers);release.set();thread.join(3)
    deadline=time.monotonic()+2
    while time.monotonic()<deadline and server.jobs.get(server.current,{}).get('status')!='error':time.sleep(.01)
    assert not sent and 'Stop requested' in server.jobs[server.current]['error']


def test_opening_history_does_not_activate_archived_run(tmp_path,monkeypatch):
    import json
    identifier='f'*32;folder=tmp_path/identifier;folder.mkdir()
    (folder/'capture.json').write_text(json.dumps({'id':identifier,'source':'live','captured_at':1}))
    monkeypatch.setattr(server,'WORK',tmp_path);monkeypatch.setattr(server,'current','a'*32);monkeypatch.setattr(server,'jobs',{})
    class NoHardware:
        def call(self,*a,**k):raise AssertionError('History must not access robot')
    monkeypatch.setattr(server,'hardware',NoHardware())
    rows=client.get('/api/runs').json()
    assert rows['runs']==[] and rows['active_capture']=='a'*32
    assert client.get('/api/plans/'+identifier).json()['id']==identifier
    assert server.current=='a'*32


def test_simulation_appearance_is_validated_and_rendered_separately(monkeypatch):
    headers={'X-Studio-Token':server.TOKEN}
    assert client.get('/api/episodes/0/video?appearance=neural').status_code==409
    assert client.get('/api/episodes/0/frame/0?appearance=neural').status_code==409
    calls=[]
    class Done:
        returncode=0;stderr=''
    def run(command,**_):
        calls.append(command);target=server.WORK/'replay-03'/'comparison-realistic.mp4'
        target.parent.mkdir(exist_ok=True);target.write_bytes(b'video');return Done()
    class Inline:
        def submit(self,function):function()
    monkeypatch.setattr(server.subprocess,'run',run);monkeypatch.setattr(server,'video_pool',Inline())
    monkeypatch.setattr(server,'video_jobs',{})
    target=server.WORK/'replay-03'/'comparison-realistic.mp4';existed=target.exists()
    try:
        if not existed:
            result=client.post('/api/episodes/3/video?appearance=realistic',json={},headers=headers).json()
            assert calls and calls[0][1].endswith('studio_episode_appearance.py') and calls[0][2:]==['video','3','realistic']
            assert server.video_jobs[(3,'realistic')]['video'].endswith('replay-03/comparison-realistic.mp4')
        assert client.get('/api/episodes/3/video?appearance=realistic').json()['status']=='ready'
        assert not client.get('/api/episodes/3/video').json().get('video','').endswith('-realistic.mp4')
    finally:
        if not existed:target.unlink(missing_ok=True)
