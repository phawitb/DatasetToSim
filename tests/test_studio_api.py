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
                return {'initial_native':[0]*6,'calibration_hash':'abc','stop_generation':0,'images':{'top':base64.b64encode(cv2.imencode('.jpg',np.zeros((2,2,3),np.uint8))[1]).decode()}}
            if command=='execute':sent.append(kwargs);return {'running':True}
    monkeypatch.setattr(server,'hardware',FakeHardware());monkeypatch.setattr(server,'WORK',tmp_path);monkeypatch.setattr(server,'current',identifier);monkeypatch.setattr(server,'jobs',{})
    monkeypatch.setattr(strict_action_scene,'estimate_scene',lambda *a: {})
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
