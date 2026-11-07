import sys,json
from pathlib import Path
from types import SimpleNamespace
from fastapi.testclient import TestClient
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import studio_server as server
client=TestClient(server.app)
headers={'X-Studio-Token':server.TOKEN}


def test_dataset_catalog_requires_valid_selection():
    data=client.get('/api/datasets').json()['datasets']
    assert data[0]['id'] and data[0]['episode_count']==72
    result=client.get('/api/datasets/'+data[0]['id']).json()
    assert len(result['episodes'])==72
    assert client.get('/api/datasets/missing').status_code==409


def test_new_session_clears_stale_capture_without_robot_commands(monkeypatch):
    calls=[]
    monkeypatch.setattr(server,'current','old');monkeypatch.setattr(server,'jobs',{'old':{'status':'error'}})
    monkeypatch.setattr(server,'hardware',SimpleNamespace(call=lambda c,**k:calls.append(c) or {'running':False}))
    assert client.post('/api/session/new',json={},headers=headers).status_code==200
    assert server.current is None and calls==['status']


def test_new_session_cannot_interrupt_active_work(monkeypatch):
    monkeypatch.setattr(server,'current','old');monkeypatch.setattr(server,'jobs',{'old':{'status':'running'}})
    assert client.post('/api/session/new',json={},headers=headers).status_code==409
    assert server.current=='old'


def test_restart_requires_token_and_cancels_pending_dispatch(monkeypatch):
    requested=[]
    monkeypatch.setattr(server,'restarting',False)
    monkeypatch.setattr(server,'schedule_restart',lambda:requested.append(True))
    old=server.stop_generation
    assert client.post('/api/server/restart',json={}).status_code==403
    assert client.post('/api/server/restart',json={},headers=headers).status_code==200
    assert requested==[True] and server.stop_generation>old


def test_already_admitted_requests_are_rejected_after_restart(monkeypatch):
    import pytest
    from starlette.requests import Request
    monkeypatch.setattr(server,'restarting',True)
    monkeypatch.setattr(server,'hardware',SimpleNamespace(call=lambda *a,**k:pytest.fail('No hardware mutation during restart')))
    request=Request({'type':'http','headers':[]})
    with pytest.raises(ValueError,match='restarting'):server.robot_command('torque_on',request)
    with pytest.raises(ValueError,match='restarting'):server.plan_operation('x','execute',request)


def test_pending_torque_command_is_rejected_after_operation_lock(monkeypatch):
    import threading,time
    from starlette.requests import Request
    monkeypatch.setattr(server,'restarting',False)
    monkeypatch.setattr(server,'operation_lock',threading.Lock())
    calls=[];errors=[]
    monkeypatch.setattr(server,'hardware',SimpleNamespace(call=lambda c,**k:calls.append(c) or {}))
    def command():
        try:server.robot_command('torque_on',Request({'type':'http','headers':[]}))
        except ValueError as e:errors.append(str(e))
    server.operation_lock.acquire();t=threading.Thread(target=command);t.start();time.sleep(.03)
    server.restarting=True;server.operation_lock.release();t.join(1)
    assert errors==['Server restarting'] and not calls
