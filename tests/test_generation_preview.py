import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_generation_preview import PreviewManager


def options():
    return {'tasks':['red_cube_to_green_cube'],'seed':30,'background_id':'calibrated-real'}


def test_preview_snapshots_settings_and_detects_failed_worker(tmp_path,monkeypatch):
    import studio_generation_preview as module
    process=SimpleNamespace(poll=lambda:None,pid=99)
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**kw:process)
    manager=PreviewManager(tmp_path)
    request=options();job=manager.start(request);request['seed']=123
    assert job['options']['seed']==30
    assert Path(job['background_asset']['path']).is_file()
    assert job['settings_hash'] and job['images']==[]
    process.poll=lambda:1
    result=manager.get(job['id']);assert result['status']=='failed' and result['error']
    with pytest.raises(ValueError):manager.get('../outside')


def test_recovery_and_cancel_do_not_leave_a_completed_preview_cancelled(tmp_path,monkeypatch):
    import studio_generation_preview as module
    process=SimpleNamespace(poll=lambda:None,pid=99)
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**kw:process)
    manager=PreviewManager(tmp_path);job=manager.start(options());folder=manager.folder(job['id'])
    completed=job|{'status':'completed','images':[{'index':0}]}
    (folder/'preview.json').write_text(json.dumps(completed))
    stopped=[]
    monkeypatch.setattr(module,'terminate_group',lambda p:stopped.append(p.pid))
    manager.close();assert stopped==[99]
    assert manager.get(job['id'])['status']=='completed'
    running=job|{'status':'running'};(folder/'preview.json').write_text(json.dumps(running))
    manager.recover_interrupted();assert manager.get(job['id'])['status']=='cancelled'


def test_preview_and_upload_api_auth_validation_and_media(tmp_path,monkeypatch):
    import studio_server as server
    import studio_generation_assets as assets
    from studio_generation import GenerationManager
    from fastapi.testclient import TestClient
    from PIL import Image
    import io
    monkeypatch.setattr(assets,'ASSET_ROOT',tmp_path/'assets')
    monkeypatch.setattr(server,'generation',GenerationManager(tmp_path/'jobs'))
    preview=PreviewManager(tmp_path/'previews');monkeypatch.setattr(server,'generation_preview',preview)
    monkeypatch.setattr(server.hardware,'call',lambda *a,**kw:{'running':False})
    client=TestClient(server.app);headers={'x-studio-token':server.TOKEN}
    assert client.post('/api/generation-preview',json={}).status_code==403
    assert client.post('/api/generation-assets/upload',content=b'bad').status_code==403
    assert client.post('/api/generation-assets/upload',content=b'bad',headers=headers).status_code==409
    buffer=io.BytesIO();Image.new('RGB',(100,100),'gray').save(buffer,format='PNG')
    uploaded=client.post('/api/generation-assets/upload',content=buffer.getvalue(),headers=headers|{'x-filename':'My%20image.png'}).json()['background']
    assert uploaded['name']=='My image.png'
    assert client.get(uploaded['url']).headers['content-type']=='image/png'
    assert len(client.get('/api/generation-assets').json()['backgrounds'])==2
    assert client.post('/api/generation-preview',json={'_background_asset':{'path':'/etc/passwd'}},headers=headers).status_code==409
    assert client.post('/api/generation-preview',json={'light_intensity':3.},headers=headers).status_code==409
    monkeypatch.setattr(preview,'start',lambda opts:{'id':'a'*32,'status':'queued','options':opts})
    result=client.post('/api/generation-preview',json={'background_id':uploaded['id'],'real_background':True},headers=headers)
    assert result.status_code==200 and result.json()['options']['background_id']==uploaded['id']
    assert client.get('/api/generation-preview/'+'a'*32+'/images/scene-0/plan.json').status_code==409
    stopped=[]
    monkeypatch.setattr(preview,'close',lambda:stopped.append(True))
    monkeypatch.setattr(server.generation,'start',lambda body:{'status':'queued'})
    assert client.post('/api/generation',json={},headers=headers).status_code==200
    assert stopped==[True]


def test_latest_preview_replaces_obsolete_render(tmp_path,monkeypatch):
    import studio_generation_preview as module
    created=[]
    def spawn(*args,**kwargs):
        p=SimpleNamespace(poll=lambda:None,pid=99+len(created));created.append(p);return p
    def stop(p):p.poll=lambda:-15
    monkeypatch.setattr(module.subprocess,'Popen',spawn)
    monkeypatch.setattr(module,'terminate_group',stop)
    manager=PreviewManager(tmp_path)
    first=manager.start(options());second=manager.start(options()|{'seed':31})
    assert first['id']!=second['id']
    assert manager.get(first['id'])['status']=='cancelled'
    assert manager.get(second['id'])['status']=='queued'
    manager.close()
