import sys
from pathlib import Path
import cv2,numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_sam_geometry import validate_point,mask_record,validate_pair,scene_from_masks


def test_prompt_must_be_finite_inside_original_image():
    assert validate_point([639,479])==[639.,479.]
    for p in [[640,20],[-1,20],[0,float('nan')],[1],[1,2,3]]:
        with pytest.raises(ValueError):validate_point(p)


def test_mask_must_include_clicked_object_and_have_correct_shape():
    m=np.zeros((480,640),np.uint8);m[170:195,440:465]=1
    r=mask_record(m,[450,180],.95)
    assert r['area']==625 and r['bbox']==[440,170,465,195]
    with pytest.raises(ValueError):mask_record(m,[200,100],.95)
    with pytest.raises(ValueError):mask_record(np.zeros((20,20)),[2,2],.95)


def test_from_to_cannot_be_same_object():
    a=np.zeros((480,640),np.uint8);a[100:150,100:150]=1
    with pytest.raises(ValueError,match='different'):validate_pair(a,a)
    b=np.zeros_like(a);b[200:300,300:400]=1;validate_pair(a,b)


def test_selection_binds_masks_to_image_and_detects_tampering(tmp_path):
    import hashlib,json
    from studio_sam_geometry import load_selection
    (tmp_path/'top.jpg').write_bytes(b'captured-image')
    a=np.zeros((480,640),np.uint8);a[100:125,100:125]=255
    b=np.zeros_like(a);b[200:300,300:400]=255
    selection={'revision':'one','image_sha256':hashlib.sha256(b'captured-image').hexdigest()}
    for role,m,p in [('from',a,[110,110]),('to',b,[350,250])]:
        cv2.imwrite(str(tmp_path/f'{role}.png'),m)
        selection[role]=mask_record(m,p,.95)|{'mask_file':f'{role}.png'}
    (tmp_path/'selection.json').write_text(json.dumps(selection))
    assert load_selection(tmp_path)[0]['revision']=='one'
    a[10,10]=255;cv2.imwrite(str(tmp_path/'from.png'),a)
    with pytest.raises(ValueError,match='mask changed'):load_selection(tmp_path)
    (tmp_path/'top.jpg').write_bytes(b'other-image')
    with pytest.raises(ValueError,match='Capture changed'):load_selection(tmp_path)


def test_selecting_new_object_invalidates_simulation_and_cannot_reuse_run(tmp_path,monkeypatch):
    import base64,hashlib,json
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    import studio_server as server
    identifier='f'*32;folder=tmp_path/identifier;folder.mkdir()
    (folder/'capture.json').write_text(json.dumps({'id':identifier,'source':'live'}))
    (folder/'top.jpg').write_bytes(b'capture')
    (folder/'plan.json').write_text(json.dumps({'outcome':'success','executed':False}))
    (folder/'plan.npz').write_bytes(b'old-actions')
    mask=np.zeros((480,640),np.uint8);mask[100:125,100:125]=255
    result=mask_record(mask,[110,110],.95)|{'png':base64.b64encode(cv2.imencode('.png',mask)[1]).decode(),'image_sha256':hashlib.sha256(b'capture').hexdigest(),'model':{'weights_sha256':'model'}}
    monkeypatch.setattr(server,'WORK',tmp_path);monkeypatch.setattr(server,'current',identifier);monkeypatch.setattr(server,'jobs',{})
    monkeypatch.setattr(server,'hardware',SimpleNamespace(call=lambda *a,**k:{'running':False}))
    monkeypatch.setattr(server,'sam',SimpleNamespace(predict=lambda *a:result))
    c=TestClient(server.app);headers={'X-Studio-Token':server.TOKEN}
    response=c.post(f'/api/captures/{identifier}/select',json={'role':'from','point':[110,110]},headers=headers)
    assert response.status_code==200 and response.json()['from']['area']==625
    assert not (folder/'plan.json').exists() and not (folder/'plan.npz').exists()
    assert c.post(f'/api/plans/{identifier}/plan',json={},headers=headers).status_code==409
    (folder/'plan.json').write_text(json.dumps({'executed':True}))
    response=c.post(f'/api/captures/{identifier}/select',json={'role':'to','point':[110,110]},headers=headers)
    assert response.status_code==409 and 'new scene' in response.json()['error']


def test_prompt_component_cannot_redirect_geometry_to_larger_other_object():
    from studio_sam_geometry import prompted_component
    m=np.zeros((480,640),np.uint8);m[100:125,100:125]=1;m[300:310,300:310]=1
    with pytest.raises(ValueError,match='disconnected'):mask_record(m,[305,305],.9)
    selected=prompted_component(m,[305,305])
    assert selected.sum()==100
    assert mask_record(selected,[305,305],.9)['bbox']==[300,300,310,310]


def test_slow_sam_preflight_cannot_authorize_motion():
    from studio_sam_geometry import validate_vision_age
    validate_vision_age(100,101)
    with pytest.raises(ValueError,match='expired'):validate_vision_age(100,105.01)


def test_home_after_transfer_only_prompts_bowl_and_handles_partial_occlusion(tmp_path,monkeypatch):
    import json,time,base64
    from types import SimpleNamespace
    import studio_sam_home as home
    reference=tmp_path/'reference';reference.mkdir();folder=tmp_path/'home';folder.mkdir()
    (reference/'capture.json').write_text(json.dumps({'calibration_hash':'cal','captured_at':time.time()}))
    cube=np.zeros((480,640),np.uint8);cube[100:125,440:465]=1
    bowl=np.zeros_like(cube);cv2.ellipse(bowl,(200,180),(60,52),0,0,360,1,-1)
    model={'weights_sha256':'sam'}
    selection={'revision':'one','model':model,'from':{'point':[450,110]},'to':{'point':[200,180]}}
    monkeypatch.setattr(home,'load_selection',lambda *a:(selection,[cube>0,bowl>0]))
    monkeypatch.setattr(home,'scene_from_masks',lambda *a:{'cube_pixel':[450,110],'bowl_pixel':[200,180]})
    prompts=[]
    def predict(image,point):
        prompts.append(point)
        assert point!=[450,110], 'Moved cube must never be prompted for Home'
        if point==[200,180]:raise ValueError('Arm occludes original bowl click')
        return mask_record(bowl,point,.95)|{'png':base64.b64encode(cv2.imencode('.png',bowl*255)[1]).decode(),'model':model,'image_sha256':'fresh'}
    result=home.home_scene(SimpleNamespace(predict=predict),folder,reference,'fresh.jpg',{'follower_calibration_sha256':'cal'})
    assert len(prompts)>1 and np.linalg.norm(np.array(result['scene']['bowl_pixel'])-[200,180])<1
    with pytest.raises(ValueError,match='clear the camera'):
        home.home_scene(SimpleNamespace(predict=lambda *a:(_ for _ in ()).throw(ValueError('fully hidden'))),folder,reference,'fresh.jpg',{'follower_calibration_sha256':'cal'})
