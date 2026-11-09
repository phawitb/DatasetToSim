"""New Run: unstack, automatic SAM roles, obstacles and their checks."""
import json,sys
from pathlib import Path
import cv2,mujoco,numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from strict_action_scene import ROOT
from studio_sam_geometry import scene_from_masks,mat_to_pixel,pixel_to_mat


def config_and_camera():
    from episode0 import Calibration
    cfg=json.loads((ROOT/'calibration/native-action-current.json').read_text())
    return cfg,Calibration(ROOT/'calibration/native-action-current.json',cfg['base_calibration']).camera('top')


def cube_mask(camera,x,y,yaw=0.,support=.003):
    from dataset_to_sim.scene_estimation import project_cube
    mask=np.zeros((480,640),np.uint8)
    cv2.fillConvexPoly(mask,np.round(project_cube(camera,[x,y,yaw],.0195,support)).astype(np.int32),1)
    return mask>0


def test_mat_pixel_projection_round_trips():
    cfg,_=config_and_camera()
    for xy in [[.2,.05],[.3,-.1]]:
        np.testing.assert_allclose(pixel_to_mat(cfg,mat_to_pixel(cfg,xy,.01),.01),xy,atol=1e-6)


def test_unstack_scene_puts_lower_cube_under_top_and_release_on_empty_mat():
    from studio_stack import scene_calibration,build_scene_model
    cfg,camera=config_and_camera()
    top=cube_mask(camera,.20,.06,support=.003+.0195+.0002)
    place=mat_to_pixel(cfg,[.22,-.06],.0195/2)
    scene=scene_from_masks(cfg,top,place,[('cube',cube_mask(camera,.30,.08))],task='unstack')
    cal=scene_calibration(cfg,scene)
    np.testing.assert_allclose(cal.world['cube']['position'][:2],[.20,.06],atol=.003)
    assert cal.world['cube']['position'][2]-cal.world['target_cube']['position'][2]==pytest.approx(.0197,abs=1e-4)
    np.testing.assert_allclose(cal.world['placement']['position'][:2],[.22,-.06],atol=1e-4)
    assert cal.world['bowl_present'] is False
    model=build_scene_model(cal,cfg)
    assert mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'bowl')==-1
    assert model.joint('target_cube_free').type==mujoco.mjtJoint.mjJNT_FREE
    # The obstacle cube is a static body with collision geometry.
    body=model.body('obstacle_0');assert body.weldid==0 and model.geom('obstacle_0_geom').contype>0


def test_unstack_release_must_be_clear_of_objects():
    cfg,camera=config_and_camera()
    top=cube_mask(camera,.20,.06,support=.003+.0195+.0002)
    with pytest.raises(ValueError,match='too close'):
        scene_from_masks(cfg,top,mat_to_pixel(cfg,[.30,.10],.01),[('cube',cube_mask(camera,.30,.08))],task='unstack')
    with pytest.raises(ValueError,match='70 mm'):
        scene_from_masks(cfg,top,mat_to_pixel(cfg,[.22,.06],.01),[],task='unstack')


def test_stack_bowl_presence_follows_selected_obstacles():
    cfg,camera=config_and_camera()
    source,target=cube_mask(camera,.16,.10),cube_mask(camera,.27,.10,.05)
    scene=scene_from_masks(cfg,source,target,[('cube',cube_mask(camera,.30,-.08))],task='stack')
    assert scene['bowl_present'] is False and [o['kind'] for o in scene['obstacles']]==['cube']
    with pytest.raises(ValueError,match='bowl obstacle'):scene_from_masks(cfg,source,target,[],task='stack',bowl_present=True)


def test_clearance_sees_arm_near_obstacle_and_allows_object_starting_beside_one():
    from studio_stack import scene_calibration,build_scene_model,obstacle_clearance,obstacle_geoms
    cfg,camera=config_and_camera()
    bowl=cv2.circle(np.zeros((480,640),np.uint8),(320,140),55,1,-1)>0
    scene=scene_from_masks(cfg,cube_mask(camera,.20,.06),bowl,[('cube',cube_mask(camera,.20,.06+.026))],task='bowl')
    cal=scene_calibration(cfg,scene);model=build_scene_model(cal,cfg);data=mujoco.MjData(model)
    blocked=obstacle_geoms(model,cal.world,'bowl');assert len(blocked)==1
    rest=obstacle_clearance(model,[data.qpos.copy()],blocked,.008)
    # Resting 6.5 mm beside the obstacle is allowed; only moving closer is not.
    assert rest['object_minimum_m']==pytest.approx(.0065,abs=.0015) and rest['cleared']
    closer=data.qpos.copy();address=model.joint('cube_free').qposadr[0];closer[address+1]+=.005
    assert not obstacle_clearance(model,[data.qpos.copy(),closer],blocked,.008)['cleared']


def test_scene_match_rejects_moved_or_missing_obstacles():
    from dataset_to_sim.studio import validate_scene_match
    base={'cube':{'position':[.2,0,.01],'quaternion':[1,0,0,0]},'bowl':{'position':[.3,.1,0]},'bowl_present':True,
          'obstacles':[{'kind':'cube','position':[.25,-.05,.01]}]}
    validate_scene_match(base,json.loads(json.dumps(base)))
    moved=json.loads(json.dumps(base));moved['obstacles'][0]['position'][0]+=.01
    with pytest.raises(ValueError,match='obstacle moved'):validate_scene_match(base,moved)
    missing=json.loads(json.dumps(base));missing['obstacles']=[]
    with pytest.raises(ValueError,match='Obstacles changed'):validate_scene_match(base,missing)


def test_roles_prefer_red_source_and_unstack_uses_stacked_cube():
    from studio_sam_auto import assign_roles
    def item(kind,color=None,score=.9,area=400):
        return {'info':{'kind':kind,'color':color},'result':{'score':score},'mask':np.ones(area,bool)}
    objects=[item('cube','green'),item('bowl',area=9000),item('cube','red',.8),item('tape')]
    assert assign_roles('bowl',objects)[:2]==(2,1)
    assert assign_roles('stack',objects)[:2]==(2,0)
    assert assign_roles('tape',objects)[:2]==(3,1)
    source,target,notes=assign_roles('stack',[item('bowl',area=9000)])
    assert source is None and target is None and len(notes)==2


def test_candidate_points_find_objects_on_dark_mat_but_not_the_arm():
    from studio_sam_auto import candidate_points
    rgb=np.full((480,640,3),60,np.uint8)
    cv2.rectangle(rgb,(0,0),(639,40),(230,230,230),-1)            # desk beyond the mat
    cv2.rectangle(rgb,(300,200),(320,220),(200,30,30),-1)          # red cube
    cv2.circle(rgb,(450,120),50,(200,220,210),-1)                  # bowl
    cv2.rectangle(rgb,(150,300),(260,420),(240,140,40),-1)         # orange arm
    points=candidate_points(rgb)
    assert any(abs(x-310)<12 and abs(y-210)<12 for x,y in points)
    assert any(np.hypot(x-450,y-120)<50 for x,y in points)
    assert not any(140<=x<=270 and 290<=y<=430 for x,y in points)


def test_capture_accepts_unstack_and_leaves_bowl_to_the_selection(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    import studio_server as server
    monkeypatch.setattr(server,'WORK',tmp_path);monkeypatch.setattr(server,'current',None);monkeypatch.setattr(server,'jobs',{})
    monkeypatch.setattr(server,'hardware',SimpleNamespace(call=lambda command,**kw: {'running':False} if command=='status' else {'source':'live','images':{},'initial_native':[0]*6}))
    client=TestClient(server.app);headers={'X-Studio-Token':server.TOKEN}
    r=client.post('/api/capture',json={'source':'live','task':'unstack'},headers=headers)
    assert r.status_code==200
    saved=json.loads((tmp_path/r.json()['capture']['id']/'capture.json').read_text())
    assert saved['task']=='unstack' and saved['bowl_present'] is None
    assert client.post('/api/capture',json={'task':'unstack','source':'dataset'},headers=headers).status_code==409


def test_new_run_page_defaults_to_full_speed_and_offers_run_all_and_pick_place():
    html=(ROOT/'web/index.html').read_text()
    assert 'id="motion-speed" type="range" min="50" max="150" step="5" value="150"' in html
    assert 'id="run-all"' in html and '<option value="pickplace" selected>Pick &amp; place</option>' in html
    js=(ROOT/'web/app.js').read_text()
    # Run all goes through the same endpoints, so every server check still applies.
    body=js[js.index('async function planUntilPass('):js.index("$('#run-all').onclick")]
    run_all=js[js.index('async function runAll('):js.index("$('#run-all').onclick")]
    assert run_all.index('returnHome(')<run_all.index('doCapture()')  # Home before every new capture
    assert "/refine" in body  # the Wrist check runs between the Look stop and the grasp
    assert "operation('execute')" in body and "result.transfer_checks_passed" in body and '/api/plans' not in body
