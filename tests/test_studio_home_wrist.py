import sys,json,hashlib
from pathlib import Path
import cv2,numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_home_wrist import released_reference,fit_bowl,project_rim


def reference(tmp_path):
    from dataset_to_sim.strict_replay import array_hash
    q=np.array([0.,0,0,0,0,40]);actions=np.vstack([q,q]);digest=array_hash(actions)
    p={'source':'live','executed':True,'outcome':'success','transfer_checks_passed':True,'calibration_hash':'cal','action_hash':digest,'simulated_action_hash':digest}
    t={'completed':True,'error':None,'samples':[{'target':q.tolist(),'measured':q.tolist()},{'target':q.tolist(),'measured':q.tolist()}]}
    (tmp_path/'plan.json').write_text(json.dumps(p));(tmp_path/'hardware-trace.json').write_text(json.dumps(t));np.savez(tmp_path/'plan.npz',actions=actions,timestamps=[0,1/30])
    return q


def test_release_recovery_requires_completed_matching_immutable_actions(tmp_path):
    q=reference(tmp_path);released_reference(tmp_path,q,'cal')
    released_reference(tmp_path,q+[6,3,0,0,0,0],'cal')
    with pytest.raises(ValueError):released_reference(tmp_path,q+[11,0,0,0,0,0],'cal')
    with pytest.raises(ValueError):released_reference(tmp_path,q+[0,0,0,0,0,4],'cal')
    with pytest.raises(ValueError):released_reference(tmp_path,q,'changed')
    t=json.loads((tmp_path/'hardware-trace.json').read_text());t['completed']=False;(tmp_path/'hardware-trace.json').write_text(json.dumps(t))
    with pytest.raises(ValueError):released_reference(tmp_path,q,'cal')


def test_release_pose_bounds_do_not_accumulate(tmp_path):
    q=reference(tmp_path);t=json.loads((tmp_path/'hardware-trace.json').read_text())
    t['samples'][-1]['measured'][0]=8
    (tmp_path/'hardware-trace.json').write_text(json.dumps(t))
    with pytest.raises(ValueError):released_reference(tmp_path,q+[16,0,0,0,0,0],'cal')
    reference(tmp_path);np.savez(tmp_path/'plan.npz',actions=np.zeros((2,6)),timestamps=[0,1/30])
    with pytest.raises(ValueError):released_reference(tmp_path,q,'cal')


def test_wrist_circle_fit_reconstructs_current_bowl_and_rejects_wrong_shape():
    camera={'position':[0,0,.25],'rotation':np.eye(3).tolist(),'fovy':52}
    bowl={'position':[.0,.0,.003],'outer_radius':.059,'height':.05}
    xy=[.008,-.007];poly=project_rim(camera,bowl,xy)
    mask=np.zeros((480,640),np.uint8);cv2.fillConvexPoly(mask,poly.astype(np.int32),1)
    pose,report=fit_bowl(mask,camera,bowl)
    assert np.linalg.norm(np.array(pose)-xy)<.002 and report['iou']>.95
    mask[:]=0;mask[140:340,150:490]=1
    with pytest.raises(ValueError):fit_bowl(mask,camera,bowl)


def test_home_clearance_contains_original_bowl_and_leaves_walls_intact():
    from strict_action_scene import build_model
    from episode0 import Calibration
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'calibration/native-action-current.json').read_text())
    cal=Calibration(root/'calibration/native-action-current.json',cfg['base_calibration'])
    base=build_model(cal,cfg);guard=build_model(cal,cfg|{'home_bowl_clearance_m':.02})
    g=guard.geom('home_bowl_clearance');r=cal.world['bowl']['outer_radius'];h=cal.world['bowl']['height']
    assert g.size[0]>=r+.02-1e-9
    assert g.pos[2]-g.size[1]<=-.02+1e-9 and g.pos[2]+g.size[1]>=h+.02-1e-9
    for i in range(32):
        original=base.geom(f'bowl_contact_{i}');retained=guard.geom(f'bowl_contact_{i}')
        assert np.array_equal(original.pos,retained.pos) and np.array_equal(original.size,retained.size)
    assert g.contype==0 and g.conaffinity==10
