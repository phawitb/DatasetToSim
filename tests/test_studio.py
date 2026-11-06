import numpy as np
import pytest
from dataset_to_sim.studio import validate_trajectory, execution_gate, safe_file


def test_trajectory_rejects_nonfinite_gripper_speed_and_time():
    q = np.zeros((3, 6)); t = np.array([0., .1, .2]); limits = np.array([[-180,180]]*5+[[0,100]])
    validate_trajectory(q, t, limits)
    for bad in [np.nan, -1, 101]:
        b=q.copy(); b[1,5]=bad
        with pytest.raises(ValueError): validate_trajectory(b,t,limits)
    b=q.copy(); b[1,0]=40
    with pytest.raises(ValueError, match='speed'): validate_trajectory(b,t,limits)
    with pytest.raises(ValueError): validate_trajectory(q,np.array([0,0,.2]),limits)


def test_gate_requires_live_success_recent_capture_and_same_pose():
    plan={'source':'live','outcome':'success','transfer_checks_passed':True,'captured_at':100.,'initial_native':[0]*6,'calibration_hash':'abc','executed':False}
    execution_gate(plan,[0]*6,'abc',now=110)
    for changes in [{'source':'dataset'},{'outcome':'missed_bowl'},{'transfer_checks_passed':False},{'captured_at':-100},{'executed':True},{'calibration_hash':'other'}]:
        with pytest.raises(ValueError): execution_gate(plan|changes,[0]*6,'abc',now=110)
    with pytest.raises(ValueError): execution_gate(plan,[10,0,0,0,0,0],'abc',now=110)


def test_media_path_blocks_traversal_and_symlink(tmp_path):
    root=tmp_path/'media'; root.mkdir(); (root/'ok.mp4').write_bytes(b'video')
    secret=tmp_path/'secret'; secret.write_text('no'); (root/'link').symlink_to(secret)
    assert safe_file(root,'ok.mp4') == root/'ok.mp4'
    for value in ['../secret','link','/etc/passwd']:
        with pytest.raises(ValueError): safe_file(root,value)


def test_initial_encoder_overshoot_can_only_move_back_into_range():
    limits=np.array([[-90,90]]*5+[[0,100]])
    q=np.zeros((3,6));q[:,0]=[91,90.5,90]
    validate_trajectory(q,np.array([0.,.1,.2]),limits)
    q[:,0]=[91,91.1,90]
    with pytest.raises(ValueError):validate_trajectory(q,np.array([0.,.1,.2]),limits)
