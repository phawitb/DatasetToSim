import sys,time
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_hardware import Hardware

class Robot:
    def __init__(self): self.commands=[]
    def get_observation(self): return {f'{k}.pos':0. for k in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']}
    def send_action(self,a): self.commands.append(a)


def test_stop_latches_before_any_future_command():
    h=Hardware();h.robot=Robot();h.connected=True
    h.stop()
    assert h.stopped.is_set()
    with pytest.raises(RuntimeError): h.send(np.ones(6))
    assert len(h.robot.commands)==1  # measured-position hold only


def test_second_execution_is_rejected_and_rehearsal_cannot_move():
    h=Hardware();h.robot=Robot();h.connected=True;h.running=True
    with pytest.raises(ValueError,match='running'): h.execute({})
    h.running=False
    with pytest.raises(ValueError): h.execute({'source':'dataset'})
    assert not h.robot.commands


def test_unrelated_status_poll_does_not_keep_execution_alive():
    h=Hardware();h.running=True;h.execution_owner='owner';h.last_heartbeat=0
    h.status('other')
    assert h.last_heartbeat==0
    h.status('owner')
    assert h.last_heartbeat>0


def test_execution_requires_fresh_camera_and_matching_stop_generation():
    h=Hardware();h.robot=Robot();h.connected=True
    h.calibration_hash=lambda:'abc'
    plan={'id':'x','source':'live','outcome':'success','transfer_checks_passed':True,'captured_at':time.time(),'initial_native':[0]*6,'calibration_hash':'abc','executed':False,'stop_generation':0,'owner':'owner'}
    with pytest.raises(ValueError,match='camera'):h.execute(plan)
    h.frames={v:(time.monotonic(),np.zeros((1,1,3))) for v in ['top','wrist']}
    h.stop()
    with pytest.raises(ValueError,match='stop'):h.execute(plan)


def test_camera_drops_mid_execution_and_prevents_remaining_targets(tmp_path,monkeypatch):
    import hashlib,studio_hardware
    h=Hardware();h.robot=Robot();h.connected=True;h.calibration_hash=lambda:'abc'
    monkeypatch.setattr(studio_hardware,'ROOT',tmp_path)
    h.frames={v:(time.monotonic(),np.zeros((1,1,3))) for v in ['top','wrist']}
    original=h.robot.send_action
    def send(action):
        original(action)
        h.frames.clear()
    h.robot.send_action=send
    actions=np.zeros((3,6));actions[:,0]=[0,1,2]
    digest=hashlib.sha256(np.ascontiguousarray(actions,dtype='<f8').tobytes()).hexdigest()
    plan={'id':'test','source':'live','outcome':'success','transfer_checks_passed':True,'captured_at':time.time(),
          'initial_native':[0]*6,'calibration_hash':'abc','executed':False,'stop_generation':0,'owner':'owner',
          'actions':actions.tolist(),'timestamps':[0,.1,.2],'action_hash':digest,'simulated_action_hash':digest}
    h.execute(plan)
    deadline=time.monotonic()+2
    while h.running and time.monotonic()<deadline:time.sleep(.01)
    assert not h.running and 'camera' in h.error
    assert all(c['shoulder_pan.pos']==0 for c in h.robot.commands)
