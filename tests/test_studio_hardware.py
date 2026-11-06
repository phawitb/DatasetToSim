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


def test_repeated_stop_does_not_ratchet_gravity_sag():
    h=Hardware();h.robot=Robot();h.connected=True
    h.stop();h.robot.get_observation=lambda:{f'{k}.pos':2. for k in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']}
    h.stop()
    assert len(h.robot.commands)==1 and h.stop_generation==2


def test_capture_rejects_drift_over_camera_age(monkeypatch):
    h=Hardware();h.robot=Robot();h.connected=True
    values=iter([np.zeros(6),np.ones(6)*2])
    h.positions=lambda:next(values)
    monkeypatch.setattr('studio_hardware.time.sleep',lambda _:None)
    with pytest.raises(ValueError,match='stable'):h.snapshot()


def test_already_uncapped_connection_does_not_rewrite_held_goal():
    from studio_hardware import prepare_follower_speed
    from types import SimpleNamespace
    calls=[];bus=SimpleNamespace(read=lambda *a,**k:0)
    prepare_follower_speed(bus,lambda bus:calls.append('reset'))
    assert not calls
    bus.read=lambda name,*a,**k:1 if name=='Goal_Velocity' else 0
    prepare_follower_speed(bus,lambda bus:calls.append('reset'))
    assert calls==['reset']


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
    plan={'id':'x','source':'live','outcome':'success','transfer_checks_passed':True,'captured_at':time.time(),'verified_image_captured_at':time.time(),'initial_native':[0]*6,'calibration_hash':'abc','executed':False,'stop_generation':0,'owner':'owner'}
    with pytest.raises(ValueError,match='camera'):h.execute(plan)
    h.frames={v:(time.monotonic(),np.zeros((1,1,3),dtype=np.uint8)) for v in ['top','wrist']}
    h.stop()
    with pytest.raises(ValueError,match='stop'):h.execute(plan)


def test_camera_drops_mid_execution_and_prevents_remaining_targets(tmp_path,monkeypatch):
    import hashlib,studio_hardware
    h=Hardware();h.robot=Robot();h.connected=True;h.calibration_hash=lambda:'abc'
    monkeypatch.setattr(studio_hardware,'ROOT',tmp_path)
    h.frames={v:(time.monotonic(),np.zeros((1,1,3),dtype=np.uint8)) for v in ['top','wrist']}
    original=h.robot.send_action
    def send(action):
        original(action)
        h.frames.clear()
    h.robot.send_action=send
    actions=np.zeros((3,6));actions[:,0]=[0,1,2]
    digest=hashlib.sha256(np.ascontiguousarray(actions,dtype='<f8').tobytes()).hexdigest()
    plan={'id':'test','source':'live','outcome':'success','transfer_checks_passed':True,'captured_at':time.time(),'verified_image_captured_at':time.time(),
          'initial_native':[0]*6,'calibration_hash':'abc','executed':False,'stop_generation':0,'owner':'owner',
          'actions':actions.tolist(),'timestamps':[0,.1,.2],'action_hash':digest,'simulated_action_hash':digest}
    h.execute(plan)
    deadline=time.monotonic()+2
    while h.running and time.monotonic()<deadline:time.sleep(.01)
    assert not h.running and 'camera' in h.error
    assert all(c['shoulder_pan.pos']==0 for c in h.robot.commands)


def test_recorder_finalization_failure_still_releases_running_state(tmp_path,monkeypatch):
    import hashlib,studio_hardware
    h=Hardware();h.robot=Robot();h.connected=True;h.calibration_hash=lambda:'abc'
    monkeypatch.setattr(studio_hardware,'ROOT',tmp_path)
    h.frames={v:(time.monotonic(),np.zeros((1,1,3),dtype=np.uint8)) for v in ['top','wrist']}
    class BrokenRecording:
        def __init__(self,*a):pass
        def start(self,*a):pass
        def finish(self):raise OSError('disk full')
    monkeypatch.setattr(studio_hardware,'Recording',BrokenRecording)
    actions=np.zeros((2,6));digest=hashlib.sha256(actions.astype('<f8').tobytes()).hexdigest()
    plan={'id':'test','source':'live','outcome':'success','transfer_checks_passed':True,'captured_at':time.time(),'verified_image_captured_at':time.time(),'initial_native':[0]*6,'calibration_hash':'abc','executed':False,'stop_generation':0,'owner':'owner','actions':actions.tolist(),'timestamps':[0,.0333333333],'action_hash':digest,'simulated_action_hash':digest}
    h.execute(plan)
    deadline=time.monotonic()+1
    while h.running and time.monotonic()<deadline:time.sleep(.01)
    assert not h.running and h.stopped.is_set() and 'disk full' in h.error

class Bus:
    def __init__(self):self.calls=[];self.goals={};self.torque=1
    def disable_torque(self):self.calls.append('off');self.torque=0
    def enable_torque(self):self.calls.append('on');self.torque=1
    def sync_read(self,name,**kwargs):return {k:2048 for k in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']}
    def write(self,name,motor,value,**kwargs):self.calls.append((name,motor,value));self.goals[motor]=value
    def read(self,name,motor,**kwargs):return self.goals.get(motor,0) if name=='Goal_Position' else self.torque if name=='Torque_Enable' else 0


def test_torque_off_blocks_motion_and_stop_does_not_reenable():
    h=Hardware();h.robot=Robot();h.robot.bus=Bus();h.connected=True
    h.torque_off()
    assert h.torque_enabled is False and h.stopped.is_set()
    h.stop()
    assert h.robot.bus.calls==['off'] and not h.robot.commands
    with pytest.raises(ValueError,match='Torque'):h.execute({})
    with pytest.raises(ValueError,match='Torque'):h.snapshot()


def test_torque_on_sets_current_position_before_enable():
    h=Hardware();h.robot=Robot();h.robot.bus=Bus();h.connected=True;h.torque_enabled=False
    h.torque_on()
    assert h.robot.bus.calls[0][0]=='Goal_Position'
    assert all(c[2]==2048 for c in h.robot.bus.calls[:6])
    assert h.robot.bus.calls[6]=='on' and h.torque_enabled


def test_torque_changes_are_rejected_during_motion():
    h=Hardware();h.robot=Robot();h.robot.bus=Bus();h.connected=True;h.running=True
    for operation in [h.torque_off,h.torque_on]:
        with pytest.raises(ValueError,match='running'):operation()
    assert not h.robot.bus.calls


def test_failed_torque_hold_verification_never_enables():
    h=Hardware();h.robot=Robot();h.robot.bus=Bus();h.connected=True;h.torque_enabled=False
    original=h.robot.bus.read
    h.robot.bus.read=lambda name,*a,**kw:0 if name=='Goal_Position' else original(name,*a,**kw)
    with pytest.raises(ValueError,match='verify'):h.torque_on()
    assert 'on' not in h.robot.bus.calls and h.torque_enabled is None
    with pytest.raises(ValueError,match='Torque'):h.snapshot()
    with pytest.raises(ValueError,match='Torque'):h.execute({})


def test_partial_torque_failure_is_unknown_and_blocks_motion():
    h=Hardware();h.robot=Robot();h.robot.bus=Bus();h.connected=True
    def broken():raise OSError('partial bus failure')
    h.robot.bus.disable_torque=broken
    with pytest.raises(OSError):h.torque_off()
    assert h.torque_enabled is None
    with pytest.raises(ValueError,match='Torque'):h.execute({})


@pytest.mark.parametrize('torque,expected',[(0,False),(1,True),(2,None)])
def test_connect_preserves_existing_motor_torque(torque,expected):
    from studio_hardware import open_calibrated_connection
    from types import SimpleNamespace
    calls=[]
    bus=SimpleNamespace(connect=lambda:calls.append('serial-only'),read=lambda *a,**k:torque)
    robot=SimpleNamespace(bus=bus,is_calibrated=True,connect=lambda **kw:pytest.fail('Must not reconfigure or toggle torque on connect'))
    assert open_calibrated_connection(robot) is expected
    assert calls==['serial-only']


def test_hardware_rejects_scene_that_expired_while_waiting_for_dispatch():
    import hashlib
    h=Hardware();h.robot=Robot();h.connected=True;h.calibration_hash=lambda:'abc'
    h.frames={v:(time.monotonic(),np.zeros((1,1,3),dtype=np.uint8)) for v in ['top','wrist']}
    actions=np.zeros((2,6));digest=hashlib.sha256(actions.astype('<f8').tobytes()).hexdigest()
    plan={'id':'test','source':'live','outcome':'success','transfer_checks_passed':True,'captured_at':time.time(),'verified_image_captured_at':time.time()-6,'initial_native':[0]*6,'calibration_hash':'abc','executed':False,'stop_generation':0,'owner':'owner','actions':actions.tolist(),'timestamps':[0,.0333333333],'action_hash':digest,'simulated_action_hash':digest}
    with pytest.raises(ValueError,match='expired'):h.execute(plan)
    assert not h.robot.commands and not h.running and not h.consumed
