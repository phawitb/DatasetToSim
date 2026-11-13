"""Tape entry geometry and observed-grasp checkpoint, without operating hardware."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_tape_visual_and_collision_share_vertical_bounds():
    from tape_object import add_tape_geometry, dimensions
    root=ET.Element('mujoco');ET.SubElement(root,'asset');body=ET.SubElement(root,'body')
    geom=ET.SubElement(body,'geom',name='cube_geom')
    add_tape_geometry(root,body,geom)
    height=dimensions()['width_m']
    for name in ('tape_roll_visual','tape_core_visual','tape_liner_visual','tape_wall_0_mesh'):
        vertices=np.fromstring(root.find(f"asset/mesh[@name='{name}']").get('vertex'),sep=' ').reshape(-1,3)
        np.testing.assert_allclose([vertices[:,2].min(),vertices[:,2].max()],[-height/2,height/2],atol=1e-6)


def test_fixed_finger_entry_has_three_mm_planar_error_margin():
    from tape_object import FIXED_PAD_RADIUS_M,dimensions
    # Minkowski clearance: all 6 x 16 mm pad corners stay inside the core
    # under any planar translation with norm <= 3 mm.
    corners=np.array([[FIXED_PAD_RADIUS_M+x,y] for x in [-.003,.003] for y in [-.008,.008]])
    clearance=dimensions()['core_inner_diameter_m']/2-np.linalg.norm(corners,axis=1).max()
    assert clearance>=.003


def grip_plan():
    actions=np.zeros((30,6));actions[:5,5]=28;actions[20:,0]=np.arange(10)/10
    expected=actions.copy();expected[5:,5]=9.
    plan={'world':{'pickplace':{'grasp':{'pads':'fixed_in_hole'}}},'grasp_check_index':19}
    return plan,actions,np.arange(30)/30,expected


def test_empty_and_obstructed_grasp_stop_but_matched_width_passes():
    from studio_grasp import grasp_check,verify_grasp
    p,a,t,e=grip_plan();check=grasp_check(p,a,t,e)
    def samples(grip):return [{'index':i,'measured':[0]*5+[grip],'observed_at':i/30} for i in range(20)]
    assert verify_grasp(check,samples(8.5))['passed']
    # Failed real run 4efb2281 closed to 0.81; sim was still open at 8.9.
    for grip in [.81,1.96,20.,float('nan')]:
        with pytest.raises(RuntimeError,match='Grasp'):verify_grasp(check,samples(grip))
    with pytest.raises(RuntimeError,match='Grasp'):verify_grasp(check,samples(8.5)[-1:])
    compressed=[s|{'observed_at':s['observed_at']/4} for s in samples(8.5)]
    with pytest.raises(RuntimeError,match='too short'):verify_grasp(check,compressed)


@pytest.mark.parametrize('change',['missing','moving','short','missing_sim','flat_sim','nan_sim'])
def test_tape_requires_a_valid_simulated_stationary_check(change):
    from studio_grasp import grasp_check
    p,a,t,e=grip_plan()
    if change=='missing':p.pop('grasp_check_index')
    if change=='moving':a[15,0]=1
    if change=='short':p['grasp_check_index']=10
    if change=='missing_sim':e=None
    if change=='flat_sim':e[:,5]=a[:,5]
    if change=='nan_sim':e[19,5]=np.nan
    with pytest.raises(ValueError,match='[Gg]rasp'):grasp_check(p,a,t,e)


def test_other_objects_do_not_inherit_tape_width_assumptions():
    from studio_grasp import grasp_check
    p,a,t,e=grip_plan();p['world']['pickplace']['grasp']['pads']='between'
    assert grasp_check(p,a,t,e) is None


def test_natural_preserves_grasp_check_before_lift():
    from studio_natural import interpolate_natural
    names=['Start','Pregrasp','Approach','Grasp','Grip check','Lift','Transfer','Release','Home']
    poses=np.zeros((len(names),6));poses[:3,5]=28;poses[5:,0]=20
    q,t,info=interpolate_natural([(pose,.4) for pose in poses],names)
    k=info['task_knot_indices']
    assert 'Grip check' in k
    assert t[k['Grip check']]-t[k['Grasp']]>=.4
    np.testing.assert_allclose(q[k['Grasp']:k['Grip check']+1],np.tile(poses[3],(k['Grip check']-k['Grasp']+1,1)))


@pytest.mark.parametrize('contact',[False,True])
def test_execution_checks_grasp_before_dispatching_lift(tmp_path,monkeypatch,contact):
    import time,json
    import studio_hardware as hw
    from dataset_to_sim.strict_replay import array_hash
    h=hw.Hardware();h.connected=True;h.calibration_hash=lambda:'abc'
    monkeypatch.setattr(hw,'ROOT',tmp_path)
    # Camera/recording fakes only; execute(), its gate and its send/stop loop are real.
    monkeypatch.setattr(h,'check_cameras',lambda:None)
    class Recording:
        def __init__(self,*a):pass
        def start(self,*a):pass
        def finish(self):return {'error':None}
    monkeypatch.setattr(hw,'Recording',Recording)
    class Robot:
        def __init__(self):self.q=np.array([0.]*5+[28.]);self.commands=[]
        def get_observation(self):return {f'{k}.pos':v for k,v in zip(hw.JOINTS,self.q)}
        def send_action(self,a):
            self.commands.append(a);self.q=np.array([a[f'{k}.pos'] for k in hw.JOINTS]);self.q[5]=max(self.q[5],8.5 if contact else .81)
    h.robot=Robot()
    actions=np.zeros((45,6));actions[:21,5]=np.linspace(28,0,21);actions[35:,0]=np.arange(1,11)/10
    expected=actions.copy();expected[:,5]=np.maximum(expected[:,5],9.)
    p={'id':'grip-test','task':'pickplace','source':'live','outcome':'success','transfer_checks_passed':True,
       'captured_at':time.time(),'verified_image_captured_at':time.time(),'initial_native':actions[0].tolist(),
       'scene_contract_hash':'same','simulated_scene_contract_hash':'same','calibration_hash':'abc',
       'executed':False,'stop_generation':0,'owner':'test','actions':actions.tolist(),'timestamps':(np.arange(45)/30).tolist(),
       'action_hash':array_hash(actions),'simulated_action_hash':array_hash(actions),'simulated_native':expected.tolist(),
       'world':{'pickplace':{'grasp':{'pads':'fixed_in_hole'}}},'grasp_check_index':34}
    h.execute(p)
    deadline=time.monotonic()+3
    while h.running and time.monotonic()<deadline:time.sleep(.01)
    assert not h.running and h.stopped.is_set()
    trace=json.loads((tmp_path/'output/studio/grip-test/hardware-trace.json').read_text())
    assert trace['completed']==contact and trace['grasp_check']['passed']==contact
    if not contact:
        assert 'without the tape' in h.error
        assert all(c['shoulder_pan.pos']==0 for c in h.robot.commands)
    else:assert h.error is None and h.progress==1


def test_tape_return_reaches_raised_corridor_before_folding_home():
    from studio_natural import interpolate_natural
    names=['Start','Pregrasp','Approach','Grasp','Grip check','Lift','Transfer','Release','Return raised','Return unfolded','Home']
    poses=np.zeros((len(names),6));poses[:3,5]=28;poses[5:8,0]=20;poses[8,1]=30;poses[9,0]=10
    q,t,info=interpolate_natural([(p,.4) for p in poses],names,lift_stop=True,return_stop=True)
    k=info['task_knot_indices']
    assert k['Grip check']<k['Lift']<k['Return raised']<k['Home']
    np.testing.assert_allclose(q[k['Return raised']],poses[8],atol=1e-10)
