import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from benchmark72 import detect_scene, read_real, DatasetIndex, DATA
from episode0_physics import Calibration, ROOT, physical_model, clear_table_targets

@pytest.mark.parametrize('episode,expected', [(8, [286, 240]), (54, [257, 262]), (65, [541, 376]), (67, [463, 314])])
def test_red_cube_is_not_robot_screw_or_pink_gripper(episode, expected):
    rgb = read_real(DatasetIndex(DATA).episode(episode), 'top')[0]
    scene = detect_scene(rgb)
    assert np.linalg.norm(np.array(scene['cube_pixel']) - expected) < 5
    assert np.linalg.norm(np.array(scene['bowl_pixel']) - [480, 178]) < 8

def test_clearance_handles_side_grasp_beyond_old_twenty_degree_cap():
    cal = Calibration.load(ROOT / 'calibration/episode0.json')
    ep = DatasetIndex(DATA).episode(33)
    m = physical_model(cal)
    q = cal.joint_convention.to_radians(ep.state_deg[95:96])
    q[:, 5] = np.deg2rad(ep.action_deg[95:96, 5] - 8)
    corrected, delta = clear_table_targets(m, q)
    assert 20 < delta[0] < 35
    assert (corrected >= m.actuator_ctrlrange[:, 0]).all()
    assert (corrected <= m.actuator_ctrlrange[:, 1]).all()

def test_vertical_clearance_preserves_grasp_xy_and_orientation():
    import mujoco
    from episode0_physics import clear_table_targets_vertical
    cal = Calibration.load(ROOT / 'calibration/episode0.json')
    ep = DatasetIndex(DATA).episode(1)
    m = physical_model(cal); d = mujoco.MjData(m)
    q = cal.joint_convention.to_radians(ep.state_deg[58:59])
    q[:, 5] = np.deg2rad(ep.action_deg[58:59, 5] - 8)
    d.qpos[:6] = q[0]; mujoco.mj_forward(m, d)
    before = d.body('gripper').xpos.copy(); rotation = d.body('gripper').xmat.copy()
    corrected, _ = clear_table_targets_vertical(m, q)
    d.qpos[:6] = corrected[0]; mujoco.mj_forward(m, d)
    assert np.linalg.norm(d.body('gripper').xpos[:2] - before[:2]) < .0005
    assert d.body('gripper').xpos[2] > before[2] + .004
    assert np.linalg.norm(d.body('gripper').xmat - rotation) < .005
    contacts = [c for c in d.contact if {int(m.geom_contype[c.geom1]), int(m.geom_contype[c.geom2])} == {2, 4} and {m.geom(c.geom1).name, m.geom(c.geom2).name} & {'table', 'work_mat', 'floor'} and c.dist < 0]
    assert not contacts

def test_gripper_opens_before_approaching_cube():
    from episode0_physics import simulate
    import copy
    cal = Calibration.load(ROOT / 'calibration/episode0.json')
    cal = copy.deepcopy(cal)
    cal.data['world']['cube']['position']=[.2327996232747993,-.03976390645150936,.0162]
    cal.data['physics'] = {'gripper_joint_damping': .1, 'gripper_armature': .003, 'gripper_servo_kv': .1}
    ep = DatasetIndex(DATA).episode(1)
    summary, _, joints, _ = simulate(cal, ep, cal.world['cube']['position'][:2], clearance_mode='vertical')
    assert abs(np.rad2deg(joints[35, 5]) - (ep.state_deg[35, 5] - 8)) < 5
    assert summary['max_cube_z'] > .08
    assert summary['bilateral_frames'] > 20

def test_bowl_containment_uses_walls_and_floor_not_centroid_circle():
    import mujoco
    from episode0_physics import bowl_containment
    cal=Calibration.load(ROOT/'calibration/episode0.json')
    m=physical_model(cal);d=mujoco.MjData(m)
    address=m.jnt_qposadr[m.joint('cube_free').id]
    center=np.array(cal.world['bowl']['position'][:2])
    d.qpos[address:address+3]=[* (center+[.030,0]),.031]
    mujoco.mj_forward(m,d)
    assert bowl_containment(m,d)['inside']
    d.qpos[address]=center[0]+.060;mujoco.mj_forward(m,d)
    assert not bowl_containment(m,d)['inside']
    d.qpos[address:address+3]=[*center,.005];mujoco.mj_forward(m,d)
    assert not bowl_containment(m,d)['inside']
    d.qpos[address:address+3]=[*center,.049];mujoco.mj_forward(m,d)
    assert not bowl_containment(m,d)['inside']  # Top corner exceeds physical rim.

@pytest.mark.parametrize('episode', [2, 30, 63])
def test_shared_trajectory_repair_lifts_failed_episode_using_contacts(episode):
    import copy
    from benchmark72 import episode_calibration, cv2, json
    from episode0_physics import simulate
    ep=DatasetIndex(DATA).episode(episode)
    scene=detect_scene(read_real(ep,'top')[0])
    cal=episode_calibration(Calibration.load(ROOT/'calibration/episode0.json'),scene)
    shared=json.loads((ROOT/'calibration/shared72.json').read_text())
    H=np.array(shared['H'])
    cal.data['world']['cube']['position'][:2]=cv2.perspectiveTransform(np.array([[scene['cube_pixel']]],float),H)[0,0].tolist()
    corners=cv2.perspectiveTransform(np.array([scene['cube_pixel_corners']],float),H)[0]
    edge=corners[1]-corners[0];yaw=np.arctan2(edge[1],edge[0])
    cal.data['world']['cube']['quaternion']=[float(np.cos(yaw/2)),0.,0.,float(np.sin(yaw/2))]
    cal.data['physics']={'gripper_joint_damping':.1,'gripper_armature':.003,'gripper_servo_kv':.1}
    raw,*_=simulate(cal,ep,clearance_mode='vertical')
    assert raw['max_cube_z']<.03
    fixed,m,joints,poses=simulate(cal,ep,clearance_mode='vertical',repair=True)
    assert fixed['final_inside_bowl'] and not fixed['final_finger_contact']
    assert fixed['max_cube_z']>.08 and fixed['bilateral_frames']>20
    assert fixed['cube_pose_writes_after_initialization']==0 and m.neq==0
    assert np.isfinite(joints).all() and np.isfinite(poses).all()
    disabled,*_=simulate(cal,ep,clearance_mode='vertical',repair=True,disable_fingers=True)
    assert disabled['max_cube_z']<.025 and not disabled['final_inside_bowl']
