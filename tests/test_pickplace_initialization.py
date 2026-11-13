"""Initial scene physics is separate from evaluating the manipulation."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def test_tape_registry_and_grasp_follow_measured_geometry_without_restart(tmp_path, monkeypatch):
    import tape_object
    from pickplace import load_registry
    measurements = json.loads((tape_object.MODEL_DIR / 'tape.json').read_text())
    (tmp_path / 'tape.json').write_text(json.dumps(measurements))
    monkeypatch.setattr(tape_object, 'MODEL_DIR', tmp_path)
    first = load_registry()['objects']['tape']
    assert first['shape']['inner_diameter_m'] == pytest.approx(.025)
    assert first['grasps'][0]['width_m'] == pytest.approx(.014)
    assert first['grasps'][0]['centre_m'][0] == pytest.approx(.0195)
    assert first['place_surfaces'][0]['inner_radius_m'] == pytest.approx(.0125)
    tape_object.dimensions()  # populate any process cache before changing the file
    measurements['core_inner_diameter_m'] = .024
    (tmp_path / 'tape.json').write_text(json.dumps(measurements))
    second = load_registry()['objects']['tape']
    assert second['shape']['inner_diameter_m'] == pytest.approx(.024)
    assert second['grasps'][0]['width_m'] == pytest.approx(.0145)
    assert tape_object.grasp_width() == pytest.approx(.0145)


def test_registry_save_cannot_override_measured_tape_geometry(tmp_path):
    from pickplace import load_registry, save_registry
    registry = load_registry()
    registry['objects']['tape']['grasps'][0]['width_m'] = .030
    with pytest.raises(ValueError, match='measured tape'):
        save_registry(registry, tmp_path / 'registry.json')
    assert not (tmp_path / 'registry.json').exists()


@pytest.fixture
def captured_scene(tmp_path):
    """Geometry of the failed tape-to-NUC capture, independent of archived output."""
    objects = [
        dict(id='p', kind='pen', position=[.290043, -.008034, 0.], yaw=-.69265, support='u'),
        dict(id='b', kind='bowl', position=[.259057, -.140548, 0.], yaw=0., support='mat'),
        dict(id='u', kind='cup', position=[.282387, -.001682, 0.], yaw=0., support='mat'),
        dict(id='t', kind='tape', position=[.120148, -.140689, 0.], yaw=0., support='mat'),
        dict(id='n', kind='nuc', position=[.250746, .123260, 0.], yaw=1.54132, support='mat'),
    ]
    capture = dict(source='live', task='pickplace', initial_native=[-6.373626, -97.626374, 96.483516, 49.230769, -1.186813, 1.760325])
    (tmp_path / 'capture.json').write_text(json.dumps(capture))
    (tmp_path / 'pickplace-scene.json').write_text(json.dumps(dict(objects=objects, pick='t', place={'object': 'n'})))
    return tmp_path


def hold_scene(model, arm, seconds=2.):
    import mujoco
    data = mujoco.MjData(model)
    data.qpos[:6] = arm
    data.ctrl[:] = arm
    for i, count in enumerate(model.actuator_history[:, 0]):
        if count:
            mujoco.mj_initCtrlHistory(model, data, i, None, np.full((int(count), 1), arm[i]))
    mujoco.mj_forward(model, data)
    address = model.joint('object_0_free').qposadr[0]
    start = data.qpos[address:address + 3].copy()
    peak = 0.
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        peak = max(peak, np.linalg.norm(data.qpos[address:address + 3] - start))
    return peak


def test_inferred_pen_is_stable_before_any_manipulation(captured_scene):
    import studio_planner as planner
    from dataset_to_sim.strict_replay import NativeJointAdapter
    capture, config, scene, cal, model = planner.load_scene(captured_scene)
    arm = NativeJointAdapter(**config['adapter']).to_model(capture['initial_native'])
    assert hold_scene(model, arm) < .001
    diagnostic = scene['initialization']
    assert diagnostic['status'] == 'stable'
    assert set(diagnostic['accepted_poses']) == {'p'}
    assert .008 < diagnostic['objects']['p']['max_displacement_m'] < .018
    assert (captured_scene / 'initial-state.json').exists()
    # The measured tape and destination are not silently relocated by initialization.
    tape = next(o for o in scene['objects'] if o['id'] == 't')
    assert tape['position'][:2] == pytest.approx([.120148, -.140689])
    again = planner.load_scene(captured_scene)
    np.testing.assert_array_equal(again[-1].qpos0, model.qpos0)


def test_physics_cache_changes_with_initial_state_and_solver():
    import mujoco
    from studio_planner import _model_key
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".01"/></body></worldbody></mujoco>')
    report = dict(action_hash='actions', timestamp_hash='time', initial_native=[0] * 6)
    key = _model_key(model, report)
    model.qpos0[0] += .01
    changed = _model_key(model, report)
    assert changed != key
    model.opt.gravity[2] = -4.
    assert _model_key(model, report) != changed


def test_measured_object_drift_is_rejected_without_relocating_it(tmp_path):
    import mujoco
    from pickplace import load_registry
    from pickplace_initialization import initialize_scene
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1" contype="4"/><body name="a" pos="0 0 .1"><freejoint/><geom type="box" size=".01 .01 .01"/></body></worldbody></mujoco>')
    scene = dict(objects=[dict(id='a', kind='cube', body='a', position=[0., 0., .1], support='mat')])
    before = copy.deepcopy(scene)
    qpos0 = model.qpos0.copy()
    with pytest.raises(ValueError, match='drifts'):
        initialize_scene(model, scene, load_registry(), np.array([]), tmp_path / 'initial.json')
    assert scene == before
    np.testing.assert_array_equal(model.qpos0, qpos0)
    assert json.loads((tmp_path / 'initial.json').read_text())['status'] == 'rejected'


def test_unsettled_scene_times_out(tmp_path, monkeypatch):
    import mujoco
    import pickplace_initialization as initialization
    from pickplace import load_registry
    monkeypatch.setattr(initialization, 'MAX_SECONDS', .01)
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1" contype="4"/><body name="a" pos="0 0 .01"><freejoint/><geom type="box" size=".01 .01 .01"/></body></worldbody></mujoco>')
    scene = dict(objects=[dict(id='a', kind='cube', body='a', support='mat')])
    with pytest.raises(ValueError, match='do not settle'):
        initialization.initialize_scene(model, scene, load_registry(), np.array([]), tmp_path / 'initial.json')


def test_initial_robot_contact_is_rejected(tmp_path):
    import mujoco
    from pickplace import load_registry
    from pickplace_initialization import initialize_scene
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="robot" type="box" size=".02 .02 .01" contype="2"/><body name="a" pos="0 0 .0199"><freejoint/><geom name="object" type="box" size=".01 .01 .01"/></body></worldbody></mujoco>')
    scene = dict(objects=[dict(id='a', kind='cube', body='a', support='mat')])
    with pytest.raises(ValueError, match='robot contact'):
        initialize_scene(model, scene, load_registry(), np.array([]), tmp_path / 'initial.json')


def test_corrupted_initialization_cache_cannot_move_an_object(captured_scene):
    import studio_planner as planner
    expected = planner.load_scene(captured_scene)[-1].qpos0.copy()
    path = captured_scene / 'initial-state.json'
    record = json.loads(path.read_text())
    record['accepted_poses']['p'][0] += .1
    path.write_text(json.dumps(record))
    actual = planner.load_scene(captured_scene)[-1].qpos0
    np.testing.assert_array_equal(actual, expected)


def test_execution_requires_the_simulated_scene_identity():
    from dataset_to_sim.studio import execution_gate
    plan = dict(task='pickplace', source='live', outcome='success', transfer_checks_passed=True,
                captured_at=100., initial_native=[0] * 6, calibration_hash='calibration')
    with pytest.raises(ValueError, match='scene'):
        execution_gate(plan, [0] * 6, 'calibration', now=110.)
    plan.update(scene_contract_hash='new', simulated_scene_contract_hash='old')
    with pytest.raises(ValueError, match='scene'):
        execution_gate(plan, [0] * 6, 'calibration', now=110.)
    plan['simulated_scene_contract_hash'] = 'new'
    execution_gate(plan, [0] * 6, 'calibration', now=110.)


def test_changed_scene_cannot_reuse_an_approved_plan(captured_scene):
    import studio_planner as planner
    from pickplace_initialization import scene_fingerprint
    loaded = planner.load_scene(captured_scene)
    report = dict(task='pickplace', scene_contract_hash=scene_fingerprint(loaded[-1], loaded[1], loaded[2]))
    planner.validate_plan_scene(captured_scene, report)
    path = captured_scene / 'pickplace-scene.json'
    raw = json.loads(path.read_text())
    next(o for o in raw['objects'] if o['id'] == 't')['position'][0] += .003
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='changed since planning'):
        planner.validate_plan_scene(captured_scene, report)


def test_inferred_object_cannot_settle_on_the_wrong_support(captured_scene):
    import studio_planner as planner
    from pickplace import load_registry
    from pickplace_initialization import initialize_scene
    from dataset_to_sim.strict_replay import NativeJointAdapter
    capture, config, scene, cal, model = planner.load_scene(captured_scene)
    next(o for o in scene['objects'] if o['id'] == 'p')['support'] = 'b'
    arm = NativeJointAdapter(**config['adapter']).to_model(capture['initial_native'])
    with pytest.raises(ValueError, match='declared support'):
        initialize_scene(model, scene, load_registry(), arm)


def test_movement_during_manipulation_still_fails_the_unchanged_verdict(captured_scene):
    """Scorer input includes a pen pushed 5 mm, then returned: peak motion matters."""
    import studio_planner as planner
    from pickplace import load_registry, verdict
    capture, config, scene, cal, model = planner.load_scene(captured_scene)
    poses = {}
    for obj in scene['objects']:
        joint_id = model.body_jntadr[model.body(obj['body']).id]
        if joint_id >= 0:
            address = model.jnt_qposadr[joint_id]
            poses[model.joint(joint_id).name] = np.tile(model.qpos0[address:address + 7], (3, 1))
    cube = poses.pop('cube_free')
    cube[1, 2] += .04
    trace = dict(cube_pose=cube, qpos=np.zeros((3, 6)), object_poses=poses,
                 audit=dict(final_finger_contact=False, final_cube_linear_speed_m_s=0., final_cube_angular_speed_rad_s=0.,
                            grasp_support=dict(longest_secure_grasp_s=.2, longest_secure_lift_s=.2)))
    assert verdict(model, trace, scene, cal.world['pickplace'], load_registry(), config)[1]['moved_objects_m'] == {}
    poses['object_0_free'][1, 0] += .005
    outcome, detail = verdict(model, trace, scene, cal.world['pickplace'], load_registry(), config)
    assert outcome == 'other_object_moved'
    assert detail['moved_objects_m']['p'] == pytest.approx(.005)


def test_resting_on_table_does_not_validate_a_declared_object_support(tmp_path):
    import mujoco
    from pickplace import load_registry
    from pickplace_initialization import initialize_scene
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="table" type="plane" size="1 1 .1" contype="4"/>
      <body name="a" pos="0 0 .01"><freejoint/><geom type="box" size=".01 .01 .01"/></body>
      <body name="b" pos=".2 0 .01"><freejoint/><geom type="box" size=".01 .01 .01"/></body>
      </worldbody></mujoco>''')
    scene = dict(objects=[dict(id='a', kind='cube', body='a', support='b'),
                          dict(id='b', kind='cube', body='b', support='mat')])
    with pytest.raises(ValueError, match='support'):
        initialize_scene(model, scene, load_registry(), np.array([]), tmp_path / 'initial.json')


@pytest.mark.parametrize('yaw', [-1.5, 0., 1.5])
def test_initialization_handles_other_pen_headings(captured_scene, yaw):
    import studio_planner as planner
    from dataset_to_sim.strict_replay import NativeJointAdapter
    path = captured_scene / 'pickplace-scene.json'
    raw = json.loads(path.read_text())
    next(o for o in raw['objects'] if o['id'] == 'p')['yaw'] = yaw
    path.write_text(json.dumps(raw))
    capture, config, scene, cal, model = planner.load_scene(captured_scene)
    arm = NativeJointAdapter(**config['adapter']).to_model(capture['initial_native'])
    assert hold_scene(model, arm) < .002
    assert scene['initialization']['free_scene_validation']['status'] == 'stable'
