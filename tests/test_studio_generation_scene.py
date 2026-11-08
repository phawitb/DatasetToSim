import copy
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from studio_generation_scene import (COLORS, TASKS, initialize_generation_scene,
                                     load_generation_scene, unstack_verdict)


@pytest.mark.parametrize('task', TASKS)
def test_seeded_scene_is_reproducible_and_has_two_contact_cubes(tmp_path, task):
    paths = [tmp_path / 'a', tmp_path / 'b']
    for folder in paths:
        initialize_generation_scene(folder, task, 481, {'physics_randomization': True})
    assert (paths[0] / 'generation-scene.json').read_bytes() == (paths[1] / 'generation-scene.json').read_bytes()
    assert (paths[0] / 'generation-config.json').read_bytes() == (paths[1] / 'generation-config.json').read_bytes()
    capture, config, scene, cal, model = load_generation_scene(paths[0])
    assert capture['source'] == 'synthetic' and capture['executed'] is False
    assert model.nu == 6
    for name in ('cube', 'target_cube'):
        assert model.joint(name + '_free').type == mujoco.mjtJoint.mjJNT_FREE
        assert model.geom(name + '_geom').contype > 0
        np.testing.assert_allclose(model.geom(name + '_geom').rgba, COLORS[cal.world[name]['color']])
    assert cal.world['cube']['color'] != cal.world['target_cube']['color']
    assert (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bowl') >= 0) == (capture['task'] == 'bowl')
    assert .9 <= scene['physics_draws']['cube_mass_scale'] <= 1.1
    assert all(.9 <= v <= 1.1 for v in scene['physics_draws']['friction_scales'].values())
    if capture['task'] == 'unstack':
        np.testing.assert_array_equal(cal.world['cube']['position'][:2], cal.world['target_cube']['position'][:2])
        assert np.linalg.norm(np.asarray(cal.world['placement']['position'][:2])-cal.world['cube']['position'][:2]) > .08


def test_synthetic_load_never_reads_camera_images(tmp_path, monkeypatch):
    import studio_planner
    initialize_generation_scene(tmp_path, 'red_cube_to_green_bowl', 0, {})
    monkeypatch.setattr(studio_planner.cv2, 'imread', lambda *a: pytest.fail('Synthetic scenes must not read camera images'))
    assert studio_planner.load_scene(tmp_path)[0]['source'] == 'synthetic'


def test_randomization_preserves_controller_and_collision_model(tmp_path):
    for name, enabled in [('fixed', False), ('random', True)]:
        initialize_generation_scene(tmp_path/name, 'red_cube_to_green_cube', 9,
                                    {'physics_randomization': enabled, 'randomize_layout': False})
    a, b = [load_generation_scene(tmp_path/name) for name in ('fixed', 'random')]
    for key in ('physical_control', 'identified_dynamics', 'contact_model', 'adapter', 'base_calibration'):
        assert a[1][key] == b[1][key]
    for attr in ('geom_contype', 'geom_conaffinity', 'geom_size', 'actuator_gainprm', 'actuator_delay'):
        np.testing.assert_array_equal(getattr(a[-1], attr), getattr(b[-1], attr))
    assert not np.array_equal(a[-1].geom_friction, b[-1].geom_friction)
    assert a[-1].body('cube').mass != b[-1].body('cube').mass


def test_unstack_verdict_rejects_unphysical_or_incomplete_success():
    edge = .0195
    destination = [.24, -.01, .01275]
    target = np.tile([.23, .12, .01275, 1, 0, 0, 0], (90, 1))
    source = np.tile([*destination, 1, 0, 0, 0], (90, 1))
    source[0, :3] = target[0, :3] + [0, 0, edge]
    source[1, 2] = .15
    trace = {'cube_pose': source, 'target_pose': target, 'target_velocity': np.zeros((90, 6)),
             'audit': {'final_finger_contact': False, 'final_cube_linear_speed_m_s': 0.,
                       'final_cube_angular_speed_rad_s': 0.,
                       'grasp_support': {'longest_secure_grasp_s': 1., 'longest_secure_lift_s': 1.}}}
    assert unstack_verdict(trace, edge, destination, .003, .04)['outcome'] == 'success'
    for problem in ('bottom_moved', 'held', 'hovering', 'wrong_destination', 'no_grasp', 'moving', 'no_support'):
        changed = copy.deepcopy(trace)
        force = .04
        if problem == 'bottom_moved': changed['target_pose'][20, 0] += .01
        elif problem == 'held': changed['audit']['final_finger_contact'] = True
        elif problem == 'hovering': changed['cube_pose'][-30:, 2] += .005
        elif problem == 'wrong_destination': changed['cube_pose'][-30:, 0] += .03
        elif problem == 'no_grasp': changed['audit']['grasp_support']['longest_secure_lift_s'] = 0.
        elif problem == 'moving': changed['audit']['final_cube_linear_speed_m_s'] = .1
        else: force = 0.
        assert unstack_verdict(changed, edge, destination, .003, force)['outcome'] != 'success', problem


def test_invalid_generation_inputs_fail_before_scene_creation(tmp_path):
    for task, seed, options in [('unknown', 0, {}), ('unstack', -1, {}), ('unstack', 0, {'unstack_top': 'blue'}),
                               ('unstack', 0, {'physics_randomization': 'yes'})]:
        with pytest.raises(ValueError):
            initialize_generation_scene(tmp_path, task, seed, options)
    assert not (tmp_path / 'capture.json').exists()


@pytest.mark.parametrize('task', TASKS)
def test_full_contact_generation_and_saved_trace_without_graphics(tmp_path, monkeypatch, task):
    import studio_planner
    from studio_generation_scene import generate_episode
    monkeypatch.setattr(studio_planner, 'render_preview', lambda *a: pytest.fail('Engine must not render'))
    report = generate_episode(tmp_path, task, 0, {'randomize_layout': False})
    assert report['outcome'] == 'success'
    assert report['transfer_checks_passed']
    assert report['audit']['post_initial_state_writes'] == 0
    assert report['audit']['target_clipping'] is False
    assert report['audit']['grasp_support']['longest_secure_lift_s'] >= .03
    plan = np.load(tmp_path / 'plan.npz')
    trace = np.load(tmp_path / 'trace.npz')
    assert plan['actions'].shape == (report['frames'], 6)
    assert trace['qpos'].shape == (report['frames'] + 1, 6)
    assert trace['cube_pose'].shape == trace['target_pose'].shape == (report['frames'] + 1, 7)
    assert trace['timestamps'][-1] == pytest.approx(report['duration'])
    np.testing.assert_allclose(trace['timestamps'][:-1], plan['timestamps'], atol=1e-10)
    assert (tmp_path / 'scene.xml').is_file()
    if report['task'] in {'stack', 'unstack'}:
        home = json.loads((studio_planner.ROOT / 'calibration/studio-home.json').read_text())['home']
        np.testing.assert_allclose(plan['actions'][-1], home)
        assert 'Home' in report['motion']['task_knot_indices']


def test_scene_contains_only_task_objects_unless_extras_are_added(tmp_path):
    import mujoco
    from studio_generation_scene import initialize_generation_scene, load_generation_scene
    from studio_generation_zones import scene_presence
    assert scene_presence('red_cube_to_green_bowl', {'extra_objects': []}) == {'other_cube': False, 'bowl': True}
    assert scene_presence('red_cube_to_green_bowl', {'extra_objects': ['green_cube']}) == {'other_cube': True, 'bowl': True}
    assert scene_presence('red_cube_to_green_cube', {'extra_objects': []}) == {'other_cube': True, 'bowl': False}
    assert scene_presence('red_cube_to_green_cube', {'extra_objects': ['bowl']}) == {'other_cube': True, 'bowl': True}
    # Saved requests without the key keep both cubes.
    assert scene_presence('red_cube_to_green_bowl', {}) == {'other_cube': True, 'bowl': True}
    initialize_generation_scene(tmp_path / 'a', 'red_cube_to_green_bowl', 0, {'extra_objects': []})
    model = load_generation_scene(tmp_path / 'a')[-1]
    geom = model.geom('target_cube_geom')
    assert geom.contype == 0 and geom.conaffinity == 0 and geom.group == 3
    assert model.body_gravcomp[model.body('target_cube').id] == 1
    initialize_generation_scene(tmp_path / 'b', 'red_cube_to_green_cube', 0, {'extra_objects': ['bowl']})
    model = load_generation_scene(tmp_path / 'b')[-1]
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bowl') >= 0
    assert model.geom('target_cube_geom').group != 3
