"""One pick-and-place algorithm: contact registry, plan numbers and full contact physics."""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pickplace import load_registry, validate_registry, plan_spec, settle, grasp_candidates

CONFIG = json.loads((ROOT / 'calibration/native-action-current.json').read_text())
HOME = json.loads((ROOT / 'calibration/studio-home.json').read_text())


def obj(id, kind, x, y, yaw=0., support='mat', color=None):
    return {'id': id, 'kind': kind, 'position': [x, y, 0.], 'yaw': yaw, 'support': support, 'color': color}


def spec(objects, pick, place):
    registry = load_registry()
    scene = settle({'objects': copy.deepcopy(objects), 'pick': pick, 'place': place}, registry, CONFIG)
    return scene, plan_spec(scene, registry, CONFIG)


def test_registry_has_every_new_run_object_and_rejects_bad_values():
    registry = load_registry()
    assert {'cube', 'tape', 'bowl', 'eraser'} <= set(registry['objects'])
    for entry in registry['objects'].values():
        assert entry['grasps'] and entry['place_surfaces']
    broken = copy.deepcopy(registry); broken['objects']['cube']['grasps'][0]['width_m'] = .2
    with pytest.raises(ValueError, match='width'):
        validate_registry(broken)
    broken = copy.deepcopy(registry); broken['objects']['bowl']['place_surfaces'][0]['type'] = 'side'
    with pytest.raises(ValueError, match='surface type'):
        validate_registry(broken)


def test_new_object_kind_needs_only_a_registry_entry():
    registry = load_registry()
    registry['objects']['block'] = {'label': 'Block', 'shape': {'type': 'box', 'size_m': [.03, .02, .02], 'mass_kg': .01},
                                    'grasps': [{'name': 'Sides', 'centre_m': [0, 0, 0], 'jaw_axis_deg': [90], 'width_m': .02,
                                                'pad_height_m': .01, 'pads': 'between', 'gripper_native': 'from_width'}],
                                    'place_surfaces': [{'name': 'Top', 'type': 'top', 'centre_m': [0, 0, .01], 'half_extents_m': [.015, .01], 'release': 'lower'}]}
    validate_registry(registry)
    scene = settle({'objects': [obj('a', 'block', .2, .05), obj('c', 'cube', .27, -.05)], 'pick': 'c', 'place': {'object': 'a'}}, registry, CONFIG)
    result = plan_spec(scene, registry, CONFIG)
    assert result['place']['bottom_z'] == pytest.approx(scene['objects'][0]['position'][2] + .01)


def test_supports_set_heights_for_stacks_and_bowl_contents():
    scene, _ = spec([obj('b', 'bowl', .24, -.1), obj('c1', 'cube', .24, -.1, support='b'), obj('c2', 'cube', .2, .1),
                     obj('c3', 'cube', .2, .1, support='c2')], 'c3', {'area': [.25, .03]})
    z = {o['id']: o['position'][2] for o in scene['objects']}
    edge = CONFIG['base_calibration']['world']['cube']['size']
    assert z['c3'] - z['c2'] == pytest.approx(edge + .0002, abs=1e-6)
    assert z['c1'] == pytest.approx(z['b'] + .004 + edge / 2, abs=1e-3)


def test_unstack_grasp_keeps_jaw_tips_above_the_lower_cube():
    _, result = spec([obj('c2', 'cube', .24, .1), obj('c1', 'cube', .24, .1, support='c2')], 'c1', {'area': [.22, -.02]})
    lower_top = .003 + CONFIG['base_calibration']['world']['cube']['size'] + .0002
    assert result['grasp']['z_cmd'] >= lower_top + .0325 - 1e-6


def test_small_object_on_a_ring_goes_on_the_wall_not_into_the_hole():
    scene, result = spec([obj('t', 'tape', .2, .1), obj('c', 'cube', .27, -.05)], 'c', {'object': 't'})
    distance = np.linalg.norm(np.asarray(result['place']['xy']) - [.2, .1])
    assert .013 < distance < .0265


def test_bowl_rim_grasp_faces_the_robot_and_lands_the_bowl_centred():
    registry = load_registry()
    bowl = obj('b', 'bowl', .24, -.1); settle({'objects': [bowl]}, registry, CONFIG)
    base = np.asarray(CONFIG['base_calibration']['world']['robot_base']['position'][:2])
    first = grasp_candidates(registry, bowl, base)[0]
    assert np.linalg.norm(np.asarray(first['xyz'][:2]) - base) < np.linalg.norm(np.asarray(bowl['position'][:2]) - base)
    _, result = spec([obj('b', 'bowl', .24, -.1)], 'b', {'area': [.27, .06]})
    offset = np.asarray(result['place']['pad_xy']) - result['place']['xy']
    assert np.linalg.norm(offset) == pytest.approx(np.linalg.norm(np.asarray(first['xyz'][:2]) - [.24, -.1]), abs=1e-6)


def test_invalid_choices_are_explained():
    with pytest.raises(ValueError, match='overlaps'):
        spec([obj('c1', 'cube', .2, .08), obj('c2', 'cube', .25, .08)], 'c1', {'area': [.255, .08]})
    with pytest.raises(ValueError, match='resting on it'):
        spec([obj('c2', 'cube', .24, .1), obj('c1', 'cube', .24, .1, support='c2')], 'c2', {'object': 'c1'})


CASES = {
    'stack_on_turned_cube': dict(objects=[obj('c1', 'cube', .20, .08, color='red'), obj('c2', 'cube', .28, .02, .3, color='green')], pick='c1', place={'object': 'c2'}),
    'cube_from_bowl_onto_tape': dict(objects=[obj('b', 'bowl', .24, -.10), obj('c1', 'cube', .24, -.10, support='b', color='red'), obj('t', 'tape', .20, .10)], pick='c1', place={'object': 't'}),
    'move_bowl_by_rim_with_cube_inside': dict(objects=[obj('b', 'bowl', .24, -.10), obj('c1', 'cube', .24, -.10, support='b', color='red')], pick='b', place={'area': [.27, .06]}),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_pick_and_place_succeeds_in_contact_physics(name, tmp_path):
    import studio_planner as planner
    folder = tmp_path / name; folder.mkdir()
    (folder / 'capture.json').write_text(json.dumps({'source': 'live', 'task': 'pickplace', 'initial_native': HOME['home'],
        'calibration_hash': CONFIG['follower_calibration_sha256'], 'captured_at': time.time(), 'id': name}))
    (folder / 'pickplace-scene.json').write_text(json.dumps(CASES[name]))
    (folder / 'planning-options.json').write_text(json.dumps({'motion_technique': 'natural', 'speed_scale': 1., 'timing_profile': 'recorded'}))
    planner.prepare_plan(folder, preview=False)
    report = planner.simulate(folder)
    assert report['outcome'] == 'success', report.get('pickplace_assessment')
    assert report['transfer_checks_passed'], report['transfer_check_failures']
    assert not report['pickplace_assessment']['moved_objects_m']


def test_workspace_correction_shifts_scene_into_the_arm_frame():
    from pickplace import correction_at, apply_correction
    correction = {'samples': [{'xy': [.1, .1], 'error': [.004, 0.]}, {'xy': [.3, -.1], 'error': [0., -.006]}]}
    np.testing.assert_allclose(correction_at(correction, [.1, .1]), [.004, 0.], atol=6e-4)
    np.testing.assert_allclose(correction_at(correction, [.3, -.1]), [0., -.006], atol=6e-4)
    assert np.linalg.norm(correction_at({'samples': [{'xy': [0, 0], 'error': [.1, 0]}]}, [0, 0])) == pytest.approx(.02)
    scene = {'objects': [obj('c', 'cube', .1, .1), obj('k', 'cube', .3, -.1)], 'pick': 'c', 'place': {'area': [.3, -.1]}}
    apply_correction(scene, correction, keep={'k'})
    assert scene['objects'][0]['position'][0] == pytest.approx(.1 - correction_at(correction, [.1, .1])[0])
    assert scene['objects'][1]['position'][:2] == [.3, -.1]
    assert scene['place']['area'][1] == pytest.approx(-.1 - correction_at(correction, [.3, -.1])[1])
