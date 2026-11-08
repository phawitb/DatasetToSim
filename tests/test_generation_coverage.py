import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from studio_generation_coverage import generation_coverage, task_counts, zone_for, _scene


def write_scene(root, index, world):
    folder = root / 'episodes' / f'{index:05d}'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'generation-scene.json').write_text(json.dumps({'world': world}))


def world():
    return {'mat': {'position': [.23, 0, .002], 'size': [.405, .405, .002]},
            'cube': {'position': [.05, -.18, .02]},
            'target_cube': {'position': [.40, .18, .02]},
            'bowl': {'position': [.23, 0, .03]},
            'placement': {'position': [.23, .18, .02]}}


def test_counts_all_accepted_tasks_even_without_targets_or_current_selection(tmp_path):
    job = {'options': {'tasks': ['green_cube_to_red_cube']}, 'episodes': [
        {'status': 'accepted', 'task_id': 'red_cube_to_green_cube', 'inherited_from_job_id': 'parent'},
        {'status': 'rejected', 'task_id': 'green_cube_to_red_cube'},
        {'status': 'accepted', 'task_id': 'green_cube_to_red_cube'}]}
    assert task_counts(job) == [{'task_id': 'green_cube_to_red_cube', 'accepted': 1, 'target': None},
                                {'task_id': 'red_cube_to_green_cube', 'accepted': 1, 'target': None}]
    for i in (0, 2):
        write_scene(tmp_path, i, world())
    result = generation_coverage(tmp_path, job)
    assert result['accepted'] == 2
    assert [e['episode_index'] for e in result['episodes']] == [0, 2]
    assert result['episodes'][0]['inherited'] is True
    assert result['episodes'][0]['source_xy_mm'] == [50, -180]
    assert result['episodes'][0]['destination_xy_mm'] == [400, 180]
    assert result['tasks'][1]['source_counts'] == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert result['tasks'][1]['destination_counts'] == [0, 0, 0, 0, 0, 0, 0, 0, 1]


def test_destination_semantics_and_missing_geometry(tmp_path):
    tasks = ['unstack', 'red_cube_to_green_bowl', 'red_cube_to_green_cube']
    job = {'episodes': [{'status': 'accepted', 'task_id': t} for t in tasks]}
    for i in (0, 1):
        write_scene(tmp_path, i, world())
    result = generation_coverage(tmp_path, job)
    assert result['episodes'][0]['destination_xy_mm'] == [230, 180]
    assert result['episodes'][1]['destination_xy_mm'] == [230, 0]
    assert result['episodes'][2]['source_xy_mm'] is None
    assert result['tasks'][2]['source_missing'] == 1
    assert result['tasks'][2]['destination_missing'] == 1
    assert sum(result['tasks'][2]['source_counts']) == 0


def test_scene_cache_reuses_small_metadata_and_invalidates_on_change(tmp_path):
    write_scene(tmp_path, 0, world())
    job = {'episodes': [{'status': 'accepted', 'task_id': 'unstack'}]}
    _scene.cache_clear()
    generation_coverage(tmp_path, job)
    with patch.object(Path, 'read_text', side_effect=AssertionError('No repeat file reads')):
        assert generation_coverage(tmp_path, job)['accepted'] == 1
    changed = world()
    changed['cube']['position'][0] = .127
    write_scene(tmp_path, 0, changed)
    assert generation_coverage(tmp_path, job)['episodes'][0]['source_xy_mm'][0] == 127


def test_grid_edges_outliers_and_nonfinite_metadata(tmp_path):
    assert zone_for([3, 3], [[0, 0], [3, 3]]) == 8
    assert zone_for([1, 1], [[0, 0], [3, 3]]) == 4
    assert zone_for([3.1, 3], [[0, 0], [3, 3]]) is None
    invalid = world()
    invalid['cube']['position'][0] = float('nan')
    invalid['target_cube']['position'][0] = 1
    write_scene(tmp_path, 0, invalid)
    result = generation_coverage(tmp_path, {'episodes': [{'status': 'accepted', 'task_id': 'red_cube_to_green_cube'}]})
    assert result['tasks'][0]['source_missing'] == 1
    assert result['tasks'][0]['destination_outside'] == 1
    json.dumps(result, allow_nan=False)


def test_robot_base_is_saved_position_not_assumed_at_mat_origin(tmp_path):
    scene = world()
    scene['robot_base'] = {'position': [-.043, -.001, .0034], 'quaternion': [1, 0, 0, 0]}
    write_scene(tmp_path, 0, scene)
    result = generation_coverage(tmp_path, {'episodes': [{'status': 'accepted', 'task_id': 'unstack'}]})
    row = result['episodes'][0]
    assert row['robot_base_xy_mm'] == [-43, -1]
    assert row['robot_base_yaw_deg'] == 0
    assert result['mat_bounds_xy_mm'][0][0] > row['robot_base_xy_mm'][0]
    scene.pop('robot_base')
    write_scene(tmp_path, 0, scene)
    assert generation_coverage(tmp_path, {'episodes': [{'status': 'accepted'}]})['episodes'][0]['robot_base_xy_mm'] is None
