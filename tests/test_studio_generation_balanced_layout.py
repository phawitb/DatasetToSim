"""Balanced requests constrain initial poses while preserving geometric safety."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from studio_generation_layout import sample_workspace_layout
from studio_generation_scene import ROOT, _robot_base_bounds, initialize_generation_scene


@pytest.fixture(scope='module')
def geometry():
    config = json.loads((ROOT / 'calibration/native-action-current.json').read_text())
    return config['base_calibration']['world'], _robot_base_bounds(json.dumps(config, sort_keys=True))


def test_requested_cells_cover_whole_mat_without_repeating_layouts(geometry):
    world, base = geometry
    layouts = set()
    for source_cell in range(9):
        for seed in range(20):
            descriptor = {'grid_shape': [3, 3], 'source_cell': source_cell,
                          'target_cell': (source_cell + 4) % 9}
            first = sample_workspace_layout(world, np.random.default_rng(seed), base, 'stack', descriptor)
            repeat = sample_workspace_layout(world, np.random.default_rng(seed), base, 'stack', descriptor)
            assert json.dumps(first[3:]) == json.dumps(repeat[3:])
            source, target, _, draws, policy = first
            assert np.linalg.norm(source[0] - target[0]) >= .08
            assert policy['grid_shape'] == [3, 3]
            np.testing.assert_allclose(policy['mat_bounds_xy_m'], [[.0275, -.2025], [.4325, .2025]])
            for name, role in [('source_cube', 'source_cell'), ('other_cube', 'target_cell')]:
                xy = np.asarray(draws[name]['position_xy_m'])
                cell = descriptor[role]
                # Hand-derived cell widths: calibrated 405 mm square / 3.
                low = np.array([.0275 + (cell % 3) * .135, -.2025 + (cell // 3) * .135])
                assert np.all(xy >= low) and np.all(xy <= low + .135)
                assert draws[name]['cell_id'] == cell
                assert np.linalg.norm(xy - np.clip(xy, *base)) >= world['cube']['size'] / np.sqrt(2) + .008
                yaw = draws[name]['yaw_rad']
                extent = world['cube']['size'] / 2 * (abs(np.cos(yaw)) + abs(np.sin(yaw)))
                assert np.all(xy - extent >= [.0295, -.2005])
                assert np.all(xy + extent <= [.4305, .2005])
            layouts.add(tuple(source[0]) + tuple(target[0]))
    assert len(layouts) == 180


def test_same_cell_sampling_retries_pair_instead_of_stranding_target(geometry):
    world, base = geometry
    for seed in range(40):
        source, target, *_ = sample_workspace_layout(world, np.random.default_rng(seed), base, 'stack',
            {'grid_shape': [3, 3], 'source_cell': 0, 'target_cell': 0})
        assert np.linalg.norm(source[0] - target[0]) >= .08


@pytest.mark.parametrize('descriptor', [
    {'grid_shape': [0, 3], 'source_cell': 0, 'target_cell': 1},
    {'grid_shape': [3, 3], 'source_cell': -1, 'target_cell': 1},
    {'grid_shape': [3, 3], 'source_cell': 0, 'target_cell': 9},
    {'grid_shape': [3, 3], 'source_cell': True, 'target_cell': 1},
])
def test_invalid_cell_requests_fail_explicitly(geometry, descriptor):
    world, base = geometry
    with pytest.raises(ValueError, match='cell|grid'):
        sample_workspace_layout(world, np.random.default_rng(0), base, 'stack', descriptor)


def test_scene_persists_requested_cells_and_actual_positions(geometry, tmp_path):
    descriptor = {'grid_shape': [3, 3], 'source_cell': 0, 'target_cell': 8}
    initialize_generation_scene(tmp_path, 'red_cube_to_green_cube', 5, {'_layout_cell': descriptor})
    scene = json.loads((tmp_path / 'generation-scene.json').read_text())
    assert scene['layout_sampling']['source_cell'] == 0
    assert scene['layout_sampling']['target_cell'] == 8
    for draw, body in [('source_cube', 'cube'), ('other_cube', 'target_cube')]:
        assert scene['layout_draws'][draw]['position_xy_m'] == scene['world'][body]['position'][:2]


def test_fixed_scene_cannot_silently_claim_a_balanced_cell(tmp_path):
    with pytest.raises(ValueError, match='randomize_layout'):
        initialize_generation_scene(tmp_path, 'red_cube_to_green_cube', 5,
            {'randomize_layout': False, '_layout_cell': {'source_cell': 0, 'target_cell': 8}})
    assert not (tmp_path / 'generation-scene.json').exists()


def test_impossible_cell_names_failed_role_and_requested_cell(geometry):
    world, _ = geometry
    # A robot base covering the entire requested source cell makes it invalid;
    # it must not silently fall back to a more convenient cell elsewhere.
    with pytest.raises(ValueError, match='source_cube.*requested cell'):
        sample_workspace_layout(world, np.random.default_rng(1), [[0., -.3], [.2, -.05]], 'stack',
            {'grid_shape': [3, 3], 'source_cell': 0, 'target_cell': 8})


def test_kinematic_screen_records_and_resamples_unreachable_candidate(geometry, tmp_path):
    # Seed 10008 in the 36-layout physical probe failed Transfer at 4.64 mm.
    descriptor = {'grid_shape': [3, 3], 'source_cell': 6, 'target_cell': 2}
    initialize_generation_scene(tmp_path, 'red_cube_to_green_cube', 10008, {'_layout_cell': descriptor})
    scene = json.loads((tmp_path / 'generation-scene.json').read_text())
    audit = scene['layout_sampling']['kinematic_screen']
    assert audit['filtered_proposal_count'] >= 1
    assert audit['filtered_proposals'][0]['failure']['stage'] == 'target_transfer'
    assert audit['filtered_proposals'][0]['failure']['error_m'] > .004
    assert audit['selected_proposal']['passed']
    assert audit['proposal_count'] == audit['filtered_proposal_count'] + 1
    assert scene['layout_draws']['source_cube']['cell_id'] == 6
    assert scene['layout_draws']['other_cube']['cell_id'] == 2
    assert json.loads((tmp_path / 'generation-layout-screen.json').read_text()) == audit


def test_kinematic_screen_preserves_known_successful_initial_pose(geometry, tmp_path):
    world, base = geometry
    descriptor = {'grid_shape': [3, 3], 'source_cell': 5, 'target_cell': 0}
    original = sample_workspace_layout(world, np.random.default_rng(10001), base, 'stack', descriptor)
    initialize_generation_scene(tmp_path, 'red_cube_to_green_cube', 10001, {'_layout_cell': descriptor})
    scene = json.loads((tmp_path / 'generation-scene.json').read_text())
    assert scene['layout_sampling']['kinematic_screen']['filtered_proposal_count'] == 0
    assert scene['layout_draws'] == original[3]


def test_balanced_pitch_risk_filter_reports_preference_separately_from_ik(tmp_path):
    initialize_generation_scene(tmp_path, 'red_cube_to_green_cube', 10009,
        {'layout_strategy': 'balanced',
         '_layout_cell': {'grid_shape': [3, 3], 'source_cell': 7, 'target_cell': 2}})
    audit = json.loads((tmp_path / 'generation-layout-screen.json').read_text())
    failure = audit['filtered_proposals'][0]['failure']
    assert failure['reason'] == 'placement_pitch_risk'
    assert failure['error_m'] <= .004  # Reachable does not imply a safe placement attitude.
    assert abs(failure['pitch_residual_deg']) > failure['threshold_deg']
    assert audit['rejection_counts']['placement_pitch_risk'] >= 1


def test_pitch_preference_keeps_known_cell2_success_and_does_not_filter_source_pitch(tmp_path):
    for seed, source, target in [(20007, 3, 2), (10001, 5, 0)]:
        descriptor = {'grid_shape': [3, 3], 'source_cell': source, 'target_cell': target}
        for mode in ['uniform', 'balanced']:
            initialize_generation_scene(tmp_path / str(seed) / mode, 'red_cube_to_green_cube', seed,
                {'layout_strategy': mode, '_layout_cell': descriptor})
        a, b = [json.loads((tmp_path / str(seed) / mode / 'generation-scene.json').read_text())
                for mode in ['uniform', 'balanced']]
        assert a['layout_draws'] == b['layout_draws']
        selected = b['layout_sampling']['kinematic_screen']['selected_proposal']['waypoints']
        lower = next(w for w in selected if w['stage'] == 'target_lower')
        assert abs(lower['pitch_residual_deg']) <= 20
        if seed == 10001:
            assert abs(next(w for w in selected if w['stage'] == 'source_grasp')['pitch_residual_deg']) > 20

