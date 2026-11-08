"""Full-workspace proposal coverage and footprint safety, independent of IK."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from studio_generation_layout import EDGE_CLEARANCE_M, OBJECT_CLEARANCE_M, sample_workspace_layout
from studio_generation_scene import ROOT, _robot_base_bounds, initialize_generation_scene


@pytest.fixture(scope='module')
def geometry():
    config = json.loads((ROOT / 'calibration/native-action-current.json').read_text())
    return config['base_calibration']['world'], _robot_base_bounds(json.dumps(config, sort_keys=True))


@pytest.mark.parametrize('task', ['stack', 'unstack', 'bowl'])
def test_full_workspace_proposals_cover_mat_and_respect_footprints(geometry, task):
    world, base = geometry
    edge = world['cube']['size']
    source_samples, destination_samples = [], []
    for seed in range(512):
        source, target, bowl, draws, policy = sample_workspace_layout(world, np.random.default_rng(seed), base, task)
        low, high = np.asarray(policy['mat_bounds_xy_m'])
        objects = []
        for xy, yaw in (source, target):
            half = edge / 2 * (abs(np.cos(yaw)) + abs(np.sin(yaw)))
            assert np.all(xy - half >= low + EDGE_CLEARANCE_M - 1e-12)
            assert np.all(xy + half <= high - EDGE_CLEARANCE_M + 1e-12)
            objects.append((xy, edge / np.sqrt(2)))
        if task == 'bowl':
            radius = world['bowl']['outer_radius']
            assert np.all(bowl - radius >= low + EDGE_CLEARANCE_M - 1e-12)
            assert np.all(bowl + radius <= high - EDGE_CLEARANCE_M + 1e-12)
            objects.append((bowl, radius))
        for i, (xy, radius) in enumerate(objects):
            assert np.linalg.norm(xy - np.clip(xy, *base)) >= radius + OBJECT_CLEARANCE_M
            for other, other_radius in objects[:i]:
                assert np.linalg.norm(xy - other) >= radius + other_radius + OBJECT_CLEARANCE_M
        if task == 'unstack':
            assert np.linalg.norm(source[0] - target[0]) >= .08
        source_samples.append(source[0])
        destination_samples.append(target[0])
    # Every source/destination role must reach all sixteen spatial bins, and
    # span >90% of the admissible center range, not merely the old 20 mm patch.
    for samples in (source_samples, destination_samples):
        samples = np.asarray(samples)
        assert np.all(np.ptp(samples, axis=0) > .36)
        cells = np.floor((samples - low) / (high - low) * 4).astype(int)
        assert len(set(map(tuple, cells))) == 16


def test_sampler_uses_calibrated_mat_translation_and_size(geometry):
    original, base = geometry
    world = copy.deepcopy(original)
    world['mat']['position'][:2] = [.7, -.3]
    world['mat']['size'][:2] = [.3, .5]
    source, target, _, _, policy = sample_workspace_layout(world, np.random.default_rng(13), base, 'stack')
    np.testing.assert_allclose(policy['mat_bounds_xy_m'], [[.55, -.55], [.85, -.05]])
    for xy, _ in (source, target):
        assert .55 < xy[0] < .85 and -.55 < xy[1] < -.05


def test_scene_saves_sampling_policy_and_seeded_poses(tmp_path):
    for folder in (tmp_path/'a', tmp_path/'b'):
        initialize_generation_scene(folder, 'red_cube_to_green_cube', 2026, {})
    first = (tmp_path/'a/generation-scene.json').read_bytes()
    assert first == (tmp_path/'b/generation-scene.json').read_bytes()
    scene = json.loads(first)
    assert scene['layout_sampling']['distribution'] == 'uniform_xy_over_full_calibrated_mat_with_footprint_rejection'
    assert scene['layout_sampling']['reachability_filter'].startswith('none_at_sampling')
    for name, key in [('source_cube', 'cube'), ('other_cube', 'target_cube')]:
        assert scene['layout_draws'][name]['position_xy_m'] == scene['world'][key]['position'][:2]
