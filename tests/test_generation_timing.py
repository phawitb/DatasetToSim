"""Generated pace must not change the hardware motion contract."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dataset_to_sim.studio import motion_speed_limits, validate_trajectory


def test_recorded_pace_is_synthetic_only():
    from studio_generation_timing import generation_timing
    assert generation_timing({'source': 'live', 'generation_options': {'timing_profile': 'recorded'}}, 1.) is None
    assert generation_timing({'source': 'synthetic'}, 1.) is None
    profile = generation_timing({'source': 'synthetic', 'generation_options': {'timing_profile': 'recorded'}}, 1.)
    assert profile['reference']['episodes'] == 72
    assert profile['reference']['median_duration_s'] == pytest.approx(5.)
    assert np.all(np.array(profile['limits'][0]) > motion_speed_limits())
    np.testing.assert_array_equal(motion_speed_limits(), [35]*5+[50])
    with pytest.raises(ValueError, match='speed limits'):
        validate_trajectory([[0]*6, [3]*6], [0, 1/30], [[-180,180]]*6)


@pytest.mark.parametrize('task', ['green_cube_to_green_bowl', 'red_cube_to_green_bowl', 'tape_to_green_bowl', 'red_cube_to_green_cube', 'green_cube_to_red_cube', 'unstack'])
def test_recorded_pace_contact_validation(tmp_path, task):
    from studio_generation_scene import generate_episode
    report = generate_episode(tmp_path, task, 0, {'randomize_layout': False, 'timing_profile': 'recorded'})
    assert report['outcome'] == 'success'
    assert report['transfer_checks_passed']
    assert report['audit']['post_initial_state_writes'] == 0
    assert report['motion']['timing_profile'] == 'recorded'
    # Every task, including place-in-bowl, now ends back at Home.
    assert report['duration'] < 20
    assert 'Home' in report['motion']['waypoints']
    actions=np.load(tmp_path/'plan.npz')
    np.testing.assert_allclose(np.diff(actions['timestamps']), 1/30)
    trace=np.load(tmp_path/'trace.npz')
    assert len(trace['native']) == len(actions['actions'])+1


def test_recorded_flow_unstack_retains_tracking_gate(tmp_path):
    from studio_generation_scene import generate_episode
    report = generate_episode(tmp_path, 'unstack', 0, {'randomize_layout': False,
        'timing_profile': 'recorded', 'motion_technique': 'flow'})
    assert report['outcome'] == 'success' and report['transfer_checks_passed']
    assert max(report['max_tracking_error'][:5]) <= 10
    assert report['duration'] < 15
