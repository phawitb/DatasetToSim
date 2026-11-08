"""Safety and lineage of building images around unchanged real action traces."""
import sys
from pathlib import Path
import json
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from studio_replay_dataset import validate_request, ReplayDatasetManager
from studio_replay_dataset_source import ReplaySource, verify_trace, DATASET, RUN


@pytest.fixture
def local_source():
    if not (DATASET / 'meta/info.json').is_file() or not (RUN / 'settings.json').is_file():
        pytest.skip('Local calibrated dataset and replay archive unavailable')
    return ReplaySource()


def test_selection_is_explicit_and_native_only():
    with pytest.raises(ValueError, match='Select'):
        validate_request({})
    for body in ({'episode_ids':[0,0]}, {'episode_ids':[True]}, {'episode_ids':[0], 'speed_scale':1.2}, {'episode_ids':[0], 'source_id':'other'}):
        with pytest.raises(ValueError):
            validate_request(body)
    result = validate_request({'episode_ids':[0,37], 'light_intensity':1.4})
    assert result['success_only'] is True
    assert result['light_intensity'] == 1.4


def test_known_source_reports_failures_and_actual_rows(local_source):
    source = local_source
    catalog = source.catalog()
    assert len(catalog['episodes']) == 72
    assert catalog['successes'] == 68
    assert catalog['failures'] == 4
    row = next(row for row in catalog['episodes'] if row['episode_index'] == 37)
    assert row['outcome'] == 'not_lifted'
    episode, trace, report, _ = source.read(0)
    np.testing.assert_array_equal(trace['native_action_applied'], episode.action_deg)
    altered = dict(trace, native_action_applied=trace['native_action_applied'].copy())
    altered['native_action_applied'][0,0] += .01
    with pytest.raises(ValueError, match='action'):
        verify_trace(episode, altered, report, source.settings)


def test_adapter_preserves_actions_and_video_offsets(tmp_path, local_source):
    source = local_source
    episode, _, _, _ = source.read(1)
    metadata = source.prepare(1, tmp_path, seed=3)
    with np.load(tmp_path / 'plan.npz') as plan:
        np.testing.assert_array_equal(plan['actions'], episode.action_deg)
        np.testing.assert_array_equal(plan['timestamps'], episode.timestamps)
    with np.load(tmp_path / 'trace.npz') as trace:
        assert len(trace['native']) == episode.frame_count + 1
        world = json.loads((tmp_path / 'generation-scene.json').read_text())['world']
        np.testing.assert_array_equal(world['cube']['position'], trace['cube_pose'][0, :3])
    assert metadata['source_episode_index'] == 1
    assert metadata['source_video_slices']['top']['from_timestamp'] == episode.video_slices['top'].start_time
    assert metadata['physical_transfer_validated'] is False


def test_manager_recovery_ignores_other_jobs(tmp_path):
    for identifier, source_type in [('a'*32,'real_replay'),('b'*32,'generated')]:
        folder = tmp_path / identifier; folder.mkdir()
        (folder / 'job.json').write_text(json.dumps({'id':identifier,'source_type':source_type,'status':'running','errors':[]}))
    manager = ReplayDatasetManager(tmp_path)
    manager.recover_interrupted()
    assert manager.get('a'*32)['status'] == 'cancelled'
    assert json.loads((tmp_path / ('b'*32) / 'job.json').read_text())['status'] == 'running'
    assert len(manager.list()) == 1
    with pytest.raises(ValueError):
        manager.folder('../bad')


def test_start_keeps_failures_explicit_and_previews_bounded(tmp_path, monkeypatch, local_source):
    class Process:
        def poll(self): return None
    monkeypatch.setattr('studio_replay_dataset.subprocess.Popen', lambda *a,**k:Process())
    manager = ReplayDatasetManager(tmp_path / 'jobs')
    job = manager.start({'episode_ids':[0,37]})
    assert job['included_episode_ids'] == [0]
    assert job['skipped'][0]['outcome'] == 'not_lifted'
    assert job['source_type'] == 'real_replay'
    diagnostic = ReplayDatasetManager(tmp_path / 'diagnostics').start({'episode_ids':[37],'success_only':False})
    assert diagnostic['included_episode_ids'] == [37]
    preview = ReplayDatasetManager(tmp_path / 'previews',media_prefix='/media/replay-previews').start_preview({'episode_ids':[0,1,2,3,4,5]})
    assert preview['kind'] == 'preview'
    assert len(preview['included_episode_ids']) == 5
    assert preview['media_prefix'] == '/media/replay-previews'
    with pytest.raises(ValueError,match='No successful'):
        ReplayDatasetManager(tmp_path / 'none').start({'episode_ids':[37]})


def test_rejects_corrupt_trace_timing_and_targets(local_source):
    source = local_source
    episode, trace, report, _ = source.read(0)
    for field in ('model_targets','sample_times','recorded_native_state'):
        altered = dict(trace); altered[field] = trace[field].copy(); altered[field].flat[0] += .001
        with pytest.raises(ValueError):
            verify_trace(episode,altered,report,source.settings)
