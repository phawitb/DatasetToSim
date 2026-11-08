"""Per-layout visibility failures must not discard a whole generation batch."""
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import studio_generation
import studio_generation_export
import studio_generation_render
import studio_generation_scene
from tests.test_studio_generation_export import episode, set_task


VISIBILITY_FAILURE = 'A task cube is never visible in the rendered reference views'


@pytest.fixture
def worker_job(tmp_path, monkeypatch):
    source = set_task(episode(tmp_path, 'fixture'), 'red_cube_to_green_cube')
    metadata = json.loads((source / 'episode.json').read_text())
    (source / 'episode.json').unlink()
    studio_generation.save(source / 'capture.json', {'instruction': 'Stack red on green'})
    studio_generation.save(source / 'generation-scene.json', {
        'world': {'cube': {'position': [1, 2, 3]}, 'target_cube': {'position': [4, 5, 6]}},
        'physics_draws': {},
    })
    folder = tmp_path / ('a' * 32)
    folder.mkdir()
    options = studio_generation.validate_request({
        'tasks': ['red_cube_to_green_cube'], 'seed': 20,
        'episodes_per_task': 1, 'randomize_layout': False,
    })
    studio_generation.save(folder / 'job.json', {
        'id': folder.name, 'name': 'visibility-retry', 'status': 'queued',
        'options': options, 'episodes': [], 'errors': [],
        'progress': {'attempts': 0, 'accepted': 0, 'rejected': 0, 'requested': 1},
    })

    def generate(destination, task_id, seed, options):
        shutil.copytree(source, destination)
        (destination / 'attempt-seed.txt').write_text(str(seed))
        return dict(outcome='success', transfer_checks_passed=True,
                    action_hash=metadata['action_hash'], frames=4, duration=4 / 30)

    monkeypatch.setattr(studio_generation_scene, 'generate_episode', generate)
    # The worker watchdog belongs to the server process, not this synchronous test.
    monkeypatch.setattr(studio_generation.threading.Thread, 'start', lambda self: None)
    original_export = studio_generation_export.export_dataset
    monkeypatch.setattr(studio_generation_export, 'export_dataset',
                        lambda *args, **kwargs: original_export(*args, **kwargs, expected_size=(32, 24)))
    return folder


def test_visibility_failure_is_journaled_and_retry_exports_only_the_next_episode(worker_job, monkeypatch):
    def render(destination, seed, options, progress):
        if seed == 20:
            (destination / 'top-raw.mp4').write_bytes(b'partial render')
            raise ValueError(VISIBILITY_FAILURE)
    monkeypatch.setattr(studio_generation_render, 'render_episode', render)

    studio_generation.work(worker_job)

    saved = json.loads((worker_job / 'job.json').read_text())
    assert saved['status'] == 'completed'
    assert saved['progress']['attempts'] == 2
    assert saved['progress']['rejected'] == 1
    assert saved['progress']['accepted'] == 1
    assert saved['lineage']['seed_highwater'] == 22
    assert [item['status'] for item in saved['episodes']] == ['rejected', 'accepted']
    assert saved['episodes'][0]['reason'] == VISIBILITY_FAILURE
    assert saved['episodes'][1]['seed'] == 21
    assert (worker_job / 'episodes/00000/attempt-seed.txt').read_text() == '20'
    assert (worker_job / 'episodes/00000/top-raw.mp4').read_bytes() == b'partial render'
    assert not (worker_job / 'episodes/00000/episode.json').exists()
    assert (worker_job / 'episodes/00001/attempt-seed.txt').read_text() == '21'
    manifest = json.loads((worker_job / 'dataset/export_manifest.json').read_text())
    assert len(manifest['episodes']) == 1
    assert Path(manifest['episodes'][0]['source']).name == '00001'
    assert saved['export']['valid'] is True


def test_visibility_retries_remain_bounded_and_never_export_failed_observations(worker_job, monkeypatch):
    def render(*args):
        raise ValueError(VISIBILITY_FAILURE)
    monkeypatch.setattr(studio_generation_render, 'render_episode', render)

    with pytest.raises(ValueError, match='after bounded retries'):
        studio_generation.work(worker_job)

    saved = json.loads((worker_job / 'job.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['progress']['attempts'] == saved['progress']['rejected'] == 4
    assert saved['progress']['accepted'] == 0
    assert len(saved['episodes']) == 4
    assert all(item['reason'] == VISIBILITY_FAILURE for item in saved['episodes'])
    assert not (worker_job / 'dataset').exists()


@pytest.mark.parametrize('error', [OSError('Disk full'), RuntimeError('Video encoder unavailable'),
                                 ValueError('Background snapshot hash mismatch')])
def test_renderer_system_and_configuration_errors_still_fail_immediately(worker_job, monkeypatch, error):
    def render(*args):
        raise error
    monkeypatch.setattr(studio_generation_render, 'render_episode', render)

    with pytest.raises(type(error), match=str(error)):
        studio_generation.work(worker_job)

    saved = json.loads((worker_job / 'job.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['progress']['attempts'] == 1
    assert saved['progress']['rejected'] == saved['progress']['accepted'] == 0
    assert saved['errors'] == [str(error)]
    assert not (worker_job / 'dataset').exists()
