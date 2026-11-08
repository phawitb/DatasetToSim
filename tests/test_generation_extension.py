"""Immutable continuation uses actual portable fixtures; only worker launch is stubbed."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from studio_generation import GenerationManager, save, validate_request
from studio_generation_export import export_dataset
from tests.test_studio_generation_export import episode, set_task


def snapshot(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


def parent_fixture(tmp_path):
    folder = tmp_path / ('a' * 32)
    (folder / 'episodes').mkdir(parents=True)
    sources = [set_task(episode(folder / 'episodes', f'{i:05d}', f'group-{i}'), 'unstack') for i in (0, 2)]
    exported = export_dataset(sources, folder / 'dataset', name='original', validation_fraction=.4, expected_size=(32, 24))
    save(folder / 'job.json', dict(id=folder.name, name='original', status='completed', errors=[],
        options=validate_request(dict(name='original', tasks=['unstack'], seed=7, speed_scale=.8)),
        progress=dict(attempts=3, accepted=2, rejected=1, requested=2),
        episodes=[dict(status='accepted', seed=7, task_id='unstack', frames=4),
                  dict(status='rejected', seed=8, task_id='unstack'),
                  dict(status='accepted', seed=9, task_id='unstack', frames=4)], export=exported))
    return folder


@pytest.fixture
def no_worker(monkeypatch):
    import studio_generation
    class Process:
        def poll(self): return 0
    monkeypatch.setattr(studio_generation.subprocess, 'Popen', lambda *a, **kw: Process())


def test_continuation_copies_dense_accepted_artifacts_and_inherits_settings(tmp_path, no_worker):
    parent = parent_fixture(tmp_path)
    before = snapshot(parent)
    manager = GenerationManager(tmp_path)
    child = manager.start({'parent_job_id': parent.name, 'episodes_per_task': 2, 'light_intensity': 1.3})
    assert child['options']['seed'] == 10
    assert child['options']['speed_scale'] == .8
    assert child['options']['light_intensity'] == 1.3
    assert child['progress']['requested'] == 4
    assert child['progress']['accepted'] == child['progress']['inherited_count'] == 2
    assert len(child['episodes']) == 2
    for index, source in enumerate((0, 2)):
        destination = tmp_path / child['id'] / 'episodes' / f'{index:05d}'
        assert (destination / 'top.mp4').read_bytes() == (parent / 'episodes' / f'{source:05d}' / 'top.mp4').read_bytes()
        assert child['episodes'][index]['preview_url'] == f'/media/generated/{child["id"]}/episodes/{index:05d}/top.jpg'
    # A queued child can be cancelled before any worker has consumed a seed.
    manager.process = None
    assert manager.cancel(child['id'])['status'] == 'cancelled'
    assert snapshot(parent) == before


@pytest.mark.parametrize('fault', ['missing', 'failed', 'corrupt_artifact', 'bad_export'])
def test_continuation_rejects_invalid_parent(tmp_path, no_worker, fault):
    parent = parent_fixture(tmp_path)
    identifier = parent.name
    if fault == 'missing': identifier = 'b' * 32
    elif fault == 'failed':
        job = json.loads((parent / 'job.json').read_text()); job['status'] = 'failed'; save(parent / 'job.json', job)
    elif fault == 'corrupt_artifact': (parent / 'episodes/00000/plan.npz').write_bytes(b'corrupt')
    else: (parent / 'dataset/export_manifest.json').write_text('{}')
    with pytest.raises(ValueError):
        GenerationManager(tmp_path).start({'parent_job_id': identifier})
    assert len(list(tmp_path.glob('*/job.json'))) == 1


def test_seed_ranges_include_failed_extension_attempts_without_mutating_parent(tmp_path, no_worker):
    parent = parent_fixture(tmp_path)
    before = snapshot(parent)
    manager = GenerationManager(tmp_path)
    with pytest.raises(ValueError, match='seed|Seed'):
        manager.start({'parent_job_id': parent.name, 'seed': 9})
    child = manager.start({'parent_job_id': parent.name})
    child['status'] = 'failed'; child['progress']['attempts'] = 4
    child['lineage']['seed_highwater'] = 14
    save(tmp_path / child['id'] / 'job.json', child)
    assert manager.get(parent.name)['next_seed'] == 14
    with pytest.raises(ValueError, match='seed|Seed'):
        manager.start({'parent_job_id': parent.name, 'seed': 12})
    retry = manager.start({'parent_job_id': parent.name})
    assert retry['options']['seed'] == 14
    assert snapshot(parent) == before


def test_combined_export_preserves_parent_groups_even_with_zero_validation(tmp_path):
    parent = parent_fixture(tmp_path)
    manifest = json.loads((parent / 'dataset/export_manifest.json').read_text())
    fixed = {item['group_id']: item['split'] for item in manifest['episodes']}
    assert set(fixed.values()) == {'train', 'val'}
    before = snapshot(parent)
    extra = set_task(episode(tmp_path, 'new', 'new-group', n=3), 'unstack')
    sources = [parent / 'episodes/00000', parent / 'episodes/00002', extra]
    output = tmp_path / 'combined'
    result = export_dataset(sources, output, name='combined', validation_fraction=0,
                            expected_size=(32, 24), inherited_group_splits=fixed,
                            lineage={'parent_job_id': parent.name, 'inherited_count': 2})
    combined = json.loads((output / 'export_manifest.json').read_text())
    assert result['total_episodes'] == 3 and result['total_frames'] == 11
    for item in combined['episodes']:
        if item['group_id'] in fixed: assert item['split'] == fixed[item['group_id']]
    assert combined['lineage']['parent_job_id'] == parent.name
    assert json.loads((output / 'meta/stats.json').read_text())['action']['count'] == [11]
    assert snapshot(parent) == before


@pytest.mark.parametrize('render_failure', [False, True])
@pytest.mark.parametrize('total_quota', [False, True])
def test_worker_exports_combined_real_fixtures_or_keeps_parent_immutable(tmp_path, no_worker, monkeypatch, render_failure, total_quota):
    import shutil
    import studio_generation
    import studio_generation_scene
    import studio_generation_render
    import studio_generation_export
    parent = parent_fixture(tmp_path)
    before = snapshot(parent)
    manager = GenerationManager(tmp_path)
    child = manager.start({'parent_job_id': parent.name, 'episodes_per_task': 300 if total_quota else 1, 'target_counts': {'unstack': 3} if total_quota else None, 'validation_fraction': 0})
    new_source = set_task(episode(tmp_path, 'worker-fixture', 'new-physical-group', n=3), 'unstack')
    meta = json.loads((new_source / 'episode.json').read_text())
    save(new_source / 'capture.json', {'instruction': 'Perform unstack'})
    save(new_source / 'generation-scene.json', {'world': {'cube': {'position': [1, 2, 3]}, 'target_cube': {'position': [4, 5, 6]}}, 'physics_draws': {}})
    report = dict(outcome='success', transfer_checks_passed=True, action_hash=meta['action_hash'], frames=3, duration=.1)
    def generate(destination, task_id, seed, options):
        assert task_id == 'unstack' and seed == 10
        shutil.copytree(new_source, destination)
        return report
    def render(destination, seed, options, progress):
        if render_failure: raise ValueError('Renderer fixture failed')
    monkeypatch.setattr(studio_generation_scene, 'generate_episode', generate)
    monkeypatch.setattr(studio_generation_render, 'render_episode', render)
    original_export = studio_generation_export.export_dataset
    monkeypatch.setattr(studio_generation_export, 'export_dataset', lambda *a, **kw: original_export(*a, **kw, expected_size=(32, 24)))
    folder = tmp_path / child['id']
    if render_failure:
        with pytest.raises(ValueError, match='Renderer fixture failed'):
            studio_generation.work(folder)
    else:
        studio_generation.work(folder)
    saved = json.loads((folder / 'job.json').read_text())
    assert saved['lineage']['seed_highwater'] == 11
    assert saved['progress']['attempts'] == 1
    if render_failure:
        assert saved['status'] == 'failed'
        assert saved['progress']['accepted'] == len(saved['episodes']) == 2
        assert not (folder / 'dataset').exists()
    else:
        assert saved['status'] == 'completed'
        assert saved['progress']['accepted'] == saved['progress']['requested'] == 3
        assert saved['export']['total_frames'] == 11
        assert (folder / 'dataset.zip').is_file()
        manifest = json.loads((folder / 'dataset/export_manifest.json').read_text())
        assert manifest['lineage']['inherited_count'] == 2
        assert len(manifest['episodes']) == 3
        assert (folder / 'episodes/00002/top.mp4').is_file()
        assert saved['episodes'][2]['seed'] == 10
    assert snapshot(parent) == before


def test_legacy_parent_keeps_its_implicit_conservative_timing(tmp_path):
    parent = parent_fixture(tmp_path)
    job = json.loads((parent / 'job.json').read_text())
    job['options'].pop('timing_profile')
    save(parent / 'job.json', job)
    options = GenerationManager(tmp_path).resolve_options({'parent_job_id': parent.name})
    assert options['timing_profile'] == 'conservative'
