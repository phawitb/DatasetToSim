"""Export regression tests use real videos/parquet, including causal state alignment."""
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def exporter():
    assert importlib.util.find_spec('studio_generation_export'), 'LeRobot v3 exporter is missing'
    return importlib.import_module('studio_generation_export')


def episode(tmp_path, name, group='one', n=4):
    from dataset_to_sim.strict_replay import array_hash
    root = tmp_path / name
    root.mkdir()
    actions = np.tile([10., 20., 30., 40., 50., 60.], (n, 1))
    actions[:, 0] += np.arange(n)
    states = np.tile([-10., -20., -30., -40., -50., 80.], (n + 1, 1))
    np.savez(root / 'plan.npz', actions=actions, timestamps=np.arange(n) / 30)
    np.savez(root / 'trace.npz', native=states)
    meta = dict(task_id='pick', instruction='Put cube in bowl', seed=7,
                group_id=group, action_hash=array_hash(actions), outcome='success', transfer_checks_passed=True)
    (root / 'episode.json').write_text(json.dumps(meta))
    for camera in ['top', 'wrist']:
        video = cv2.VideoWriter(str(root / f'{camera}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (32, 24))
        assert video.isOpened()
        for _ in range(n):
            video.write(np.full((24, 32, 3), [0, 0, 255], dtype=np.uint8))
        video.release()
    return root


def test_export_preserves_causal_state_ranges_rgb_and_group_split(tmp_path):
    mod = exporter()
    sources = [episode(tmp_path, 'a', 'one'), episode(tmp_path, 'b', 'two', 3), episode(tmp_path, 'c', 'one', 5)]
    dst = tmp_path / 'export'
    result = mod.export_dataset(sources, dst, name='test', validation_fraction=.5, expected_size=(32, 24))
    info = json.loads((dst / 'meta/info.json').read_text())
    assert info['codebase_version'] == 'v3.0'
    assert info['total_frames'] == 12
    eps = pq.read_table(dst / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    assert eps[0]['dataset_from_index'] == 0
    assert eps[-1]['dataset_to_index'] == 12
    for prev, nxt in zip(eps, eps[1:]):
        assert prev['dataset_to_index'] == nxt['dataset_from_index']
    rows = pq.read_table(sorted((dst / 'data').rglob('*.parquet'))).to_pylist()
    assert rows[0]['observation.state'] == [-10, -20, -30, -40, -50, 80]
    assert rows[0]['action'] == [10, 20, 30, 40, 50, 60]
    manifest = json.loads((dst / 'export_manifest.json').read_text())
    grouped = {}
    for item in manifest['episodes']:
        grouped.setdefault(item['group_id'], set()).add(item['split'])
        assert (dst / item['videos']['top']).read_bytes() == (Path(item['source']) / 'top.mp4').read_bytes()
    assert all(len(splits) == 1 for splits in grouped.values())
    assert {item['split'] for item in manifest['episodes']} == {'train', 'val'}
    stats = json.loads((dst / 'meta/stats.json').read_text())['observation.images.top']
    assert np.asarray(stats['mean']).shape == (3, 1, 1)
    assert stats['mean'][0][0][0] > .9
    assert stats['mean'][2][0][0] < .1
    assert result['total_episodes'] == 3
    assert mod.validate_export(dst, expected_size=(32, 24))['valid']
    with pytest.raises(FileExistsError):
        mod.export_dataset(sources, dst, name='test', expected_size=(32, 24))


@pytest.mark.parametrize('fault', ['failed', 'nonfinite', 'hash', 'state_length', 'video_length', 'timestamps'])
def test_rejects_invalid_or_unverified_episode_without_partial_export(tmp_path, fault):
    mod = exporter()
    root = episode(tmp_path, 'episode')
    if fault == 'failed':
        meta = json.loads((root / 'episode.json').read_text()); meta['outcome'] = 'failed'
        (root / 'episode.json').write_text(json.dumps(meta))
    elif fault in ('nonfinite', 'state_length'):
        state = np.zeros((2 if fault == 'state_length' else 5, 6))
        if fault == 'nonfinite': state[0, 0] = np.nan
        np.savez(root / 'trace.npz', native=state)
    elif fault == 'video_length':
        (root / 'top.mp4').write_bytes(b'invalid video')
    else:
        data = dict(np.load(root / 'plan.npz'))
        if fault == 'hash': data['actions'][0, 0] += 1
        else: data['timestamps'][1] = .9
        np.savez(root / 'plan.npz', **data)
    dst = tmp_path / 'export'
    with pytest.raises(ValueError):
        mod.export_dataset([root], dst, name='bad', expected_size=(32, 24))
    assert not dst.exists()


def test_audit_rejects_stale_episode_bounds(tmp_path):
    mod = exporter()
    dst = tmp_path / 'export'
    mod.export_dataset([episode(tmp_path, 'a'), episode(tmp_path, 'b')], dst, name='test', expected_size=(32, 24))
    path = dst / 'meta/episodes/chunk-000/file-000.parquet'
    import pyarrow as pa
    rows = pq.read_table(path).to_pylist(); rows[1]['dataset_from_index'] = 0
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match='bound|range'):
        mod.validate_export(dst, expected_size=(32, 24))


@pytest.mark.parametrize('fault', ['split_range', 'task_mapping', 'state_corrupt', 'success_claim'])
def test_audit_rejects_corrupted_semantics(tmp_path, fault):
    mod = exporter()
    dst = tmp_path / 'export'
    mod.export_dataset([episode(tmp_path, 'a')], dst, name='test', expected_size=(32, 24))
    import pyarrow as pa
    if fault == 'split_range':
        path = dst / 'meta/info.json'; value = json.loads(path.read_text()); value['splits']['train'] = '0:99'
        path.write_text(json.dumps(value))
    elif fault == 'success_claim':
        path = dst / 'export_manifest.json'; value = json.loads(path.read_text()); value['episodes'][0]['outcome'] = 'failed'
        path.write_text(json.dumps(value))
    else:
        path = dst / 'data/chunk-000/file-000.parquet'; table = pq.read_table(path); rows = table.to_pylist()
        if fault == 'task_mapping': rows[0]['task_index'] = -1
        else: rows[0]['observation.state'][0] = 999.
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError):
        mod.validate_export(dst, expected_size=(32, 24))


def test_qpos_adapter_uses_pre_action_rows(tmp_path):
    mod = exporter()
    from dataset_to_sim.strict_replay import NativeJointAdapter
    source = episode(tmp_path, 'a')
    cfg = dict(arm_offset_deg=[1, 2, 3, 4, 5], arm_sign=[1, -1, 1, -1, 1],
               gripper_range_min=0, gripper_range_max=1000, encoder_max_tick=4096, gripper_closed_model_deg=-3)
    native = np.tile([0., 0., 0., 0., 0., 25.], (5, 1)); native[-1] = [9, 9, 9, 9, 9, 99]
    np.savez(source / 'trace.npz', qpos=NativeJointAdapter(**cfg).to_model(native))
    metadata = json.loads((source / 'episode.json').read_text()); metadata['adapter'] = cfg
    (source / 'episode.json').write_text(json.dumps(metadata))
    dst = tmp_path / 'export'
    mod.export_dataset([source], dst, name='test', expected_size=(32, 24))
    state = pq.read_table(dst / 'data/chunk-000/file-000.parquet').to_pydict()['observation.state']
    np.testing.assert_allclose(state, [[0, 0, 0, 0, 0, 25]] * 4, atol=1e-6)


def set_task(source, task):
    path = source / 'episode.json'
    metadata = json.loads(path.read_text())
    metadata.update(task_id=task, instruction=f'Perform {task}')
    path.write_text(json.dumps(metadata))
    return source


def test_singleton_tasks_remain_train_with_explanation(tmp_path):
    mod = exporter()
    sources = [set_task(episode(tmp_path, str(i), f'group-{i}'), f'task-{i}') for i in range(5)]
    dst = tmp_path / 'export'
    result = mod.export_dataset(sources, dst, name='test', expected_size=(32, 24))
    assert result['splits']['val'] == []
    manifest = json.loads((dst / 'export_manifest.json').read_text())
    assert set(manifest['validation_unavailable_tasks']) == {f'task-{i}' for i in range(5)}
    assert 'singleton' in manifest['split_note'].lower()


def test_cross_task_groups_can_be_held_out_only_if_all_tasks_retain_training(tmp_path):
    mod = exporter()
    # "other" sorts ahead as a holdout candidate, but contains b's only group.
    sources = [set_task(episode(tmp_path, 'shared-a', 'other'), 'a'),
               set_task(episode(tmp_path, 'shared-b', 'other'), 'b'),
               set_task(episode(tmp_path, 'other-a', 'shared'), 'a')]
    dst = tmp_path / 'export'
    mod.export_dataset(sources, dst, name='test', validation_fraction=.5, expected_size=(32, 24))
    manifest = json.loads((dst / 'export_manifest.json').read_text())
    entries = manifest['episodes']
    assert {e['split'] for e in entries if e['group_id'] == 'other'} == {'train'}
    assert {e['split'] for e in entries if e['group_id'] == 'shared'} == {'val'}
    assert {e['task_id'] for e in entries if e['split'] == 'train'} == {'a', 'b'}


def test_portable_provenance_is_copied_and_audited_without_source_folder(tmp_path):
    mod = exporter()
    source = episode(tmp_path, 'a')
    for name in ('generation-config.json', 'generation-scene.json', 'visual-provenance.json', 'plan.json'):
        (source / name).write_text('{}')
    for name in ('scene.xml', 'render-scene.xml'):
        (source / name).write_text('<mujoco/>')
    dst = tmp_path / 'export'
    mod.export_dataset([source], dst, name='test', expected_size=(32, 24))
    manifest = json.loads((dst / 'export_manifest.json').read_text())
    files = manifest['episodes'][0]['provenance_files']
    assert set(files) == {'generation-config.json', 'generation-scene.json', 'visual-provenance.json', 'plan.json',
                          'scene.xml', 'render-scene.xml', 'episode.json', 'plan.npz', 'trace.npz'}
    for name, record in files.items():
        assert (dst / record['path']).read_bytes() == (source / name).read_bytes()
        assert len(record['sha256']) == 64
    import shutil
    shutil.rmtree(source)
    assert mod.validate_export(dst, expected_size=(32, 24))['valid']
    (dst / files['trace.npz']['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='provenance'):
        mod.validate_export(dst, expected_size=(32, 24))
