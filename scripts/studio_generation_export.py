"""Audited, local-only LeRobot v3 export of accepted simulated episodes.

No hardware or Hub operations. Videos are copied byte for byte. Observation row
k is the simulated state *before* action k; terminal states never become labels.
Use ``python scripts/studio_generation_export.py audit PATH --official-reader``
in a Python environment with LeRobot installed for temporal/decode verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dataset_to_sim.joints import FEATURE_JOINT_NAMES
from dataset_to_sim.strict_replay import NativeJointAdapter, array_hash

FPS = 30
CAMERAS = ('top', 'wrist')
DATA_PATH = 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'
VIDEO_PATH = 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'
PROVENANCE_FILES = ('generation-config.json', 'generation-scene.json', 'scene.xml',
                    'render-scene.xml', 'visual-provenance.json', 'episode.json',
                    'plan.json', 'report.json', 'plan.npz', 'trace.npz',
                    'real-appearance.json', 'real-background.png', 'real-background-valid.png')


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    result = {key: fn(values, axis=0).tolist() for key, fn in
              [('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std)]}
    result['count'] = [len(values)]
    for q in (.01, .1, .5, .9, .99):
        result[f'q{int(q * 100):02d}'] = np.quantile(values, q, axis=0).tolist()
    return result


def _image_stats(histogram, frames):
    count = histogram.sum(axis=1)
    levels = np.arange(256) / 255.
    mean = (histogram * levels).sum(axis=1) / count
    variance = (histogram * levels**2).sum(axis=1) / count - mean**2
    result = {'mean': mean, 'std': np.sqrt(np.maximum(0, variance)),
              'min': np.argmax(histogram > 0, axis=1) / 255.,
              'max': (255 - np.argmax(histogram[:, ::-1] > 0, axis=1)) / 255.}
    for q in (.01, .1, .5, .9, .99):
        result[f'q{int(q * 100):02d}'] = np.array([
            np.searchsorted(np.cumsum(channel), q * total) / 255.
            for channel, total in zip(histogram, count)])
    return {**{key: val.reshape(3, 1, 1).tolist() for key, val in result.items()}, 'count': [frames]}


def _video(path, n, expected_size, *, collect_stats=True):
    capture = cv2.VideoCapture(str(path))
    histogram = np.zeros((3, 256), dtype=np.int64)
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot decode video: {path}')
        if abs(capture.get(cv2.CAP_PROP_FPS) - FPS) > .001:
            raise ValueError(f'Video must be {FPS} fps: {path}')
        width, height = expected_size
        if (round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (width, height):
            raise ValueError(f'Video must be {width}x{height}: {path}')
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = ''.join(chr((fourcc >> (8 * i)) & 255) for i in range(4))
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (height, width, 3):
                raise ValueError(f'Invalid RGB frame shape: {path}')
            if collect_stats:
                for rgb, bgr in enumerate((2, 1, 0)):
                    histogram[rgb] += np.bincount(frame[:, :, bgr].ravel(), minlength=256)
            count += 1
        if count != n:
            raise ValueError(f'Video frame count {count} differs from actions {n}: {path}')
        return histogram, {'video.fps': FPS, 'video.height': height, 'video.width': width,
                           'video.channels': 3, 'video.codec': {'avc1': 'h264', 'mp4v': 'mpeg4'}.get(codec, codec),
                           'video.is_depth_map': False, 'has_audio': False}
    finally:
        capture.release()


def _native(values, n, label):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (n, 6) or not np.isfinite(values).all():
        raise ValueError(f'{label} must be finite with shape ({n}, 6), got {values.shape}')
    if np.any((values[:, 5] < 0) | (values[:, 5] > 100)):
        raise ValueError(f'{label} gripper must be native normalized 0–100; refusing to clip')
    return values


def _read_episode(root, expected_size):
    root = Path(root).resolve()
    metadata = {}
    for file in ('plan.json', 'report.json', 'episode.json'):
        path = root / file
        if path.exists():
            metadata.update(json.loads(path.read_text()))
    if metadata.get('outcome') != 'success' or metadata.get('transfer_checks_passed') is not True:
        raise ValueError(f'Episode lacks successful task and transfer checks: {root}')
    for key in ('task_id', 'instruction', 'group_id', 'action_hash', 'seed'):
        if key not in metadata or metadata[key] is None or (key != 'seed' and not str(metadata[key]).strip()):
            raise ValueError(f'Episode requires {key}: {root}')
    with np.load(root / 'plan.npz', allow_pickle=False) as plan:
        raw_actions = plan['actions']
        n = len(raw_actions)
        if not n:
            raise ValueError('Cannot export an empty episode')
        actions = _native(raw_actions, n, 'actions')
        timestamps = np.asarray(plan['timestamps'], dtype=float)
    if timestamps.shape != (n,) or not np.isfinite(timestamps).all() or not np.allclose(timestamps, np.arange(n) / FPS, atol=1e-5, rtol=0):
        raise ValueError(f'Episode timestamps must start at zero and advance at {FPS} Hz')
    if array_hash(actions) != metadata['action_hash']:
        raise ValueError(f'Action provenance hash mismatch: {root}')
    with np.load(root / 'trace.npz', allow_pickle=False) as trace:
        if 'native' in trace:
            states = trace['native']
            if len(states) == n + 1:
                states = states[:-1]
        elif 'state' in trace:
            states = trace['state']
        elif 'qpos' in trace and 'adapter' in metadata:
            qpos = trace['qpos']
            if qpos.ndim != 2 or qpos.shape[0] != n + 1 or qpos.shape[1] < 6:
                raise ValueError('qpos trace must contain N+1 states and six joint coordinates')
            states = NativeJointAdapter(**metadata['adapter']).to_native(qpos[:-1, :6])
        else:
            raise ValueError('Trace requires pre-action native/state, or N+1 qpos plus adapter calibration')
        states = _native(states, n, 'observation.state')
    videos = {camera: _video(root / f'{camera}.mp4', n, expected_size) for camera in CAMERAS}
    return {'source': root, 'metadata': metadata, 'actions': actions.astype(np.float32),
            'states': states.astype(np.float32), 'timestamps': timestamps.astype(np.float32), 'videos': videos, 'n': n}


def export_dataset(episodes: list[Path], destination: Path, *, name: str,
                   validation_fraction: float = .2, expected_size: tuple[int, int] = (640, 480),
                   inherited_group_splits: dict | None = None, lineage: dict | None = None) -> dict:
    """Write a complete new dataset atomically; expected_size overrides are for fixtures.

    Grouped variants stay in one split. Every task retains a training group;
    singleton tasks remain train-only. Episodes are reordered train-first so the
    standard LeRobot split range syntax remains accurate.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    if not episodes or not 0 <= validation_fraction < 1:
        raise ValueError('Provide episodes and validation_fraction in [0, 1)')
    if not name.strip():
        raise ValueError('Dataset name is required')
    if len(set(Path(p).resolve() for p in episodes)) != len(episodes):
        raise ValueError('Duplicate source episode')
    loaded = [_read_episode(path, expected_size) for path in episodes]
    groups = sorted({str(item['metadata']['group_id']) for item in loaded}, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    requested_val_count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction))) if validation_fraction and len(groups) > 1 else 0
    task_groups = {}
    for item in loaded:
        task_groups.setdefault(str(item['metadata']['task_id']), set()).add(str(item['metadata']['group_id']))
    fixed = dict(inherited_group_splits or {})
    if any(group not in groups or split not in ('train', 'val') for group, split in fixed.items()):
        raise ValueError('Invalid inherited group split assignment')
    val_groups = {group for group, split in fixed.items() if split == 'val'}
    if not all(task_group - val_groups for task_group in task_groups.values()):
        raise ValueError('Inherited validation groups leave a task without a training group')
    for group in reversed(groups):
        if len(val_groups) >= requested_val_count:
            break
        if group in fixed:
            continue
        candidate = val_groups | {group}
        # A group can span tasks; check all tasks before removing it from train.
        if all(task_group - candidate for task_group in task_groups.values()):
            val_groups = candidate
    val_count = len(val_groups)
    for item in loaded:
        item['split'] = 'val' if str(item['metadata']['group_id']) in val_groups else 'train'
    loaded.sort(key=lambda x: x['split'] == 'val')
    train_count = sum(item['split'] == 'train' for item in loaded)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
    try:
        width, height = expected_size
        features = {key: {'dtype': 'float32', 'shape': [6], 'names': list(FEATURE_JOINT_NAMES)}
                    for key in ('action', 'observation.state')}
        features.update({key: {'dtype': 'float32' if key == 'timestamp' else 'int64', 'shape': [1], 'names': None}
                         for key in ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index')})
        for camera in CAMERAS:
            features[f'observation.images.{camera}'] = {'dtype': 'video', 'shape': [height, width, 3],
                'names': ['height', 'width', 'channels'], 'info': loaded[0]['videos'][camera][1]}
        tasks = list(dict.fromkeys(item['metadata']['instruction'] for item in loaded))
        table = pa.table({'task': tasks, 'task_index': pa.array(range(len(tasks)), type=pa.int64())})
        (temporary / 'meta').mkdir()
        pq.write_table(table, temporary / 'meta/tasks.parquet')
        ep_rows, provenance, numeric = [], [], {key: [] for key in features if not key.startswith('observation.images.')}
        histogram = {camera: np.zeros((3, 256), dtype=np.int64) for camera in CAMERAS}
        offset = 0
        for idx, item in enumerate(loaded):
            n, meta = item['n'], item['metadata']
            chunk, file = divmod(idx, 1000)
            task_index = tasks.index(meta['instruction'])
            columns = {'action': item['actions'], 'observation.state': item['states'], 'timestamp': item['timestamps'],
                       'frame_index': np.arange(n, dtype=np.int64), 'episode_index': np.full(n, idx, dtype=np.int64),
                       'index': np.arange(offset, offset + n, dtype=np.int64), 'task_index': np.full(n, task_index, dtype=np.int64)}
            table = pa.table({key: pa.array(value.tolist(), type=pa.list_(pa.float32(), 6)) if value.ndim == 2 else pa.array(value)
                              for key, value in columns.items()})
            path = temporary / DATA_PATH.format(chunk_index=chunk, file_index=file)
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path)
            ep_stats = {key: _stats(value) for key, value in columns.items()}
            row = {'episode_index': idx, 'tasks': [meta['instruction']], 'length': n,
                   'dataset_from_index': offset, 'dataset_to_index': offset + n, 'data/chunk_index': chunk,
                   'data/file_index': file, 'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0}
            video_paths, hashes = {}, {}
            for camera in CAMERAS:
                key = f'observation.images.{camera}'
                relative = VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)
                out = temporary / relative
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item['source'] / f'{camera}.mp4', out)
                video_paths[camera], hashes[camera] = relative, _sha(out)
                row.update({f'videos/{key}/chunk_index': chunk, f'videos/{key}/file_index': file,
                            f'videos/{key}/from_timestamp': 0., f'videos/{key}/to_timestamp': n / FPS})
                histogram[camera] += item['videos'][camera][0]
                ep_stats[key] = _image_stats(item['videos'][camera][0], n)
            for key, stats in ep_stats.items():
                row.update({f'stats/{key}/{kind}': value for kind, value in stats.items()})
            ep_rows.append(row)
            provenance_files = {}
            for filename in PROVENANCE_FILES:
                source = item['source'] / filename
                if source.is_file():
                    relative = f'provenance/episode-{idx:06d}/{filename}'
                    out = temporary / relative
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, out)
                    provenance_files[filename] = {'path': relative, 'sha256': _sha(out)}
            provenance.append({'episode_index': idx, 'source': str(item['source']), 'group_id': str(meta['group_id']),
                'split': item['split'], 'task_id': meta['task_id'], 'seed': meta['seed'],
                'action_hash': meta['action_hash'], 'exported_action_hash': array_hash(item['actions']),
                'exported_state_hash': array_hash(item['states']),
                'source_plan_sha256': _sha(item['source'] / 'plan.npz'), 'source_trace_sha256': _sha(item['source'] / 'trace.npz'),
                'outcome': 'success', 'transfer_checks_passed': True, 'videos': video_paths,
                'video_sha256': hashes, 'provenance_files': provenance_files, 'provenance': meta})
            for key, value in columns.items():
                numeric[key].append(value)
            offset += n
        ep_path = temporary / 'meta/episodes/chunk-000/file-000.parquet'
        ep_path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(ep_rows), ep_path)
        stats = {key: _stats(np.concatenate(values)) for key, values in numeric.items()}
        stats.update({f'observation.images.{camera}': _image_stats(histogram[camera], offset) for camera in CAMERAS})
        _json(temporary / 'meta/stats.json', stats)
        splits = {'train': f'0:{train_count}'}
        if val_count:
            splits['val'] = f'{train_count}:{len(loaded)}'
        info = {'codebase_version': 'v3.0', 'robot_type': 'so101_follower', 'total_episodes': len(loaded),
                'total_frames': offset, 'total_tasks': len(tasks), 'chunks_size': 1000,
                'data_files_size_in_mb': 100, 'video_files_size_in_mb': 200,
                'fps': FPS, 'splits': splits, 'data_path': DATA_PATH, 'video_path': VIDEO_PATH, 'features': features}
        _json(temporary / 'meta/info.json', info)
        manifest = {'schema_version': 1, 'name': name, 'format': 'LeRobot v3.0', 'synthetic': True,
                    'physical_transfer_validated': False,
                    'state_alignment': 'pre-action state k paired with action k; terminal state excluded',
                    'units': {'arm': 'degrees', 'gripper': 'normalized 0–100'},
                    'requested_validation_fraction': validation_fraction,
                    'lineage': dict(lineage or {}),
                    'split_note': 'Inherited group assignments are preserved across revisions. Grouped variants never cross splits. Every task retains at least one training group; singleton task groups remain train-only. Validation may be smaller than requested to preserve task coverage. New group assignment is deterministic within this export.',
                    'validation_unavailable_tasks': sorted(task for task, task_group in task_groups.items() if not task_group & val_groups),
                    'provenance_note': 'Available generation settings, scene XML, native plans and traces are bundled with hashes. Model mesh and other assets referenced by scene XML may use external paths and are not bundled; rendering/re-simulation requires those original dependencies.',
                    'splits': {split: [p['episode_index'] for p in provenance if p['split'] == split] for split in ('train', 'val')},
                    'episodes': provenance}
        _json(temporary / 'export_manifest.json', manifest)
        audit = validate_export(temporary, expected_size=expected_size)
        _json(temporary / 'export_audit.json', audit)
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite {destination}')
        temporary.rename(destination)
        return {'path': str(destination), 'name': name, 'total_episodes': len(loaded), 'total_frames': offset,
                'splits': manifest['splits'], 'valid': True, 'format': 'LeRobot v3.0'}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def validate_export(root: Path, *, expected_size=(640, 480), official_reader=False) -> dict:
    """Re-read all rows and decode all video frames; reject stale temporal ranges."""
    root = Path(root)
    info = json.loads((root / 'meta/info.json').read_text())
    manifest = json.loads((root / 'export_manifest.json').read_text())
    if info['codebase_version'] != 'v3.0' or info['fps'] != FPS:
        raise ValueError('Invalid LeRobot version/fps')
    episodes = pq.read_table(sorted((root / 'meta/episodes').rglob('*.parquet'))).to_pylist()
    tasks = pq.read_table(root / 'meta/tasks.parquet').to_pylist()
    if len(episodes) != info['total_episodes'] or len(tasks) != info['total_tasks'] or len(manifest['episodes']) != len(episodes):
        raise ValueError('Episode/task totals disagree')
    if [task['task_index'] for task in tasks] != list(range(len(tasks))):
        raise ValueError('Task index mapping is not contiguous')
    members = []
    for split, indices in manifest['splits'].items():
        if indices:
            if info['splits'].get(split) != f'{indices[0]}:{indices[-1] + 1}' or indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError('Split ranges disagree with group membership')
        elif split in info['splits']:
            raise ValueError('Empty split must not declare a range')
        members.extend(indices)
    if sorted(members) != list(range(len(episodes))) or set(info['splits']) != {s for s, ids in manifest['splits'].items() if ids}:
        raise ValueError('Split membership must cover every episode once')
    offset, groups = 0, {}
    all_tasks, train_tasks = set(), set()
    for idx, (ep, provenance) in enumerate(zip(episodes, manifest['episodes'])):
        n = ep['length']
        if provenance['episode_index'] != idx or provenance['outcome'] != 'success' or provenance['transfer_checks_passed'] is not True:
            raise ValueError('Episode lacks successful outcome/provenance')
        files = provenance.get('provenance_files', {})
        if not {'plan.npz', 'trace.npz'} <= files.keys():
            raise ValueError('Missing portable plan/trace provenance')
        for filename, record in files.items():
            expected_path = f'provenance/episode-{idx:06d}/{filename}'
            if filename not in PROVENANCE_FILES or record['path'] != expected_path:
                raise ValueError('Invalid portable provenance path')
            copied = root / expected_path
            if not copied.is_file() or _sha(copied) != record['sha256']:
                raise ValueError(f'Portable provenance hash mismatch: {filename}')
        if files['plan.npz']['sha256'] != provenance['source_plan_sha256'] or files['trace.npz']['sha256'] != provenance['source_trace_sha256']:
            raise ValueError('Portable plan/trace provenance differs from source hashes')
        if ep['episode_index'] != idx or ep['dataset_from_index'] != offset or ep['dataset_to_index'] != offset + n:
            raise ValueError(f'Invalid episode bounds/ranges at episode {idx}')
        path = root / info['data_path'].format(chunk_index=ep['data/chunk_index'], file_index=ep['data/file_index'])
        data = pq.read_table(path).to_pydict()
        if len(data['action']) != n or data['episode_index'] != [idx] * n or data['index'] != list(range(offset, offset + n)) or data['frame_index'] != list(range(n)):
            raise ValueError('Data row count/indices disagree with episode bounds')
        for key in ('action', 'observation.state'):
            _native(data[key], n, key)
            if info['features'][key]['names'] != list(FEATURE_JOINT_NAMES):
                raise ValueError('Native joint names mismatch')
        if array_hash(np.asarray(data['action'])) != provenance['exported_action_hash']:
            raise ValueError('Exported action hash mismatch')
        if array_hash(np.asarray(data['observation.state'])) != provenance['exported_state_hash']:
            raise ValueError('Exported pre-action state hash mismatch')
        if not np.allclose(data['timestamp'], np.arange(n) / FPS, atol=1e-5, rtol=0):
            raise ValueError('Data timestamps mismatch')
        if any(t < 0 or t >= len(tasks) for t in data['task_index']):
            raise ValueError('Invalid task indices')
        if any(tasks[t]['task'] != provenance['provenance']['instruction'] for t in data['task_index']) or ep['tasks'] != [provenance['provenance']['instruction']]:
            raise ValueError('Task instruction mapping mismatch')
        for camera in CAMERAS:
            key = f'observation.images.{camera}'
            if ep[f'videos/{key}/from_timestamp'] != 0 or not np.isclose(ep[f'videos/{key}/to_timestamp'], n / FPS):
                raise ValueError('Video timestamp offset/bounds mismatch')
            relative = info['video_path'].format(video_key=key, chunk_index=ep[f'videos/{key}/chunk_index'], file_index=ep[f'videos/{key}/file_index'])
            video = root / relative
            _video(video, n, expected_size, collect_stats=False)
            if relative != provenance['videos'][camera] or _sha(video) != provenance['video_sha256'][camera]:
                raise ValueError('Video provenance mismatch')
        group, split = provenance['group_id'], provenance['split']
        if group in groups and groups[group] != split:
            raise ValueError('Group leakage across splits')
        groups[group] = split
        all_tasks.add(str(provenance['task_id']))
        if split == 'train':
            train_tasks.add(str(provenance['task_id']))
        if idx not in manifest['splits'][split]:
            raise ValueError('Split membership mismatch')
        offset += n
    if train_tasks != all_tasks:
        raise ValueError('Every selected task must retain a training group')
    if offset != info['total_frames']:
        raise ValueError('Total frame count mismatch')
    stats = json.loads((root / 'meta/stats.json').read_text())
    for key, feature in info['features'].items():
        shape = (3, 1, 1) if feature['dtype'] == 'video' else tuple(feature['shape'])
        for kind in ('mean', 'std', 'min', 'max'):
            value = np.asarray(stats[key][kind])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid statistics {key}/{kind}')
        if stats[key]['count'] != [offset]:
            raise ValueError('Statistics count mismatch')
    result = {'valid': True, 'total_frames': offset, 'total_episodes': len(episodes),
              'video_frames_decoded': offset * len(CAMERAS), 'official_reader_checked': False,
              'physical_transfer_validated': False}
    if official_reader:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset('local/sim-export', root=root, video_backend='pyav',
                                  delta_timestamps={'action': [-1 / FPS, 0., 1 / FPS]})
        for ep in episodes:
            path = root / info['data_path'].format(chunk_index=ep['data/chunk_index'], file_index=ep['data/file_index'])
            actions = np.asarray(pq.read_table(path, columns=['action']).to_pydict()['action'])
            for index, pad in ((ep['dataset_from_index'], [True, False, ep['length'] == 1]),
                               (ep['dataset_to_index'] - 1, [ep['length'] == 1, False, True])):
                sample = dataset[index]
                if sample['action_is_pad'].tolist() != pad or sample['action'].shape != (3, 6):
                    raise ValueError('Official reader temporal boundary mismatch')
                relative_index = index - ep['dataset_from_index']
                expected = actions[np.clip(np.array([-1, 0, 1]) + relative_index, 0, ep['length'] - 1)]
                if not np.allclose(sample['action'].numpy(), expected):
                    raise ValueError('Official reader action chunk crossed episode boundary')
                for camera in CAMERAS:
                    if tuple(sample[f'observation.images.{camera}'].shape) != (3, expected_size[1], expected_size[0]):
                        raise ValueError('Official reader image decode mismatch')
        result['official_reader_checked'] = True
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit'])
    parser.add_argument('path', type=Path)
    parser.add_argument('--official-reader', action='store_true')
    args = parser.parse_args()
    print(json.dumps(validate_export(args.path, official_reader=args.official_reader), indent=2))
