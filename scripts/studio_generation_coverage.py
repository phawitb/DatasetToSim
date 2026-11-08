"""Read-only accepted-episode coverage from saved initial scene geometry.

Only small scene JSON files are inspected; videos, traces and plans are never
opened. The bounded cache invalidates whenever a scene's size or mtime changes.
"""
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path


def task_counts(job):
    """Include accepted inherited tasks even when absent from batch options."""
    counts = Counter(item.get('task_id', 'unknown') for item in job.get('episodes', [])
                     if item.get('status') == 'accepted')
    tasks = dict.fromkeys([*job.get('options', {}).get('tasks', []), *counts])
    targets = job.get('options', {}).get('target_counts') or {}
    return [{'task_id': task, 'accepted': counts[task], 'target': targets.get(task)} for task in tasks]


def _xy(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value[:2]):
        return None
    return [float(v) for v in value[:2]]


@lru_cache(maxsize=8192)
def _scene(path, mtime_ns, size):
    # Signature arguments are intentional cache invalidators.
    try:
        data = json.loads(Path(path).read_text())
        world = data.get('world', {})
        return world if isinstance(world, dict) else {}
    except (OSError, ValueError, AttributeError):
        return {}


def _world(folder):
    # generation-scene is authoritative; config.world supports older exports.
    for name in ('generation-scene.json', 'generation-config.json'):
        path = folder / name
        try:
            stat = path.stat()
        except OSError:
            continue
        world = _scene(str(path), stat.st_mtime_ns, stat.st_size)
        if world:
            return world, name
    return {}, None


def _position(world, key):
    item = world.get(key)
    return _xy(item.get('position')) if isinstance(item, dict) else None


def _base_yaw(world):
    base = world.get('robot_base')
    q = base.get('quaternion') if isinstance(base, dict) else None
    if not isinstance(q, (list, tuple)) or len(q) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in q):
        return None
    norm = math.sqrt(sum(v*v for v in q))
    if norm < 1e-10:
        return None
    w, x, y, z = (v/norm for v in q)
    return math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def _bounds(world):
    mat = world.get('mat', {})
    if not isinstance(mat, dict):
        return None
    center, size = _xy(mat.get('position')), _xy(mat.get('size'))
    if center is None or size is None or min(size) <= 0:
        return None
    return [[center[i] - size[i] / 2 for i in range(2)],
            [center[i] + size[i] / 2 for i in range(2)]]


def zone_for(position, bounds):
    """Row-major full-mat cells, x increases first; outliers are not clamped."""
    if position is None or bounds is None:
        return None
    low, high = bounds
    if any(position[i] < low[i] - 1e-10 or position[i] > high[i] + 1e-10 for i in range(2)):
        return None
    cells = [min(2, max(0, int((position[i] - low[i]) / (high[i] - low[i]) * 3))) for i in range(2)]
    return cells[1] * 3 + cells[0]


def generation_coverage(folder, job=None):
    """Endpoint payload in world XY millimeters; episode_index uses job journal indices."""
    folder = Path(folder)
    if job is None:
        job = json.loads((folder / 'job.json').read_text())
    records = []
    bounds = None
    different_mats = False
    for index, episode in enumerate(job.get('episodes', [])):
        if episode.get('status') != 'accepted':
            continue
        world, filename = _world(folder / 'episodes' / f'{index:05d}')
        mat = _bounds(world)
        if mat is not None:
            if bounds is None:
                bounds = mat
            elif any(abs(mat[j][i] - bounds[j][i]) > 1e-8 for j in range(2) for i in range(2)):
                different_mats = True
        task = episode.get('task_id', 'unknown')
        key = 'placement' if task == 'unstack' else 'bowl' if task.endswith('_to_green_bowl') else 'target_cube'
        source, destination = _position(world, 'cube'), _position(world, key)
        base = _position(world, 'robot_base')
        records.append({'episode_index': index, 'task_id': task,
                        'instruction': episode.get('instruction', task),
                        'inherited': bool(episode.get('inherited_from_job_id')),
                        'source_xy_mm': [v * 1000 for v in source] if source else None,
                        'destination_xy_mm': [v * 1000 for v in destination] if destination else None,
                        'destination_kind': key, 'metadata_file': filename,
                        'robot_base_xy_mm': [v * 1000 for v in base] if base else None,
                        'robot_base_yaw_deg': _base_yaw(world)})
    bounds_mm = [[v * 1000 for v in row] for row in bounds] if bounds else None
    tasks = task_counts(job)
    summaries = {row['task_id']: row for row in tasks}
    for row in tasks:
        row.update(source_counts=[0] * 9, destination_counts=[0] * 9,
                   source_missing=0, destination_missing=0, source_outside=0, destination_outside=0)
    for record in records:
        row = summaries[record['task_id']]
        for role in ('source', 'destination'):
            point = record[f'{role}_xy_mm']
            zone = zone_for(point, bounds_mm)
            record[f'{role}_zone'] = zone
            if point is None:
                row[f'{role}_missing'] += 1
            elif zone is None:
                row[f'{role}_outside'] += 1
            else:
                row[f'{role}_counts'][zone] += 1
    return {'job_id': job.get('id', folder.name), 'accepted': len(records),
            'units': 'mm', 'coordinate_system': 'Saved scene world XY; initial object centers',
            'grid_shape': [3, 3], 'mat_bounds_xy_mm': bounds_mm,
            'different_mats': different_mats, 'tasks': tasks, 'episodes': records,
            'notes': ['Source is the initial pick-cube center. Destination is the initial target-cube or bowl center; unstack uses the saved empty-mat placement.',
                      'Inherited and new accepted episodes are included. Positions are read from saved scene geometry, not trajectory endpoints.']}
