"""Measured motion statistics. Live use requires the Studio timing contract."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'calibration/generation-recorded-pace.json'


def generation_timing(capture, speed_scale):
    if capture.get('source') != 'synthetic':
        return None
    name = capture.get('generation_options', {}).get('timing_profile', 'conservative')
    if name not in ('recorded', 'conservative'):
        raise ValueError('Choose recorded or conservative generation timing')
    if name == 'conservative':
        return None
    return recorded_timing(speed_scale)


def recorded_timing(speed_scale):
    from dataset_to_sim.studio import validate_speed_scale
    speed_scale = validate_speed_scale(speed_scale)
    raw = PROFILE.read_bytes()
    reference = json.loads(raw)
    limits = np.asarray(reference['derivative_limits'], float)
    if limits.shape != (3, 6) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError('Invalid recorded motion profile')
    limits[0, :5] = np.minimum(limits[0, :5] * speed_scale, [100., 90., 75., 100., 100.])
    return {'name': 'recorded', 'reference': reference, 'profile_sha256': hashlib.sha256(raw).hexdigest(),
            'limits': limits.tolist(), 'tracking_velocity_ceiling_deg_s': [100., 90., 75., 100., 100.],
            'note': 'Recorded derivative envelope capped per joint at 100/90/75/100/100 deg/s for the unchanged 10-degree simulated tracking check. All episodes still require physical validation.'}


def generation_durations(waypoints, names):
    """Shorter task holds; derivative limits determine actual travel duration."""
    holds = {'Grasp settle': .15, 'Grasp': .45, 'Placement settle': .3,
             'Release': .45, 'Release on cube': .45, 'Release on table': .45,
             'Final settle': 1.0}
    return [(q, holds.get(name, seconds * .20)) for (q, seconds), name in zip(waypoints, names)]


def measure_reference(root):
    """Rebuild the profile from native 30 Hz commands, grouped by episode identity."""
    from dataset_to_sim.dataset import DatasetIndex
    from scipy.signal import savgol_filter
    root = Path(root)
    dataset = DatasetIndex(root)
    episodes = [dataset.episode(i) for i in range(dataset.episode_count)]
    # Nine-frame local cubic derivatives suppress encoder quantization. No
    # cross-episode differences or metadata-global index ranges are used.
    derivatives = [np.percentile(np.concatenate([
        abs(savgol_filter(e.action_deg, 9, 3, deriv=order, delta=1/30, axis=0)[4:-4])
        for e in episodes]), 95, axis=0).tolist() for order in (1, 2, 3)]
    files = [root/'meta/info.json', *sorted((root/'data').glob('chunk-*/*.parquet'))]
    return {'schema_version': 1, 'dataset': root.name, 'episodes': len(episodes), 'fps': dataset.fps,
            'task_coverage': ['Put the red cube in the green bowl.'],
            'median_duration_s': float(np.median([e.frame_count/30 for e in episodes])),
            'duration_p10_p90_s': np.percentile([e.frame_count/30 for e in episodes], [10,90]).tolist(),
            'derivative_limits': derivatives,
            'method': 'Per-joint 95th percentile absolute native-command derivatives; Savitzky-Golay 9-frame cubic, interior samples only. Degrees for arm; 0–100 gripper units. Simulation-only timing heuristic, not hardware limits or a fitted human motor model.',
            'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, default=PROFILE)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure_reference(args.dataset), indent=2)+'\n')
