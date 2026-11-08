"""Aggregate per-episode Realistic calibrations into one shared profile.

Each sampled real episode is calibrated exactly as the Datasets Realistic view
does it (per camera object colors from hue-filtered real pixels, and a faint
soft shadow fitted to the Top camera). Generate data and Build Simulation
dataset have no real frames per episode, so they use these pooled targets.
"""
from pathlib import Path
import json, sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'scripts'))
import mujoco, numpy as np
from studio_episode_appearance import APPEARANCE_VERSION, EpisodeAppearance, MAX_SHADOW, RUN

TARGET = ROOT / 'calibration/real-appearance-v1/realistic-v2.json'


def pooled(stats):
    means = np.array([s['mean'] for s in stats]); stds = np.array([s['std'] for s in stats]); n = np.array([s['pixels'] for s in stats], float)
    w = n / n.sum(); mean = (w[:, None] * means).sum(0)
    # Pooled variance: within-episode spread plus between-episode spread.
    var = (w[:, None] * (stds ** 2 + (means - mean) ** 2)).sum(0)
    return {'mean': mean.tolist(), 'std': np.sqrt(var).tolist(), 'pixels': int(n.sum()), 'episodes': len(stats)}


def build(episodes):
    measured = {'top': {}, 'wrist': {}}; shadows = []
    for index in episodes:
        if not (RUN / f'episode-{index:02d}' / 'report.json').is_file(): continue
        e = EpisodeAppearance(index, 'realistic')
        with mujoco.Renderer(e.model, height=480, width=640) as renderer: e.calibrate(renderer)
        cache = json.loads((e.work / f'episode-calibration-v{APPEARANCE_VERSION}.json').read_text())
        shadows.append(cache['shadow'] | {'episode': index})
        for view, kinds in cache['measured'].items():
            for kind in kinds: measured[view].setdefault(kind, []).append(cache['profile'][view][kind]['real'])
        print(index, cache['shadow'], cache['measured'], flush=True)
    appearance = {view: {kind: pooled(stats) for kind, stats in kinds.items() if len(stats) >= 3} for view, kinds in measured.items()}
    # Light direction: the candidate with the largest summed correlation.
    votes = {}
    for s in shadows:
        if s['correlation'] > .05: votes[(s['azimuth_degrees'], s['elevation_degrees'])] = votes.get((s['azimuth_degrees'], s['elevation_degrees']), 0) + s['correlation']
    azimuth, elevation = max(votes, key=votes.get) if votes else (0., 85.)
    good = [s for s in shadows if s['correlation'] > .05]
    shadow = {'azimuth_degrees': azimuth, 'elevation_degrees': elevation,
              'blur_sigma_px': float(np.median([s['blur_sigma_px'] for s in good])) if good else 16.,
              'strength': float(min(MAX_SHADOW, np.median([s['strength'] for s in good]))) if good else .1,
              'episodes': shadows}
    result = {'schema_version': 2, 'episodes': [s['episode'] for s in shadows], 'appearance': appearance, 'shadow': shadow,
              'method': 'Per-episode Datasets Realistic calibrations pooled across sampled real episodes: per camera Lab targets '
                        '(pixel-weighted mean, pooled std) and the shadow light direction with the largest summed correlation, '
                        'median blur and strength (capped at %.2f).' % MAX_SHADOW}
    TARGET.write_text(json.dumps(result, indent=1))
    return result


if __name__ == '__main__':
    result = build(range(0, 72, 6))
    print(json.dumps({k: v for k, v in result['shadow'].items() if k != 'episodes'}), {v: list(k) for v, k in result['appearance'].items()})
