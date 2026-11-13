"""Pick-and-place benchmark: random workcell scenes and round trips, plan + contact simulation.

    python scripts/pickplace_bench.py SCENES SEED OUT.json [SPEED]   (WORKERS=10 by default)

Each scene places 2-4 modelled objects at random on the mat (away from the arm's Home
pose), picks one and puts it on a sensible destination (another object or an empty
spot). A passing leg is followed by the way back: the object is picked from where the
simulation left it and returned to its first place. Every leg is planned and simulated
exactly as New Run does (prepare_plan + simulate), so the numbers compare planners.
A "refused" leg is a planner ValueError with a reason; the robot would not move.
"""
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
CONFIG = json.loads((ROOT / 'calibration/native-action-current.json').read_text())
HOME = json.loads((ROOT / 'calibration/studio-home.json').read_text())
BASE = np.array(CONFIG['base_calibration']['world']['robot_base']['position'][:2])
# Sensible destinations for each picked kind (a cup is not set on a cube).
ALLOWED = {'bowl': (), 'cup': (), 'pen': ('cup',), 'tape': ('nuc', 'bowl'),
           'cube': ('cube', 'eraser', 'tape', 'nuc', 'bowl', 'cup'), 'eraser': ('nuc', 'bowl', 'cube')}


def under_home(xy, r):
    return xy[0] < .27 and abs(xy[1]) < .075 + r


def scenes(n, seed):
    from pickplace import load_registry, footprint_radius
    reg = load_registry(); rng = np.random.default_rng(seed); out = []
    pool = ['cube', 'cube', 'bowl', 'tape', 'eraser', 'pen', 'cup', 'nuc']
    while len(out) < n:
        kinds = list(rng.choice(pool, rng.integers(2, 5), replace=False))
        objs = []; ok = True
        for i, kind in enumerate(kinds):
            r0 = footprint_radius(reg['objects'][kind]['shape']) if kind != 'pen' else .03
            for _ in range(300):
                ang = rng.uniform(-1.1, 1.1); rad = rng.uniform(.17, .32)
                xy = BASE + rad * np.array([np.cos(ang), np.sin(ang)])
                if not (.06 < xy[0] < .41 and -.18 < xy[1] < .18) or under_home(xy, r0):
                    continue
                if all(np.linalg.norm(xy - o['_xy']) > r0 + o['_r'] + .025 for o in objs):
                    break
            else:
                ok = False; break
            objs.append({'id': f'o{i + 1}', 'kind': kind, 'position': [*xy, 0.], 'support': 'mat',
                         'yaw': float(rng.uniform(-np.pi, np.pi)) if kind in ('cube', 'eraser', 'pen', 'nuc') else 0.,
                         'color': ['red', 'green', 'blue', 'yellow'][i % 4] if kind == 'cube' else None, '_xy': xy, '_r': r0})
        if not ok:
            continue
        pickable = [o for o in objs if reg['objects'][o['kind']]['grasps']]
        if not pickable:
            continue
        pick = pickable[rng.integers(len(pickable))]
        dests = [o for o in objs if o is not pick and o['kind'] in ALLOWED.get(pick['kind'], ())]
        if dests and rng.random() < .7:
            place = {'object': dests[rng.integers(len(dests))]['id']}
        else:
            for _ in range(300):
                ang = rng.uniform(-1.0, 1.0); rad = rng.uniform(.18, .30); xy = BASE + rad * np.array([np.cos(ang), np.sin(ang)])
                if (.07 < xy[0] < .40 and -.16 < xy[1] < .16) and not under_home(xy, .035) and \
                        all(np.linalg.norm(xy - o['_xy']) > o['_r'] + pick['_r'] + .03 for o in objs if o is not pick):
                    break
            else:
                continue
            place = {'area': [float(xy[0]), float(xy[1])]}
        for o in objs:
            o.pop('_xy'); o.pop('_r')
        out.append({'objects': objs, 'pick': pick['id'], 'place': place})
    return out


def run(scene, speed=1.5, technique='natural', tag='x'):
    import studio_planner as P
    folder = Path(tempfile.mkdtemp(prefix='bench-')) / tag; folder.mkdir()
    (folder / 'capture.json').write_text(json.dumps({'source': 'live', 'task': 'pickplace', 'initial_native': HOME['home'],
                                                    'calibration_hash': CONFIG['follower_calibration_sha256'], 'captured_at': time.time(), 'id': tag}))
    (folder / 'pickplace-scene.json').write_text(json.dumps(scene))
    (folder / 'planning-options.json').write_text(json.dumps({'motion_technique': technique, 'speed_scale': speed, 'timing_profile': 'recorded'}))
    t0 = time.time(); rec = {'tag': tag, 'folder': str(folder)}
    try:
        P.prepare_plan(folder, preview=False); t1 = time.time()
        r = P.simulate(folder, render=False); t2 = time.time()
        rec.update(outcome=r['outcome'], passed=bool(r['outcome'] == 'success' and r['transfer_checks_passed']),
                   failures=r['transfer_check_failures'], duration=r['duration'], speed=r.get('speed_scale'),
                   plan_s=t1 - t0, sim_s=t2 - t1, assess=r.get('pickplace_assessment'))
        rec['final'] = np.load(folder / 'trace.npz')['cube_pose'][-1].tolist()
    except Exception as e:
        rec.update(outcome='refused' if isinstance(e, ValueError) else 'error', passed=False, error=str(e)[:300], plan_s=time.time() - t0)
    return rec


def job(args):
    i, scene, speed = args
    out = {'i': i, 'scene': scene, 'go': run(scene, speed, tag=f's{i}')}
    a = out['go']
    if a.get('passed'):
        back = json.loads(json.dumps(scene)); po = next(o for o in back['objects'] if o['id'] == scene['pick'])
        q = a['final']; po['position'] = [q[0], q[1], 0.]
        po['yaw'] = float(2 * np.arctan2(q[6], q[3])) if po['kind'] in ('cube', 'eraser', 'pen', 'nuc') else 0.
        if 'object' in scene['place']:
            po['support'] = scene['place']['object']
        first = next(o for o in scene['objects'] if o['id'] == scene['pick'])
        back['place'] = {'area': first['position'][:2]}
        out['back'] = run(back, speed, tag=f's{i}b')
    return out


def summary(results):
    legs = [r[k] for r in results for k in ('go', 'back') if k in r]
    ok = [l for l in legs if l.get('passed')]; refused = [l for l in legs if l.get('outcome') == 'refused']
    lines = [f"legs {len(legs)}: passed {len(ok)}, refused {len(refused)}, failed {len(legs) - len(ok) - len(refused)}"
             f"  (go {sum(r['go'].get('passed', False) for r in results)}/{len(results)}, back {sum(r['back'].get('passed', False) for r in results if 'back' in r)}/{sum('back' in r for r in results)})"]
    if ok:
        lines.append(f"mean motion {np.mean([l['duration'] for l in ok]):.1f} s, mean plan {np.mean([l['plan_s'] for l in legs]):.1f} s")
    lines.append(str(Counter((l.get('outcome'), (l.get('failures') or [''])[0][:45], (l.get('error') or '')[:70])
                             for l in legs if not l.get('passed')).most_common(12)))
    return '\n'.join(lines)


if __name__ == '__main__':
    from multiprocessing import Pool
    n, seed, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    speed = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
    with Pool(int(os.environ.get('WORKERS', '10'))) as pool:
        results = pool.map(job, [(i, s, speed) for i, s in enumerate(scenes(n, seed))], chunksize=1)
    Path(name).write_text(json.dumps(results, indent=1, default=float))
    print(summary(results))
