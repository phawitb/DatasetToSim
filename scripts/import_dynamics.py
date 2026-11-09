"""Dataset-level arm motor identification for an imported dataset (code only).

The shared motor model was identified on SIMCUBE. A new recording can move
faster/slower or under different loads, so the five arm joints are refitted
here: per joint kp, kv, armature, frictionloss and actuation delay, by
coordinate-wise Nelder-Mead on recorded actions -> simulated state vs the
recorded state (soft-L1, degrees). Training uses a subset of episodes; the
remaining episodes are a held-out check, and the result is only accepted
when the held-out error improves. The gripper keeps the shared model.
Recorded actions are never changed; no task outcome enters the objective.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from dataset_to_sim.calibration import Calibration  # noqa: E402
from dataset_to_sim.strict_replay import NativeJointAdapter  # noqa: E402
import import_replay as ir  # noqa: E402

KEYS = ('kp', 'kv', 'armature', 'frictionloss', 'delay_s')
BOUNDS = [(np.log(3), np.log(200)), (np.log(.015), np.log(8)), (np.log(.0001), np.log(.15)), (0, .4), (0, .12)]


def fast_trace(model, episode, adapter):
    """Free integration of the recorded actions (no contact bookkeeping); returns joint states per frame."""
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data); mujoco.mj_resetData(model, data)
    data.qpos[:6] = adapter.to_model(episode.state_deg[0]); target = adapter.to_model(episode.action_deg)
    for j, n in enumerate(model.actuator_history[:, 0]):
        if n: mujoco.mj_initCtrlHistory(model, data, j, None, np.full((int(n), 1), data.qpos[j]))
    mujoco.mj_forward(model, data); states = []; dt = float(model.opt.timestep)
    try:
        for i, t in enumerate(target):
            states.append(data.qpos[:6].copy()); data.ctrl[:] = t
            end = float(episode.timestamps[i + 1]) if i + 1 < len(target) else float(episode.timestamps[i] + 1 / 30)
            n = int((end - data.time + 1e-10) / dt)
            if n: mujoco.mj_step(model, data, nstep=n)
            remaining = end - data.time
            if remaining > 1e-10:
                model.opt.timestep = remaining; mujoco.mj_step(model, data); model.opt.timestep = dt
    finally:
        model.opt.timestep = dt
    return np.array(states)


def set_motor(model, joint, x):
    kp, kv, armature, friction, delay = np.exp(x[0]), np.exp(x[1]), np.exp(x[2]), x[3], x[4]
    model.actuator_gainprm[joint, 0] = kp; model.actuator_biasprm[joint, 1] = -kp; model.actuator_biasprm[joint, 2] = -kv
    model.dof_armature[joint] = armature; model.dof_frictionloss[joint] = friction
    # The delay buffer length is fixed at build time; keep inside its capacity.
    capacity = model.actuator_history[joint, 0] * model.opt.timestep * .9
    model.actuator_delay[joint] = min(delay, capacity)


def identify(calibration, folder, log=print, train_count=6, budget=70):
    config = ir.replay_config(ir.ic.load_config(), calibration); adapter = NativeJointAdapter(**config['adapter'])
    from dataset_to_sim.dataset import DatasetIndex
    dataset = DatasetIndex(calibration['dataset'])
    episodes = sorted(int(k) for k in calibration['episode_poses'])
    if len(episodes) < 4: return None, {'used': False, 'reason': 'too few episodes'}
    train = episodes[::max(1, len(episodes) // train_count)][:train_count]; held = [e for e in episodes if e not in train][:8]

    def items(ids):
        out = []
        for index in ids:
            cal = ir.episode_calibration(config, calibration['episode_poses'][str(index)], calibration['manipulated'], calibration['container'])
            out.append((dataset.episode(index), ir.scene_model(cal, config, calibration)))
        return out
    training, holdout = items(train), items(held)
    dynamics = copy.deepcopy(config['identified_dynamics'])

    def error(item_list):
        values = []
        for episode, model in item_list:
            q = adapter.to_native(fast_trace(model, episode, adapter))
            values.append(np.sqrt(np.mean((q[8:, :5] - episode.state_deg[8:, :5]) ** 2, axis=0)))
        return np.mean(values, axis=0)

    before_train, before_held = error(training), error(holdout)
    log(f'motor model: arm RMSE before {before_train.mean():.3f} deg (train), {before_held.mean():.3f} deg (held-out)')
    history = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for joint in (1, 2, 3, 0, 4):
            best = [np.inf, None]

            def one(item, joint=joint):
                episode, model = item
                q = adapter.to_native(fast_trace(model, episode, adapter))
                e = q[8:, joint] - episode.state_deg[8:, joint]
                return float(np.mean(2 * (np.sqrt(1 + e ** 2) - 1)))

            def objective(x, joint=joint):
                for _, model in training: set_motor(model, joint, x)
                value = float(np.mean(list(pool.map(one, training))))
                if value < best[0]: best[:] = value, x.copy()
                return value
            x0 = np.array([np.log(dynamics['kp'][joint]), np.log(dynamics['kv'][joint]), np.log(dynamics['armature'][joint]),
                           dynamics['frictionloss'][joint], dynamics['delay_s'][joint]])
            start = objective(x0); simplex = np.tile(x0, (6, 1)); simplex[1:] += np.diag([.25, .3, .3, .03, .012])
            minimize(objective, x0, method='Nelder-Mead', bounds=BOUNDS,
                     options={'maxfev': budget, 'initial_simplex': simplex, 'xatol': .001, 'fatol': 1e-5})
            x = best[1]
            for _, model in training + holdout: set_motor(model, joint, x)
            for key, value in zip(KEYS, [np.exp(x[0]), np.exp(x[1]), np.exp(x[2]), x[3], x[4]]): dynamics[key][joint] = float(value)
            history.append({'joint': joint, 'loss_before': start, 'loss_after': best[0]})
            log(f'motor model: joint {joint} loss {start:.3f} -> {best[0]:.3f}')
    after_train, after_held = error(training), error(holdout)
    report = {'training_episodes': train, 'held_out_episodes': held, 'joints': history,
              'arm_rmse_deg_before': {'train': before_train.tolist(), 'held_out': before_held.tolist()},
              'arm_rmse_deg_after': {'train': after_train.tolist(), 'held_out': after_held.tolist()},
              'method': 'coordinate-wise Nelder-Mead on kp, kv, armature, frictionloss, delay per arm joint; recorded actions -> simulated vs recorded state; gripper unchanged',
              'used': bool(after_held.mean() < before_held.mean() - .02)}
    log(f'motor model: arm RMSE after {after_train.mean():.3f} deg (train), {after_held.mean():.3f} deg (held-out) · ' + ('accepted' if report['used'] else 'kept shared model'))
    return (dynamics if report['used'] else None), report
