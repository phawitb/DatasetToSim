"""Encoder evidence at a simulated, stationary tape-grasp checkpoint.

This rejects empty/over-open closure before lift. Width agreement is evidence of
contact, not proof of a correct object or of final placement success.
"""
import numpy as np

HOLD_SECONDS = .35
OBSERVE_SECONDS = .2


def checkpoint_index(actions, timestamps, pose):
    matches=np.flatnonzero(np.all(np.isclose(actions,pose,atol=1e-10,rtol=0),axis=1))
    for run in np.split(matches,np.flatnonzero(np.diff(matches)>1)+1):
        if len(run) and timestamps[run[-1]]-timestamps[run[0]]>=HOLD_SECONDS:
            return int(run[-1])
    raise ValueError('Grasp check needs a stationary hold before lift; generate a new plan')


def grasp_check(plan, actions, timestamps, expected):
    world=plan.get('world',{})
    tape=(world.get('pickplace',{}).get('grasp',{}).get('pads')=='fixed_in_hole'
          or world.get('cube',{}).get('object')=='tape')
    if not tape:return None
    end=plan.get('grasp_check_index')
    if not isinstance(end,int) or isinstance(end,bool) or not 0<end<len(actions)-1:
        raise ValueError('Grasp check missing; generate and simulate a new tape plan')
    expected=np.asarray(expected,float) if expected is not None else np.empty((0,6))
    if expected.shape!=actions.shape or not np.isfinite(expected).all():
        raise ValueError('Grasp check requires the matching simulated joint trace')
    start=end
    while start>0 and np.allclose(actions[start-1],actions[end],atol=1e-10,rtol=0):start-=1
    if timestamps[end]-timestamps[start]<HOLD_SECONDS:
        raise ValueError('Grasp check must follow a stationary closed-jaw hold')
    indices=np.arange(start,end+1)[timestamps[start:end+1]>=timestamps[end]-OBSERVE_SECONDS-1e-9]
    gap=expected[indices,5]-actions[indices,5]
    if len(indices)<6 or np.any(gap<3):
        raise ValueError('Grasp check: simulated tape grip is not distinguishable from empty closure')
    # Native units are calibrated 0–100 gripper readings, not millimetres.
    lower=actions[indices,5]+np.maximum(2.,.45*gap)
    upper=expected[indices,5]+np.maximum(3.,.5*gap)
    return {'index':end,'indices':indices.tolist(),'lower':lower.tolist(),'upper':upper.tolist(),
            'expected':expected[indices,5].tolist()}


def verify_grasp(check, samples):
    observed={s['index']:s for s in samples}
    selected=[observed.get(i,{}) for i in check['indices']]
    times=np.asarray([s.get('observed_at',np.nan) for s in selected],float)
    values=np.asarray([s.get('measured',[np.nan]*6)[5] for s in selected],float)
    if not np.isfinite(values).all():
        raise RuntimeError('Grasp not verified: missing gripper readings; stopped before lift')
    # Catch-up scheduling can compress nominal 30 Hz samples after a delayed read.
    # Allow 20 ms sampling jitter, not the worker's full 150 ms deadline tolerance.
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0) or times[-1]-times[0]<OBSERVE_SECONDS-.02:
        raise RuntimeError('Grasp not verified: contact observation was too short; stopped before lift')
    if np.any(values<check['lower']):
        raise RuntimeError('Grasp failed: jaws closed without the tape; stopped before lift')
    if np.any(values>check['upper']):
        raise RuntimeError('Grasp failed: jaw opening differs from simulated tape contact; stopped before lift')
    return {'passed':True,'index':check['index'],'measured':values.tolist(),'expected':check['expected']}
