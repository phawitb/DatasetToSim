"""Observation selectors for offline identification, never replay control."""
import numpy as np


def held_gripper_frames(state, action):
    """Stable blocked native gripper observations, including openings below25.

    This is evidence of a blocked gripper, not a physical success oracle. A
    closing transient must not substitute for the stable object-holding plateau.
    Thresholds are native0–100 units per recorded frame, for this30Hz dataset.
    """
    state=np.asarray(state,dtype=float);action=np.asarray(action,dtype=float)
    if state.ndim!=1 or action.shape!=state.shape:
        raise ValueError('Matching one-dimensional native gripper arrays required')
    if len(state)<3:return np.zeros(len(state),dtype=bool)
    stable=np.abs(np.gradient(state))<.25
    candidate=(state>3)&(state<18)&(action<3)&(state-action>3)&stable
    # Require three successive samples, so braking through a value is excluded.
    mask=np.zeros(len(state),dtype=bool)
    boundaries=np.flatnonzero(np.diff(np.r_[False,candidate,False]))
    for start,end in zip(boundaries[::2],boundaries[1::2]):
        if end-start>=3:mask[start:end]=True
    return mask
