"""Causal camera observation timing, separate from unchanged actuator commands."""
import numbers
import numpy as np


def camera_sample_indices(frame_count,delay_frames=0):
    """Use an earlier integrated state; unavailable prehistory holds state zero.

    Delays are whole frames at the source 30 Hz. No future state, image blending,
    or joint/cube pose correction is used. Fractional fit estimates must be
    explicitly rounded and documented by the caller's frozen configuration.
    """
    if not isinstance(delay_frames,numbers.Integral) or delay_frames<0:
        raise ValueError('Camera latency must be a nonnegative integer frame count')
    if not isinstance(frame_count,numbers.Integral) or frame_count<1:
        raise ValueError('Camera trace requires at least one frame')
    return np.maximum(np.arange(frame_count)-delay_frames,0)
