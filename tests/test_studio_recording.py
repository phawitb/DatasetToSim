import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_recording import causal_frame_indices


def test_irregular_real_camera_frames_use_past_frame_on_simulation_clock():
    np.testing.assert_array_equal(causal_frame_indices([0,.041,.067,.105],[0,.033,.066,.1,.133]),[0,0,1,2,3])


def test_recording_tail_is_bounded_by_last_simulated_frame():
    from studio_recording import comparison_times
    times=comparison_times(8.3,241)
    assert len(times)==241 and times[-1]==8.
