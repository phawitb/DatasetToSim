import numpy as np
import pytest
from dataset_to_sim.observations import camera_sample_indices


def test_camera_latency_uses_only_recorded_past_physical_states():
    np.testing.assert_array_equal(camera_sample_indices(6,2),[0,0,0,1,2,3])
    np.testing.assert_array_equal(camera_sample_indices(6,0),np.arange(6))
    with pytest.raises(ValueError):camera_sample_indices(6,-1)
    with pytest.raises(ValueError):camera_sample_indices(6,1.5)
