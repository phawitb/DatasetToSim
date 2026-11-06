import math

import numpy as np
import pytest

from dataset_to_sim.joints import JointConvention, JointRangeError


def test_converts_recorded_degrees_to_model_radians_with_explicit_offsets():
    convention = JointConvention(offset_deg=[1, 0, -2, 0, 0, 0])

    actual = convention.to_radians([89, -90, 2, 180, 0, 10])

    np.testing.assert_allclose(actual, [math.pi / 2, -math.pi / 2, 0, math.pi, 0, math.pi / 18])


def test_range_check_names_the_out_of_range_joint():
    convention = JointConvention()

    with pytest.raises(JointRangeError, match="shoulder_pan"):
        convention.validate_model_ranges([200, 0, 0, 0, 0, 0], [[-1.9, 1.9]] * 6, frame=12)
