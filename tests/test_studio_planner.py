import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_planner import interpolate_native
from dataset_to_sim.studio import validate_trajectory


def test_interpolation_preserves_endpoints_and_obeys_speed_limits():
    start=np.array([0,0,0,0,0,5.]); end=np.array([80,-40,20,10,-15,60.])
    q,t=interpolate_native([(start,.5),(end,1.)])
    np.testing.assert_array_equal(q[0],start)
    np.testing.assert_allclose(q[-1],end)
    validate_trajectory(q,t,np.array([[-180,180]]*5+[[0,100]]))
    assert t[-1]>3
