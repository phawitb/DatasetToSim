"""Live speed profiles must be revalidated at simulation and execution."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))


def recorded_plan():
    from studio_generation_timing import recorded_timing
    from studio_timing import timing_contract_hash
    reference=recorded_timing(1.)
    plan={'source':'live','timing_profile':'recorded','speed_scale':1.,'motion':{'timing_reference':reference}}
    plan['timing_contract_hash']=timing_contract_hash(plan)
    return plan


def test_live_profile_is_fixed_and_requires_matching_simulation():
    from studio_timing import validated_motion_limits
    p=recorded_plan()
    limits=validated_motion_limits(p)
    assert np.all(np.asarray(limits)>[35]*5+[50])
    with pytest.raises(ValueError,match='simulated timing profile'):
        validated_motion_limits(p,require_simulated=True)
    p.update(simulated_timing_profile='recorded',simulated_timing_contract_hash=p['timing_contract_hash'])
    np.testing.assert_array_equal(validated_motion_limits(p,require_simulated=True),limits)


def test_timing_changes_or_forged_envelope_are_rejected():
    from studio_timing import validated_motion_limits
    for change in ['limits','profile_hash','speed','contract']:
        p=recorded_plan()
        if change=='limits':p['motion']['timing_reference']['limits'][0][0]=999
        elif change=='profile_hash':p['motion']['timing_reference']['profile_sha256']='changed'
        elif change=='speed':p['speed_scale']=.5
        else:p['timing_contract_hash']='changed'
        with pytest.raises(ValueError):validated_motion_limits(p)


def test_legacy_limits_remain_default_and_home_cannot_use_fast_profile():
    from studio_timing import validated_motion_limits
    assert validated_motion_limits({'source':'live'}) is None
    p=recorded_plan();p['purpose']='home'
    with pytest.raises(ValueError,match='Home'):validated_motion_limits(p)
    with pytest.raises(ValueError):validated_motion_limits({'timing_profile':'unknown'})
