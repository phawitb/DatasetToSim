"""Fixed motion-profile contract shared by planning, simulation and hardware."""
import hashlib
import json
import numpy as np
from dataset_to_sim.studio import validate_speed_scale
from studio_generation_timing import recorded_timing


def validate_timing_profile(value):
    if not isinstance(value,str) or value not in ('conservative','recorded'):
        raise ValueError('Choose recorded or conservative motion timing')
    return value


def timing_contract_hash(plan):
    profile=validate_timing_profile(plan.get('timing_profile','conservative'))
    speed=validate_speed_scale(plan.get('speed_scale',1.))
    if profile=='conservative':return None
    if plan.get('purpose')=='home':raise ValueError('Home uses conservative motion timing')
    if plan.get('source') not in ('live','synthetic','dataset'):raise ValueError('Unknown motion source')
    expected=recorded_timing(speed)
    saved=plan.get('motion',{}).get('timing_reference',{})
    if saved.get('profile_sha256')!=expected['profile_sha256']:
        raise ValueError('Motion timing reference changed; generate and simulate a new plan')
    actual=np.asarray(saved.get('limits',[]),float)
    if actual.shape!=(3,6) or not np.array_equal(actual,np.asarray(expected['limits'])):
        raise ValueError('Motion timing limits changed; generate and simulate a new plan')
    contract={'profile':profile,'speed_scale':speed,'profile_sha256':expected['profile_sha256'],
              'limits':expected['limits']}
    return hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validated_motion_limits(plan, *, require_simulated=False):
    """Never accept velocity limits supplied by a browser or saved report alone."""
    profile=validate_timing_profile(plan.get('timing_profile','conservative'))
    if profile=='conservative':
        if plan.get('motion',{}).get('timing_profile')=='recorded' or plan.get('simulated_timing_profile')=='recorded':
            raise ValueError('Motion timing profile changed; generate and simulate a new plan')
        return None
    digest=timing_contract_hash(plan)
    if plan.get('timing_contract_hash')!=digest:
        raise ValueError('Motion timing contract changed; generate and simulate a new plan')
    if require_simulated and (plan.get('simulated_timing_profile')!=profile or plan.get('simulated_timing_contract_hash')!=digest):
        raise ValueError('Commands do not match the simulated timing profile')
    return recorded_timing(plan.get('speed_scale',1.))['limits'][0]
