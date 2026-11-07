import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_planner import interpolate_native
from dataset_to_sim.strict_replay import array_hash


def test_classic_commands_remain_exactly_compatible():
    q,t=interpolate_native([(np.zeros(6),0),(np.array([20,-10,15,5,0,30]),1),(np.array([40,0,-5,10,5,0]),1.5)])
    assert array_hash(q)=='875fdc2097135bff8181f33691c92a6d2fe3539c804151af5960e0ad7e59549e'
    assert array_hash(t)=='51d1ce63000fd0489409c83dcc45d3c63e207f7c126157e9f165befa318cf6fe'


@pytest.mark.parametrize('engine',['interpolate_flow','interpolate_ruckig'])
def test_flow_passes_travel_knots_but_stops_at_contact_and_holds(engine):
    import studio_motion
    interpolate_flow=getattr(studio_motion,engine)
    poses=np.zeros((5,6));poses[:,0]=[0,20,40,40,10]
    q,t,info=interpolate_flow([(p,d) for p,d in zip(poses,[0,1,1,.7,1])],blend={1})
    k=info['knot_indices'];v=np.diff(q,axis=0)*30
    assert min(v[k[1]-1:k[1]+1,0])>5
    assert abs(v[k[2]-1,0])<.15 and abs(v[k[2],0])<.15
    np.testing.assert_allclose(q[k[2]:k[3]+1],np.tile(poses[2],(k[3]-k[2]+1,1)),atol=1e-12)
    assert t[k[3]]-t[k[2]]>=.7-1e-10
    np.testing.assert_array_equal(q[k],poses)
    assert np.all(np.diff(t)>0) and np.isfinite(q).all()
    for i in range(4):
        assert np.all(q[k[i]:k[i+1]+1]>=np.minimum(poses[i],poses[i+1])-1e-10)
        assert np.all(q[k[i]:k[i+1]+1]<=np.maximum(poses[i],poses[i+1])+1e-10)


@pytest.mark.parametrize('engine',['interpolate_flow','interpolate_ruckig'])
def test_flow_bounds_velocity_acceleration_jerk_and_has_no_start_stop_impulse(engine):
    import studio_motion
    from studio_motion import FLOW_ACCELERATION,FLOW_JERK
    interpolate_flow=getattr(studio_motion,engine)
    from dataset_to_sim.studio import SPEED_LIMITS
    poses=np.array([[0]*6,[100,-50,30,25,-5,70],[-20,20,30,50,5,0],[0]*6],float)
    q,t,info=interpolate_flow([(p,.05) for p in poses],blend={1,2})
    padded=np.vstack([np.tile(q[0],(3,1)),q,np.tile(q[-1],(3,1))])
    for order,limit in [(1,SPEED_LIMITS),(2,FLOW_ACCELERATION),(3,FLOW_JERK)]:
        assert np.all(np.max(np.abs(np.diff(padded,n=order,axis=0))*30**order,axis=0)<=limit*1.01)
    assert info['max_velocity'][0]<=35
    assert abs(q[1,0]-q[0,0])<.01


def test_plan_api_freezes_technique_and_rejects_mismatched_execution(tmp_path,monkeypatch):
    import json,time
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    import studio_server as server
    identifier='e'*32;folder=tmp_path/identifier;folder.mkdir()
    (folder/'capture.json').write_text('{}')
    (folder/'plan.json').write_text(json.dumps({'outcome':'success','transfer_checks_passed':True,'motion_technique':'classic','executed':False}))
    np.savez(folder/'plan.npz',actions=np.zeros((2,6)),timestamps=[0,.1])
    monkeypatch.setattr(server,'WORK',tmp_path);monkeypatch.setattr(server,'current',identifier);monkeypatch.setattr(server,'jobs',{})
    def hardware(command,**kw):
        assert command=='status','Mismatched technique must not reach hardware preflight'
        return {'running':False}
    monkeypatch.setattr(server,'hardware',SimpleNamespace(call=hardware))
    monkeypatch.setattr(server.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1))
    client=TestClient(server.app);headers={'X-Studio-Token':server.TOKEN}
    for value in ['unknown',3,[],None]:
        r=client.post(f'/api/plans/{identifier}/plan',json={'motion_technique':value},headers=headers)
        assert r.status_code==409
    for operation in ['simulate','execute']:
        r=client.post(f'/api/plans/{identifier}/{operation}',json={'motion_technique':'flow'},headers=headers)
        assert r.status_code==409 and 'technique' in r.json()['error'].lower()
    for value in [None,True,'1.25',0,1.51]:
        r=client.post(f'/api/plans/{identifier}/plan',json={'speed_scale':value},headers=headers)
        assert r.status_code==409
    for operation in ['simulate','execute']:
        r=client.post(f'/api/plans/{identifier}/{operation}',json={'motion_technique':'classic','speed_scale':1.25},headers=headers)
        assert r.status_code==409 and 'speed' in r.json()['error'].lower()
    r=client.post(f'/api/plans/{identifier}/plan',json={'motion_technique':'flow','speed_scale':1.25},headers=headers)
    assert r.status_code==200
    assert json.loads((folder/'planning-options.json').read_text())['motion_technique']=='flow'
    assert json.loads((folder/'planning-options.json').read_text())['speed_scale']==1.25
    old=json.loads((folder/'plan.json').read_text())
    assert old['outcome']=='not_simulated' and not old['transfer_checks_passed']
    deadline=time.monotonic()+2
    while server.jobs[identifier]['status']=='running' and time.monotonic()<deadline:time.sleep(.01)
    assert server.jobs[identifier]['status']=='error'


@pytest.mark.parametrize('engine',['interpolate_flow','interpolate_ruckig'])
def test_adjustable_speed_shortens_travel_without_raising_acceleration_or_jerk(engine):
    import studio_motion
    from dataset_to_sim.studio import motion_speed_limits,validate_trajectory
    interpolate=getattr(studio_motion,engine)
    poses=np.zeros((4,6));poses[:,0]=[0,90,120,120]
    waypoints=[(p,d) for p,d in zip(poses,[0,.1,.1,.7])]
    base=interpolate(waypoints,blend={1})
    q,t,info=interpolate(waypoints,blend={1},speed_scale=1.5)
    assert t[-1]<base[1][-1]
    assert np.max(np.abs(np.diff(q[:,0]))*30)>35
    assert np.all(np.asarray(info['max_velocity'])<=motion_speed_limits(1.5)*.92+1e-6)
    assert np.all(np.asarray(info['max_acceleration'])<=studio_motion.FLOW_ACCELERATION+1e-6)
    assert np.all(np.asarray(info['max_jerk'])<=studio_motion.FLOW_JERK+1e-6)
    k=info['knot_indices'];assert t[k[3]]-t[k[2]]>=.7-1e-8
    limits=np.array([[-180,180]]*5+[[0,100]])
    validate_trajectory(q,t,limits,speed_scale=1.5)
    with pytest.raises(ValueError,match='speed'):validate_trajectory(q,t,limits)
    np.testing.assert_array_equal(base[0],interpolate(waypoints,blend={1},speed_scale=1.)[0])


@pytest.mark.parametrize('value',[None,True,False,'1.2',[],float('nan'),float('inf'),0,.49,1.51])
def test_speed_setting_rejects_invalid_values(value):
    from dataset_to_sim.studio import validate_speed_scale
    with pytest.raises(ValueError,match='speed'):validate_speed_scale(value)


def test_simulated_timing_is_bound_to_speed_and_timestamps():
    from dataset_to_sim.studio import validate_simulated_timing
    times=np.array([0.,1/30,2/30])
    report={'speed_scale':1.25,'simulated_speed_scale':1.25,'timestamp_hash':array_hash(times),'simulated_timestamp_hash':array_hash(times)}
    validate_simulated_timing(report,times)
    with pytest.raises(ValueError,match='timing'):validate_simulated_timing(report,times*.8)
    with pytest.raises(ValueError,match='speed'):validate_simulated_timing(report|{'speed_scale':1.5},times)
    with pytest.raises(ValueError,match='timing'):validate_simulated_timing({'speed_scale':1.25},times)
    validate_simulated_timing({},times) # Legacy default-speed runs retain compatibility.
