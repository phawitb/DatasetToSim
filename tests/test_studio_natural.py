import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_natural_rounds_travel_corners_and_coordinates_axes():
    from studio_natural import coordinated_curve
    guides=np.array([[0,0,0,0,0,0],[0,30,0,0,0,0],[40,30,0,0,0,35],[40,0,0,0,0,35]],float)
    q,t,info=coordinated_curve(guides,minimum=1,speed_scale=1.,grip_window=(0,.4))
    np.testing.assert_allclose(q[[0,-1]],guides[[0,-1]],atol=1e-10)
    assert q[len(q)//4,0]>1 and q[len(q)//4,1]>1 # Lift and turn together.
    assert min(np.linalg.norm(q[:,:2]-guides[1,:2],axis=1))>1 # Soft guide, not mandatory corner.
    assert np.all(q>=guides.min(axis=0)-1e-9) and np.all(q<=guides.max(axis=0)+1e-9)
    assert np.allclose(q[int(np.ceil(.4*(len(q)-1))):,5],35)
    from studio_motion import FLOW_ACCELERATION,FLOW_JERK
    from dataset_to_sim.studio import motion_speed_limits
    padded=np.vstack([np.tile(q[0],(3,1)),q,np.tile(q[-1],(3,1))])
    for order,limit in [(1,motion_speed_limits()),(2,FLOW_ACCELERATION),(3,FLOW_JERK)]:
        assert np.all(np.max(abs(np.diff(padded,n=order,axis=0))*30**order,axis=0)<=limit*1.001)
    assert info['max_acceleration'][0]<=80 and info['max_jerk'][0]<=600


def test_natural_gripper_close_waits_for_retreat():
    from studio_natural import coordinated_curve
    guides=np.array([[0,0,0,0,0,40],[10,20,0,0,0,40],[30,30,0,0,0,1]],float)
    q,t,info=coordinated_curve(guides,minimum=2,speed_scale=1.5,grip_window=(.5,.9))
    assert np.allclose(q[:int(.5*(len(q)-1))+1,5],40)
    assert q[-1,5]==1
    assert t[-1]>=2


def test_natural_rejects_bad_inputs():
    from studio_natural import coordinated_curve
    for g in [np.zeros((1,6)),np.ones((2,5)),np.full((2,6),np.nan)]:
        with pytest.raises(ValueError):coordinated_curve(g,minimum=1)
    with pytest.raises(ValueError):coordinated_curve(np.zeros((2,6)),minimum=1,speed_scale=2)


@pytest.mark.parametrize('speed',[.5,1.,1.5])
@pytest.mark.parametrize('guarded',[False,True])
def test_natural_macro_assembly_retains_task_endpoints_holds_and_bounds(speed,guarded):
    from studio_natural import interpolate_natural
    from studio_motion import FLOW_ACCELERATION,FLOW_JERK
    from dataset_to_sim.studio import motion_speed_limits
    names=['Start','Unfold from rest','Raise','Open above cube','Approach','Grasp','Lift','Transfer','Lower onto cube','Placement settle','Release on cube','Retreat','Close after retreat','Return raised','Return unfolded','Home','Final settle']
    poses=np.zeros((len(names),6));poses[:,0]=[0,0,0,30,30,30,30,-30,-30,-30,-30,-30,-30,0,0,0,0]
    poses[:,1]=[0,0,40,40,10,10,40,40,20,20,20,40,40,40,0,0,0]
    poses[:,5]=[1,1,1,35,35,0,0,0,0,0,40,40,1,1,1,1,1]
    seconds=np.ones(len(names));seconds[9]=.7;seconds[-1]=2.
    if guarded:
        pregrasp=(poses[3]+poses[4])/2
        names.insert(4,'Pregrasp');poses=np.insert(poses,4,pregrasp,axis=0);seconds=np.insert(seconds,4,.6)
        names.insert(6,'Grasp settle');poses=np.insert(poses,6,poses[5],axis=0);seconds=np.insert(seconds,6,.4)
    q,t,info=interpolate_natural(list(zip(poses,seconds)),names,speed_scale=speed)
    assert len(q)==len(t) and t[-1]==(len(q)-1)/30
    for name,index in info['task_knot_indices'].items():np.testing.assert_array_equal(q[index],poses[names.index(name)])
    k=info['task_knot_indices'];a,b=k['Lower onto cube'],k['Placement settle']
    assert t[b]-t[a]>=.7 and np.allclose(q[a:b+1],poses[names.index('Placement settle')])
    if guarded:
        a,b=k['Approach'],k['Grasp settle']
        assert t[b]-t[a]>=.4 and np.allclose(q[a:b+1],poses[names.index('Approach')])
    assert t[k['Final settle']]-t[k['Home']]>=2
    padded=np.vstack([np.tile(q[0],(3,1)),q,np.tile(q[-1],(3,1))])
    for order,limit in [(1,motion_speed_limits(speed)),(2,FLOW_ACCELERATION),(3,FLOW_JERK)]:
        assert np.all(np.max(abs(np.diff(padded,n=order,axis=0))*30**order,axis=0)<=limit*1.001)
