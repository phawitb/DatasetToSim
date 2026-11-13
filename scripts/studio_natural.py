"""Coordinated soft-guide paths; human-inspired, not a human motor model.

Quintic B-splines round travel corners, retaining exact task endpoints. Endpoint
velocity/acceleration are zero; internal knots are C4. Convex-hull bounds and
analytic derivative extrema constrain commands before full contact simulation.
"""
import numpy as np
from scipy.interpolate import BSpline,PPoly
from dataset_to_sim.studio import motion_speed_limits
from studio_motion import FLOW_ACCELERATION,FLOW_JERK


def _curve(guides):
    controls=np.vstack([np.tile(guides[0],(3,1)),guides[1:-1],np.tile(guides[-1],(3,1))])
    degree=5;interior=len(controls)-degree-1
    knots=np.r_[np.zeros(degree+1),np.arange(1,interior+1)/(interior+1),np.ones(degree+1)]
    return BSpline(knots,controls,degree)


def _derivative_peaks(curve):
    peaks=np.zeros((3,6))
    for joint in range(6):
        scalar=BSpline(curve.t,curve.c[:,joint],curve.k)
        for order in range(1,4):
            polynomial=PPoly.from_spline(scalar.derivative(order))
            roots=polynomial.derivative().roots(extrapolate=False)
            points=np.r_[0.,1.,curve.t,roots[np.isfinite(roots)&(roots>=0)&(roots<=1)]]
            peaks[order-1,joint]=np.max(abs(polynomial(points)))
    return peaks


def _time_optimal(curve,guides,limits,peaks,duration,minimum,grip_window):
    """The same quintic path, with a time-optimal speed profile instead of one uniform
    time scale (studio_topp). Kept only when it is shorter and every sampled velocity,
    acceleration and jerk stays within the same limits; the gripper keeps Natural's
    minimum-jerk schedule over its own window."""
    import os
    # Off by default: in 30-scene benchmarks it shortened motions only ~5% because the
    # loaded shoulder then trails its command and Run all has to slow down anyway.
    if not os.environ.get('NATURAL_TIME_OPTIMAL'):return None
    from studio_topp import time_optimal,_derivatives
    result=time_optimal(curve,limits,minimum=minimum,joints=range(5))
    if result is None or not result[2]['within_limits'] or result[2]['duration']>=duration-1/30:return None
    q,times,info=result
    # The gripper needs its own minimum time for the same schedule.
    if peaks[0,5] and any((peaks[o,5]/limits[o,5])**(1/(o+1))>times[-1] for o in range(3)):return None
    low,high=grip_window;u=times/times[-1];s=np.clip((u-low)/(high-low),0,1);s=10*s**3-15*s**4+6*s**5
    q=q.copy();q[:,5]=guides[0,5]+(guides[-1,5]-guides[0,5])*s;q[0]=guides[0];q[-1]=guides[-1]
    vel,acc,jerk=_derivatives(q,30.)
    if np.any(vel>limits[0]+1e-6) or np.any(acc>limits[1]+1e-6) or np.any(jerk>limits[2]+1e-6):return None
    return q,times,{'max_velocity':vel.tolist(),'max_acceleration':acc.tolist(),'max_jerk':jerk.tolist(),'duration':float(times[-1]),
                    'grip_window':list(grip_window),'guide_count':len(guides),'timing':'time-optimal along the path (TOPP, sampled limits)',
                    'uniform_duration':float(duration),'acceleration_fraction':info['acceleration_fraction']}


def coordinated_curve(guides,minimum=0.,speed_scale=1.,grip_window=(0.,1.), derivative_limits=None):
    guides=np.asarray(guides,float);limits=[motion_speed_limits(speed_scale)*.92,FLOW_ACCELERATION,FLOW_JERK] if derivative_limits is None else np.asarray(derivative_limits,float)*np.array([[.92],[1.],[1.]])
    if guides.ndim!=2 or guides.shape[1]!=6 or len(guides)<2 or not np.isfinite(guides).all():raise ValueError('Natural requires finite six-joint guides')
    if not np.isfinite(minimum) or minimum<0:raise ValueError('Invalid Natural minimum duration')
    low,high=grip_window
    if not 0<=low<high<=1:raise ValueError('Invalid gripper timing window')
    curve=_curve(guides);peaks=_derivative_peaks(curve)
    # A separate minimum-jerk schedule opens early in the reach. Clearance
    # still requires contact simulation; timing fractions alone do not prove it.
    delta=guides[-1,5]-guides[0,5]
    peaks[:,5]=abs(delta)*np.array([1.875,10*np.sqrt(3)/3,60.])/np.array([(high-low)**n for n in [1,2,3]])
    duration=max(float(minimum),1/30,*[float(np.max(p/limit))**(1/order) for order,(p,limit) in enumerate(zip(peaks,limits),1)])
    ticks=max(1,int(np.ceil(duration*30+1e-9)));duration=ticks/30
    u=np.arange(ticks+1)/ticks;actions=curve(u)
    s=np.clip((u-low)/(high-low),0,1);s=10*s**3-15*s**4+6*s**5
    actions[:,5]=guides[0,5]+delta*s
    actions[0]=guides[0];actions[-1]=guides[-1]
    fast=_time_optimal(curve,guides,limits,peaks,duration,minimum,grip_window)
    if fast is not None:return fast
    return actions,np.arange(ticks+1)/30,{'max_velocity':(peaks[0]/duration).tolist(),'max_acceleration':(peaks[1]/duration**2).tolist(),
        'max_jerk':(peaks[2]/duration**3).tolist(),'duration':duration,'grip_window':list(grip_window),'guide_count':len(guides)}


def interpolate_natural(waypoints,names,speed_scale=1.,derivative_limits=None,lift_stop=False,return_stop=False):
    """Merge travel stages while keeping grasp/release/settle exact and stopped."""
    poses=np.asarray([q for q,_ in waypoints],float);seconds=np.asarray([s for _,s in waypoints],float)
    travel={'Unfold from rest','Raise','Open above cube','Lift','Transfer','Retreat','Return raised','Return unfolded','Home'}
    floors=np.array([s*.55/speed_scale if name in travel else s for s,name in zip(seconds,names)])
    lower=next((name for name in ['Lower onto cube','Lower onto table'] if name in names),'Transfer')
    carry=names.index('Grip check') if 'Grip check' in names else names.index('Grasp')
    spans={0:names.index('Pregrasp') if 'Pregrasp' in names else names.index('Approach'),carry:names.index(lower)}
    if lift_stop and 'Lift' in names:
        # Rise straight to travel height first (a wide held object, such as a bowl held by
        # its rim, otherwise starts moving sideways while still low and sweeps neighbours).
        spans[carry]=names.index('Lift');spans[names.index('Lift')]=names.index(lower)
    if 'Home' in names:
        # From the release (or the jaw retreat out of a tape roll) to Home is one coordinated path.
        release=next(name for name in ['Release on cube','Release on table','Lift out of roll','Turn jaws down','Release'] if name in names)
        spans[names.index(release)]=names.index('Home')
        if return_stop and 'Return raised' in names:
            # Reach the raised corridor before folding toward Home. Blending it
            # away can sweep neighbours during the empty-handed return.
            spans[names.index(release)]=names.index('Return raised')
            spans[names.index('Return raised')]=names.index('Home')
    out=[poses[:1]];knots=[0];end_names=['Start'];phases=[];task_knots={'Start':0}
    index=0
    while index<len(poses)-1:
        end=spans.get(index,index+1);guides=poses[index:end+1].copy()
        window=(0.,.4) if index==0 else (.4,.85) if names[end]=='Home' else (0.,1.)
        q,t,info=coordinated_curve(guides,minimum=float(floors[index+1:end+1].sum()),speed_scale=speed_scale,grip_window=window,derivative_limits=derivative_limits)
        out.append(q[1:]);knots.append(knots[-1]+len(q)-1);end_names.append(names[end]);task_knots[names[end]]=knots[-1]
        phases.append({'from':names[index],'to':names[end],'soft_guides':names[index+1:end],**info})
        index=end
    actions=np.concatenate(out);timestamps=np.arange(len(actions))/30
    return actions,timestamps,{'engine':'Quintic B-spline soft-guide paths with bounded derivatives','waypoints':end_names,'knot_indices':knots,
        'task_knot_indices':task_knots,'phases':phases,'max_velocity':np.max([p['max_velocity'] for p in phases],axis=0).tolist(),
        'max_acceleration':np.max([p['max_acceleration'] for p in phases],axis=0).tolist(),'max_jerk':np.max([p['max_jerk'] for p in phases],axis=0).tolist(),
        'acceleration_limits':(FLOW_ACCELERATION if derivative_limits is None else np.asarray(derivative_limits)[1]).tolist(),'jerk_limits':(FLOW_JERK if derivative_limits is None else np.asarray(derivative_limits)[2]).tolist(),
        'continuity':'Arm C4 within travel; gripper C2; C2 task stops; soft guides; exact task endpoints','cloud':False}
