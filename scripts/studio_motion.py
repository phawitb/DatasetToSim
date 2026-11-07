"""C2, shape-preserving live motion. Does not alter benchmark actions or limits."""
import numpy as np
from numpy.polynomial import polynomial as poly
from dataset_to_sim.studio import SPEED_LIMITS

TECHNIQUES={'classic':'Classic — step by step','gentle':'Gentle — quintic blend','flow':'Flow — blended S-curve'}
FLOW_ACCELERATION=np.array([80.]*5+[120.])
FLOW_JERK=np.array([600.]*5+[900.])


def validate_technique(value):
    if not isinstance(value,str) or value not in TECHNIQUES:
        raise ValueError('Choose Classic, Gentle or Flow motion technique')
    return value


def _coefficients(poses,durations,blend):
    slopes=np.diff(poses,axis=0)/durations[:,None]
    velocities=np.zeros_like(poses)
    for i in blend:
        if not 0<i<len(poses)-1:raise ValueError('Blend knot must be internal')
        left,right=slopes[i-1],slopes[i]
        # Same-direction derivatives only. Each Bezier control stays within
        # its segment endpoints, so joint positions cannot overshoot a knot.
        velocities[i]=np.where(left*right>0,.8*np.sign(left)*np.minimum(abs(left),abs(right)),0.)
    result=[]
    for i,dt in enumerate(durations):
        delta=poses[i+1]-poses[i];a=dt*velocities[i];b=dt*velocities[i+1]
        result.append(np.array([poses[i],a,np.zeros(6),10*delta-6*a-4*b,-15*delta+8*a+7*b,6*delta-3*a-3*b]))
    return result


def _peaks(coefficients,duration,order):
    derivative=poly.polyder(coefficients,m=order,axis=0)/duration**order
    extrema=poly.polyder(derivative,axis=0)
    peaks=[]
    for joint in range(6):
        roots=poly.polyroots(extrema[:,joint])
        points=[0.,1.]+[float(r.real) for r in roots if abs(r.imag)<1e-9 and 0<r.real<1]
        peaks.append(np.max(np.abs(poly.polyval(points,derivative[:,joint]))))
    return np.array(peaks)


def interpolate_flow(waypoints,blend=()):
    """Return 30 Hz native commands and auditable knot/derivative diagnostics.

    Input durations are minimums. Segment time is expanded until analytic
    velocity/acceleration/jerk extrema pass; exact task knots are retained.
    Velocity and acceleration are continuous, with zero acceleration at knots.
    """
    poses=np.array([q for q,_ in waypoints],float)
    durations=np.array([duration for _,duration in waypoints[1:]],float)
    if poses.ndim!=2 or poses.shape[1]!=6 or len(poses)<2 or not np.isfinite(poses).all() or not np.isfinite(durations).all() or np.any(durations<0):
        raise ValueError('Expected finite six-joint waypoints and nonnegative durations')
    durations=np.maximum(durations,1/30)
    limits=[SPEED_LIMITS*.92,FLOW_ACCELERATION,FLOW_JERK]
    for attempt in range(24):
        durations=np.ceil(durations*30-1e-9)/30
        coefficients=_coefficients(poses,durations,blend)
        peaks=np.array([[_peaks(c,dt,order) for order in [1,2,3]] for c,dt in zip(coefficients,durations)])
        ratios=np.array([max(float(np.max(p/limit))**(1/order) for order,(p,limit) in enumerate(zip(segment,limits),1)) for segment in peaks])
        if np.max(ratios)<=1+1e-8:break
        durations*=np.where(ratios>1,ratios*1.01,1)
    else:raise ValueError('Flow timing did not converge within motion limits')
    result=[poses[0:1]];knots=[0]
    for i,(c,dt) in enumerate(zip(coefficients,durations)):
        n=int(round(dt*30));samples=poly.polyval(np.arange(1,n+1)/n,c).T
        samples[-1]=poses[i+1];result.append(samples);knots.append(knots[-1]+n)
    actions=np.concatenate(result);timestamps=np.arange(len(actions))/30
    info={'knot_indices':knots,'segment_seconds':durations.tolist(),
          'blended_knots':sorted(blend),'max_velocity':peaks[:,0].max(axis=0).tolist(),
          'max_acceleration':peaks[:,1].max(axis=0).tolist(),'max_jerk':peaks[:,2].max(axis=0).tolist(),
          'acceleration_limits':FLOW_ACCELERATION.tolist(),'jerk_limits':FLOW_JERK.tolist(),
          'continuity':'C2 quintic; zero acceleration at knots; bounded joint positions'}
    return actions,timestamps,info


def interpolate_ruckig(waypoints,blend=()):
    """Local state-to-state Ruckig sections with shared pass-through velocities.

    Never sets intermediate_positions (Community's cloud waypoint interface).
    Full geometric extrema are checked against each joint's waypoint interval.
    """
    try:import ruckig
    except ImportError as exc:raise ValueError('Flow requires the Ruckig package; choose Classic or install Studio dependencies') from exc
    poses=np.array([q for q,_ in waypoints],float)
    floors=np.array([d for _,d in waypoints[1:]],float)
    if poses.ndim!=2 or poses.shape[1]!=6 or len(poses)<2 or not np.isfinite(poses).all() or not np.isfinite(floors).all() or np.any(floors<0):raise ValueError('Expected finite six-joint waypoints and nonnegative durations')
    vmax=SPEED_LIMITS*.92
    estimates=np.maximum(np.maximum(floors,1/30),np.max(abs(np.diff(poses,axis=0))/vmax,axis=1)+.6)
    slopes=np.diff(poses,axis=0)/estimates[:,None];velocities=np.zeros_like(poses)
    for i in blend:
        if not 0<i<len(poses)-1:raise ValueError('Blend knot must be internal')
        left,right=slopes[i-1],slopes[i]
        velocities[i]=np.where(left*right>0,.65*np.sign(left)*np.minimum(abs(left),abs(right)),0.)
    generator=ruckig.Ruckig(6,1/30)
    # Reduce pass-through speed only if an exact position extremum overshoots.
    # Each retry rebuilds the whole chain so neighboring sections share velocity.
    for attempt in range(9):
        trajectories=[];violations=set()
        for i,minimum in enumerate(floors):
            inp=ruckig.InputParameter(6);inp.current_position=poses[i].tolist();inp.target_position=poses[i+1].tolist()
            inp.current_velocity=velocities[i].tolist();inp.target_velocity=velocities[i+1].tolist()
            inp.current_acceleration=[0.]*6;inp.target_acceleration=[0.]*6
            inp.max_velocity=vmax.tolist();inp.max_acceleration=FLOW_ACCELERATION.tolist();inp.max_jerk=FLOW_JERK.tolist()
            inp.minimum_duration=max(float(minimum),1/30);inp.duration_discretization=ruckig.DurationDiscretization.Discrete
            trajectory=ruckig.Trajectory(6)
            if generator.calculate(inp,trajectory).value<0:raise ValueError('Ruckig could not generate a bounded motion')
            for joint,extrema in enumerate(trajectory.position_extrema):
                low,high=sorted([poses[i,joint],poses[i+1,joint]])
                if extrema.min<low-1e-7 or extrema.max>high+1e-7:violations.update([i,i+1])
            trajectories.append(trajectory)
        if not violations:break
        for i in violations:velocities[i]*=.5
    else:raise ValueError('Flow path exceeds waypoint bounds; use Gentle or Classic')
    result=[poses[0:1]];knots=[0];maxima=np.zeros((3,6))
    for i,trajectory in enumerate(trajectories):
        n=int(round(trajectory.duration*30));samples=np.array([trajectory.at_time(j/30)[0] for j in range(1,n+1)])
        samples[-1]=poses[i+1];result.append(samples);knots.append(knots[-1]+n)
        for joint,profile in enumerate(trajectory.profiles[0]):
            if profile.brake.duration>1e-9:raise ValueError('Unexpected braking phase in planned Flow motion')
            speeds=list(map(abs,profile.v))
            for dt,a,v,j in zip(profile.t,profile.a,profile.v,profile.j):
                if j and 0<-a/j<dt:speeds.append(abs(v-a*a/(2*j)))
            maxima[:,joint]=np.maximum(maxima[:,joint],[max(speeds),max(map(abs,profile.a)),max(map(abs,profile.j))])
    actions=np.concatenate(result);timestamps=np.arange(len(actions))/30
    return actions,timestamps,{'engine':'Ruckig '+ruckig.__version__,'knot_indices':knots,'segment_seconds':[t.duration for t in trajectories],
        'blended_knots':[i for i in sorted(blend) if np.any(abs(velocities[i])>1e-6)],'knot_velocities':velocities.tolist(),
        'max_velocity':maxima[0].tolist(),'max_acceleration':maxima[1].tolist(),'max_jerk':maxima[2].tolist(),
        'acceleration_limits':FLOW_ACCELERATION.tolist(),'jerk_limits':FLOW_JERK.tolist(),
        'continuity':'C2 local S-curves; shared pass-through velocities; exact position-extrema bounds','cloud':False}
