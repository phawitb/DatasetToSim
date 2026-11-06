"""Shared image-conditioned arm target repair; never edits object states."""
import numpy as np
import mujoco
from scipy.optimize import least_squares, brentq


def repair_targets(m, targets, ep, cube, bowl):
    d = mujoco.MjData(m)
    g = ep.action_deg[:, 5]
    closes = np.flatnonzero((g[1:] < 5) & (g[:-1] >= 5)) + 1
    if not len(closes):
        raise ValueError('No recorded grasp-closing event')
    close = int(closes[0])
    opens = np.flatnonzero((np.arange(len(g)) > close + 5) & (g > 20))
    if not len(opens):
        raise ValueError('No recorded release-opening event')
    release = int(opens[0])
    reference = min(close + 3, release - 1)
    # Evaluate the center with a nominal contact aperture, not fully closed jaws.
    cube_R = np.empty(9)
    mujoco.mju_quat2Mat(cube_R, np.array(cube.get('quaternion', [1., 0., 0., 0.])))
    cube_R = cube_R.reshape(3, 3)
    d.qpos[:6] = targets[reference]
    d.qpos[5] = .15
    mujoco.mj_forward(m, d)
    axis = d.geom('fixed_pad').xmat.reshape(3, 3)[:, 0].copy()
    if axis @ (d.geom('moving_pad').xpos - d.geom('fixed_pad').xpos) < 0:
        axis = -axis
    width = cube['size'] * np.abs(axis @ cube_R).sum()
    def gap(angle):
        d.qpos[5] = angle; mujoco.mj_forward(m, d)
        return np.linalg.norm(d.geom('moving_pad').xpos-d.geom('fixed_pad').xpos) - width - .006
    aperture = brentq(gap, -.1, 1.)
    roll_offset=None
    roll_blend=0.
    def hand(q):
        d.qpos[:6] = q; d.qpos[5] = aperture
        if roll_offset is not None:d.qpos[4]=(1-roll_blend)*q[4]+roll_blend*(q[0]+roll_offset)
        mujoco.mj_forward(m, d)
        return (d.geom('fixed_pad').xpos + d.geom('moving_pad').xpos) / 2
    # Choose among quarter-turn grasp directions using static collision checks,
    # never a per-episode simulation success search.
    choices=[];roll_blend=1.
    for turn in [0,1,-1,2,-2]:
        roll_offset=targets[reference,4]-targets[reference,0]+turn*np.pi/2
        rolls=targets[:,0]+roll_offset
        if rolls.min()<m.actuator_ctrlrange[4,0] or rolls.max()>m.actuator_ctrlrange[4,1]:continue
        q=targets[reference].copy();q[3]=np.pi/2-q[1]-q[2]
        hand(q)
        candidate_axis=d.geom('fixed_pad').xmat.reshape(3,3)[:,0].copy()
        if candidate_axis@(d.geom('moving_pad').xpos-d.geom('fixed_pad').xpos)<0:candidate_axis=-candidate_axis
        width=cube['size']*np.abs(candidate_axis@cube_R).sum()
        aperture=brentq(gap,-.1,1.)
        collision_depth=0.
        for height in [.024,.04,.06,.08]:
            destination=np.r_[np.array(cube['position'][:2])-.006*candidate_axis[:2],height]
            def pose_residual(x):
                trial=q.copy();trial[:4]=x
                return np.r_[hand(trial)-destination,.003*(x[1:4].sum()-np.pi/2)]
            seed=np.clip(q[:4],m.actuator_ctrlrange[:4,0]+1e-8,m.actuator_ctrlrange[:4,1]-1e-8)
            fit=least_squares(pose_residual,seed,bounds=(m.actuator_ctrlrange[:4,0],m.actuator_ctrlrange[:4,1]),max_nfev=50)
            pose_residual(fit.x);d.qpos[5]=np.deg2rad(35);mujoco.mj_forward(m,d)
            for contact in d.contact:
                bits={int(m.geom_contype[contact.geom1]),int(m.geom_contype[contact.geom2])}
                if 1 not in bits and bits&{2,8} and (4 in bits or bits<={2,8}):collision_depth+=max(0.,-contact.dist)
        choices.append((collision_depth*100+abs(turn)*.001,roll_offset,aperture,candidate_axis.copy(),turn,collision_depth))
    if not choices:raise ValueError('No grasp orientation within wrist joint limits')
    _,roll_offset,aperture,axis,selected_turn,collision_depth=min(choices,key=lambda entry:entry[0])
    roll_blend=0.
    positions = np.array([hand(q).copy() for q in targets])
    cube_xy = np.asarray(cube['position'][:2])
    bowl_xy = np.asarray(bowl['position'][:2])
    start=0
    overhead=max(11,close-25)
    approach=max(overhead+1,close-12)
    lifted=min(close+13,release-6)
    hover=max(lifted+1,release-3)
    end=min(len(targets)-1,release+23)
    hold_end=min(len(targets)-1,release+10)
    approach_xy=cube_xy-.006*axis[:2]/np.linalg.norm(axis[:2])
    knots=[start,10,overhead,approach,close-4,close+5,lifted,hover,hold_end]
    points=[positions[start],np.r_[positions[start,:2],max(.10,positions[start,2])],np.r_[approach_xy,.075],np.r_[approach_xy,.024],np.r_[cube_xy,.024],np.r_[cube_xy,.024],np.r_[cube_xy,.105],np.r_[bowl_xy,.105],np.r_[bowl_xy,.105]]
    desired = positions.copy()
    for lo,hi,p0,p1 in zip(knots[:-1],knots[1:],points[:-1],points[1:]):
        for f in range(lo,hi+1):
            a=(f-lo)/max(1,hi-lo);a=a*a*(3-2*a)
            desired[f]=(1-a)*p0+a*p1
    result = targets.copy()
    ik_errors=[]
    for f, q in enumerate(targets):
        if f<start or f>end:
            continue
        if f>hold_end:
            a=(f-hold_end)/max(1,end-hold_end);a=a*a*(3-2*a)
            result[f]=(1-a)*result[hold_end]+a*q
            roll_blend=0.;desired[f]=hand(result[f]).copy()
            continue
        roll_blend=min(1.,f/10)
        blend=min(np.clip((f-start)/max(1,overhead-start),0,1),np.clip((end-f)/max(1,end-release-10),0,1))
        pitch=(1-blend)*q[1:4].sum()+blend*np.pi/2
        def residual(x):
            candidate = q.copy(); candidate[:4] = x
            center = hand(candidate)
            return np.r_[center-desired[f], .003*(x[1:4].sum()-pitch), .00001*(x-q[:4])]
        fit = least_squares(residual, q[:4], bounds=(m.actuator_ctrlrange[:4, 0],m.actuator_ctrlrange[:4, 1]), max_nfev=50, gtol=1e-10)
        result[f, :4] = fit.x
        result[f,4]=(1-roll_blend)*q[4]+roll_blend*(fit.x[0]+roll_offset)
        ik_errors.append(float(np.linalg.norm(hand(result[f])-desired[f])))
        if ik_errors[-1] > (.03 if desired[f,2]>.04 else .002):
            raise ValueError(f'Unreachable repaired hand target at frame {f}: error {np.linalg.norm(hand(result[f])-desired[f]):.4f} m, point {desired[f]}')
        if f<close-10:result[f,5]=max(q[5],np.deg2rad(35)*min(1,(f-start)/5))
        elif f<=close:result[f,5]=np.deg2rad(35-(f-close+10)*4.3)
        elif f<release:result[f,5]=np.deg2rad(-8)
        elif f<=release+10:result[f,5]=np.deg2rad(-8+48*min(1,(f-release)/8))
    return result, {'enabled': True, 'close_frame': close, 'release_frame': release,
                    'nominal_contact_aperture_deg': float(np.rad2deg(aperture)),
                    'max_hand_translation_m': float(np.linalg.norm(desired-positions, axis=1).max()),
                    'max_joint_correction_deg': float(np.abs(np.rad2deg(result-targets)).max()),
                    'max_ik_position_residual_m': max(ik_errors),
                    'grasp_quarter_turn':selected_turn,'grasp_orientation_collision_depth_m':collision_depth,
                    'world_jaw_heading_locked_during_transport':True,
                    'method': 'shared image-conditioned vertical approach and bowl-centered release; arm targets modified, object states untouched'}
