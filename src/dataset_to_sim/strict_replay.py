"""Native-action replay with explicit units and causal zero-order hold.

No recorded states after frame zero, IK, clearance correction or interpolated
future commands are used by the integrator. Physical limits remain in MuJoCo.
"""
from dataclasses import dataclass
import hashlib

import mujoco
import numpy as np

from .dataset import Episode


@dataclass(frozen=True)
class NativeJointAdapter:
    arm_offset_deg: list[float]
    arm_sign: list[float]
    gripper_range_min: int
    gripper_range_max: int
    encoder_max_tick: int
    gripper_closed_model_deg: float
    gripper_sign: float = 1.

    def __post_init__(self):
        if len(self.arm_offset_deg)!=5 or len(self.arm_sign)!=5:
            raise ValueError('Exactly five arm offsets/signs are required')
        if not np.isfinite([*self.arm_offset_deg,*self.arm_sign,self.gripper_closed_model_deg,self.gripper_sign]).all():
            raise ValueError('Adapter parameters must be finite')
        if not all(x in (-1,1) for x in [*self.arm_sign,self.gripper_sign]):
            raise ValueError('Joint signs must be +1 or -1')
        if self.gripper_range_max<=self.gripper_range_min or self.encoder_max_tick<=0:
            raise ValueError('Invalid encoder calibration')

    @property
    def gripper_degrees_per_native_unit(self):
        return (self.gripper_range_max-self.gripper_range_min)*360/self.encoder_max_tick/100

    def to_model(self, native):
        source=np.asarray(native,dtype=float)
        if source.shape[-1:]!=(6,) or not np.isfinite(source).all():
            raise ValueError('Native positions must be finite with final dimension six')
        if np.any(source[...,5]<0) or np.any(source[...,5]>100):
            raise ValueError('Native gripper values outside 0–100; refusing to clip')
        result=source.copy()
        result[...,:5]=(source[...,:5]+self.arm_offset_deg)*self.arm_sign
        result[...,5]=self.gripper_closed_model_deg+self.gripper_sign*source[...,5]*self.gripper_degrees_per_native_unit
        return np.deg2rad(result)

    def to_native(self, model_radians):
        q=np.rad2deg(np.asarray(model_radians,dtype=float))
        if q.shape[-1:]!=(6,) or not np.isfinite(q).all():
            raise ValueError('Model positions must be finite with final dimension six')
        result=q.copy()
        result[...,:5]=q[...,:5]/self.arm_sign-self.arm_offset_deg
        result[...,5]=(q[...,5]-self.gripper_closed_model_deg)/(self.gripper_sign*self.gripper_degrees_per_native_unit)
        return result


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array,dtype='<f8').tobytes()).hexdigest()


def integrate_actions(model, episode: Episode, adapter: NativeJointAdapter, final_interval=1/30):
    """Sample at observation timestamps, execute all N actions, then sample terminal state.

    Trace arrays have N+1 state samples and N applied native actions. The last
    interval has no following real observation and is explicitly reported.
    """
    native=np.array(episode.action_deg,dtype=float,copy=True)  # historical field name; mixed native units
    times=np.array(episode.timestamps,dtype=float,copy=True)
    if len(native)!=len(times) or len(times)==0 or abs(times[0])>1e-7 or np.any(np.diff(times)<=0):
        raise ValueError('Actions require increasing timestamps starting at zero')
    if not np.isfinite(times).all() or not np.isfinite(final_interval) or final_interval<=0:
        raise ValueError('Invalid replay duration')
    if model.nu!=6 or np.any(model.actuator_ctrllimited):
        raise ValueError('Strict replay requires six actuators with control clipping explicitly disabled')
    targets=adapter.to_model(native)
    joint_ids=model.actuator_trnid[:,0]
    if len(set(joint_ids))!=6 or np.any(model.jnt_type[joint_ids]!=mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError('Expected six distinct hinge position actuators')
    qadr=model.jnt_qposadr[joint_ids];vadr=model.jnt_dofadr[joint_ids]
    data=mujoco.MjData(model)
    data.qpos[qadr]=adapter.to_model(episode.state_deg[0])
    # Pre-recording commands are unknown: initialize motor history to the observed initial pose.
    for actuator,count in enumerate(model.actuator_history[:,0]):
        if count:
            if model.actuator_delay[actuator]>count*model.opt.timestep*.9:
                raise ValueError('Motor history needs margin for shortened timestamp-boundary steps')
            mujoco.mj_initCtrlHistory(model,data,actuator,None,np.full((int(count),1),data.qpos[qadr[actuator]]))
    # Free-object qpos0 belongs to the scene; never write an object pose here.
    mujoco.mj_forward(model,data)
    def body_id(name):return mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
    cube=body_id('cube');fixed=body_id('gripper');moving=body_id('moving_jaw_so101_v1')
    cube_joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'cube_free')
    cube_addr=int(model.jnt_qposadr[cube_joint]) if cube_joint>=0 else None
    cube_vaddr=int(model.jnt_dofadr[cube_joint]) if cube_joint>=0 else None
    samples=[];velocities=[];poses=[];cube_velocities=[];sample_times=[];step_times=[];step_ends=[];indices=[];controls=[]
    bilateral=[];peaks=[]
    support_samples=[];secure_run=lift_run=secure_max=lift_max=secure_total=lift_total=0.
    initial_cube_z=float(data.qpos[cube_addr+2]) if cube_addr is not None else 0.
    audit={key:{'contacts':0,'max_penetration_m':0.,'max_normal_force_N':0.} for key in ['robot_self','finger_table','robot_workcell','cube']}
    initial_penetration=max([0.]+[-float(c.dist) for c in data.contact])
    def sample():
        samples.append(data.qpos[qadr].copy());velocities.append(data.qvel[vadr].copy());sample_times.append(float(data.time))
        if cube_addr is not None:
            poses.append(data.qpos[cube_addr:cube_addr+7].copy())
            cube_velocities.append(data.qvel[cube_vaddr:cube_vaddr+6].copy())
    sample();nominal_dt=float(model.opt.timestep)
    try:
        for f,target in enumerate(targets):
            boundary=float(times[f+1]) if f+1<len(times) else float(times[-1]+final_interval)
            data.ctrl[:]=target
            both=False;peak=0.
            while data.time<boundary-1e-12:
                model.opt.timestep=min(nominal_dt,boundary-data.time)
                step_times.append(float(data.time));indices.append(f);controls.append(data.ctrl.copy())
                mujoco.mj_step(model,data);step_ends.append(float(data.time))
                sides=set();jaw_force={'fixed':0.,'moving':0.};jaw_depth=0.
                for j,c in enumerate(data.contact):
                    g1,g2=int(c.geom1),int(c.geom2)
                    bodies={int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])}
                    bits={int(model.geom_contype[g1]),int(model.geom_contype[g2])}
                    names={model.geom(g1).name,model.geom(g2).name}
                    category='cube' if cube>=0 and cube in bodies else ('robot_self' if bits<={2,8} else ('finger_table' if 2 in bits and names&{'table','work_mat','floor'} else 'robot_workcell'))
                    force=np.zeros(6);mujoco.mj_contactForce(model,data,j,force)
                    item=audit[category];item['contacts']+=1
                    item['max_penetration_m']=max(item['max_penetration_m'],max(0.,-float(c.dist)))
                    item['max_normal_force_N']=max(item['max_normal_force_N'],float(abs(force[0])))
                    if cube>=0 and cube in bodies:
                        for side,body in [('fixed',fixed),('moving',moving)]:
                            if body>=0 and body in bodies:
                                sides.add(side);jaw_force[side]+=float(abs(force[0]));jaw_depth=max(jaw_depth,max(0.,-float(c.dist)))
                        peak=max(peak,float(abs(force[0])))
                both|=len(sides)==2
                z=float(data.qpos[cube_addr+2]) if cube_addr is not None else 0.
                secure=min(jaw_force.values())>=.02 and jaw_depth<=.0015
                lifted=secure and z>=initial_cube_z+.04
                dt=float(model.opt.timestep)
                secure_run=secure_run+dt if secure else 0.;lift_run=lift_run+dt if lifted else 0.
                secure_max=max(secure_max,secure_run);lift_max=max(lift_max,lift_run)
                secure_total+=dt*secure;lift_total+=dt*lifted
                support_samples.append([jaw_force['fixed'],jaw_force['moving'],jaw_depth,z])
            if not np.isfinite(data.qpos).all() or abs(data.time-boundary)>1e-8:
                raise RuntimeError(f'Non-finite or reset simulation at action frame {f}')
            if any(w.number for w in data.warning):
                raise RuntimeError(f'MuJoCo numerical warning at action frame {f}')
            bilateral.append(both);peaks.append(peak);sample()
    finally:
        model.opt.timestep=nominal_dt
    if array_hash(episode.action_deg)!=array_hash(native):raise RuntimeError('Source action array changed')
    mujoco.mj_forward(model,data)
    final_finger_contact=any(cube in {int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])} and 2 in {int(model.geom_contype[c.geom1]),int(model.geom_contype[c.geom2])} for c in data.contact) if cube>=0 else False
    return {'qpos':np.array(samples),'qvel':np.array(velocities),'cube_pose':np.array(poses),'cube_velocity':np.array(cube_velocities),
            'sample_times':np.array(sample_times),'native_action_applied':native,'model_targets':targets,
            'step_start_times':np.array(step_times),'step_end_times':np.array(step_ends),
            'step_action_index':np.array(indices,dtype=np.int32),'step_controls':np.array(controls),
            'bilateral':np.array(bilateral),'contact_peak':np.array(peaks),
            'step_grasp_support':np.array(support_samples),
            'audit':{'action_sha256':array_hash(native),'target_clipping':False,'trajectory_repair':False,
                     'control_source':'recorded action only; zero-order hold','post_initial_state_writes':0,
                     'initial_max_penetration_m':initial_penetration,'contact_audit':audit,
                     'final_finger_contact':bool(final_finger_contact),'last_action_interval_seconds':final_interval,
                     'grasp_support':{'minimum_force_each_jaw_N':.02,'maximum_jaw_penetration_m':.0015,'lift_above_initial_m':.04,'longest_secure_grasp_s':secure_max,'longest_secure_lift_s':lift_max,'total_secure_grasp_s':secure_total,'total_secure_lift_s':lift_total,'step_array_columns':['fixed_normal_force_N','moving_normal_force_N','jaw_cube_max_penetration_m','cube_z_m']},
                     'final_cube_linear_speed_m_s':float(np.linalg.norm(data.qvel[cube_vaddr:cube_vaddr+3])) if cube_vaddr is not None else None,
                     'final_cube_angular_speed_rad_s':float(np.linalg.norm(data.qvel[cube_vaddr+3:cube_vaddr+6])) if cube_vaddr is not None else None,
                     'nominal_timestep_seconds':nominal_dt,'actuator_delay_seconds':model.actuator_delay.tolist(),'prehistory':'initial observed joint pose; pre-recording commands unavailable','timing_note':'Dataset timestamps are nominal recording times; hardware latency is not identified.'}}
