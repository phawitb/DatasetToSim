"""Fit shared physical motor parameters from original actions and recorded states."""
import sys,json,copy,time
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from strict_action_replay import DATA,ROOT,DatasetIndex
from strict_action_scene import build_model
from episode0 import Calibration
from dataset_to_sim.strict_replay import NativeJointAdapter

OUT=ROOT/'output/system-identification';OUT.mkdir(exist_ok=True)
BASE=ROOT/'output/strict-action'

def fast_trace(m,e,adapter):
    d=mujoco.MjData(m);d.qpos[:6]=adapter.to_model(e.state_deg[0]);target=adapter.to_model(e.action_deg)
    for j,n in enumerate(m.actuator_history[:,0]):
        if n:mujoco.mj_initCtrlHistory(m,d,j,None,np.full((int(n),1),d.qpos[j]))
    mujoco.mj_forward(m,d);states=[];poses=[];dt=float(m.opt.timestep)
    for i,t in enumerate(target):
        states.append(d.qpos[:6].copy());poses.append(d.qpos[6:13].copy());d.ctrl[:]=t
        end=float(e.timestamps[i+1]) if i+1<len(target) else float(e.timestamps[i]+1/30)
        n=int((end-d.time+1e-10)/dt)
        if n:mujoco.mj_step(m,d,nstep=n)
        remaining=end-d.time
        if remaining>1e-10:
            m.opt.timestep=remaining;mujoco.mj_step(m,d);m.opt.timestep=dt
    return np.array(states),np.array(poses)


def load(ids):
    ds=DatasetIndex(DATA);config=json.loads((BASE/'settings.json').read_text())
    config['identified_dynamics']={'kp':[150]*5+[35],'kv':[4]*5+[.1],'delay_s':[0]*6}
    adapter=NativeJointAdapter(**config['adapter']);items=[]
    for i in ids:
        ep=ds.episode(i);cal=Calibration.load(BASE/f'episode-{i:02d}/calibration.json');m=build_model(cal,config);items.append((ep,m))
    return config,adapter,items


def grid():
    ids=[1,4,8,13,20,28,40,61] # Training only: indices not divisible by 3.
    config,adapter,items=load(ids);rows=[]
    for kp in [15.,30.,60.,150.]:
        for tau in [.008,.02,.04]:
            for delay in [.03,.06,.09,.12]:
                errors=[]
                for e,m in items:
                    m.actuator_gainprm[:5,0]=kp;m.actuator_biasprm[:5,1]=-kp;m.actuator_biasprm[:5,2]=-kp*tau;m.actuator_delay[:]=delay
                    states,_=fast_trace(m,e,adapter);errors.append(adapter.to_native(states)[8:,:5]-e.state_deg[8:,:5])
                rmse=np.sqrt(np.mean(np.concatenate(errors)**2,axis=0))
                row={'kp':kp,'kv':kp*tau,'delay_s':delay,'rmse':rmse.tolist(),'mean_rmse':float(np.sqrt(np.mean(rmse**2)))};rows.append(row)
                print(row,flush=True);(OUT/'motor-grid.json').write_text(json.dumps(sorted(rows,key=lambda x:x['mean_rmse']),indent=2))
    return rows

if __name__=='__main__':grid()


def refine():
    from scipy.optimize import minimize
    config,adapter,items=load([1,4,8,13,20,28,40,61]);calls=0;best=[float('inf')]
    def objective(x):
        nonlocal calls
        kp,kv,delay,gkp,gkv,gdelay=np.exp(x[:2]).tolist()+[x[2]]+np.exp(x[3:5]).tolist()+[x[5]]
        errors=[];grip_errors=[]
        for e,m in items:
            gains=np.r_[[kp]*5,gkp];damping=np.r_[[kv]*5,gkv]
            m.actuator_gainprm[:,0]=gains;m.actuator_biasprm[:,1]=-gains;m.actuator_biasprm[:,2]=-damping;m.actuator_delay[:]=[delay]*5+[gdelay]
            states,_=fast_trace(m,e,adapter);native=adapter.to_native(states)
            errors.append((native[8:,:5]-e.state_deg[8:,:5]).ravel())
            mask=(e.state_deg[:,5]>15)&(e.action_deg[:,5]>15);mask[:8]=False
            grip_errors.extend((native[mask,5]-e.state_deg[mask,5])*adapter.gripper_degrees_per_native_unit)
        residual=np.r_[np.concatenate(errors),grip_errors];value=float(np.mean(np.minimum(residual**2,25)))
        calls+=1
        if value<best[0]:
            best[0]=value;config['identified_dynamics']={'kp':[kp]*5+[gkp],'kv':[kv]*5+[gkv],'delay_s':[delay]*5+[gdelay]}
            config['identification']={'training_episodes':[e.index for e,m in items],'held_out_rule':'episode index divisible by 3','objective':'real state error, capped squared loss 25 deg²; free gripper states >15','loss':value,'fit_calls':calls}
            (OUT/'dynamics-fit.json').write_text(json.dumps(config,indent=2));print('best',calls,value,config['identified_dynamics'],flush=True)
        return value
    fit=minimize(objective,[np.log(15),np.log(.6),.03,np.log(10),np.log(.1),.07],method='Nelder-Mead',bounds=[(np.log(5),np.log(150)),(np.log(.05),np.log(8)),(0,.15),(np.log(1),np.log(80)),(np.log(.02),np.log(3)),(0,.2)],options={'maxfev':180,'xatol':.002,'fatol':.0001})
    print(fit,flush=True)


def refine_individual():
    from scipy.optimize import minimize
    config=json.loads((OUT/'combined-v3.json').read_text());adapter=NativeJointAdapter(**config['adapter']);ds=DatasetIndex(DATA)
    ids=[1,4,8,13,20,28,40,61];items=[]
    for i in ids:
        e=ds.episode(i);cal=Calibration.load(ROOT/'output/identified-v3'/f'episode-{i:02d}/calibration.json');items.append((e,build_model(cal,config)))
    dynamics=config['identified_dynamics'];calls=0
    for j in range(5):
        best=[float('inf')];chosen=[None]
        def objective(x):
            nonlocal calls
            kp,kv,delay=np.exp(x[0]),np.exp(x[1]),x[2];errors=[]
            for e,m in items:
                m.actuator_gainprm[j,0]=kp;m.actuator_biasprm[j,1]=-kp;m.actuator_biasprm[j,2]=-kv;m.actuator_delay[j]=delay
                states,_=fast_trace(m,e,adapter);native=adapter.to_native(states);errors.extend(native[8:,j]-e.state_deg[8:,j])
            errors=np.array(errors);loss=float(np.mean(np.minimum(errors**2,16)));calls+=1
            if loss<best[0]:best[0]=loss;chosen[0]=[float(kp),float(kv),float(delay)]
            return loss
        x=[np.log(dynamics['kp'][j]),np.log(dynamics['kv'][j]),dynamics['delay_s'][j]]
        result=minimize(objective,x,method='Nelder-Mead',bounds=[(np.log(3),np.log(100)),(np.log(.02),np.log(6)),(0,.15)],options={'maxfev':100,'xatol':.001,'fatol':.0001})
        kp,kv,delay=chosen[0];dynamics['kp'][j]=kp;dynamics['kv'][j]=kv;dynamics['delay_s'][j]=delay
        for e,m in items:
            m.actuator_gainprm[j,0]=kp;m.actuator_biasprm[j,1]=-kp;m.actuator_biasprm[j,2]=-kv;m.actuator_delay[j]=delay
        print('joint',j,'loss',best[0],'kp kv delay',chosen[0],flush=True)
        config['identification']['individual_motor_fit']={'training_episodes':ids,'completed_joints':j+1,'calls':calls,'objective':'per-joint original-action rollout versus recorded state, capped squared error 16 deg²; no outcome objective'}
        (OUT/'individual-dynamics.json').write_text(json.dumps(config,indent=2))
