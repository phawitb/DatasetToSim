"""Fit wrist image silhouettes across an episode using the physical jaw geometry."""
from episode0_physics import *

def mask(rgb,kind):
    h=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    if kind=='jaw':return (h[:,:,0]<30)&(h[:,:,1]>100)&(h[:,:,2]>35)
    return (h[:,:,0]>65)&(h[:,:,0]<112)&(h[:,:,1]>25)&(h[:,:,1]<230)&(h[:,:,2]>45)

def fit():
    cal=Calibration.load(ROOT/'calibration/episode0.json');ep=DatasetIndex(DATA).episode(0)
    m=physical_model(cal);d=mujoco.MjData(m);cid=m.camera('wrist_cam').id
    previous=json.loads((ROOT/'calibration/wrist-physics.json').read_text())
    m.cam_pos[cid]=previous['position'];m.cam_quat[cid]=previous['quaternion'];m.cam_fovy[cid]=previous['fovy']
    oldp=m.cam_pos[cid].copy();oldR=Rotation.from_quat(m.cam_quat[cid][[1,2,3,0]]).as_matrix()
    trace=np.load(DEST/'physics_trace.npz');addr=m.jnt_qposadr[m.joint('cube_free').id]
    frames=read_real(ep,'wrist');indices=[0,110,145,175,190,220];targets=[[mask(cv2.resize(frames[f],(320,240)),kind) for kind in ['jaw','bowl']] for f in indices]
    opt=mujoco.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0
    evaluations=0
    with mujoco.Renderer(m,240,320) as renderer:
        def cost(x):
            nonlocal evaluations
            evaluations+=1
            m.cam_pos[cid]=oldp+x[:3];m.cam_quat[cid]=Rotation.from_matrix(oldR@Rotation.from_rotvec(x[3:6]).as_matrix()).as_quat()[[3,0,1,2]];m.cam_fovy[cid]=x[6]
            scores=[]
            for f,t in zip(indices,targets):
                d.qpos[:6]=trace['qpos'][f];d.qpos[addr:addr+7]=trace['cube_pose'][f];mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera='wrist_cam',scene_option=opt);rgb=renderer.render()
                for k,target in zip(['jaw','bowl'],t):
                    sim=mask(rgb,k);scores.append(np.logical_and(sim,target).sum()/max(1,np.logical_or(sim,target).sum()))
            return 1-float(np.mean(scores))
        initial=[0,0,0,0,0,0,previous['fovy']];before=cost(initial)
        result=minimize(cost,initial,method='Powell',bounds=[(-.035,.035)]*3+[(-.4,.4)]*3+[(45,95)],options={'maxfev':1200,'maxiter':12,'ftol':.0002,'xtol':.0003})
        if result.fun>before:result.x=np.array(initial);result.fun=before
        cost(result.x)
        out={'position':m.cam_pos[cid].tolist(),'quaternion':m.cam_quat[cid].tolist(),'fovy':float(m.cam_fovy[cid]),'mean_mask_iou_before':1-before,'mean_mask_iou_after':1-result.fun,'fit_frames':indices,'evaluations':evaluations}
    (ROOT/'calibration/wrist-full-contact.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':fit()
