"""Reproducible episode-0 calibration and native MuJoCo video comparison."""
import json
from pathlib import Path
import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation
from dataset_to_sim.dataset import DatasetIndex
from dataset_to_sim.calibration import Calibration
from dataset_to_sim.scene import build_scene_xml

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/Users/phawit/Projects/VLA-Benchmark/data/SIMCUBE_1O1B_VF')
ROBOT=Path('/Users/phawit/Projects/SO-101 MuJoCo 3D Manipulation Scene Editor/assets/so101/so101.xml')
OUT=ROOT/'output/episode0'
OUT.mkdir(parents=True,exist_ok=True)

def camera_fit():
    K=np.array([[650.,0,320],[0,650,240],[0,0,1]])
    xyz=np.array([[.43,.20,0],[.43,-.20,0],[.03,-.20,0],[.03,.20,0]])
    uv=np.array([[175.,34],[524,35],[565,402],[119,400]])
    _,r,t=cv2.solvePnP(xyz,uv,K,None)
    R=cv2.Rodrigues(r)[0]; p=(-R.T@t).ravel()
    return K,R,p,xyz,uv

def ground_point(uv,K,R,p,z):
    ray=R.T@np.linalg.solve(K,np.r_[uv,1.])
    return p+ray*((z-p[2])/ray[2])

def make_calibration():
    K,R,p,xyz,uv=camera_fit()
    c=json.loads((ROOT/'calibration/simcube-draft.json').read_text())
    w=c['world'];w['robot_base']['position']=[-.03,.01,0]
    c['joint_convention']['offset_deg']=[4,0,-2.5,-5,0,0]
    w['table']={'size':[1.2,1.2],'height':.001}
    w['mat']={'position':[.23,0,.003],'size':[.4,.4,.004]}
    w['cube']={'position':ground_point([401,243],K,R,p,.011).tolist(),'size':.022}
    w['bowl']={'position':ground_point([480,180],K,R,p,.025).tolist(),'outer_radius':.06,'height':.05}
    w['bowl']['position'][2]=.005
    c['cameras']['top']={'position':p.tolist(),'target':(p+R.T@np.array([0,0,1])).tolist(),'fovy':float(np.degrees(2*np.arctan(240/650))), 'quaternion':Rotation.from_matrix(R.T@np.diag([1,-1,-1])).as_quat()[[3,0,1,2]].tolist()}
    c['calibration_notes']={'method':'manual mat corner correspondences, solvePnP; dimensions estimated','mat_world_points':xyz.tolist(),'mat_image_points':uv.tolist(),'intrinsics':K.tolist(),'object_motion':'static initial poses; not yet reconstructed'}
    path=ROOT/'calibration/episode0.json';path.write_text(json.dumps(c,indent=2))
    return Calibration.load(path)

def model(cal, return_xml=False):
    import xml.etree.ElementTree as ET
    root=ET.fromstring(build_scene_xml(cal,ROBOT))
    root.find(".//camera[@name='top_cam']").set('quat',' '.join(map(str,cal.camera('top')['quaternion'])))
    world=root.find('worldbody')
    root.find(".//geom[@name='floor']").set('pos','0 0 -.8')
    ET.SubElement(world,'light',pos='0 0 2',dir='0 0 -1',diffuse='.5 .5 .5',ambient='.2 .2 .2')
    visual=root.find('visual');ET.SubElement(visual,'headlight',diffuse='.3 .3 .3',ambient='.4 .4 .4',specular='0 0 0')
    for material in root.findall('.//material'):
        if material.get('rgba','').startswith('1 0.82'):
            material.set('rgba','1 .38 .055 1')
    root.find(".//geom[@name='work_mat']").set('rgba','.13 .12 .15 1')
    bowl=root.find(".//body[@name='bowl']")
    for geom in list(bowl): bowl.remove(geom)
    vertices=[];faces=[];n=96
    for radius,z in [(.042,0),(.06,.05),(.057,.05),(.039,.004)]:
        vertices.extend([[radius*np.cos(a),radius*np.sin(a),z] for a in np.arange(n)*2*np.pi/n])
    for ring in range(3):
        for j in range(n):
            a=ring*n+j;b=ring*n+(j+1)%n;c=(ring+1)*n+j;dd=(ring+1)*n+(j+1)%n
            faces.extend([[a,b,c],[b,dd,c]])
    vertices.append([0,0,.004])
    faces.extend([[4*n,3*n+j,3*n+(j+1)%n] for j in range(n)])
    ET.SubElement(root.find('asset'),'mesh',name='bowl_visual',vertex=' '.join(map(str,np.ravel(vertices))),face=' '.join(map(str,np.ravel(faces))))
    ET.SubElement(bowl,'geom',type='mesh',mesh='bowl_visual',rgba='.5 .65 .6 1',contype='0',conaffinity='0')
    ET.SubElement(bowl,'geom',type='cylinder',size='.04 .002',pos='0 0 .004',rgba='.5 .65 .6 1',contype='0',conaffinity='0')
    xml=ET.tostring(root,encoding='unicode')
    return xml if return_xml else mujoco.MjModel.from_xml_string(xml)

def fit_wrist(m,d,ep,cal):
    d.qpos[:6]=cal.joint_convention.to_radians(ep.state_deg[0]);mujoco.mj_forward(m,d)
    cid=m.camera('wrist_cam').id;bid=m.cam_bodyid[cid]
    bodyR=d.xmat[bid].reshape(3,3).copy();bodyp=d.xpos[bid].copy()
    oldp=m.cam_pos[cid].copy();oldR=Rotation.from_quat(m.cam_quat[cid][[1,2,3,0]]).as_matrix()
    points=np.array([d.geom('fixed_jaw_sph_tip1').xpos,d.geom('moving_jaw_sph_tip1').xpos,d.body('cube').xpos])
    target=np.array([[255,365],[281,371],[431,369]])
    def project(x):
        camR=bodyR@oldR@Rotation.from_rotvec(x[:3]).as_matrix();camp=bodyp+bodyR@(oldp+x[3:6])
        local=(points-camp)@camR
        f=240/np.tan(np.deg2rad(x[6]/2))
        return np.c_[320+f*local[:,0]/(-local[:,2]),240-f*local[:,1]/(-local[:,2])]
    d.qpos[:6]=cal.joint_convention.to_radians(ep.state_deg[190]);mujoco.mj_forward(m,d)
    br190=d.xmat[bid].reshape(3,3).copy();bp190=d.xpos[bid].copy()
    center=np.array(cal.world['bowl']['position'])+np.array([0,0,.05]);angles=np.linspace(0,2*np.pi,100)
    rim=center+np.c_[.06*np.cos(angles),.06*np.sin(angles),np.zeros(100)]
    def bounds(x,br,bp):
        cr=br@oldR@Rotation.from_rotvec(x[:3]).as_matrix();cp=bp+br@(oldp+x[3:6]);v=(rim-cp)@cr
        f=240/np.tan(np.deg2rad(x[6]/2));uv=np.c_[320+f*v[:,0]/(-v[:,2]),240-f*v[:,1]/(-v[:,2])]
        return np.r_[uv.min(axis=0),uv.max(axis=0)]
    def residual(x):
        return np.r_[(project(x)-target).ravel()*[.2,.2,.2,.2,1,1],(bounds(x,br190,bp190)-[80,60,485,440]),(bounds(x,bodyR,bodyp)[:2]-[380,320])*.5,x[:3]*5,x[3:6]*150,(x[6]-61)*.3]
    fit=least_squares(residual,[0,0,0,0,0,0,61],bounds=([-.7]*3+[-.04]*3+[45],[.7]*3+[.04]*3+[95]))
    x=fit.x;m.cam_pos[cid]=oldp+x[3:6];m.cam_quat[cid]=Rotation.from_matrix(oldR@Rotation.from_rotvec(x[:3]).as_matrix()).as_quat()[[3,0,1,2]];m.cam_fovy[cid]=x[6]
    cal.data['cameras']['wrist'].update(position=m.cam_pos[cid].tolist(),quaternion=m.cam_quat[cid].tolist(),fovy=float(x[6]))
    cal.data['wrist_fit']={'frames':[0,190],'manual_pixel_points':target.tolist(),'point_order':['fixed_jaw_tip','moving_jaw_tip','cube_center'],'landmark_rmse_px':float(np.sqrt(np.mean((project(x)-target)**2))),'bowl_bbox_target_190':[80,60,485,440],'bowl_bbox_projected_190':bounds(x,br190,bp190).tolist(),'limitation':'manual approximate landmarks and estimated dimensions, not a measured intrinsic calibration'}

def full_video(cal,ep):
    m=model(cal);d=mujoco.MjData(m);fit_wrist(m,d,ep,cal)
    real={v:read_real(ep,v) for v in ['top','wrist']}
    K,R,p,_,_=camera_fit();cube_start=np.array(cal.world['cube']['position']);cube_start[2]=.016
    cube_final=ground_point([474,168],K,R,p,.020)
    tcp=[]
    for q in ep.state_deg:
        d.qpos[:6]=cal.joint_convention.to_radians(q);mujoco.mj_forward(m,d);tcp.append(d.site('gripperframe').xpos.copy())
    tcp=np.array(tcp);cube=[];modes=[]
    for f in range(ep.frame_count):
        if f<168: xyz=cube_start.copy();mode='initial observed pose'
        elif f<198:
            xyz=cube_start+tcp[f]-tcp[168];xyz[2]=max(.016,xyz[2]);mode='occluded: inferred rigid grasp'
        elif f<214:
            start=cube_start+tcp[197]-tcp[168];a=(f-198)/15;xyz=(1-a)*start+a*cube_final;mode='release: inferred interpolation'
        else: xyz=cube_final.copy();mode='final observed pose'
        cube.append(xyz);modes.append(mode)
    opt=mujoco.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0
    addr=m.jnt_qposadr[m.joint('cube_free').id]
    writer=cv2.VideoWriter(str(OUT/'comparison-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,1024))
    if not writer.isOpened():raise RuntimeError('Video encoder did not open')
    metrics=[]
    with mujoco.Renderer(m,480,640) as renderer:
        for f in range(ep.frame_count):
            d.qpos[:6]=cal.joint_convention.to_radians(ep.state_deg[f]);d.qpos[addr:addr+3]=cube[f];d.time=float(ep.timestamps[f]);mujoco.mj_forward(m,d)
            rows=[];entry={'frame':f,'time':float(ep.timestamps[f]),'cube_source':modes[f]}
            for v,cam in [('top','top_cam'),('wrist','wrist_cam')]:
                renderer.update_scene(d,camera=cam,scene_option=opt);sim=renderer.render().copy()
                canvas=np.zeros((512,1280,3),np.uint8);canvas[32:,:640]=real[v][f];canvas[32:,640:]=sim
                for text,pos in [(f'REAL | {v} | {f:03d}/234 | {f/30:.2f}s',(10,23)),(f'MuJoCo | measured joints + reconstructed cube | {v}',(650,23))]:
                    cv2.putText(canvas,text,pos,cv2.FONT_HERSHEY_SIMPLEX,.51,(255,255,255),1,cv2.LINE_AA)
                rows.append(canvas)
                if v=='top':
                    a,b=orange(real[v][f]),orange(sim);entry['robot_orange_iou']=float(np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum()))
            combined=np.vstack(rows);writer.write(cv2.cvtColor(combined,cv2.COLOR_RGB2BGR));metrics.append(entry)
            if f in [0,130,190,230]:cv2.imwrite(str(OUT/f'comparison-{f:03d}.jpg'),cv2.cvtColor(combined,cv2.COLOR_RGB2BGR))
    writer.release()
    cal.data['calibration_notes']['object_motion']='Initial/final from top image; frames 168..213 inferred from TCP and release interpolation. Kinematic reconstruction, contact physics not validated.'
    cal.path.write_text(json.dumps(cal.data,indent=2))
    np.savez_compressed(OUT/'replay.npz',timestamps=ep.timestamps,qpos=cal.joint_convention.to_radians(ep.state_deg),cube_position=cube)
    report={'frames':ep.frame_count,'fps':30,'duration_seconds':ep.frame_count/30,'render_backend':'CGL / MuJoCo 3.17','calibration':cal.data,'mean_robot_orange_iou':float(np.mean([r['robot_orange_iou'] for r in metrics])),'per_frame':metrics}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print('Rendered',ep.frame_count,'frames; mean IoU',report['mean_robot_orange_iou'],flush=True)

def read_real(ep,view):
    cap=cv2.VideoCapture(str(ep.video_slices[view].path));frames=[]
    cap.set(cv2.CAP_PROP_POS_FRAMES,round(ep.video_slices[view].start_time*30))
    for _ in range(ep.frame_count):
        ok,bgr=cap.read()
        if not ok: raise RuntimeError('Incomplete source video')
        frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release();return frames

def orange(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    return ((hsv[:,:,0]<26)&(hsv[:,:,0]>3)&(hsv[:,:,1]>110)&(hsv[:,:,2]>100)).astype(np.uint8)

def fit_arm(cal,ep):
    m=model(cal);d=mujoco.MjData(m);frames=read_real(ep,'top');indices=[0,130,190]
    targets=[orange(cv2.resize(frames[f],(320,240))) for f in indices]
    opt=mujoco.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0
    base=m.body('base').id
    with mujoco.Renderer(m,240,320) as renderer:
        def cost(x):
            m.body_pos[base,:2]=x[:2];errors=[]
            for f,target in zip(indices,targets):
                d.qpos[:6]=np.deg2rad(ep.state_deg[f]+[x[2],x[3],x[4],x[5],0,0]);mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera='top_cam',scene_option=opt);mask=orange(renderer.render())
                errors.append(1-np.logical_and(mask,target).sum()/max(1,np.logical_or(mask,target).sum()))
            return np.mean(errors)
        x0=[-.03,.01,4,0,-2.5,-5];before=cost(x0)
        result=minimize(cost,x0,method='Powell',bounds=[(-.08,.02),(-.04,.06),(-10,15),(-15,15),(-15,15),(-15,15)],options={'maxiter':6,'maxfev':600,'xtol':.001,'ftol':.001})
    x=result.x;cal.data['world']['robot_base']['position'][:2]=x[:2].tolist()
    cal.data['joint_convention']['offset_deg']=[*x[2:].tolist(),0,0]
    cal.data['arm_fit']={'frames':indices,'before_mean_iou':1-before,'after_mean_iou':1-result.fun,'evaluations':result.nfev,'method':'orange silhouette IoU / Powell','held_out_frame':230}
    cal.path.write_text(json.dumps(cal.data,indent=2));print(cal.data['arm_fit'],flush=True)
    return cal

def main():
    import sys
    ep=DatasetIndex(DATA).episode(0)
    if '--video' in sys.argv:
        full_video(Calibration.load(ROOT/'calibration/episode0.json'),ep);return
    cal=make_calibration();cal=fit_arm(cal,ep);m=model(cal);d=mujoco.MjData(m)
    opt=mujoco.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0
    with mujoco.Renderer(m,480,640) as renderer:
        for f in [0,130,190,230]:
            d.qpos[:6]=cal.joint_convention.to_radians(ep.state_deg[f]);mujoco.mj_forward(m,d)
            renderer.update_scene(d,camera='top_cam',scene_option=opt)
            cv2.imwrite(str(OUT/f'top-{f}.png'),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
            renderer.update_scene(d,camera='wrist_cam',scene_option=opt)
            cv2.imwrite(str(OUT/f'wrist-{f}.png'),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))

if __name__=='__main__':main()
