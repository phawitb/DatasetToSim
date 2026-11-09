"""Real-camera background plate and semantic photometric matching.

Static planes are reprojected in world coordinates for both cameras. Robot and
objects always retain rendered depth/occlusion. This is not neural sim-to-real.
"""
import hashlib
import json
from pathlib import Path
import warnings
import cv2
import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
from studio_generation_visuals import reference_lighting

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'calibration/real-appearance-v1'
REALISTIC=REFERENCE/'realistic-v2.json'
APPEARANCE_PROFILES=('standard','realistic')
REALISTIC_OBJECTS=('robot','dark_robot','bowl')
MAX_SOFT_SHADOW=.35


def geom_classes(model):
    classes={}; names={}
    for i in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
        names[i]=name
        color=model.mat_rgba[model.geom_matid[i],:3] if model.geom_matid[i]>=0 else model.geom_rgba[i,:3]
        if name in ('table','work_mat','floor'):kind=name
        # Library objects of an imported scene (near-white eraser and its sleeve band).
        elif name.startswith('eraser_') or (name=='cube_geom' and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX and min(color)>.7):kind='object'
        elif name in ('cube_geom','target_cube_geom') and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX:kind='red' if color[0]>color[1] else 'green'
        elif name=='cube_geom':kind='tape'
        # The tape roll keeps its own measured material colors (no real-image profile yet).
        elif name.startswith('tape_'):kind='tape'
        elif 'bowl' in name or 'bowl' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[i]) or ''):
            # The visible bowl mesh is unnamed; its body identifies it.
            kind='bowl'
        elif color[0]>.5 and color[0]>color[1]*1.5:kind='robot'
        else:kind='dark_robot'
        classes[i]=kind
    return classes


def segmentation_ids(renderer):
    renderer.enable_segmentation_rendering()
    try:
        seg=renderer.render().copy()
        return np.where(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM),seg[:,:,0],-1)
    finally:renderer.disable_segmentation_rendering()


def camera_matrix(position,quaternion,fovy):
    rotation=Rotation.from_quat(np.asarray(quaternion)[[1,2,3,0]]).as_matrix()
    return np.asarray(position),rotation,240/np.tan(np.deg2rad(fovy)/2)


def project_world(points,position,rotation,focal):
    local=(np.asarray(points)-position)@rotation
    return np.stack([320+focal*local[...,0]/(-local[...,2]),240-focal*local[...,1]/(-local[...,2])],axis=-1)


def background_maps(data,model,view,reference,z):
    cam=model.camera(view+'_cam').id
    position=data.cam_xpos[cam];rotation=data.cam_xmat[cam].reshape(3,3)
    focal=240/np.tan(np.deg2rad(float(model.cam_fovy[cam]))/2)
    yy,xx=np.mgrid[:480,:640]
    rays=np.stack([(xx-320)/focal,-(yy-240)/focal,-np.ones_like(xx)],axis=-1)@rotation.T
    distance=(z-position[2])/np.where(abs(rays[:,:,2])>1e-8,rays[:,:,2],1e-8)
    world=position+rays*distance[:,:,None]
    top=reference['camera'];p,r,f=camera_matrix(top['position'],top['quaternion'],top['fovy'])
    uv=project_world(world,p,r,f)
    valid=(distance>0)&(uv[:,:,0]>=0)&(uv[:,:,0]<639)&(uv[:,:,1]>=0)&(uv[:,:,1]<479)
    return uv[:,:,0].astype('float32'),uv[:,:,1].astype('float32'),valid


def surface_height(model,data,gid):
    """Plane size[2] is grid spacing, whereas a box uses half-height."""
    z=float(data.geom_xpos[gid,2])
    return z if int(model.geom_type[gid])==int(mujoco.mjtGeom.mjGEOM_PLANE) else z+float(model.geom_size[gid,2])


def robust_stats(rgb):
    values=np.asarray(rgb,dtype=np.uint8).reshape(-1,1,3)
    lab=cv2.cvtColor(values,cv2.COLOR_RGB2LAB).reshape(-1,3).astype(float)
    lo,hi=np.percentile(lab,[5,95],axis=0);lab=np.clip(lab,lo,hi)
    return {'mean':lab.mean(0).tolist(),'std':np.maximum(lab.std(0),1).tolist(),'pixels':len(lab)}


def complete_background(rgb,valid,mat_domain):
    """Fill foreground holes using nearby real surface illumination.

    Dilated exclusions also remove contact shadows and mask/calibration error.
    Normalized convolution never borrows pixels across material boundaries.
    """
    missing=cv2.dilate((~valid).astype('uint8'),np.ones((71,71),np.uint8))>0
    valid=~missing
    result=rgb.astype(float);surfaces={}
    blend=np.exp(-distance_transform_edt(~missing)**2/(2*12.**2))
    for kind,domain in [('work_mat',mat_domain),('table',~mat_domain)]:
        available=valid&domain
        if available.sum()<100:raise ValueError('Insufficient clean real background samples')
        weight=available.astype('float32')
        denominator=cv2.GaussianBlur(weight,(0,0),45)
        estimate=cv2.GaussianBlur(rgb.astype('float32')*weight[:,:,None],(0,0),45)/np.maximum(denominator[:,:,None],1e-6)
        alpha=(blend*domain)[:,:,None]
        result=result*(1-alpha)+estimate*alpha
        surfaces[kind]=np.median(rgb[available],axis=0).tolist()
    return np.clip(result,0,255).astype('uint8'),valid,surfaces


def preserve_lighting(matched, illuminated, reference):
    """Reapply achromatic rendered exposure/shadows after semantic matching.

    Both renders have identical geometry and camera. The neutral reference is
    matched to real colors first so per-scene color fitting cannot erase light
    changes; luminance ratios retain red/green labels and foreground occlusion.
    """
    actual = cv2.cvtColor(illuminated, cv2.COLOR_RGB2GRAY).astype(np.float32)
    neutral = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ratio = np.clip((actual + 4.) / (neutral + 4.), .08, 5.)
    return np.clip(matched.astype(np.float32) * ratio[:, :, None], 0, 255).astype(np.uint8)


class RealAppearance:
    def __init__(self,folder,model,options,lighting=None,reference=None):
        reference=Path(reference) if reference else REFERENCE;self.reference=reference
        self.background=bool(options.get('real_background',False));self.processing=bool(options.get('real_image_processing',False))
        # Optional: plate brightness from the real camera only, without rendered light falloff/shadows.
        self.shading=bool(options.get('rendered_shading',True))
        self.model=model;self.classes=geom_classes(model);self.maps={};self.lighting=lighting or {}
        self.profile=json.loads((reference/'profile.json').read_text())
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        from studio_generation_assets import resolve_background
        import shutil
        identifier=options.get('background_id','calibrated-real')
        asset=options.get('_background_asset') or resolve_background(identifier)
        if asset['id'] != identifier:
            raise ValueError('Background snapshot does not match selected background')
        source=Path(asset['path'])
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != asset['sha256']:
            raise ValueError('Background snapshot hash mismatch')
        self.background_mode=asset['mode']
        self.plate=cv2.imread(str(source))
        if self.plate is None or self.plate.shape != (480,640,3):
            raise ValueError('Background must be a normalized 640 × 480 image')
        self.plate=cv2.cvtColor(self.plate,cv2.COLOR_BGR2RGB)
        if source.resolve() != (self.folder/'real-background.png').resolve():
            shutil.copyfile(source,self.folder/'real-background.png')
        self.profile['background_asset']={key:asset[key] for key in ('id','name','mode','sha256')}
        if self.background_mode=='calibrated':
            shutil.copyfile(reference/'valid-mask.png',self.folder/'real-background-valid.png')
        else:
            self.profile['background_mapping']='Uploaded image mapped onto the workcell planes using the reference camera; image is not camera-calibrated.'
            self.profile['surface_rgb']={kind:np.median(self.plate.reshape(-1,3),axis=0).tolist() for kind in ('work_mat','table','floor')}
            self.profile['filled_fraction']=0.
            self.profile['background_sha256']=digest
        (self.folder/'real-appearance.json').write_text(json.dumps(self.profile,indent=2))
        self.provenance={'reference_sha256':hashlib.sha256((reference/'profile.json').read_bytes()).hexdigest(),
            'background_sha256':digest,'background_asset':self.profile['background_asset'],
            'real_background':self.background,'real_image_processing':self.processing,
            'method':'Plane reprojection of selected plate; rendered robot/object occlusion; per-camera semantic Lab matching followed by requested rendered illumination.',
            'limitations':('Single calibrated workcell and planar background; occluded plate pixels filled from same real surface.' if self.background_mode=='calibrated' else self.profile['background_mapping'])+' No neural translation, texture reconstruction or hardware-transfer guarantee.',
            'wrist_cube_color_estimate':'Top segmented cube colors adjusted by measured Top-to-Wrist mat Lab shift.',
            'background_filled_fraction':self.profile['filled_fraction'],'color_sources':self.profile.get('color_sources',{})}
        if self.background_mode=='uploaded':
            self.provenance['uploaded_background_camera_response']={
                'method':'Estimated Wrist response: additive real Wrist-minus-Top work-mat Lab mean; uploaded texture and color variation retained, without material-statistics fitting.',
                'applied':self.background and self.processing,
                'wrist_lab_shift':(np.asarray(self.profile['appearance']['wrist']['work_mat']['real']['mean'])-
                    np.asarray(self.profile['appearance']['top']['work_mat']['real']['mean'])).tolist()}
        self.realistic=options.get('appearance_profile','standard')=='realistic'
        if self.realistic:self._use_realistic_profile()

    def _use_realistic_profile(self):
        """Shared Realistic profile: per camera object targets pooled from real
        episodes, flat real-plate background and faint blurred shadows."""
        real=json.loads(REALISTIC.read_text());appearance=self.profile['appearance']
        self.shading=False;self.min_gain=.35;self.soft_shadow=real['shadow']
        for view,kinds in real['appearance'].items():
            for kind,stats in kinds.items():
                # Wrist background pixels come from the Top plate: Top real stats are the source.
                sim=appearance['top'][kind]['real'] if view=='wrist' and kind in ('work_mat','table','floor') else None
                appearance[view][kind]={'real':{k:stats[k] for k in ('mean','std','pixels')},'sim':sim,'measured':True,'wrist_measured':view=='wrist'}
        # No real green cube was recorded (the profile's 'green' reference is the
        # tan New Run cube), so the green cube keeps its rendered color.
        for view in ('top','wrist'):appearance[view].pop('green',None)
        # Hard MuJoCo shadows are never rendered directly, including the
        # neutral reference light used for color matching.
        self.model.light_castshadow[:]=0
        if self.lighting.get('reference'):
            import copy
            self.lighting=copy.deepcopy(self.lighting);self.lighting['reference']['castshadow']=[0]*len(self.lighting['reference']['castshadow'])
        self.provenance['appearance_profile']={'name':'realistic','sha256':hashlib.sha256(REALISTIC.read_bytes()).hexdigest(),
            'episodes':real['episodes'],'shadow':{k:v for k,v in real['shadow'].items() if k!='episodes'},'method':real['method']}


    def _calibrate_poses(self,renderer,data,poses,option,allow_unseen=False):
        kinds=('red','green')+(REALISTIC_OBJECTS if getattr(self,'realistic',False) else ())
        samples={v:{k:[] for k in kinds} for v in ('top','wrist')}
        original=data.qpos.copy()
        try:
            with reference_lighting(self.model,data,self.lighting):
                for pose in poses:
                    data.qpos[:]=pose
                    mujoco.mj_forward(self.model,data)
                    for view in samples:
                        renderer.update_scene(data,camera=view+'_cam',scene_option=option)
                        rgb=renderer.render().copy();ids=segmentation_ids(renderer)
                        for kind in kinds:
                            mask=np.isin(ids,[j for j,c in self.classes.items() if c==kind]).astype('uint8')
                            mask=cv2.erode(mask,np.ones((3,3),np.uint8))>0
                            if mask.sum()>10:samples[view][kind].append(rgb[mask])
        finally:
            data.qpos[:]=original
            mujoco.mj_forward(self.model,data)
        # Objects left out of the scene are hidden (geom group 3) and have no colors to match.
        hidden={c for c in set(self.classes.values()) if all(self.model.geom_group[j]>=3 for j,k in self.classes.items() if k==c)}
        for view in samples:
            for kind in kinds:
                if kind not in self.profile['appearance'][view]:continue
                if kind in hidden:self.profile['appearance'][view].pop(kind);continue
                if kind in REALISTIC_OBJECTS:
                    # Shared-profile object never visible in this view: leave it unprocessed.
                    if samples[view][kind]:self.profile['appearance'][view][kind]['sim']=robust_stats(np.concatenate(samples[view][kind]))
                    else:self.profile['appearance'][view].pop(kind)
                    continue
                pixels=samples[view][kind] or samples['top'][kind]
                if not pixels:
                    if not allow_unseen:
                        raise ValueError('A task cube is never visible in the rendered reference views')
                    # A still can legitimately hide one cube (e.g. unstack).
                    # Estimate only its unavailable color stats; never move it.
                    gids=[j for j,c in self.classes.items() if c==kind]
                    color=self.model.geom_rgba[gids[0],:3] if gids else np.array([.5,.5,.5])
                    pixels=[np.tile(np.clip(color*180,0,255).astype(np.uint8),(20,1))]
                    self.provenance.setdefault('unseen_preview_color_estimates',[]).append(view+':'+kind)
                self.profile['appearance'][view][kind]['sim']=robust_stats(np.concatenate(pixels))
        (self.folder/'real-appearance.json').write_text(json.dumps(self.profile,indent=2))
        self.provenance['episode_color_profile_sha256']=hashlib.sha256((self.folder/'real-appearance.json').read_bytes()).hexdigest()

    def calibrate_objects(self,renderer,data,trace,cube,other,option):
        """Freeze reference-light color stats on sparse integrated states."""
        poses=[]
        for i in np.unique(np.linspace(0,max(0,len(trace['qpos'])-2),12,dtype=int)):
            pose=data.qpos.copy();pose[:6]=trace['qpos'][i]
            pose[cube:cube+7]=trace['cube_pose'][i];pose[other:other+7]=trace['target_pose'][i]
            poses.append(pose)
        self._calibrate_poses(renderer,data,poses,option)

    def calibrate_frame(self,renderer,data,option):
        """Still-only appearance calibration; initial geometry remains intact."""
        self._calibrate_poses(renderer,data,[data.qpos.copy()],option,allow_unseen=True)

    def render_frame(self,renderer,data,view,option,rgb=None):
        if rgb is None:
            renderer.update_scene(data,camera=view+'_cam',scene_option=option)
            rgb=renderer.render().copy()
        result=self._render_frame(renderer,data,view,option,rgb)
        return self.apply_soft_shadow(renderer,data,view,option,rgb,result) if getattr(self,'realistic',False) else result

    def apply_soft_shadow(self,renderer,data,view,option,plain,result):
        """Faint, blurred shadows: luminance ratio of shadowed/unshadowed renders
        under the configured lights, scaled by the shadow-strength control."""
        strength=min(MAX_SOFT_SHADOW,self.soft_shadow['strength']*float(self.lighting.get('shadow_strength',.5))/.5)
        if strength<=0:return result
        self.model.light_castshadow[:]=1
        try:
            renderer.update_scene(data,camera=view+'_cam',scene_option=option);shaded=renderer.render().copy()
        finally:self.model.light_castshadow[:]=0
        renderer.update_scene(data,camera=view+'_cam',scene_option=option)
        ratio=(cv2.cvtColor(shaded,cv2.COLOR_RGB2GRAY).astype(np.float32)+4)/(cv2.cvtColor(plain,cv2.COLOR_RGB2GRAY).astype(np.float32)+4)
        ratio=cv2.GaussianBlur(np.clip(ratio,.2,1.),(0,0),self.soft_shadow['blur_sigma_px'])
        return np.clip(result.astype(np.float32)*(1-strength*(1-ratio))[:,:,None],0,255).astype(np.uint8)

    def _render_frame(self,renderer,data,view,option,rgb):
        if not self.lighting.get('reference'):
            return self.apply(rgb,segmentation_ids(renderer),data,view)
        with reference_lighting(self.model,data,self.lighting):
            renderer.update_scene(data,camera=view+'_cam',scene_option=option)
            neutral=renderer.render().copy();ids=segmentation_ids(renderer)
            matched=self.apply(neutral,ids,data,view)
        renderer.update_scene(data,camera=view+'_cam',scene_option=option)
        return preserve_lighting(matched,rgb,neutral)

    def apply(self,rgb,ids,data,view):
        result=rgb.copy()
        if self.background:
            for kind in ('work_mat','table','floor'):
                gids=[i for i,c in self.classes.items() if c==kind];mask=np.isin(ids,gids)
                if not mask.any():continue
                gid=gids[0];z=surface_height(self.model,data,gid)
                key=(view,kind)
                if view!='top' or key not in self.maps:self.maps[key]=background_maps(data,self.model,view,self.profile,z)
                x,y,valid=self.maps[key]
                plate=cv2.remap(self.plate,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                # Preserve rendered shading/contact shadows; real plate already
                # provides the calibrated exposure and material appearance.
                gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)
                baseline=max(float(np.percentile(gray[mask],90)),1.)
                if getattr(self,'shading',True):
                    shade=cv2.GaussianBlur(np.clip(gray/baseline,.45,1.05),(0,0),2.)
                    replacement=np.clip(plate*shade[:,:,None],0,255).astype('uint8')
                else:replacement=plate.copy()
                fallback=np.asarray(self.profile['surface_rgb'].get(kind,self.profile['surface_rgb']['table']),np.uint8)
                replacement[~valid]=fallback
                result[mask]=replacement[mask]
        if self.processing:
            lab=cv2.cvtColor(result,cv2.COLOR_RGB2LAB).astype(float)
            for kind,stats in self.profile['appearance'][view].items():
                if self.background and kind in ('work_mat','table','floor') and (view=='top' or getattr(self,'background_mode','calibrated')=='uploaded'):continue
                mask=np.isin(ids,[i for i,c in self.classes.items() if c==kind])
                if not mask.any():continue
                source_stats=stats['sim']
                if self.background and kind in ('work_mat','table','floor'):
                    top_stats=self.profile['appearance']['top']
                    source_stats=top_stats.get(kind,top_stats['table'])['real']
                source=np.asarray(source_stats['mean']);target=np.asarray(stats['real']['mean'])
                if view=='wrist' and kind in ('red','green','bowl','object') and not stats.get('wrist_measured'):
                    # Only Top has segmented cube references. Estimate the
                    # wrist camera exposure/color shift from the shared mat.
                    top_mat=self.profile['appearance']['top']['work_mat']['real']
                    wrist_mat=self.profile['appearance']['wrist']['work_mat']['real']
                    target=target+np.asarray(wrist_mat['mean'])-np.asarray(top_mat['mean'])
                gain=np.clip(np.asarray(stats['real']['std'])/np.asarray(source_stats['std']),getattr(self,'min_gain',.7),1.3)
                lab[mask]=(lab[mask]-source)*gain+target
            if self.background and view=='wrist' and getattr(self,'background_mode','calibrated')=='uploaded':
                # Uploaded pixels have unknown material statistics. Transfer only
                # the observed camera response, preserving their texture/colors.
                top_mean=np.asarray(self.profile['appearance']['top']['work_mat']['real']['mean'])
                wrist_mean=np.asarray(self.profile['appearance']['wrist']['work_mat']['real']['mean'])
                mask=np.isin(ids,[i for i,c in self.classes.items() if c in ('work_mat','table','floor')])
                lab[mask]+=wrist_mean-top_mean
            result=cv2.cvtColor(np.clip(lab,0,255).astype('uint8'),cv2.COLOR_LAB2RGB)
            # Mild fixed antialiasing, coherent over the episode. No independent
            # brightness draws or per-frame color flicker are introduced.
            result=cv2.GaussianBlur(result,(3,3),.35)
        return result


def build_reference(dataset_root):
    """Build an auditable local reference from paired geometry and real frames."""
    from dataset_to_sim.dataset import DatasetIndex
    from dataset_to_sim.calibration import Calibration
    from dataset_to_sim.strict_replay import NativeJointAdapter
    from strict_action_scene import build_model
    dataset=DatasetIndex(dataset_root);config=json.loads((ROOT/'calibration/native-action-current.json').read_text())
    adapter=NativeJointAdapter(**config['adapter']);REFERENCE.mkdir(parents=True,exist_ok=True)
    plates=[];samples=[];real_colors={v:{} for v in ('top','wrist')};sim_colors={v:{} for v in ('top','wrist')};surfaces=[]
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    for index in range(0,dataset.episode_count,6):
        ep=dataset.episode(index);cal=Calibration.load(ROOT/f'output/shared-calibration-v10/episode-{index:02d}/calibration.json')
        model=build_model(cal,config);data=mujoco.MjData(model);classes=geom_classes(model)
        videos={v:cv2.VideoCapture(str(ep.video_slices[v].path)) for v in ('top','wrist')}
        with mujoco.Renderer(model,height=480,width=640) as renderer:
            for frac in (0,.2,.4,.6,.8,.95):
                frame=int((ep.frame_count-1)*frac);data.qpos[:6]=adapter.to_model(ep.state_deg[frame]);mujoco.mj_forward(model,data)
                for view,video in videos.items():
                    timestamp=ep.video_slices[view].start_time+frame/30;video.set(cv2.CAP_PROP_POS_MSEC,timestamp*1000);ok,bgr=video.read()
                    if not ok:raise ValueError('Could not decode real appearance reference')
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);renderer.update_scene(data,camera=view+'_cam',scene_option=opt);sim=renderer.render().copy();ids=segmentation_ids(renderer)
                    for kind in set(classes.values()):
                        mask=np.isin(ids,[i for i,c in classes.items() if c==kind]).astype('uint8')
                        mask=cv2.erode(mask,np.ones((5,5),np.uint8))>0
                        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
                        if kind=='robot':mask &= (hsv[:,:,0]<30)&(hsv[:,:,1]>90)&(hsv[:,:,2]>90)
                        if kind=='dark_robot':mask &= hsv[:,:,2]<90
                        if kind in ('red','green','bowl'):continue
                        if mask.sum()>50:
                            real_colors[view].setdefault(kind,[]).append(rgb[mask][::4]);sim_colors[view].setdefault(kind,[]).append(sim[mask][::4])
                    if view=='top':
                        static=np.isin(ids,[i for i,c in classes.items() if c in ('work_mat','table','floor')]).astype('uint8')
                        static=cv2.erode(static,np.ones((25,25),np.uint8))>0
                        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);static &= ~(((hsv[:,:,0]<32)|(hsv[:,:,0]>170))&(hsv[:,:,1]>85))
                        plate=rgb.astype(float);plate[~static]=np.nan;plates.append(plate)
                        surfaces.append(ids)
                        samples.append({'episode':index,'frame':frame,'top_timestamp':timestamp,'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
                        if index==0 and frame==0:cv2.imwrite(str(REFERENCE/'reference-top.jpg'),bgr)
                    elif index==0 and frame==0:cv2.imwrite(str(REFERENCE/'reference-wrist.jpg'),bgr)
        for video in videos.values():video.release()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);plate=np.nanmedian(plates,axis=0)
    valid=np.isfinite(plate).all(2);clean=np.nan_to_num(plate).astype('uint8')
    # Plane material domains come from calibrated mat projection, not an
    # appearance threshold that could mistake a colored task object for floor.
    camera=config['base_calibration']['cameras']['top'];pos,rot,focal=camera_matrix(camera['position'],camera['quaternion'],camera['fovy'])
    mat=config['base_calibration']['world']['mat'];center=np.asarray(mat['position']);size=np.asarray(mat['size']);z=center[2]+size[2]/2
    corners=np.array([[center[0]+sx*size[0]/2,center[1]+sy*size[1]/2,z] for sx,sy in [(-1,-1),(-1,1),(1,1),(1,-1)]])
    mat_mask=np.zeros((480,640),np.uint8);cv2.fillConvexPoly(mat_mask,project_world(corners,pos,rot,focal).round().astype('int32'),1)
    clean,valid,surface_rgb=complete_background(clean,valid,mat_mask>0)
    # Calibrated cube colors from actual red/green stack capture SAM masks.
    stack=ROOT/'output/studio/dc7fe3a0b44040d6838eb254eb28e2b0';selection=json.loads((stack/'selection.json').read_text());stack_rgb=cv2.cvtColor(cv2.imread(str(stack/'top.jpg')),cv2.COLOR_BGR2RGB)
    color_sources={};cube_stats={}
    for role,kind in [('from','red'),('to','green')]:
        mask=cv2.imread(str(stack/selection[role]['mask_file']),cv2.IMREAD_GRAYSCALE)>0
        mask=cv2.erode(mask.astype('uint8'),np.ones((3,3),np.uint8))>0
        cube_stats[kind]=robust_stats(stack_rgb[mask]);color_sources[kind]={'capture':stack.name,'role':role,'image_sha256':hashlib.sha256((stack/'top.jpg').read_bytes()).hexdigest(),'mask_sha256':hashlib.sha256((stack/selection[role]['mask_file']).read_bytes()).hexdigest(),'note':'Top-camera color reference; applied to both views.'}
    appearance={}
    for view in ('top','wrist'):
        appearance[view]={kind:{'real':robust_stats(np.concatenate(pixels)),'sim':robust_stats(np.concatenate(sim_colors[view][kind]))} for kind,pixels in real_colors[view].items()}
        # Cube source means are measured during rendering on each actual episode
        # before fixed transfer parameters are frozen (see calibrate_objects).
        for kind in ('red','green'):appearance[view][kind]={'real':cube_stats[kind],'sim':None}
    cv2.imwrite(str(REFERENCE/'background.png'),cv2.cvtColor(clean,cv2.COLOR_RGB2BGR));cv2.imwrite(str(REFERENCE/'valid-mask.png'),valid.astype('uint8')*255)
    sources={str(p.relative_to(dataset.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((dataset.root/'videos').glob('*/chunk-*/*.mp4'))}
    profile={'schema_version':1,'dataset':dataset.root.name,'camera':camera,'samples':samples,'source_videos_sha256':sources,'background_sha256':hashlib.sha256((REFERENCE/'background.png').read_bytes()).hexdigest(),'filled_fraction':float((~valid).mean()),'surface_rgb':surface_rgb,'appearance':appearance,'color_sources':color_sources,'method':'Masked temporal median of 72 real top frames; geometry and chroma foreground exclusion; missing pixels filled from normalized interpolation of nearby observed same-material illumination. Appearance stats from 144 paired top/wrist real/sim frames, grouped by rendered semantic surface.'}
    (REFERENCE/'profile.json').write_text(json.dumps(profile,indent=2))
    return profile


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('dataset',type=Path);args=parser.parse_args()
    result=build_reference(args.dataset);print(json.dumps({'samples':len(result['samples']),'filled_fraction':result['filled_fraction'],'output':str(REFERENCE)}))
