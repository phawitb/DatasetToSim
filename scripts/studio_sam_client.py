"""SAM worker transport and immutable capture-bound mask artifacts."""
import base64,hashlib,json,os,queue,subprocess,threading,uuid
from pathlib import Path
import cv2,numpy as np
from studio_sam_geometry import validate_point,validate_pair
ROOT=Path(__file__).resolve().parents[1]


class SAMClient:
    def __init__(self):self.process=None;self.lock=threading.Lock();self.replies=queue.Queue()
    def predict(self,image,point):
        point=validate_point(point)
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self.replies=queue.Queue();log=open(ROOT/'output/studio/sam-worker.log','a')
                runtime=os.environ.get('STUDIO_SAM_PYTHON','/opt/miniconda3/envs/lerobot/bin/python')
                self.process=subprocess.Popen([runtime,str(ROOT/'scripts/studio_sam_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1);log.close()
                process=self.process;replies=self.replies
                def read():
                    for line in process.stdout:
                        try:replies.put(json.loads(line))
                        except ValueError:pass
                    replies.put({'ok':False,'error':'SAM worker exited; inspect sam-worker.log'})
                threading.Thread(target=read,daemon=True).start()
            self.process.stdin.write(json.dumps({'image':str(Path(image).resolve()),'point':point})+'\n');self.process.stdin.flush()
            try:reply=self.replies.get(timeout=90)
            except queue.Empty:
                self.process.terminate();self.process=None
                raise ValueError('SAM timed out; retry object selection')
            if not reply['ok']:raise ValueError(reply['error'])
            return reply['result']
    def close(self):
        if self.process and self.process.poll() is None:self.process.stdin.close()


def decode_mask(result):return cv2.imdecode(np.frombuffer(base64.b64decode(result['png']),np.uint8),cv2.IMREAD_GRAYSCALE)>0


def save_role(folder,role,result,revision):
    folder=Path(folder);mask=decode_mask(result)
    name=f'sam-{role}-{revision}.png';cv2.imwrite(str(folder/name),mask.astype(np.uint8)*255)
    overlay=np.zeros((480,640,4),np.uint8);overlay[mask]=[210,156,50,100] if role=='from' else [91,167,109,100] if role.startswith('to') else [70,70,225,110]
    overlay_name=f'sam-{role}-{revision}-overlay.png';cv2.imwrite(str(folder/overlay_name),overlay)
    return {k:v for k,v in result.items() if k not in ['png','model','image_sha256']}|{'mask_file':name,'overlay_file':overlay_name}


def refresh_pair(client,folder,selection,image,tag):
    """Re-segment From, To and every obstacle at their saved points on a fresh image."""
    from studio_sam_geometry import selection_obstacles,is_place_point
    capture=json.loads((Path(folder)/'capture.json').read_text()) if (Path(folder)/'capture.json').exists() else None
    place=selection['to'].get('kind')=='place'
    roles=['from']+([] if place else ['to'])
    obstacles=selection_obstacles(selection,capture)
    results=[client.predict(image,selection[role]['point']) for role in roles]+[client.predict(image,item['point']) for item in obstacles]
    masks=[decode_mask(r) for r in results]
    for i,a in enumerate(masks):
        for b in masks[i+1:]:validate_pair(a,b)
    revision=uuid.uuid4().hex;report={'revision':revision,'image_sha256':results[0]['image_sha256'],'model':results[0]['model']}
    for role,result in zip(roles,results):report[role]=save_role(folder,role,result,revision)
    report['obstacles']=[save_role(folder,f'obstacle-{i}',result,revision)|{'kind':item['kind']} for i,(item,result) in enumerate(zip(obstacles,results[len(roles):]))]
    (Path(folder)/f'{tag}.json').write_text(json.dumps(report,indent=2))
    target=list(selection['to']['point']) if place else masks[1]
    return report,[masks[0],target,[(item['kind'],m) for item,m in zip(obstacles,masks[len(roles):])]]


def _overlap(a,b):
    a,b=np.asarray(a,bool),np.asarray(b,bool);return np.count_nonzero(a&b)/max(1,min(a.sum(),b.sum()))


def finalize_selection(folder,selection):
    """Every modelled object that is neither From nor To becomes an obstacle."""
    folder=Path(folder)
    def mask(record):return cv2.imread(str(folder/record['mask_file']),cv2.IMREAD_GRAYSCALE)>0
    taken=[mask(selection[role]) for role in ['from','to'] if role in selection and 'mask_file' in selection[role]]
    selection['obstacles']=[item for item in selection.get('objects',[]) if not any(_overlap(mask(item),t)>.1 for t in taken)]
    selection.pop('obstacle',None)
    (folder/'selection.json').write_text(json.dumps(selection,indent=2))
    return selection


def object_record(folder,index,result,info,revision):
    return save_role(folder,f'object-{index}',result,revision)|{'kind':info['kind'],'color':info.get('color')}


def place_record(config,folder,point,selection):
    """Unstack To: an empty mat spot, checked against the stack and every obstacle."""
    from studio_sam_geometry import validate_point,cube_pose_from_mask,obstacle_world,pixel_to_mat,validate_placement
    folder=Path(folder);point=validate_point(point)
    if 'from' not in selection:raise ValueError('Select From (the top cube) first')
    source=cv2.imread(str(folder/selection['from']['mask_file']),cv2.IMREAD_GRAYSCALE)>0
    top,_,_=cube_pose_from_mask(config,source,elevated=True)
    taken=[source]
    obstacles=[obstacle_world(config,item['kind'],cv2.imread(str(folder/item['mask_file']),cv2.IMREAD_GRAYSCALE)>0)
               for item in selection.get('objects',[]) if _overlap(cv2.imread(str(folder/item['mask_file']),cv2.IMREAD_GRAYSCALE)>0,source)<=.1]
    edge=config['base_calibration']['world']['cube']['size'];xy=pixel_to_mat(config,point,edge/2)
    validate_placement(config,xy,top['position'],obstacles)
    return {'kind':'place','point':point,'position':[float(xy[0]),float(xy[1])],'score':1.}


def auto_place_record(config,folder,selection):
    """Unstack To chosen automatically: a clear, reachable empty mat spot near the stack."""
    from studio_sam_auto import auto_place,free_mat
    from studio_sam_geometry import cube_pose_from_mask,obstacle_world,mat_to_pixel
    folder=Path(folder);load=lambda record:cv2.imread(str(folder/record['mask_file']),cv2.IMREAD_GRAYSCALE)>0
    source=load(selection['from']);top,_,_=cube_pose_from_mask(config,source,elevated=True)
    objects=[(item,load(item)) for item in selection.get('objects',[])]
    others=[obstacle_world(config,item['kind'],m) for item,m in objects if _overlap(m,source)<=.1]
    rgb=cv2.cvtColor(cv2.imread(str(folder/'top.jpg')),cv2.COLOR_BGR2RGB)
    xy=auto_place(config,top['position'],others,free_mat(rgb,[source]+[m for _,m in objects]))
    if xy is None:return None
    edge=config['base_calibration']['world']['cube']['size']
    return place_record(config,folder,mat_to_pixel(config,xy,edge/2),selection)|{'auto':True}


def auto_selection(client,folder,task,config):
    """SAM every candidate object after Capture and assign From, To and obstacles."""
    from studio_sam_auto import detect_objects,assign_roles,auto_place
    from studio_sam_geometry import cube_pose_from_mask,obstacle_world,mat_to_pixel
    folder=Path(folder);image=folder/'top.jpg'
    rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB)
    found=detect_objects(client,image,config,rgb,decode_mask)
    if not found:raise ValueError('SAM found no cube, bowl or tape roll on the mat; click the objects')
    image_sha=hashlib.sha256(image.read_bytes()).hexdigest()
    if any(item['result']['image_sha256']!=image_sha for item in found):raise ValueError('Capture changed during segmentation')
    revision=uuid.uuid4().hex
    selection={'revision':revision,'image_sha256':image_sha,'model':found[0]['result']['model'],
               'objects':[object_record(folder,i,item['result'],item['info'],revision) for i,item in enumerate(found)]}
    source,target,notes=assign_roles(task,found,rgb)
    if source is not None:selection['from']=save_role(folder,'from',found[source]['result'],revision)|{'kind':found[source]['info']['kind'],'color':found[source]['info'].get('color'),'auto':True}
    if target is not None:selection['to']=save_role(folder,'to',found[target]['result'],revision)|{'kind':found[target]['info']['kind'],'color':found[target]['info'].get('color'),'auto':True}
    if task=='unstack' and source is not None:
        try:
            place=auto_place_record(config,folder,selection)
            if place is None:notes.append('No clear empty spot found; click an empty spot on the mat as To')
            else:selection['to']=place
        except ValueError as exc:notes.append(f'{exc}; click the top cube as From')
    selection['auto']={'objects':[{'kind':item['info']['kind'],'color':item['info'].get('color'),'point':item['point'],'score':item['result']['score']} for item in found],'notes':notes}
    return finalize_selection(folder,selection)
