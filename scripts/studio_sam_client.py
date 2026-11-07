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
    overlay=np.zeros((480,640,4),np.uint8);overlay[mask]=[210,156,50,100] if role=='from' else [91,167,109,100]
    overlay_name=f'sam-{role}-{revision}-overlay.png';cv2.imwrite(str(folder/overlay_name),overlay)
    return {k:v for k,v in result.items() if k not in ['png','model','image_sha256']}|{'mask_file':name,'overlay_file':overlay_name}


def refresh_pair(client,folder,selection,image,tag):
    roles=['from','to']+(['obstacle'] if 'obstacle' in selection else [])
    results=[client.predict(image,selection[role]['point']) for role in roles]
    masks=[decode_mask(r) for r in results]
    for i,a in enumerate(masks):
        for b in masks[i+1:]:validate_pair(a,b)
    revision=uuid.uuid4().hex;report={'revision':revision,'image_sha256':results[0]['image_sha256'],'model':results[0]['model']}
    for role,result in zip(roles,results):report[role]=save_role(folder,role,result,revision)
    (Path(folder)/f'{tag}.json').write_text(json.dumps(report,indent=2))
    return report,masks
