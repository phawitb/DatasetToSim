"""Persistent private SAM worker in the existing LeRobot / Torch environment."""
import base64,hashlib,json,os,sys,time
from pathlib import Path
import cv2,numpy as np
from studio_sam_geometry import validate_point,mask_record,prompted_component
ROOT=Path(__file__).resolve().parents[1]


def main():
    protocol=sys.stdout;sys.stdout=sys.stderr
    import torch,ultralytics
    from ultralytics.models.sam import SAM2Predictor
    weights=Path(os.environ.get('STUDIO_SAM_WEIGHTS','/Users/phawit/Projects/Hybridge-VLA/sam2.1_b.pt'))
    if not weights.is_file():raise ValueError('SAM 2.1 weights are missing; configure STUDIO_SAM_WEIGHTS')
    model_info={'name':'SAM 2.1 base','weights':weights.name,'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'ultralytics':ultralytics.__version__,'device':'mps' if torch.backends.mps.is_available() else 'cpu'}
    predictor=SAM2Predictor(overrides={'model':str(weights),'task':'segment','mode':'predict','imgsz':1024,'verbose':False,'save':False,'device':model_info['device']})
    cached=None
    for line in sys.stdin:
        try:
            request=json.loads(line);path=Path(request['image']).resolve();point=validate_point(request['point'])
            if not path.is_relative_to(ROOT/'output/studio') or path.suffix.lower() not in ['.jpg','.png']:raise ValueError('Invalid captured image')
            digest=hashlib.sha256(path.read_bytes()).hexdigest();start=time.monotonic()
            # Key on the path too: the predictor keeps the source path, which may be deleted.
            if (digest,str(path))!=cached:
                predictor.reset_image();predictor.set_image(str(path));cached=(digest,str(path))
            result=predictor(points=[point],labels=[1])[0]
            if result.masks is None or not len(result.masks.data):raise ValueError('SAM returned no object')
            mask=prompted_component(result.masks.data[0].cpu().numpy()>0,point);score=float(result.boxes.conf[0])
            info=mask_record(mask,point,score);ok,png=cv2.imencode('.png',mask.astype(np.uint8)*255)
            if not ok:raise ValueError('Could not encode mask')
            response={'ok':True,'result':info|{'png':base64.b64encode(png).decode(),'image_sha256':digest,'model':model_info,'seconds':time.monotonic()-start}}
        except Exception as exc:response={'ok':False,'error':str(exc)}
        protocol.write(json.dumps(response)+'\n');protocol.flush()

if __name__=='__main__':main()
