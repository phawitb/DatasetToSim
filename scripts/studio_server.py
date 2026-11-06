"""Local-only workbench: run .venv/bin/python scripts/studio_server.py."""
import atexit
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import secrets
import subprocess
import sys
import threading
import time
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import cv2
import numpy as np
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from dataset_to_sim.dataset import DatasetIndex
from dataset_to_sim.studio import safe_file,execution_gate,validate_scene_match,JOINTS

RUN=ROOT/'output/shared-calibration-v10';WORK=ROOT/'output/studio';WORK.mkdir(exist_ok=True)
DATA=Path(os.environ.get('STUDIO_DATASET','/Users/phawit/Projects/VLA-Benchmark/data/SIMCUBE_1O1B_VF'))
TOKEN=secrets.token_urlsafe(32);jobs={};current=None;operation_lock=threading.Lock();frames_lock=threading.Lock();stop_lock=threading.Lock();stop_generation=0
client_heartbeats={}
video_jobs={};video_lock=threading.Lock();video_pool=ThreadPoolExecutor(max_workers=1)
app=FastAPI(title='Real / Sim Studio')
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])
app.mount('/assets',StaticFiles(directory=ROOT/'web'),name='assets')


class HardwareClient:
    def __init__(self):self.process=None;self.lock=threading.Lock();self.replies=queue.Queue();self.failed=False
    def call(self,command,**payload):
        with self.lock:
            if self.failed:raise ValueError('Hardware worker timed out; restart Studio before reconnecting')
            if self.process is None:
                if command!='connect':
                    if command=='status':return {'connected':False,'running':False,'progress':0}
                    if command=='stop':return {'stopped':True}
                    raise ValueError('Connect the robot first')
                runtime=os.environ.get('STUDIO_HARDWARE_PYTHON','/opt/miniconda3/envs/lerobot/bin/python')
                log=open(WORK/'hardware-worker.log','a')
                self.process=subprocess.Popen([runtime,str(ROOT/'scripts/studio_hardware.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
                log.close()
                def reader():
                    for line in self.process.stdout:
                        try:self.replies.put(json.loads(line))
                        except ValueError:pass
                    self.replies.put({'ok':False,'error':'Hardware worker exited; inspect output/studio/hardware-worker.log'})
                threading.Thread(target=reader,daemon=True).start()
            if self.process.poll() is not None:raise ValueError('Hardware worker is not running; restart Studio')
            self.process.stdin.write(json.dumps({'command':command,**payload})+'\n');self.process.stdin.flush()
            try:reply=self.replies.get(timeout=45 if command=='connect' else 8)
            except queue.Empty:
                self.failed=True
                self.process.stdin.write('{"command":"stop"}\n');self.process.stdin.flush()
                raise ValueError('Hardware response timed out; stop requested')
            if not reply['ok']:raise ValueError(reply['error'])
            return reply['result']
    def close(self):
        if self.process and self.process.poll() is None:
            try:self.process.stdin.write('{"command":"stop"}\n');self.process.stdin.flush();self.process.stdin.close()
            except OSError:pass

hardware=HardwareClient();atexit.register(hardware.close)

@app.middleware('http')
async def protect(request:Request,call_next):
    if request.method in ['POST','PUT','DELETE','PATCH']:
        origin=request.headers.get('origin')
        if request.headers.get('x-studio-token')!=TOKEN or (origin and origin!=str(request.base_url).rstrip('/')):
            return JSONResponse({'error':'Open Studio on this computer to use controls'},status_code=403)
    response=await call_next(request)
    response.headers['Cache-Control']='no-store' if request.url.path.startswith('/api') else 'no-cache'
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='same-origin'
    return response

@app.exception_handler(ValueError)
async def invalid(request,exc):return JSONResponse({'error':str(exc)},status_code=409)

@app.get('/')
def index():return FileResponse(ROOT/'web/index.html')

@app.get('/api/bootstrap')
def bootstrap():
    results=json.loads((RUN/'results.json').read_text())
    episodes=[]
    for r in results['results']:
        path=RUN/f"episode-{r['episode']:02d}"/'real-vs-sim-strict.mp4'
        episodes.append({k:r.get(k) for k in ['episode','outcome','frames','joint_state_rmse_native']}|{'has_video':path.is_file()})
    return {'token':TOKEN,'dataset':DATA.name,'episodes':episodes,'successes':results['successes'],'total':results['completed'],'joints':JOINTS,'profile':'shared-calibration-v10','current':current}

@app.get('/api/status')
def status(request:Request):
    client=request.headers.get('x-studio-client')
    if client:client_heartbeats[client]=time.monotonic()
    try:robot=hardware.call('status',client_id=request.headers.get('x-studio-client'))
    except ValueError as exc:robot={'connected':False,'running':False,'error':str(exc)}
    finished_id=robot.get('execution')
    if finished_id and jobs.get(finished_id,{}).get('status')=='executing' and not robot.get('running'):
        with operation_lock:
            if jobs.get(finished_id,{}).get('status')=='executing':
                identifier=finished_id;folder=folder_for(identifier)
                stage=jobs[identifier].get('stage','execute')
                jobs[identifier]={'id':identifier,'status':'recording','stage':stage}
                def finalize():
                    try:
                        result=subprocess.run([sys.executable,str(ROOT/'scripts/studio_recording.py'),str(folder)],capture_output=True,text=True,timeout=120)
                        if result.returncode:raise ValueError(result.stderr.strip().splitlines()[-1])
                        report=json.loads((folder/'plan.json').read_text());report['execution']=json.loads((folder/'execution.json').read_text())
                        (folder/'plan.json').write_text(json.dumps(report,indent=2))
                        jobs[identifier]={'id':identifier,'status':'complete','stage':stage,'plan':report}
                    except Exception as exc:jobs[identifier]={'id':identifier,'status':'error','stage':stage,'error':str(exc)}
                threading.Thread(target=finalize,daemon=True).start()
    return {'robot':robot,'job':jobs.get(current),'current':current}

@app.get('/api/episodes/{index}')
def episode(index:int):
    ds=DatasetIndex(DATA)
    if not 0<=index<ds.episode_count:raise ValueError('Episode not found')
    ep=ds.episode(index);folder=RUN/f'episode-{index:02d}'
    report=json.loads((folder/'report.json').read_text())
    with np.load(folder/'strict_trace.npz') as trace:sim=trace['simulated_native_state'][:ep.frame_count].tolist()
    video=episode_video_status(index).get('video')
    return {'episode':index,'frames':ep.frame_count,'duration':ep.frame_count/30,'timestamps':ep.timestamps.tolist(),'real':ep.state_deg.tolist(),'sim':sim,'actions':ep.action_deg.tolist(),'report':report,'video':video}

def episode_video_status(index):
    folder=RUN/f'episode-{index:02d}'
    if index<0 or not (folder/'report.json').is_file():raise ValueError('Episode not found')
    if (folder/'real-vs-sim-strict.mp4').is_file():return {'status':'ready','video':f'/media/benchmark/episode-{index:02d}/real-vs-sim-strict.mp4'}
    if (WORK/f'replay-{index:02d}'/'comparison.mp4').is_file():return {'status':'ready','video':f'/media/studio/replay-{index:02d}/comparison.mp4'}
    return video_jobs.get(index,{'status':'idle'})

@app.get('/api/episodes/{index}/video')
def video_status(index:int):return episode_video_status(index)

@app.post('/api/episodes/{index}/video')
def prepare_video(index:int):
    with video_lock:
        status=episode_video_status(index)
        if status['status'] in ['ready','rendering']:return status
        video_jobs[index]={'status':'rendering'}
        def render():
            try:
                result=subprocess.run([sys.executable,str(ROOT/'scripts/studio_video.py'),str(index)],capture_output=True,text=True,timeout=240)
                if result.returncode:raise ValueError(result.stderr.strip().splitlines()[-1] if result.stderr else 'Render failed')
                video_jobs[index]={'status':'ready','video':f'/media/studio/replay-{index:02d}/comparison.mp4'}
            except Exception as exc:video_jobs[index]={'status':'error','error':str(exc)}
        video_pool.submit(render)
        return video_jobs[index]

@app.get('/api/episodes/{index}/frame/{frame}')
def frame(index:int,frame:int):
    folder=RUN/f'episode-{index:02d}'
    if index<0 or frame<0 or not folder.is_dir():raise ValueError('Frame unavailable')
    report=json.loads((folder/'report.json').read_text())
    if frame>=report['frames']:raise ValueError('Frame out of bounds')
    target=WORK/f'frame-{index}-{frame}.jpg'
    with frames_lock:
        if not target.exists():
            existing=folder/f'comparison-{frame:03d}.jpg'
            if existing.exists():return FileResponse(existing)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/studio_frame.py'),str(index),str(frame),str(target)],capture_output=True,text=True,timeout=40)
            if result.returncode:raise ValueError(result.stderr.strip().splitlines()[-1] if result.stderr else 'Frame render failed')
    return FileResponse(target)

@app.get('/media/{area}/{path:path}')
def media(area:str,path:str):
    root={'benchmark':RUN,'studio':WORK}.get(area)
    if root is None or Path(path).suffix not in ['.jpg','.mp4','.json','.npz']:raise ValueError('Media unavailable')
    return FileResponse(safe_file(root,path))

@app.post('/api/robot/{command}')
def robot_command(command:str,request:Request):
    global stop_generation,current
    requested_generation=stop_generation
    if command not in ['connect','stop','disconnect','torque_off','torque_on','home']:raise ValueError('Unsupported robot command')
    if command=='home' and not request.headers.get('x-studio-client'):raise ValueError('Home requires an owning control page')
    if command=='stop':
        with stop_lock:stop_generation+=1
        return hardware.call(command)
    with operation_lock:
        if current and jobs.get(current,{}).get('status')=='running':raise ValueError('Wait for the current operation')
        if command=='home':
            if hardware.call('status').get('running'):raise ValueError('Robot is executing')
            references=[]
            for path in WORK.glob('*/plan.json'):
                try:
                    data=json.loads(path.read_text())
                    if data.get('source')=='live' and data.get('purpose')!='home' and (path.parent/'scene.xml').exists():references.append(path)
                except (ValueError,OSError):pass
            if not references:raise ValueError('Home needs a previously captured workcell model')
            reference=max(references,key=lambda p:p.stat().st_mtime).parent
            snapshot=hardware.call('snapshot');identifier=uuid.uuid4().hex;folder=WORK/identifier;folder.mkdir()
            for view,data in snapshot.pop('images').items():(folder/f'{view}.jpg').write_bytes(base64.b64decode(data))
            snapshot.update(id=identifier,purpose='home');(folder/'capture.json').write_text(json.dumps(snapshot,indent=2))
            generation=requested_generation;owner=request.headers.get('x-studio-client')
            current=identifier;jobs[identifier]={'id':identifier,'status':'running','stage':'prepare'}
            def return_home():
                try:
                    result=subprocess.run([sys.executable,str(ROOT/'scripts/studio_home.py'),str(folder),str(reference)],capture_output=True,text=True,timeout=120)
                    if result.returncode:raise ValueError(result.stderr.strip().splitlines()[-1])
                    report=json.loads((folder/'plan.json').read_text())
                    with operation_lock:
                        fresh=hardware.call('snapshot');execution_gate(report,fresh['initial_native'],fresh['calibration_hash'])
                        from studio_scene import detect_bowl,verify_bowl_occlusion
                        rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(fresh['images']['top']),np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
                        try:observed_bowl=detect_bowl(rgb)
                        except ValueError:
                            if report.get('bowl_source')!='recent verified capture; currently occluded':raise
                            if not 0<=time.time()-report.get('reference_captured_at',0)<=600:raise ValueError('Hidden bowl reference expired')
                            verify_bowl_occlusion(rgb,report['bowl_pixel'])
                            observed_bowl=report['bowl_pixel']
                        if np.linalg.norm(np.array(observed_bowl)-report['bowl_pixel'])>5:raise ValueError('Bowl moved during Home preparation; try Home again')
                        saved=np.load(folder/'plan.npz')
                        payload=report|{'actions':saved['actions'].tolist(),'timestamps':saved['timestamps'].tolist(),'owner':owner,'stop_generation':fresh['stop_generation']}
                        with stop_lock:
                            if generation!=stop_generation:raise ValueError('Stop requested; Home cancelled')
                            if time.monotonic()-client_heartbeats.get(owner,0)>3:raise ValueError('Control page closed; Home cancelled')
                            hardware.call('execute',plan=payload)
                        report['executed']=True;(folder/'plan.json').write_text(json.dumps(report,indent=2))
                        jobs[identifier]={'id':identifier,'status':'executing','stage':'prepare'}
                except Exception as exc:jobs[identifier]={'id':identifier,'status':'error','stage':'prepare','error':str(exc)}
            threading.Thread(target=return_home,daemon=True).start()
            return {'preparing_home':True,'id':identifier}
        result=hardware.call(command)
        if command in ['torque_off','torque_on']:current=None
        return result

@app.get('/api/live/{view}')
def live(view:str):
    if view not in ['top','wrist']:raise ValueError('Unknown camera')
    snapshot=hardware.call('preview',view=view)
    from fastapi.responses import Response
    return Response(base64.b64decode(snapshot['image']),media_type='image/jpeg')


def folder_for(identifier):
    if not isinstance(identifier,str) or len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):raise ValueError('Unknown capture')
    path=WORK/identifier
    if not (path/'capture.json').exists():raise ValueError('Capture not found')
    return path

@app.post('/api/capture')
def capture(body:dict):
    global current
    with operation_lock:
        if current and jobs.get(current,{}).get('status')=='running':raise ValueError('Wait for the current operation')
        if hardware.call('status').get('running'):raise ValueError('Stop the robot before capturing')
        identifier=uuid.uuid4().hex;folder=WORK/identifier;folder.mkdir()
        source=body.get('source','live')
        if source=='live':
            snapshot=hardware.call('snapshot')
            for view,image in snapshot.pop('images').items():(folder/f'{view}.jpg').write_bytes(base64.b64decode(image))
        elif source=='dataset':
            ds=DatasetIndex(DATA);index=int(body.get('episode',0))
            if not 0<=index<ds.episode_count:raise ValueError('Unknown episode')
            ep=ds.episode(index)
            for view in ['top','wrist']:
                s=ep.video_slices[view];cap=cv2.VideoCapture(str(s.path));cap.set(cv2.CAP_PROP_POS_FRAMES,round(s.start_time*30));ok,image=cap.read();cap.release()
                if not ok:raise ValueError('Could not decode source frame')
                cv2.imwrite(str(folder/f'{view}.jpg'),image)
            cfg=json.loads((ROOT/'calibration/native-action-current.json').read_text())
            snapshot={'source':'dataset','episode':index,'captured_at':time.time(),'initial_native':ep.state_deg[0].tolist(),'calibration_hash':cfg['follower_calibration_sha256']}
        else:raise ValueError('Choose live or dataset capture')
        snapshot['id']=identifier;(folder/'capture.json').write_text(json.dumps(snapshot,indent=2))
        current=identifier;jobs[identifier]={'id':identifier,'status':'captured','stage':'capture','capture':snapshot}
        return jobs[identifier]

@app.get('/api/plans/{identifier}')
def get_plan(identifier:str):
    folder=folder_for(identifier);path=folder/'plan.json'
    return json.loads(path.read_text()) if path.exists() else json.loads((folder/'capture.json').read_text())

@app.post('/api/plans/{identifier}/{operation}')
def plan_operation(identifier:str,operation:str,request:Request):
    global current
    if operation not in ['plan','simulate','execute']:raise ValueError('Unknown operation')
    folder=folder_for(identifier)
    if json.loads((folder/'capture.json').read_text()).get('purpose')=='home':raise ValueError('Capture a new scene after Home')
    requested_generation=stop_generation
    with operation_lock:
        if current!=identifier:raise ValueError('This capture was replaced; use the current capture')
        if jobs.get(identifier,{}).get('status')=='running':raise ValueError('Operation already running')
        if hardware.call('status').get('running'):raise ValueError('Robot is executing')
        if operation=='execute':
            report=json.loads((folder/'plan.json').read_text())
            fresh=hardware.call('snapshot');execution_gate(report,fresh['initial_native'],fresh['calibration_hash'])
            # Re-detect the live scene just before dispatch; no stale object coordinates.
            from strict_action_scene import episode_scene
            from studio_scene import estimate_live_scene
            cfg=json.loads((ROOT/'calibration/native-action-current.json').read_text())
            image=cv2.imdecode(np.frombuffer(base64.b64decode(fresh['images']['top']),np.uint8),cv2.IMREAD_COLOR)
            observed=episode_scene(cfg,estimate_live_scene(cfg,cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))
            validate_scene_match(report['world'],observed.world)
            saved=np.load(folder/'plan.npz');payload=report|{'actions':saved['actions'].tolist(),'timestamps':saved['timestamps'].tolist(),'stop_generation':fresh['stop_generation'],'owner':request.headers.get('x-studio-client')}
            with stop_lock:
                if requested_generation!=stop_generation:raise ValueError('Stop requested during preflight; execution cancelled')
                result=hardware.call('execute',plan=payload)
            report['executed']=True;(folder/'plan.json').write_text(json.dumps(report,indent=2))
            jobs[identifier]={'id':identifier,'status':'executing','stage':'execute'}
            return result
        if operation=='simulate' and not (folder/'plan.npz').exists():raise ValueError('Generate a plan first')
        old=folder/'plan.json'
        if old.exists():
            invalid=json.loads(old.read_text())
            if invalid.get('executed'):raise ValueError('Capture a new scene after an execution')
            invalid.update(outcome='not_simulated',transfer_checks_passed=False,simulated_action_hash=None)
            invalid.pop('video',None);old.write_text(json.dumps(invalid,indent=2))
        jobs[identifier]={'id':identifier,'status':'running','stage':operation}
        def run():
            try:
                with open(folder/f'{operation}.log','w') as log:
                    result=subprocess.run([sys.executable,str(ROOT/'scripts/studio_planner.py'),operation,str(folder)],stdout=log,stderr=subprocess.STDOUT,timeout=240)
                if result.returncode:
                    lines=(folder/f'{operation}.log').read_text().splitlines();raise ValueError(lines[-1] if lines else 'Planning failed')
                jobs[identifier]={'id':identifier,'status':'complete','stage':operation,'plan':json.loads((folder/'plan.json').read_text())}
            except Exception as exc:jobs[identifier]={'id':identifier,'status':'error','stage':operation,'error':str(exc)}
        threading.Thread(target=run,daemon=True).start()
        return jobs[identifier]

if __name__=='__main__':
    import argparse,uvicorn
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    print(f'Real / Sim Studio: http://127.0.0.1:{args.port}',flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='warning')
