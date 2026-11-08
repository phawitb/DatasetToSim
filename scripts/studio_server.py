"""Local-only workbench: run .venv/bin/python scripts/studio_server.py."""
import atexit
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
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
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import cv2
import numpy as np
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse,JSONResponse,Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from studio_history import list_runs
from studio_generation import GenerationManager,TASKS
from studio_generation_view import GenerationViewer
from studio_generation_trash import GenerationTrash
from studio_generation_coverage import generation_coverage
from studio_objects import catalog as object_catalog, object_mesh, object_thumbnail
from studio_generation_preview import PreviewManager
from studio_replay_dataset import ReplayDatasetManager
from studio_generation_assets import list_backgrounds, resolve_background, upload_background, public_background, MAX_UPLOAD_BYTES
from studio_runtime import request_restart,supervise
from studio_sam_home import home_scene
from studio_home_wrist import released_reference,wrist_home_scene
from studio_sam_client import SAMClient,decode_mask,save_role,refresh_pair
from studio_sam_geometry import load_selection,validate_pair,scene_from_masks,validate_vision_age
from dataset_to_sim.dataset import DatasetIndex
from dataset_to_sim.studio import safe_file,execution_gate,validate_scene_match,JOINTS

RUN=ROOT/'output/shared-calibration-v10';WORK=ROOT/'output/studio';WORK.mkdir(exist_ok=True)
DATA=Path(os.environ.get('STUDIO_DATASET','/Users/phawit/Projects/VLA-Benchmark/data/SIMCUBE_1O1B_VF'))
TOKEN=secrets.token_urlsafe(32);jobs={};current=None;operation_lock=threading.Lock();frames_lock=threading.Lock();stop_lock=threading.Lock();stop_generation=0;restarting=False
client_heartbeats={}
video_jobs={};video_lock=threading.Lock();video_pool=ThreadPoolExecutor(max_workers=1)
generation=GenerationManager(ROOT/'output/generated');atexit.register(generation.close)
generation_trash=GenerationTrash(ROOT/'output/generated')
generation_viewer=GenerationViewer();atexit.register(generation_viewer.close)
generation_preview=PreviewManager(ROOT/'output/generation-previews');atexit.register(generation_preview.close)
replay_datasets=ReplayDatasetManager(ROOT/'output/generated');atexit.register(replay_datasets.close)
replay_previews=ReplayDatasetManager(ROOT/'output/replay-previews',media_prefix='/media/replay-previews');atexit.register(replay_previews.close)

def dataset_build_busy():return generation.busy() or replay_datasets.busy() or replay_previews.busy()

@asynccontextmanager
async def studio_lifespan(app):
    generation.recover_interrupted()
    generation_preview.recover_interrupted()
    replay_datasets.recover_interrupted()
    replay_previews.recover_interrupted()
    try:yield
    finally:
        generation.close()
        generation_preview.close()
        replay_datasets.close()
        replay_previews.close()

app=FastAPI(title='Real / Sim Studio',lifespan=studio_lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])
app.mount('/assets',StaticFiles(directory=ROOT/'web'),name='assets')

@app.get('/api/generation')
def generation_index():
    return {'tasks':[{'id':key,'label':value[2],'instruction':value[2]} for key,value in TASKS.items()],
            'jobs':[replay_datasets.get(item['id']) | {'summary':item['summary']} if item.get('source_type')=='real_replay' else item for item in generation.list()],'busy':dataset_build_busy(),'trash':generation_trash.list()}

@app.post('/api/generation')
def generation_start(body:dict):
    with operation_lock:
        if restarting:raise ValueError('Server restarting')
        if hardware.call('status').get('running'):raise ValueError('Wait for the robot to finish before generating data')
        if current and jobs.get(current,{}).get('status') in ['running','executing','recording']:raise ValueError('Wait for the active operation to finish')
        if replay_datasets.busy() or replay_previews.busy():raise ValueError('Wait for the real replay build to finish')
        generation_preview.close()
        if body.get('parent_job_id') and generation.get(body['parent_job_id']).get('source_type')=='real_replay':raise ValueError('Choose source recordings in Datasets to build another real replay dataset')
        return generation.start(body)

@app.get('/api/generation/{identifier}')
def generation_status(identifier:str):
    job=generation.get(identifier)
    return replay_datasets.get(identifier) | {'summary':job['summary']} if job.get('source_type')=='real_replay' else job

@app.get('/api/generation/{identifier}/coverage')
def generation_spatial_coverage(identifier:str):
    return generation_coverage(generation.folder(identifier),generation.get(identifier))

@app.get('/api/objects')
def objects_catalog():return object_catalog()

@app.get('/api/objects/{identifier}/thumbnail')
def objects_thumbnail(identifier:str):return Response(object_thumbnail(identifier),media_type='image/png')

@app.get('/api/objects/{identifier}/mesh')
def objects_mesh(identifier:str):return object_mesh(identifier)

@app.get('/api/generation/{identifier}/episodes/{index}')
def generation_episode(identifier:str,index:int):
    with operation_lock:return generation_viewer.get(generation.folder(identifier),index)

@app.post('/api/generation/{identifier}/delete')
def generation_delete(identifier:str):
    with operation_lock:
        if dataset_build_busy():raise ValueError('Wait for dataset builds to finish before deleting a dataset')
        folder=generation.folder(identifier)
        with generation_viewer.lock:
            if any(not future.done() and Path(path).is_relative_to(folder) for path,future in generation_viewer.pending.items()):
                raise ValueError('Episode preview is being prepared. Retry deleting in a moment.')
            return generation_trash.delete(identifier)

@app.post('/api/generation-trash/{identifier}/restore')
def generation_restore(identifier:str):
    with operation_lock:return generation_trash.restore(identifier)

@app.post('/api/generation/{identifier}/cancel')
def generation_cancel(identifier:str):
    return replay_datasets.cancel(identifier) if generation.get(identifier).get('source_type')=='real_replay' else generation.cancel(identifier)

@app.get('/api/generation/{identifier}/download')
def generation_download(identifier:str):
    job=generation.get(identifier)
    if job['status']!='completed' or not job.get('export',{}).get('valid'):raise ValueError('Dataset is not ready')
    return FileResponse(generation.folder(identifier)/'dataset.zip',filename=job['name']+'.zip',media_type='application/zip')

@app.get('/media/generated/{identifier}/{path:path}')
def generation_media(identifier:str,path:str):
    return FileResponse(safe_file(generation.folder(identifier),path))


@app.get('/api/replay-datasets/sources')
def replay_sources():return {'sources':replay_datasets.sources()}

@app.get('/api/replay-datasets')
def replay_index():return {'jobs':replay_datasets.list(),'busy':replay_datasets.busy()}

def start_replay_build(body,preview=False):
    with operation_lock:
        if restarting:raise ValueError('Server restarting')
        if generation.busy() or replay_datasets.busy():raise ValueError('Wait for the active dataset build to finish')
        if hardware.call('status').get('running'):raise ValueError('Wait for the robot to finish')
        if current and jobs.get(current,{}).get('status') in ['running','executing','recording']:raise ValueError('Wait for the active operation to finish')
        generation_preview.close()
        replay_previews.close()
        return replay_previews.start_preview(body) if preview else replay_datasets.start(body)

@app.post('/api/replay-datasets')
def replay_start(body:dict):return start_replay_build(body)

@app.post('/api/replay-datasets/preview')
def replay_preview_start(body:dict):return start_replay_build(body,True)

@app.get('/api/replay-datasets/previews/{identifier}')
def replay_preview_status(identifier:str):return replay_previews.get(identifier)

@app.get('/media/replay-previews/{identifier}/{path:path}')
def replay_preview_media(identifier:str,path:str):return FileResponse(safe_file(replay_previews.folder(identifier),path))

@app.get('/api/replay-datasets/{identifier}')
def replay_status(identifier:str):return replay_datasets.get(identifier)

@app.post('/api/replay-datasets/{identifier}/cancel')
def replay_cancel(identifier:str):return replay_datasets.cancel(identifier)

@app.get('/api/replay-datasets/{identifier}/download')
def replay_download(identifier:str):
    job=replay_datasets.get(identifier)
    return FileResponse(replay_datasets.download(identifier),filename=job['name']+'.zip',media_type='application/zip')

@app.get('/api/generation-assets')
def generation_assets():
    return {'backgrounds': list_backgrounds()}

@app.post('/api/generation-assets/upload')
async def generation_asset_upload(request: Request):
    content = bytearray()
    async for part in request.stream():
        if len(content) + len(part) > MAX_UPLOAD_BYTES:
            raise ValueError('Choose an image up to 12 MB')
        content.extend(part)
    item = upload_background(bytes(content), unquote(request.headers.get('x-filename', 'Uploaded background')))
    return {'background': public_background(item)}

@app.get('/api/generation-assets/{identifier}/image')
def generation_asset_image(identifier: str):
    return FileResponse(resolve_background(identifier)['path'], media_type='image/png')

@app.post('/api/generation-preview')
def generation_preview_start(body: dict):
    with operation_lock:
        if restarting: raise ValueError('Server restarting')
        if dataset_build_busy(): raise ValueError('Wait for dataset generation to finish before rendering a preview')
        if hardware.call('status').get('running'): raise ValueError('Wait for the robot to finish before rendering a preview')
        if current and jobs.get(current, {}).get('status') in ['running','executing','recording']:
            raise ValueError('Wait for the active operation to finish')
        return generation_preview.start(generation.resolve_options(body))

@app.get('/api/generation-preview/{identifier}')
def generation_preview_status(identifier: str):
    return generation_preview.get(identifier)

@app.get('/api/generation-preview/{identifier}/images/{path:path}')
def generation_preview_image(identifier: str, path: str):
    import re
    if not re.fullmatch(r'scene-[0-4]/(?:top|wrist)\.jpg', path):
        raise ValueError('Preview image not found')
    return FileResponse(safe_file(generation_preview.folder(identifier), path), media_type='image/jpeg')


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
sam=SAMClient();atexit.register(sam.close)

@app.middleware('http')
async def protect(request:Request,call_next):
    if request.method in ['POST','PUT','DELETE','PATCH']:
        origin=request.headers.get('origin')
        if request.headers.get('x-studio-token')!=TOKEN or (origin and origin!=str(request.base_url).rstrip('/')):
            return JSONResponse({'error':'Open Studio on this computer to use controls'},status_code=403)
        if restarting and request.url.path!='/api/server/restart':return JSONResponse({'error':'Server restarting; reconnect after reload'},status_code=409)
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
    return {'token':TOKEN,'dataset':DATA.name,'episodes':episodes,'successes':results['successes'],'total':results['completed'],'joints':JOINTS,'profile':'shared-calibration-v10','current':current,'instance':os.environ.get('STUDIO_INSTANCE',TOKEN),'restart_available':bool(os.environ.get('STUDIO_RESTART_FILE'))}

@app.get('/api/datasets')
def datasets():
    data=bootstrap()
    return {'datasets':[{'id':DATA.name,'name':DATA.name,'episode_count':data['total'],'successes':data['successes'],'fps':30,'profile':data['profile']}]}

@app.get('/api/datasets/{identifier}')
def selected_dataset(identifier:str):
    if identifier!=DATA.name:raise ValueError('Dataset not found')
    data=bootstrap()
    return {k:data[k] for k in ['dataset','episodes','successes','total','profile']}

@app.post('/api/session/new')
def new_session():
    global current
    with operation_lock:
        if restarting:raise ValueError('Server restarting')
        if current and jobs.get(current,{}).get('status') in ['running','executing','recording']:raise ValueError('Wait for the active run to finish')
        if hardware.call('status').get('running'):raise ValueError('Stop the robot before starting a new run')
        current=None
    return {'current':None}

def schedule_restart():
    if not os.environ.get('STUDIO_RESTART_FILE'):raise ValueError('Restart unavailable: launch Studio with scripts/studio_server.py')
    def restart():
        generation.close()
        generation_preview.close()
        replay_datasets.close()
        replay_previews.close()
        # Give an active executor a normal hold first. A stuck worker must not
        # prevent the requested full restart; the supervisor kills its group.
        done=threading.Event()
        def stop_worker():
            try:hardware.call('stop')
            except Exception:pass
            finally:done.set()
        threading.Thread(target=stop_worker,daemon=True).start();done.wait(2)
        time.sleep(.3);request_restart()
    threading.Thread(target=restart,daemon=True).start()

@app.post('/api/server/restart')
def restart_server():
    global restarting,stop_generation
    if restarting:return {'restarting':True}
    with stop_lock:
        stop_generation+=1
        schedule_restart();restarting=True
    return {'restarting':True}

@app.get('/api/runs')
def run_history():return {'runs':list_runs(WORK,jobs),'active_capture':current}


def job_error(identifier,stage,exc):
    value={'id':identifier,'status':'error','stage':stage,'error':str(exc)}
    jobs[identifier]=value
    (WORK/identifier/'job.json').write_text(json.dumps(value,indent=2))

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
                    except Exception as exc:job_error(identifier,stage,exc)
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

APPEARANCES=('raw','background','realistic')

def appearance_mode(value):
    if value not in APPEARANCES:raise ValueError('Unknown Simulation appearance')
    return value

def episode_video_status(index,appearance='raw'):
    folder=RUN/f'episode-{index:02d}'
    if index<0 or not (folder/'report.json').is_file():raise ValueError('Episode not found')
    if appearance!='raw':
        if (WORK/f'replay-{index:02d}'/f'comparison-{appearance}.mp4').is_file():return {'status':'ready','video':f'/media/studio/replay-{index:02d}/comparison-{appearance}.mp4'}
        return video_jobs.get((index,appearance),{'status':'idle'})
    if (folder/'real-vs-sim-strict.mp4').is_file():return {'status':'ready','video':f'/media/benchmark/episode-{index:02d}/real-vs-sim-strict.mp4'}
    if (WORK/f'replay-{index:02d}'/'comparison.mp4').is_file():return {'status':'ready','video':f'/media/studio/replay-{index:02d}/comparison.mp4'}
    return video_jobs.get((index,'raw'),{'status':'idle'})

@app.get('/api/episodes/{index}/video')
def video_status(index:int,appearance:str='raw'):return episode_video_status(index,appearance_mode(appearance))

@app.post('/api/episodes/{index}/video')
def prepare_video(index:int,appearance:str='raw'):
    appearance=appearance_mode(appearance);key=(index,appearance)
    with video_lock:
        status=episode_video_status(index,appearance)
        if status['status'] in ['ready','rendering']:return status
        video_jobs[key]={'status':'rendering'}
        def render():
            try:
                command=[sys.executable,str(ROOT/'scripts/studio_video.py'),str(index)] if appearance=='raw' else [sys.executable,str(ROOT/'scripts/studio_episode_appearance.py'),'video',str(index),appearance]
                result=subprocess.run(command,capture_output=True,text=True,timeout=600)
                if result.returncode:raise ValueError(result.stderr.strip().splitlines()[-1] if result.stderr else 'Render failed')
                name='comparison.mp4' if appearance=='raw' else f'comparison-{appearance}.mp4'
                if not (WORK/f'replay-{index:02d}'/name).is_file():raise ValueError('Render produced no video')
                video_jobs[key]={'status':'ready','video':f'/media/studio/replay-{index:02d}/{name}'}
            except Exception as exc:video_jobs[key]={'status':'error','error':str(exc)}
        video_pool.submit(render)
        return video_jobs[key]

@app.get('/api/episodes/{index}/frame/{frame}')
def frame(index:int,frame:int,appearance:str='raw'):
    appearance=appearance_mode(appearance)
    folder=RUN/f'episode-{index:02d}'
    if index<0 or frame<0 or not folder.is_dir():raise ValueError('Frame unavailable')
    report=json.loads((folder/'report.json').read_text())
    if frame>=report['frames']:raise ValueError('Frame out of bounds')
    target=WORK/(f'frame-{index}-{frame}.jpg' if appearance=='raw' else f'frame-{index}-{frame}-{appearance}.jpg')
    with frames_lock:
        if not target.exists():
            existing=folder/f'comparison-{frame:03d}.jpg'
            if appearance=='raw' and existing.exists():return FileResponse(existing)
            command=[sys.executable,str(ROOT/'scripts/studio_frame.py'),str(index),str(frame),str(target)] if appearance=='raw' else [sys.executable,str(ROOT/'scripts/studio_episode_appearance.py'),'frame',str(index),appearance,str(frame),str(target)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=40)
            if result.returncode:raise ValueError(result.stderr.strip().splitlines()[-1] if result.stderr else 'Frame render failed')
    return FileResponse(target)

@app.get('/media/{area}/{path:path}')
def media(area:str,path:str):
    root={'benchmark':RUN,'studio':WORK}.get(area)
    if root is None or Path(path).suffix not in ['.jpg','.png','.mp4','.json','.npz']:raise ValueError('Media unavailable')
    return FileResponse(safe_file(root,path))

@app.post('/api/robot/{command}')
def robot_command(command:str,request:Request):
    global stop_generation,current
    requested_generation=stop_generation
    if restarting and command!='stop':raise ValueError('Server restarting')
    if command not in ['connect','stop','disconnect','torque_off','torque_on','home']:raise ValueError('Unsupported robot command')
    if command=='home' and not request.headers.get('x-studio-client'):raise ValueError('Home requires an owning control page')
    if command=='stop':
        with stop_lock:stop_generation+=1
        return hardware.call(command)
    with operation_lock:
        if restarting:raise ValueError('Server restarting')
        if command in ['home','torque_on'] and dataset_build_busy():raise ValueError('Finish or cancel data generation before moving the robot')
        if current and jobs.get(current,{}).get('status')=='running':raise ValueError('Wait for the current operation')
        if command=='home':
            if hardware.call('status').get('running'):raise ValueError('Robot is executing')
            home=json.loads((ROOT/'calibration/studio-home.json').read_text())
            pose=hardware.call('status').get('positions')
            if pose is not None and np.all(np.abs(np.asarray(pose)-home['home'])<3):
                current=None
                return {'already_home':True}
            captures=[]
            for path in WORK.glob('*/capture.json'):
                try:
                    saved=json.loads(path.read_text())
                    if saved.get('source')=='live' and saved.get('purpose')!='home':captures.append(saved)
                except (OSError,ValueError):pass
            if captures and max(captures,key=lambda c:c.get('captured_at',0)).get('task')=='stack':
                raise ValueError('Interrupted stack: automatic Home cannot verify both cubes from this pose. Keep torque on and recover the scene before starting another run. A completed stack returns Home within its validated plan.')
            snapshot=hardware.call('snapshot')
            references=[]
            for path in WORK.glob('*/selection.json'):
                try:
                    load_selection(path.parent);references.append(path)
                except (ValueError,OSError):pass
            if not references:raise ValueError('Select From and To in a capture first, or support the arm and use Torque off to clear the camera')
            references.sort(key=lambda p:json.loads((p.parent/'capture.json').read_text()).get('captured_at',0),reverse=True)
            reference=references[0].parent
            for candidate in references:
                try:
                    old=json.loads((candidate.parent/'plan.json').read_text());trace=json.loads((candidate.parent/'hardware-trace.json').read_text())
                    if old.get('source')=='live' and old.get('purpose')!='home' and old.get('executed') and trace.get('completed'):
                        # Never search older placements merely because their pose matches.
                        reference=candidate.parent;break
                except (OSError,ValueError):pass
            identifier=uuid.uuid4().hex;folder=WORK/identifier;folder.mkdir()
            for view,data in snapshot.pop('images').items():(folder/f'{view}.jpg').write_bytes(base64.b64decode(data))
            snapshot.update(id=identifier,purpose='home');(folder/'capture.json').write_text(json.dumps(snapshot,indent=2))
            home_generation=requested_generation;owner=request.headers.get('x-studio-client')
            current=identifier;jobs[identifier]={'id':identifier,'status':'running','stage':'prepare'}
            def return_home():
                try:
                    cfg=json.loads((ROOT/'calibration/native-action-current.json').read_text())
                    try:released_reference(reference,snapshot['initial_native'],snapshot['calibration_hash']);use_wrist=True
                    except ValueError as exc:use_wrist=False;association_error=str(exc)
                    if use_wrist:
                        try:wrist_home_scene(sam,folder,reference,folder/'wrist.jpg',cfg,snapshot['initial_native'])
                        except ValueError as wrist_error:
                            try:home_scene(sam,folder,reference,folder/'top.jpg',cfg)
                            except ValueError as top_error:raise ValueError(f'Wrist: {wrist_error}; Top: {top_error}') from top_error
                    else:
                        try:home_scene(sam,folder,reference,folder/'top.jpg',cfg)
                        except ValueError as top_error:raise ValueError(f'Wrist recovery: {association_error}; Top: {top_error}') from top_error
                    result=subprocess.run([sys.executable,str(ROOT/'scripts/studio_home.py'),str(folder),str(reference)],capture_output=True,text=True,timeout=120)
                    if result.returncode:raise ValueError(result.stderr.strip().splitlines()[-1])
                    report=json.loads((folder/'plan.json').read_text())
                    with operation_lock:
                        fresh=hardware.call('snapshot');execution_gate(report,fresh['initial_native'],fresh['calibration_hash'])
                        verify_home_scene(folder,reference,report,fresh,cfg)
                        latest=hardware.call('snapshot');execution_gate(report,latest['initial_native'],latest['calibration_hash'])
                        validate_vision_age(fresh.get('image_captured_at',{}).get(report.get('home_view','top'),fresh['captured_at']),time.time())
                        saved=np.load(folder/'plan.npz')
                        payload=report|{'actions':saved['actions'].tolist(),'timestamps':saved['timestamps'].tolist(),'owner':owner,'stop_generation':fresh['stop_generation'],'verified_image_captured_at':fresh.get('image_captured_at',{}).get(report.get('home_view','top'),fresh['captured_at'])}
                        with stop_lock:
                            if restarting:raise ValueError('Server restarting; Home cancelled')
                            if home_generation!=stop_generation:raise ValueError('Stop requested; Home cancelled')
                            if time.monotonic()-client_heartbeats.get(owner,0)>3:raise ValueError('Control page closed; Home cancelled')
                            hardware.call('execute',plan=payload)
                        report['executed']=True;(folder/'plan.json').write_text(json.dumps(report,indent=2))
                        jobs[identifier]={'id':identifier,'status':'executing','stage':'prepare'}
                except Exception as exc:job_error(identifier,'prepare',exc)
            threading.Thread(target=return_home,daemon=True).start()
            return {'preparing_home':True,'id':identifier}
        if command in ['torque_on','torque_off','disconnect']:
            with stop_lock:
                if restarting:raise ValueError('Server restarting')
                result=hardware.call(command)
        else:result=hardware.call(command)
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
        task=body.get('task','bowl')
        if task not in ['bowl','stack']:raise ValueError('Choose Place in bowl or Stack on cube')
        if 'bowl_present' in body and not isinstance(body['bowl_present'],bool):raise ValueError('Bowl presence must be true or false')
        if restarting:raise ValueError('Server restarting')
        if current and jobs.get(current,{}).get('status')=='running':raise ValueError('Wait for the current operation')
        if hardware.call('status').get('running'):raise ValueError('Stop the robot before capturing')
        identifier=uuid.uuid4().hex;folder=WORK/identifier;folder.mkdir()
        source=body.get('source','live')
        if task=='stack' and source!='live':raise ValueError('Stacking requires a live two-cube capture')
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
        snapshot.update(id=identifier,task=task,bowl_present=body.get('bowl_present',True) if task=='stack' else True)
        (folder/'capture.json').write_text(json.dumps(snapshot,indent=2))
        current=identifier;jobs[identifier]={'id':identifier,'status':'captured','stage':'capture','capture':snapshot}
        return jobs[identifier]

@app.get('/api/plans/{identifier}')
def get_plan(identifier:str):
    folder=folder_for(identifier);path=folder/'plan.json'
    result=json.loads(path.read_text()) if path.exists() else json.loads((folder/'capture.json').read_text())
    if (folder/'selection.json').exists():result['selection']=json.loads((folder/'selection.json').read_text())
    return result

@app.post('/api/captures/{identifier}/select')
def select_object(identifier:str,body:dict):
    folder=folder_for(identifier);role=body.get('role')
    if role not in ['from','to','obstacle']:raise ValueError('Choose From, To or Bowl obstacle')
    with operation_lock:
        if restarting:raise ValueError('Server restarting')
        if current!=identifier:raise ValueError('Capture was replaced; use the current capture')
        if jobs.get(identifier,{}).get('status') in ['running','executing','recording']:raise ValueError('Wait for the current operation')
        if hardware.call('status').get('running'):raise ValueError('Stop the robot before selecting objects')
        capture=json.loads((folder/'capture.json').read_text())
        if role=='obstacle' and (capture.get('task')!='stack' or not capture.get('bowl_present',True)):raise ValueError('Bowl obstacle is only used for stacking with a bowl present')
        if capture.get('source')!='live' or capture.get('purpose')=='home':raise ValueError('Capture a live scene to select objects')
        plan=folder/'plan.json'
        if plan.exists() and json.loads(plan.read_text()).get('executed'):raise ValueError('Capture a new scene after execution')
        path=folder/'selection.json';selection=json.loads(path.read_text()) if path.exists() else {}
        result=sam.predict(folder/'top.jpg',body.get('point'))
        if result['image_sha256']!=hashlib.sha256((folder/'top.jpg').read_bytes()).hexdigest():raise ValueError('Capture changed during segmentation')
        if selection and selection['model']!=result['model']:raise ValueError('SAM model changed; capture a new scene')
        revision=uuid.uuid4().hex
        selected=save_role(folder,role,result,revision)
        for other in [r for r in ['from','to','obstacle'] if r!=role and r in selection]:
            mask=cv2.imread(str(folder/selection[other]['mask_file']),cv2.IMREAD_GRAYSCALE)
            if mask is None:raise ValueError('Previous mask missing; capture again')
            validate_pair(decode_mask(result),mask>0)
        selection.update({role:selected,'revision':revision,'image_sha256':result['image_sha256'],'model':result['model']})
        path.write_text(json.dumps(selection,indent=2))
        # A new selection invalidates every old trajectory and simulation.
        for name in ['plan.json','plan.npz']:
            (folder/name).unlink(missing_ok=True)
        jobs[identifier]={'id':identifier,'status':'captured','stage':'capture','capture':capture}
        return selection


def verify_selected_scene(folder,report,fresh):
    from studio_stack import scene_calibration
    selection,_=load_selection(folder)
    if report.get('selection_revision')!=selection['revision']:raise ValueError('Selection changed; regenerate and simulate the plan')
    image=folder/f'preflight-{uuid.uuid4().hex}.jpg'
    image.write_bytes(base64.b64decode(fresh['images']['top']))
    checked,masks=refresh_pair(sam,folder,selection,image,image.stem)
    if checked['model']!=selection['model']:raise ValueError('SAM model changed; capture and select again')
    cfg=json.loads((ROOT/'calibration/native-action-current.json').read_text())
    observed=scene_calibration(cfg,scene_from_masks(cfg,*masks,task=report.get('task','bowl'),bowl_present=report.get('bowl_present',True)))
    validate_scene_match(report['world'],observed.world)


def verify_home_scene(folder,reference,report,fresh,cfg):
    from strict_action_scene import episode_scene
    view=report.get('home_view','top')
    image=folder/f'home-preflight-{uuid.uuid4().hex}-{view}.jpg';image.write_bytes(base64.b64decode(fresh['images'][view]))
    if view=='wrist':checked=wrist_home_scene(sam,folder,reference,image,cfg,fresh['initial_native'],image.stem)
    else:checked=home_scene(sam,folder,reference,image,cfg,image.stem)
    observed=episode_scene(cfg,checked['scene'])
    if 'bowl_position_from_wrist' in checked['scene']:observed.world['bowl']['position']=checked['scene']['bowl_position_from_wrist']
    validate_scene_match(report['world'],observed.world)


@app.post('/api/plans/{identifier}/{operation}')
def plan_operation(identifier:str,operation:str,request:Request,body:dict|None=None):
    global current
    if restarting:raise ValueError('Server restarting')
    if operation not in ['plan','simulate','execute']:raise ValueError('Unknown operation')
    from studio_motion import validate_technique
    from dataset_to_sim.studio import validate_speed_scale
    body=body or {}
    from studio_timing import validate_timing_profile
    timing_profile=validate_timing_profile(body.get('timing_profile','conservative'))
    technique=validate_technique(body.get('motion_technique','classic'))
    speed_scale=validate_speed_scale(body.get('speed_scale',1.))
    folder=folder_for(identifier)
    if json.loads((folder/'capture.json').read_text()).get('purpose')=='home':raise ValueError('Capture a new scene after Home')
    requested_generation=stop_generation
    with operation_lock:
        if restarting:raise ValueError('Server restarting')
        if current!=identifier:raise ValueError('This capture was replaced; use the current capture')
        if dataset_build_busy():raise ValueError('Finish or cancel data generation before planning or running the robot')
        if jobs.get(identifier,{}).get('status')=='running':raise ValueError('Operation already running')
        if hardware.call('status').get('running'):raise ValueError('Robot is executing')
        if json.loads((folder/'capture.json').read_text()).get('source')=='live':
            selection,_=load_selection(folder)
            if operation!='plan' and (folder/'plan.json').exists():
                if json.loads((folder/'plan.json').read_text()).get('selection_revision')!=selection['revision']:raise ValueError('Selection changed; regenerate the plan')
        if operation!='plan' and 'motion_technique' in body:
            saved_technique=json.loads((folder/'plan.json').read_text()).get('motion_technique','classic')
            if technique!=saved_technique:raise ValueError('Motion technique changed; generate and simulate a new plan')
        if operation!='plan' and 'speed_scale' in body:
            if speed_scale!=json.loads((folder/'plan.json').read_text()).get('speed_scale',1.):raise ValueError('Travel speed changed; generate and simulate a new plan')
        if operation!='plan' and 'timing_profile' in body:
            if timing_profile!=json.loads((folder/'plan.json').read_text()).get('timing_profile','conservative'):raise ValueError('Motion timing changed; generate and simulate a new plan')
        if operation=='execute':
            report=json.loads((folder/'plan.json').read_text())
            fresh=hardware.call('snapshot');execution_gate(report,fresh['initial_native'],fresh['calibration_hash'])
            verify_selected_scene(folder,report,fresh)
            # SAM may take several seconds; re-check robot state immediately before dispatch.
            latest=hardware.call('snapshot');execution_gate(report,latest['initial_native'],latest['calibration_hash'])
            validate_vision_age(fresh.get('image_captured_at',{}).get('top',fresh['captured_at']),time.time())
            if latest['stop_generation']!=fresh['stop_generation']:raise ValueError('Stop requested during segmentation')
            saved=np.load(folder/'plan.npz');payload=report|{'actions':saved['actions'].tolist(),'timestamps':saved['timestamps'].tolist(),'stop_generation':fresh['stop_generation'],'verified_image_captured_at':fresh.get('image_captured_at',{}).get('top',fresh['captured_at']),'owner':request.headers.get('x-studio-client')}
            with stop_lock:
                if restarting:raise ValueError('Server restarting; execution cancelled')
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
            invalid.update(outcome='not_simulated',transfer_checks_passed=False,simulated_action_hash=None,simulated_timestamp_hash=None,simulated_speed_scale=None,simulated_timing_profile=None,simulated_timing_contract_hash=None)
            invalid.pop('video',None);old.write_text(json.dumps(invalid,indent=2))
        if operation=='plan':(folder/'planning-options.json').write_text(json.dumps({'motion_technique':technique,'speed_scale':speed_scale,'timing_profile':timing_profile}))
        (folder/'job.json').unlink(missing_ok=True)
        jobs[identifier]={'id':identifier,'status':'running','stage':operation}
        def run():
            try:
                with open(folder/f'{operation}.log','w') as log:
                    result=subprocess.run([sys.executable,str(ROOT/'scripts/studio_planner.py'),operation,str(folder)],stdout=log,stderr=subprocess.STDOUT,timeout=240)
                if result.returncode:
                    lines=(folder/f'{operation}.log').read_text().splitlines();raise ValueError(lines[-1] if lines else 'Planning failed')
                jobs[identifier]={'id':identifier,'status':'complete','stage':operation,'plan':json.loads((folder/'plan.json').read_text())}
            except Exception as exc:job_error(identifier,operation,exc)
        threading.Thread(target=run,daemon=True).start()
        return jobs[identifier]

if __name__=='__main__':
    import argparse,uvicorn
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if not args.worker:
        supervise(args.port);sys.exit(0)
    print(f'Real / Sim Studio: http://127.0.0.1:{args.port}',flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='warning')
