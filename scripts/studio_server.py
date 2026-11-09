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
from studio_imports import ImportManager
from studio_generation_assets import list_backgrounds, resolve_background, upload_background, public_background, MAX_UPLOAD_BYTES
from studio_runtime import request_restart,supervise
from studio_episode_appearance import APPEARANCE_VERSION,video_name as appearance_video_name
from studio_sam_home import home_scene
from studio_home_wrist import released_reference,wrist_home_scene
from studio_sam_client import SAMClient,decode_mask,save_role,refresh_pair,auto_selection,auto_place_record,finalize_selection,object_record,place_record
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
imports=ImportManager();atexit.register(imports.close)
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

@app.get('/api/affordances')
def affordances():
    from pickplace import load_registry
    return load_registry()

@app.post('/api/affordances')
def save_affordances(body:dict):
    """Contact areas edited on the 3D-Objects page; validated before New Run uses them."""
    from pickplace import save_registry
    with operation_lock:
        if current and jobs.get(current,{}).get('status')=='running':raise ValueError('Wait for the current plan or simulation to finish')
        return save_registry(body)

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
    imported=[{'id':item['id'],'name':item['name'],'episode_count':item['completed'],'successes':item['successes'],'fps':30,'profile':'auto-calibrated import','source':'import'}
              for item in imports.list(complete_only=True)]
    return {'datasets':[{'id':DATA.name,'name':DATA.name,'episode_count':data['total'],'successes':data['successes'],'fps':30,'profile':data['profile']}]+imported}

def imported_dataset(identifier):
    """Folder, request and results of a completed import, or None for the built-in dataset."""
    if not identifier or identifier==DATA.name:return None
    folder=imports.folder(identifier);request=json.loads((folder/'request.json').read_text())
    results=json.loads((folder/'results.json').read_text()) if (folder/'results.json').is_file() else None
    if results is None:raise ValueError('Import has no replay results yet')
    return folder,request,results

def imported_episode_folder(folder,index):
    path=folder/f'episode-{index:03d}'
    if index<0 or not (path/'report.json').is_file():raise ValueError('Episode not found')
    return path

@app.get('/api/imports')
def import_index():return {'imports':imports.list(),'busy':imports.busy()}

@app.get('/api/imports/library')
def import_library():return {'objects':imports.library()}

@app.post('/api/imports')
def import_start(body:dict):return imports.start(body)

@app.get('/api/imports/{identifier}')
def import_status(identifier:str):return imports.get(identifier)

@app.post('/api/imports/{identifier}/cancel')
def import_cancel(identifier:str):return imports.cancel(identifier)

@app.get('/api/datasets/{identifier}')
def selected_dataset(identifier:str):
    found=imported_dataset(identifier)
    if found:
        folder,request,results=found
        episodes=[{'episode':r['episode'],'outcome':r['outcome'],'frames':r.get('frames'),'joint_state_rmse_native':r.get('joint_state_rmse_native'),
                   'has_video':(folder/f"episode-{r['episode']:03d}/real-vs-sim.mp4").is_file()} for r in results['results'] if r.get('frames')]
        return {'dataset':request['name'],'episodes':episodes,'successes':results['successes'],'total':len(episodes),'profile':'auto-calibrated import'}
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
def episode(index:int,dataset:str|None=None):
    found=imported_dataset(dataset)
    if found:
        folder,request,_=found;path=imported_episode_folder(folder,index)
        ep=DatasetIndex(request['dataset_path']).episode(index);report=json.loads((path/'report.json').read_text())
        with np.load(path/'trace.npz') as trace:sim=trace['native'][:ep.frame_count].tolist()
        video=f'/media/imports/{folder.name}/episode-{index:03d}/real-vs-sim.mp4' if (path/'real-vs-sim.mp4').is_file() else None
        return {'episode':index,'frames':ep.frame_count,'duration':ep.frame_count/30,'timestamps':ep.timestamps.tolist(),'real':ep.state_deg.tolist(),'sim':sim,'actions':ep.action_deg.tolist(),'report':report,'video':video}
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
        name=appearance_video_name(appearance)
        if (WORK/f'replay-{index:02d}'/name).is_file():return {'status':'ready','video':f'/media/studio/replay-{index:02d}/{name}'}
        return video_jobs.get((index,appearance),{'status':'idle'})
    if (folder/'real-vs-sim-strict.mp4').is_file():return {'status':'ready','video':f'/media/benchmark/episode-{index:02d}/real-vs-sim-strict.mp4'}
    if (WORK/f'replay-{index:02d}'/'comparison.mp4').is_file():return {'status':'ready','video':f'/media/studio/replay-{index:02d}/comparison.mp4'}
    return video_jobs.get((index,'raw'),{'status':'idle'})

@app.get('/api/episodes/{index}/video')
def video_status(index:int,appearance:str='raw',dataset:str|None=None):return imported_video(index,dataset,appearance_mode(appearance)) or episode_video_status(index,appearance_mode(appearance))

def imported_video(index,dataset,appearance='raw',start=False):
    """Imported episodes: replay video, or Real background / Realistic views rendered on demand."""
    found=imported_dataset(dataset)
    if not found:return None
    folder=imported_episode_folder(found[0],index);base=f'/media/imports/{found[0].name}/episode-{index:03d}/'
    if appearance=='raw':
        return {'status':'ready','video':base+'real-vs-sim.mp4'} if (folder/'real-vs-sim.mp4').is_file() else {'status':'error','error':'Replay video missing'}
    from import_appearance import video_name as import_video_name
    name=import_video_name(appearance);key=(found[0].name,index,appearance)
    if (folder/name).is_file():return {'status':'ready','video':base+name}
    with video_lock:
        if not start or video_jobs.get(key,{}).get('status')=='rendering':return video_jobs.get(key,{'status':'idle'})
        video_jobs[key]={'status':'rendering'}
        def render():
            try:
                result=subprocess.run([sys.executable,str(ROOT/'scripts/import_appearance.py'),'video',found[0].name,str(index),appearance],capture_output=True,text=True,timeout=1200)
                if result.returncode:raise ValueError(result.stderr.strip().splitlines()[-1] if result.stderr else 'Render failed')
                video_jobs[key]={'status':'ready','video':base+name}
            except Exception as exc:video_jobs[key]={'status':'error','error':str(exc)}
        video_pool.submit(render)
        return video_jobs[key]

@app.post('/api/episodes/{index}/video')
def prepare_video(index:int,appearance:str='raw',dataset:str|None=None):
    status=imported_video(index,dataset,appearance_mode(appearance),start=True)
    if status:return status
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
                name='comparison.mp4' if appearance=='raw' else appearance_video_name(appearance)
                if not (WORK/f'replay-{index:02d}'/name).is_file():raise ValueError('Render produced no video')
                video_jobs[key]={'status':'ready','video':f'/media/studio/replay-{index:02d}/{name}'}
            except Exception as exc:video_jobs[key]={'status':'error','error':str(exc)}
        video_pool.submit(render)
        return video_jobs[key]

@app.get('/api/episodes/{index}/frame/{frame}')
def frame(index:int,frame:int,appearance:str='raw',dataset:str|None=None):
    found=imported_dataset(dataset)
    if found:
        path=imported_episode_folder(found[0],index)
        if appearance!='raw':
            from import_appearance import REFERENCE_VERSION
            target=path/f'frame-{frame:03d}-{appearance}-v{APPEARANCE_VERSION}-i{REFERENCE_VERSION}.jpg'
            with frames_lock:
                if not target.is_file():
                    result=subprocess.run([sys.executable,str(ROOT/'scripts/import_appearance.py'),'frame',found[0].name,str(index),appearance,str(frame),str(target)],capture_output=True,text=True,timeout=300)
                    if result.returncode:raise ValueError(result.stderr.strip().splitlines()[-1] if result.stderr else 'Frame render failed')
            return FileResponse(target)
        target=path/f'frame-{frame:03d}.jpg'
        if not target.is_file():
            capture=cv2.VideoCapture(str(path/'real-vs-sim.mp4'));capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=capture.read();capture.release()
            if not ok:raise ValueError('Frame out of bounds')
            cv2.imwrite(str(target),image)
        return FileResponse(target)
    appearance=appearance_mode(appearance)
    folder=RUN/f'episode-{index:02d}'
    if index<0 or frame<0 or not folder.is_dir():raise ValueError('Frame unavailable')
    report=json.loads((folder/'report.json').read_text())
    if frame>=report['frames']:raise ValueError('Frame out of bounds')
    target=WORK/(f'frame-{index}-{frame}.jpg' if appearance=='raw' else f'frame-{index}-{frame}-{appearance}-v{APPEARANCE_VERSION}.jpg')
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
    root={'benchmark':RUN,'studio':WORK,'imports':imports.root}.get(area)
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
            # Requested by the user: Home goes straight back from any pose, with no camera,
            # scene or collision checks. Stop still interrupts it.
            current=None
            with stop_lock:
                if restarting:raise ValueError('Server restarting')
                return hardware.call('go_home',target=home['home'])
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
        task=body.get('task','pickplace' if body.get('source','live')=='live' else 'bowl')
        if task not in ['pickplace','bowl','stack','unstack','tape']:raise ValueError('Choose Pick & place')
        if task=='pickplace' and body.get('source','live')!='live':raise ValueError('Pick & place requires a live capture')
        if 'bowl_present' in body and not isinstance(body['bowl_present'],bool):raise ValueError('Bowl presence must be true or false')
        if restarting:raise ValueError('Server restarting')
        if current and jobs.get(current,{}).get('status')=='running':raise ValueError('Wait for the current operation')
        if hardware.call('status').get('running'):raise ValueError('Stop the robot before capturing')
        identifier=uuid.uuid4().hex;folder=WORK/identifier;folder.mkdir()
        source=body.get('source','live')
        if task=='stack' and source!='live':raise ValueError('Stacking requires a live two-cube capture')
        if task=='unstack' and source!='live':raise ValueError('Unstacking requires a live capture of the stacked cubes')
        if task=='tape' and source!='live':raise ValueError('Place tape in bowl requires a live capture with the tape roll')
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
        # Stack/unstack: an explicit bowl choice is kept; otherwise the SAM obstacles decide.
        snapshot.update(id=identifier,task=task,bowl_present=body.get('bowl_present') if task in ('stack','unstack') else True)
        (folder/'capture.json').write_text(json.dumps(snapshot,indent=2))
        current=identifier;jobs[identifier]={'id':identifier,'status':'captured','stage':'capture','capture':snapshot}
        return jobs[identifier]

@app.get('/api/plans/{identifier}')
def get_plan(identifier:str):
    folder=folder_for(identifier);path=folder/'plan.json'
    result=json.loads(path.read_text()) if path.exists() else json.loads((folder/'capture.json').read_text())
    if (folder/'selection.json').exists():result['selection']=json.loads((folder/'selection.json').read_text())
    return result

def selection_guard(folder,identifier):
    if restarting:raise ValueError('Server restarting')
    if current!=identifier:raise ValueError('Capture was replaced; use the current capture')
    if jobs.get(identifier,{}).get('status') in ['running','executing','recording']:raise ValueError('Wait for the current operation')
    if hardware.call('status').get('running'):raise ValueError('Stop the robot before selecting objects')
    capture=json.loads((folder/'capture.json').read_text())
    if capture.get('source')!='live' or capture.get('purpose')=='home':raise ValueError('Capture a live scene to select objects')
    plan=folder/'plan.json'
    if plan.exists() and json.loads(plan.read_text()).get('executed'):raise ValueError('Capture a new scene after execution')
    return capture


def invalidate_plan(folder,identifier,capture):
    # A new selection invalidates every old trajectory and simulation.
    for name in ['plan.json','plan.npz']:
        (folder/name).unlink(missing_ok=True)
    jobs[identifier]={'id':identifier,'status':'captured','stage':'capture','capture':capture}


@app.post('/api/captures/{identifier}/auto_select')
def auto_select_objects(identifier:str,body:dict|None=None):
    """SAM every modelled object after Capture: From, To and all obstacles."""
    folder=folder_for(identifier)
    with operation_lock:
        capture=selection_guard(folder,identifier)
        cfg=json.loads((ROOT/'calibration/native-action-current.json').read_text())
        if capture.get('task')=='pickplace':
            from pickplace_vision import detect,remap,scene_file
            selection=detect(sam,folder,cfg,decode_mask)
            # Run all re-captures: keep the user's From/To on the same objects.
            remap(selection,(body or {}).get('previous'),cfg)
            (folder/'selection.json').write_text(json.dumps(selection,indent=2))
            if selection.get('from') and selection.get('to'):scene_file(folder,selection,cfg)
            else:(folder/'pickplace-scene.json').unlink(missing_ok=True)
            invalidate_plan(folder,identifier,capture)
            return selection
        selection=auto_selection(sam,folder,capture.get('task','bowl'),cfg)
        invalidate_plan(folder,identifier,capture)
        return selection


@app.post('/api/captures/{identifier}/select')
def select_object(identifier:str,body:dict):
    """Change one role by clicking: From, To (unstack: an empty mat spot) or add an obstacle."""
    from studio_sam_auto import classify
    folder=folder_for(identifier);role=body.get('role')
    if role not in ['from','to','obstacle']:raise ValueError('Choose From, To or Obstacle')
    with operation_lock:
        capture=selection_guard(folder,identifier)
        cfg=json.loads((ROOT/'calibration/native-action-current.json').read_text())
        path=folder/'selection.json';selection=json.loads(path.read_text()) if path.exists() else {}
        if capture.get('task')=='pickplace':
            from pickplace_vision import choose,scene_file
            if role not in ['from','to']:raise ValueError('Choose From or To')
            choose(selection,folder,role,body.get('point'),cfg)
            selection['revision']=uuid.uuid4().hex
            path.write_text(json.dumps(selection,indent=2))
            if selection.get('from') and selection.get('to'):scene_file(folder,selection,cfg)
            else:(folder/'pickplace-scene.json').unlink(missing_ok=True)
            invalidate_plan(folder,identifier,capture)
            return selection
        if 'obstacle' in selection and 'objects' not in selection:selection['objects']=[{**selection.pop('obstacle'),'kind':'bowl'}]
        revision=uuid.uuid4().hex
        if role=='to' and capture.get('task')=='unstack':
            selected=place_record(cfg,folder,body.get('point'),selection)
            selection.update({'to':selected,'revision':revision})
        else:
            result=sam.predict(folder/'top.jpg',body.get('point'))
            if result['image_sha256']!=hashlib.sha256((folder/'top.jpg').read_bytes()).hexdigest():raise ValueError('Capture changed during segmentation')
            if selection.get('model') and selection['model']!=result['model']:raise ValueError('SAM model changed; capture a new scene')
            mask=decode_mask(result)
            if role=='obstacle':
                rgb=cv2.cvtColor(cv2.imread(str(folder/'top.jpg')),cv2.COLOR_BGR2RGB);info=classify(cfg,rgb,mask)
                if not info:raise ValueError('Obstacles must be a modelled object: a 19.5 mm cube, the tape roll or the bowl')
                for item in selection.get('objects',[]):
                    other=cv2.imread(str(folder/item['mask_file']),cv2.IMREAD_GRAYSCALE)>0
                    if np.count_nonzero(mask&other)/max(1,min(mask.sum(),other.sum()))>.3:raise ValueError('That object is already selected')
                selection.setdefault('objects',[]).append(object_record(folder,len(selection.get('objects',[])),result,info,revision))
            else:
                selected=save_role(folder,role,result,revision)
                for other in [r for r in ['from','to'] if r!=role and r in selection and 'mask_file' in selection[r]]:
                    validate_pair(mask,cv2.imread(str(folder/selection[other]['mask_file']),cv2.IMREAD_GRAYSCALE)>0)
                selection[role]=selected
                if role=='from' and capture.get('task')=='unstack':
                    # The empty spot was checked against the old stack: choose a new one.
                    selection.pop('to',None)
                    try:
                        place=auto_place_record(cfg,folder,selection)
                        if place:selection['to']=place
                    except ValueError:pass
            selection.update({'revision':revision,'image_sha256':result['image_sha256'],'model':result['model']})
        selection=finalize_selection(folder,selection)
        invalidate_plan(folder,identifier,capture)
        return selection


def wrist_recheck(folder,report,fresh):
    """After the Wrist check: the object must still be where the Wrist camera saw it."""
    from pickplace import load_registry,settle
    from pickplace_wrist import measure,colour_filter,RECHECK_SHIFT_M,RECHECK_TURN_RAD
    cfg=json.loads((ROOT/'calibration/native-action-current.json').read_text());registry=load_registry()
    scene=settle(json.loads((folder/'pickplace-scene.json').read_text()),registry,cfg)
    picked=next(o for o in scene['objects'] if o['id']==scene['pick'])
    image=folder/f'wrist-recheck-{uuid.uuid4().hex}.jpg';image.write_bytes(base64.b64decode(fresh['images']['wrist']))
    rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB)
    seen=measure(sam,image,cfg,registry,picked,fresh['initial_native'],decode_mask,colour_filter(rgb,picked.get('color')) if picked.get('support','mat')!='mat' else None)
    if seen['shift_m']>RECHECK_SHIFT_M or abs(seen['turn_rad'])>RECHECK_TURN_RAD:raise ValueError(f"The {picked['kind']} moved {seen['shift_m']*1000:.0f} mm since the Wrist check; capture again")


def verify_selected_scene(folder,report,fresh):
    if report.get('purpose')=='refine':return wrist_recheck(folder,report,fresh)
    if report.get('task')=='pickplace':
        # Every object is segmented again on a fresh image: none may have moved.
        from pickplace_vision import refresh,compare
        selection=json.loads((folder/'selection.json').read_text())
        if report.get('selection_revision')!=selection['revision']:raise ValueError('Selection changed; regenerate and simulate the plan')
        image=folder/f'preflight-{uuid.uuid4().hex}.jpg';image.write_bytes(base64.b64decode(fresh['images']['top']))
        cfg=json.loads((ROOT/'calibration/native-action-current.json').read_text())
        observed=refresh(sam,folder,selection,image,cfg,decode_mask,image.stem)
        planned=json.loads((folder/'pickplace-scene.json').read_text())
        compare(planned['objects'],observed,planned['pick'])
        return
    from studio_stack import scene_calibration
    selection,_=load_selection(folder)
    if report.get('selection_revision')!=selection['revision']:raise ValueError('Selection changed; regenerate and simulate the plan')
    image=folder/f'preflight-{uuid.uuid4().hex}.jpg'
    image.write_bytes(base64.b64decode(fresh['images']['top']))
    checked,masks=refresh_pair(sam,folder,selection,image,image.stem)
    if checked['model']!=selection['model']:raise ValueError('SAM model changed; capture and select again')
    cfg=json.loads((ROOT/'calibration/native-action-current.json').read_text())
    # Every obstacle is segmented again too: none may have moved since simulation.
    rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB)
    observed=scene_calibration(cfg,scene_from_masks(cfg,*masks,task=report.get('task','bowl'),bowl_present=report.get('bowl_present'),rgb=rgb))
    validate_scene_match(report['world'],observed.world)


@app.post('/api/plans/{identifier}/refine')
def refine_after_look(identifier:str):
    """At the Look pose: measure the picked object with the Wrist camera and start a
    new capture from this pose with the measured object, to plan and simulate again."""
    global current
    from pickplace import load_registry,settle
    from pickplace_wrist import measure,colour_filter
    folder=folder_for(identifier)
    with operation_lock:
        if restarting:raise ValueError('Server restarting')
        if current!=identifier:raise ValueError('This capture was replaced')
        report=json.loads((folder/'plan.json').read_text())
        if not (report.get('look_pending') and report.get('executed')):raise ValueError('No Wrist check is pending for this run')
        if hardware.call('status').get('running'):raise ValueError('Wait for the arm to stop at the Look pose')
        trace=json.loads((folder/'hardware-trace.json').read_text()) if (folder/'hardware-trace.json').exists() else {}
        if not trace.get('completed') or trace.get('error'):raise ValueError('The arm did not reach the Look pose: '+str(trace.get('error') or 'motion incomplete'))
        look=np.load(folder/'plan.npz')['actions'][report['look_index']]
        fresh=hardware.call('snapshot');q=np.asarray(fresh['initial_native'],float)
        if np.any(np.abs(q[:5]-look[:5])>4):raise ValueError('The arm is not at the Look pose; return Home and capture again')
        cfg=json.loads((ROOT/'calibration/native-action-current.json').read_text());registry=load_registry()
        child_id=uuid.uuid4().hex;child=WORK/child_id;child.mkdir()
        for view,image in fresh['images'].items():(child/f'{view}.jpg').write_bytes(base64.b64decode(image))
        scene=json.loads((folder/'pickplace-scene.json').read_text());settled=settle(json.loads(json.dumps(scene)),registry,cfg)
        picked=next(o for o in settled['objects'] if o['id']==settled['pick'])
        rgb=cv2.cvtColor(cv2.imread(str(child/'wrist.jpg')),cv2.COLOR_BGR2RGB)
        seen=measure(sam,child/'wrist.jpg',cfg,registry,picked,q,decode_mask,colour_filter(rgb,picked.get('color')) if picked.get('support','mat')!='mat' else None)
        save_role(child,'wrist-object',seen['result'],child_id)
        for o in scene['objects']:
            if o['id']==scene['pick']:o['position']=[seen['position'][0],seen['position'][1],0.];o['yaw']=seen['yaw']
        refinement={k:v for k,v in seen.items() if k not in ('mask','result')}|{'parent':identifier}
        scene['wrist_refinement']=refinement
        (child/'pickplace-scene.json').write_text(json.dumps(scene,indent=2))
        for name in ['selection.json','planning-options.json']:
            if (folder/name).exists():(child/name).write_bytes((folder/name).read_bytes())
        snapshot={k:v for k,v in fresh.items() if k!='images'}
        snapshot.update(id=child_id,task='pickplace',purpose='refine',parent=identifier,source='live',bowl_present=None)
        (child/'capture.json').write_text(json.dumps(snapshot,indent=2))
        report['look_pending']=False;report['wrist_refinement']=refinement;report['refined_capture']=child_id;(folder/'plan.json').write_text(json.dumps(report,indent=2))
        current=child_id;jobs[child_id]={'id':child_id,'status':'captured','stage':'capture','capture':snapshot}
        return {'capture':snapshot,'refinement':refinement}


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
        if json.loads((folder/'capture.json').read_text()).get('task')=='pickplace':
            if not (folder/'pickplace-scene.json').exists():raise ValueError('Choose From and To first')
            selection=json.loads((folder/'selection.json').read_text())
            if json.loads((folder/'pickplace-scene.json').read_text()).get('selection_revision')!=selection['revision']:raise ValueError('Selection changed; regenerate the plan')
            if operation!='plan' and (folder/'plan.json').exists() and json.loads((folder/'plan.json').read_text()).get('selection_revision')!=selection['revision']:raise ValueError('Selection changed; regenerate the plan')
        elif json.loads((folder/'capture.json').read_text()).get('source')=='live':
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
            if report.get('task')=='pickplace' and report.get('look_index') is not None:
                # Stop still above the object; the Wrist check then re-plans the grasp.
                payload['stop_index']=int(report['look_index']);report['look_pending']=True
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
