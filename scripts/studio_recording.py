"""Timestamped camera recording and real/sim playback on the command clock."""
from pathlib import Path
import json,threading,time,subprocess,sys
import cv2
import numpy as np


def causal_frame_indices(sample_times, query_times):
    samples=np.asarray(sample_times,float);queries=np.asarray(query_times,float)
    if samples.ndim!=1 or len(samples)==0 or not np.isfinite(samples).all() or np.any(np.diff(samples)<0):raise ValueError('Camera timestamps must be finite and ordered')
    return np.maximum(0,np.searchsorted(samples,queries,side='right')-1)


class Recording:
    def __init__(self,folder,frames,abort):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.frames=frames;self.abort=abort;self.done=threading.Event();self.samples=[];self.error=None
        self.writer=cv2.VideoWriter(str(self.folder/'hardware-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,480))
        if not self.writer.isOpened():raise ValueError('Could not open hardware video recorder')
    def start(self,start):
        self.origin=start
        def loop():
            try:
                next_frame=start
                while not self.done.is_set():
                    now=time.monotonic();images=[];stamps={}
                    for name in ['top','wrist']:
                        stamp,frame=self.frames[name];images.append(cv2.resize(frame,(640,480)));stamps[name]=stamp-start
                    self.writer.write(np.hstack(images))
                    self.samples.append({'time':now-start,'camera_times':stamps})
                    next_frame+=1/30
                    self.done.wait(max(0,next_frame-time.monotonic()))
            except Exception as exc:self.error=str(exc);self.abort.set()
            finally:self.writer.release()
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
    def finish(self):
        self.done.set();self.thread.join(timeout=5)
        if self.thread.is_alive():self.error='Camera recorder did not finish'
        report={'fps':30,'samples':self.samples,'error':self.error,'time_origin':'same monotonic origin as hardware commands','alignment':'nearest preceding recorded sample; no dynamic time warping'}
        (self.folder/'hardware-video.json').write_text(json.dumps(report,indent=2))
        return report


def comparison_times(real_duration,simulated_frame_count):
    if simulated_frame_count<1:raise ValueError('Simulation contains no frames')
    return np.arange(min(simulated_frame_count,max(1,int(real_duration*30)+1)))/30


def comparison(folder):
    folder=Path(folder);metadata=json.loads((folder/'hardware-video.json').read_text());trace=json.loads((folder/'hardware-trace.json').read_text())
    times=[f['time'] for f in metadata['samples']]
    if not times or not trace['samples']:raise ValueError('No recorded motion to compare')
    duration=max(times[-1],trace['samples'][-1]['time'])
    real=cv2.VideoCapture(str(folder/'hardware-raw.mp4'));sim=cv2.VideoCapture(str(folder/'simulation.mp4'))
    queries=comparison_times(duration,int(sim.get(cv2.CAP_PROP_FRAME_COUNT)));indices=causal_frame_indices(times,queries)
    output=folder/'hardware-comparison-raw.mp4';writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,1024))
    if not writer.isOpened():raise ValueError('Comparison encoder unavailable')
    previous=-1;real_frame=None
    try:
        for i,(t,idx) in enumerate(zip(queries,indices)):
            if idx!=previous:
                real.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,real_frame=real.read();previous=idx
                if not ok:raise ValueError('Recorded real video is incomplete')
            sim.set(cv2.CAP_PROP_POS_FRAMES,int(round(t*30)));ok,sim_frame=sim.read()
            if not ok:raise ValueError('Simulation video is incomplete')
            canvas=np.zeros((1024,1280,3),np.uint8)
            for view,(name,y) in enumerate([('top',0),('wrist',512)]):
                canvas[y+32:y+512,:640]=real_frame[:,view*640:(view+1)*640]
                canvas[y+32:y+512,640:]=sim_frame[:,view*640:(view+1)*640]
                cv2.putText(canvas,f'REAL EXECUTION | {name} | {t:.2f}s',(10,y+23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                cv2.putText(canvas,f'SIM PLAN | {name} | same command clock',(650,y+23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            writer.write(canvas)
    finally:writer.release();real.release();sim.release()
    partial=folder/'hardware-comparison-partial.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(output),'-c:v','libx264','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)],check=True)
    partial.replace(folder/'hardware-comparison.mp4');output.unlink()
    report={'video':'hardware-comparison.mp4','frames':len(queries),'duration':len(queries)/30,'completed':trace['completed'],'error':trace['error'],'camera_sample_count':len(times),'alignment':metadata['alignment'],'real_recording_duration':duration,'unpaired_tail_seconds':max(0,duration-queries[-1])}
    (folder/'execution.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':comparison(Path(sys.argv[1]))
