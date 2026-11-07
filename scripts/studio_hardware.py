"""Private stdin/stdout worker. Owns the SO-101 connection in its LeRobot environment.

No socket listener. JSON requests arrive only from the parent Studio process.
Opening the website never connects hardware or sends targets.
"""
import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request
import fcntl
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2  # Load OpenCV before LeRobot/Torch to avoid duplicate OpenMP on macOS.
import numpy as np
from studio_recording import Recording
from dataset_to_sim.studio import JOINTS, validate_trajectory, execution_gate, native_limits, validate_simulated_timing


def open_calibrated_connection(robot):
    # LeRobot robot.connect() configures motors inside torque_disabled().
    # This workcell is already configured: preserve torque across handoffs.
    robot.bus.connect()
    if not robot.is_calibrated:raise ValueError('Motor calibration does not match the saved calibration')
    torque=[robot.bus.read('Torque_Enable',name,normalize=False) for name in JOINTS]
    return True if all(v==1 for v in torque) else False if all(v==0 for v in torque) else None


def prepare_follower_speed(bus,reset):
    if any(bus.read('Operating_Mode',name,normalize=False)!=0 for name in JOINTS):raise ValueError('Position mode required')
    # Re-staging a measured goal on every connection accumulates gravity sag.
    # Preserve the existing hold when the required speed setting is already set.
    if any(bus.read('Goal_Velocity',name,normalize=False)!=0 for name in JOINTS):reset(bus)


class Hardware:
    def __init__(self):
        self.robot=None;self.connected=False;self.running=False;self.torque_enabled=True
        self.io=threading.RLock();self.stopped=threading.Event();self.stopped.set()
        self.hold_latched=False
        self.frames={};self.cameras=[];self.last_heartbeat=time.monotonic()
        self.progress=0.;self.error=None;self.execution=None;self.lockfile=None
        self.config=json.loads((ROOT/'calibration/native-action-current.json').read_text())
        self.started_at=None;self.recording_error=None;self.history=[];self.stop_generation=0;self.execution_owner=None;self.consumed=set()

    def positions(self):
        if not self.connected:raise ValueError('Robot is not connected')
        with self.io:
            obs=self.robot.get_observation()
            q=np.array([obs[f'{j}.pos'] for j in JOINTS],float)
        if not np.isfinite(q).all():raise ValueError('Non-finite encoder reading')
        return q

    def calibration_hash(self):
        return hashlib.sha256(Path(self.config['follower_calibration_source']).read_bytes()).hexdigest()

    def connect(self):
        if self.connected:return self.status()
        # Never steal an active VLA-Benchmark session's serial/camera resources.
        try:
            with urllib.request.urlopen('http://127.0.0.1:8080/api/robot/status',timeout=2) as response:status=json.load(response)
        except OSError:status={}
        if status.get('connected') or status.get('operation_owner'):
            raise ValueError('VLA-Benchmark is using the robot; disconnect its control session first')
        import yaml,cv2
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from importlib.util import spec_from_file_location,module_from_spec
        benchmark=Path(os.environ.get('STUDIO_VLA_ROOT','/Users/phawit/Projects/VLA-Benchmark'))
        cfg=yaml.safe_load((benchmark/'config.yaml').read_text())['robot']
        if self.calibration_hash()!=self.config['follower_calibration_sha256']:
            raise ValueError('Current robot calibration differs from the simulator profile')
        self.lockfile=open(ROOT/'output/studio/hardware.lock','w')
        fcntl.flock(self.lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
        robot=make_robot_from_config(SOFollowerRobotConfig(port=cfg['port'],id=cfg['id'],cameras={},use_degrees=True))
        try:
            torque=open_calibrated_connection(robot)
            spec=spec_from_file_location('studio_handoff',benchmark/'robot_handoff.py');mod=module_from_spec(spec);spec.loader.exec_module(mod)
            prepare_follower_speed(robot.bus,mod.reset_follower_speed)
            self.robot=robot;self.connected=True;self.torque_enabled=torque;self.error=None;self.hold_latched=False
            for name,camera in cfg.get('cameras',{}).items():
                cap=cv2.VideoCapture(camera['index']);cap.set(cv2.CAP_PROP_FRAME_WIDTH,camera.get('w',640));cap.set(cv2.CAP_PROP_FRAME_HEIGHT,camera.get('h',480))
                self.cameras.append(cap)
                def loop(cap=cap,name=name):
                    while self.connected:
                        ok,image=cap.read()
                        if ok:self.frames[name]=(time.monotonic(),image.copy())
                        else:time.sleep(.1)
                threading.Thread(target=loop,daemon=True).start()
            return self.status()
        except Exception:
            if getattr(robot,'is_connected',False):
                robot.config.disable_torque_on_disconnect=False;robot.disconnect()
            self.connected=False;self.robot=None
            self.lockfile.close();self.lockfile=None
            raise

    def status(self,client_id=None):
        if client_id and client_id==self.execution_owner:self.last_heartbeat=time.monotonic()
        result={'connected':self.connected,'torque_enabled':self.torque_enabled,'running':self.running,'stopped':self.stopped.is_set(),'progress':self.progress,'error':self.error,'execution':self.execution,'elapsed':time.monotonic()-self.started_at if self.running and self.started_at else None,'recording_error':self.recording_error}
        if self.connected:
            result['positions']=self.positions().tolist()
            result['cameras']={name:time.monotonic()-stamp<.75 for name,(stamp,_) in self.frames.items()}
        return result

    def snapshot(self):
        if self.torque_enabled is not True:raise ValueError('Torque is off; enable torque before capture')
        import cv2
        before=self.positions();stable_since=time.monotonic();poses=[before]
        # Cover the entire possible camera age, not just two adjacent bus reads.
        for _ in range(10):
            time.sleep(.1);poses.append(self.positions())
            if np.max(np.ptp(np.asarray(poses),axis=0))>1:raise ValueError('Robot must be stable for one second before capture')
        now=time.monotonic();frames={};image_times={}
        for name in ['top','wrist']:
            frame=self.frames.get(name)
            if frame is None or now-frame[0]>.75 or frame[0]<stable_since:raise ValueError(f'{name} camera has no fresh frame')
            ok,encoded=cv2.imencode('.jpg',frame[1],[cv2.IMWRITE_JPEG_QUALITY,95])
            if not ok:raise ValueError('Camera encoding failed')
            frames[name]=base64.b64encode(encoded).decode()
            image_times[name]=time.time()-(time.monotonic()-frame[0])
        after=self.positions()
        poses.append(after)
        if np.max(np.ptp(np.asarray(poses),axis=0))>1:raise ValueError('Robot moved during capture')
        return {'source':'live','captured_at':time.time(),'initial_native':after.tolist(),'calibration_hash':self.calibration_hash(),'images':frames,'image_captured_at':image_times,'stop_generation':self.stop_generation}

    def preview(self,view):
        if view not in ['top','wrist']:raise ValueError('Unknown camera')
        self.check_cameras()
        ok,image=cv2.imencode('.jpg',self.frames[view][1],[cv2.IMWRITE_JPEG_QUALITY,80])
        if not ok:raise ValueError('Camera encoding failed')
        return {'image':base64.b64encode(image).decode()}

    def check_cameras(self):
        for name in ['top','wrist']:
            if name not in self.frames or time.monotonic()-self.frames[name][0]>.75:
                raise ValueError(f'{name} camera stream lost; motion stopped')

    def send(self,q):
        with self.io:
            if self.stopped.is_set():raise RuntimeError('Execution stopped')
            self.hold_latched=False
            self.robot.send_action({f'{j}.pos':float(v) for j,v in zip(JOINTS,q)})

    def stop(self):
        # Latch under the same lock as sending, then hold actual measured pose.
        with self.io:
            self.stopped.set();self.stop_generation+=1
            if self.connected and self.torque_enabled and not self.hold_latched:
                q=self.positions()
                self.robot.send_action({f'{j}.pos':float(v) for j,v in zip(JOINTS,q)})
                self.hold_latched=True
        return {'stopped':True,'note':'Software stop; holds measured pose with torque enabled'}

    def torque_off(self):
        with self.io:
            if not self.connected:raise ValueError('Robot is not connected')
            if self.running:raise ValueError('Stop the running motion first')
            self.stopped.set();self.stop_generation+=1
            self.torque_enabled=None
            self.robot.bus.disable_torque()
            if any(self.robot.bus.read('Torque_Enable',name,normalize=False)!=0 for name in JOINTS):raise ValueError('Torque state unknown; retry Torque off or Torque on')
            self.torque_enabled=False
        return self.status()

    def torque_on(self):
        with self.io:
            if not self.connected:raise ValueError('Robot is not connected')
            if self.running:raise ValueError('Stop the running motion first')
            self.torque_enabled=None
            bus=self.robot.bus
            q=bus.sync_read('Present_Position',normalize=False)
            if set(q)!=set(JOINTS) or any(not isinstance(x,(int,float)) or not np.isfinite(x) or not 0<=x<=4095 or int(x)!=x for x in q.values()):raise ValueError('Invalid raw encoder positions')
            if any(bus.read('Operating_Mode',name,normalize=False)!=0 for name in JOINTS):raise ValueError('Position mode required')
            for name in JOINTS:bus.write('Goal_Position',name,int(q[name]),normalize=False)
            if any(bus.read('Goal_Position',name,normalize=False)!=q[name] for name in JOINTS):raise ValueError('Could not verify hold positions; torque state unknown')
            bus.enable_torque()
            if any(bus.read('Torque_Enable',name,normalize=False)!=1 for name in JOINTS):raise ValueError('Torque state unknown; retry Torque on')
            self.torque_enabled=True
            self.stopped.set();self.stop_generation+=1
        return self.status()

    def execute(self,plan):
        with self.io:
            if self.torque_enabled is not True:raise ValueError('Torque is off; enable torque first')
            if self.running:raise ValueError('Robot execution already running')
            execution_gate(plan,self.positions(),self.calibration_hash())
            if plan.get('stop_generation')!=self.stop_generation:raise ValueError('A stop was requested; start a new execution explicitly')
            self.check_cameras()
            if not plan.get('owner'):raise ValueError('Execution requires an owning browser page')
            if plan['id'] in self.consumed:raise ValueError('Plan already consumed')
            actions=np.asarray(plan['actions'],float);ts=np.asarray(plan['timestamps'],float)
            validate_trajectory(actions,ts,native_limits(self.config),speed_scale=plan.get('speed_scale',1.))
            validate_simulated_timing(plan,ts)
            actual_hash=hashlib.sha256(np.ascontiguousarray(actions,dtype='<f8').tobytes()).hexdigest()
            if actual_hash!=plan['action_hash'] or actual_hash!=plan.get('simulated_action_hash'):
                raise ValueError('Commands do not match the simulated trajectory')
            if np.max(np.abs(actions[0]-plan['initial_native']))>1e-6:raise ValueError('First command must equal captured pose')
            from studio_sam_geometry import validate_vision_age
            validate_vision_age(plan.get('verified_image_captured_at',0),time.time())
            recorder=Recording(ROOT/'output/studio'/plan['id'],self.frames,self.stopped)
            self.running=True;self.recording_error=None;self.stopped.clear();self.error=None;self.progress=0.;self.last_heartbeat=time.monotonic();self.execution=plan['id'];self.history=[];self.execution_owner=plan['owner'];self.consumed.add(plan['id'])
        def run():
            start=time.monotonic();self.started_at=start;journal=None
            try:
                recorder.start(start)
                journal_path=ROOT/'output/studio'/plan['id']/'hardware-commands.jsonl'
                journal_path.parent.mkdir(parents=True,exist_ok=True)
                journal=journal_path.open('a',buffering=1)
                for i,(q,t) in enumerate(zip(actions,ts)):
                    if self.stopped.wait(max(0,start+float(t)-time.monotonic())):break
                    if time.monotonic()-self.last_heartbeat>3:raise RuntimeError('Control page heartbeat lost')
                    if time.monotonic()-(start+float(t))>.15:raise RuntimeError('Command timing deadline missed')
                    self.check_cameras()
                    measured=self.positions()
                    previous=actions[max(0,i-1)]
                    if np.any(np.abs(measured-previous)>np.array([10.]*5+[25.])):raise RuntimeError('Robot deviated from planned motion')
                    self.send(q);self.progress=(i+1)/len(actions)
                    self.history.append({'time':time.monotonic()-start,'target':q.tolist(),'measured':measured.tolist()})
                    journal.write(json.dumps(self.history[-1])+'\n')
                if not self.stopped.is_set():self.stopped.wait(1/30)
            except Exception as exc:self.error=str(exc)
            finally:
                try:self.stop()
                except Exception as exc:self.error=f'Hold failed: {exc}'
                try:
                    if journal:journal.close()
                except Exception as exc:
                    if not self.error:self.error=f'Command journal failed: {exc}'
                try:
                    recording=recorder.finish();self.recording_error=recording['error']
                except Exception as exc:self.recording_error=str(exc)
                if self.recording_error and not self.error:self.error='camera recording failed: '+self.recording_error
                path=ROOT/'output/studio'/plan['id']/'hardware-trace.json'
                try:path.write_text(json.dumps({'error':self.error,'samples':self.history,'completed':self.progress==1. and self.error is None}))
                except OSError as exc:
                    if not self.error:self.error='Could not save hardware trace: '+str(exc)
                finally:self.running=False
        threading.Thread(target=run,daemon=True).start()
        return {'running':True,'id':self.execution}

    def disconnect(self):
        if self.running:raise ValueError('Stop the running motion before disconnecting')
        self.stop();self.connected=False
        for camera in self.cameras:camera.release()
        self.cameras=[];self.frames={}
        if self.robot:
            self.robot.config.disable_torque_on_disconnect=False;self.robot.disconnect();self.robot=None
        if self.lockfile:self.lockfile.close();self.lockfile=None
        return {'connected':False}


def main():
    protocol=sys.stdout;sys.stdout=sys.stderr
    hardware=Hardware()
    try:
        for line in sys.stdin:
            try:
                request=json.loads(line);command=request['command']
                if command not in ['connect','status','snapshot','preview','execute','stop','disconnect','torque_off','torque_on']:raise ValueError('Unknown command')
                result=hardware.execute(request['plan']) if command=='execute' else hardware.status(request.get('client_id')) if command=='status' else hardware.preview(request.get('view')) if command=='preview' else getattr(hardware,command)()
                response={'ok':True,'result':result}
            except Exception as exc:response={'ok':False,'error':str(exc)}
            protocol.write(json.dumps(response)+'\n');protocol.flush()
    finally:
        try:hardware.stop()
        except Exception:pass

if __name__=='__main__':main()
