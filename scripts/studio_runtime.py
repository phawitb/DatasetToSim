"""Supervise one local Studio process group; restart never targets unrelated apps."""
import json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def request_restart():
    path=os.environ.get('STUDIO_RESTART_FILE');instance=os.environ.get('STUDIO_INSTANCE')
    if not path or not instance:raise ValueError('Restart unavailable: launch Studio with scripts/studio_server.py')
    # The restart file is watched by the supervisor named in it; if that process is
    # gone (e.g. its terminal was closed) nobody would act on the request.
    try:
        supervisor=int(Path(path).stem.rsplit('-',1)[1]);os.kill(supervisor,0)
    except (ValueError,IndexError):pass
    except ProcessLookupError:raise ValueError('The Studio supervisor has stopped, so Restart cannot work; start Studio again from the terminal: python3 scripts/studio_server.py')
    target=Path(path);temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps({'instance':instance}));temporary.replace(target)


def terminate_group(process,grace=2.):
    # The child is always started with its own session. Never signal our group.
    if process.pid==os.getpgrp():raise RuntimeError('Refusing to terminate supervisor group')
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    deadline=time.monotonic()+grace
    while process.poll() is None and time.monotonic()<deadline:time.sleep(.05)
    # The server may exit before a stuck camera/SAM/render worker does.
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait(timeout=5)


def supervise(port):
    runtime=ROOT/'output/studio/.runtime';runtime.mkdir(parents=True,exist_ok=True)
    request=runtime/f'restart-{os.getpid()}.json';process=None
    def stopped(*_):raise KeyboardInterrupt
    old=signal.signal(signal.SIGTERM,stopped)
    try:
        while True:
            request.unlink(missing_ok=True);instance=uuid.uuid4().hex
            env=os.environ|{'STUDIO_INSTANCE':instance,'STUDIO_RESTART_FILE':str(request)}
            process=subprocess.Popen([sys.executable,str(ROOT/'scripts/studio_server.py'),'--worker','--port',str(port)],env=env,start_new_session=True)
            restarting=False
            while process.poll() is None:
                if request.exists():
                    try:restarting=json.loads(request.read_text()).get('instance')==instance
                    except (OSError,ValueError):pass
                    if restarting:break
                time.sleep(.15)
            terminate_group(process)
            if not restarting:break
            print('Studio: all server workers stopped; starting a new instance.',flush=True)
    except KeyboardInterrupt:
        if process is not None:terminate_group(process)
    finally:
        request.unlink(missing_ok=True);signal.signal(signal.SIGTERM,old)
