import sys,os,json,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_runtime import terminate_group,request_restart


def test_restart_request_is_bound_to_current_instance(tmp_path,monkeypatch):
    path=tmp_path/'request.json'
    monkeypatch.setenv('STUDIO_RESTART_FILE',str(path));monkeypatch.setenv('STUDIO_INSTANCE','boot-one')
    request_restart()
    assert json.loads(path.read_text())['instance']=='boot-one'


def test_kill_cleans_stubborn_worker_group(tmp_path):
    marker=tmp_path/'child.pid'
    code="import subprocess,sys,time,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); p=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); open(sys.argv[1],'w').write(str(p.pid)); time.sleep(60)"
    p=subprocess.Popen([sys.executable,'-c',code,str(marker)],start_new_session=True)
    try:
        deadline=time.monotonic()+3
        while not marker.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert marker.exists()
        terminate_group(p,grace=.1)
        assert p.poll() is not None
        # Check OS process state: a reaped or defunct descendant is no longer alive.
        status=subprocess.run(['ps','-p',marker.read_text(),'-o','stat='],capture_output=True,text=True)
        assert not status.stdout.strip() or status.stdout.strip().startswith('Z')
    finally:
        if p.poll() is None:os.killpg(p.pid,9)
