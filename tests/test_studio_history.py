import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_history import list_runs


def save(folder,name,value):
    folder.mkdir(exist_ok=True);(folder/name).write_text(json.dumps(value))


def test_only_executed_robot_runs_are_kept_in_history():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);a=root/('a'*32);b=root/('b'*32);c=root/('c'*32)
        save(a,'capture.json',{'captured_at':1,'source':'live'});save(a,'plan.json',{'outcome':'success','executed':True});save(a,'hardware-trace.json',{'samples':[{'target':[0]*6}],'completed':True});save(a,'physical-review.json',{'task_outcome':'not_transferred'})
        save(b,'capture.json',{'captured_at':3,'source':'live','purpose':'home'});save(b,'job.json',{'status':'error','error':'Before dispatch'})
        save(c,'capture.json',{'captured_at':2,'source':'dataset'});save(c,'plan.json',{'outcome':'success'})
        rows=list_runs(root,{})
        assert len(rows)==1 and rows[0]['id']==a.name
        assert rows[0]['hardware_completed'] and rows[0]['physical_outcome']=='failed'
        save(a,'job.json',{'status':'error','error':'Recording failed'})
        assert len(list_runs(root,{}))==1


def test_partial_execution_survives_restart_and_truncated_journal(tmp_path):
    a=tmp_path/('a'*32);save(a,'capture.json',{'captured_at':1,'source':'live'})
    (a/'hardware-commands.jsonl').write_text(json.dumps({'target':[0]*6,'time':0})+'\n{"truncated')
    rows=list_runs(tmp_path,{})
    assert len(rows)==1 and rows[0]['command_count']==1 and rows[0]['hardware_completed'] is False
    assert 'interrupted' in rows[0]['error'].lower()


def test_corrupt_plan_does_not_hide_executed_run(tmp_path):
    a=tmp_path/('a'*32);save(a,'capture.json',{'captured_at':1,'source':'live'});(a/'plan.json').write_text('{broken')
    save(a,'hardware-trace.json',{'samples':[{'target':[0]*6}],'completed':False,'error':'Stop requested'})
    jobs={'a'*32:{'status':'error','error':'Stopped'}}
    assert list_runs(tmp_path,jobs)[0]['status']=='error'
    assert jobs=={'a'*32:{'status':'error','error':'Stopped'}}


def test_active_command_journal_is_not_reported_as_restarted(tmp_path):
    a=tmp_path/('a'*32);save(a,'capture.json',{'captured_at':1,'source':'live'})
    (a/'hardware-commands.jsonl').write_text(json.dumps({'target':[0]*6,'time':0})+'\n')
    rows=list_runs(tmp_path,{a.name:{'status':'executing','stage':'execute'}})
    assert rows[0]['status']=='executing' and rows[0]['hardware_completed'] is None and rows[0]['error'] is None
