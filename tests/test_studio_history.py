import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_history import list_runs


def save(folder,name,value):
    folder.mkdir(exist_ok=True);(folder/name).write_text(json.dumps(value))


def test_catalog_lists_capture_sim_home_and_real_without_conflating_success(tmp_path):
    a=tmp_path/('a'*32);b=tmp_path/('b'*32);c=tmp_path/('c'*32)
    save(a,'capture.json',{'captured_at':1,'source':'live'});save(a,'plan.json',{'outcome':'success','frames':10,'duration':1,'executed':True});save(a,'execution.json',{'completed':True});save(a,'physical-review.json',{'assessment':'failed'})
    save(b,'capture.json',{'captured_at':3,'source':'live','purpose':'home'});save(b,'job.json',{'status':'error','stage':'prepare','error':'Cannot verify camera'})
    save(c,'capture.json',{'captured_at':2,'source':'dataset'})
    save(tmp_path/'sam-smoke','capture.json',{})
    rows=list_runs(tmp_path,{})
    assert [r['id'] for r in rows]==['b'*32,'c'*32,'a'*32]
    assert rows[0]['error']=='Cannot verify camera' and rows[0]['purpose']=='home'
    assert rows[-1]['hardware_completed'] is True and rows[-1]['physical_outcome']=='failed' and rows[-1]['simulation_outcome']=='success'
    assert rows[1]['status']=='captured'
    save(a,'physical-review.json',{'task_outcome':'not_transferred'})
    assert list_runs(tmp_path,{})[-1]['physical_outcome']=='failed'


def test_bad_old_report_does_not_hide_other_runs_and_active_status_is_read_only(tmp_path):
    a=tmp_path/('a'*32);save(a,'capture.json',{'captured_at':1,'source':'live'});(a/'plan.json').write_text('{broken')
    b=tmp_path/('b'*32);save(b,'capture.json',{'captured_at':2,'source':'live'})
    jobs={'b'*32:{'status':'running','stage':'simulate'}}
    rows=list_runs(tmp_path,jobs)
    assert len(rows)==2 and rows[0]['status']=='running' and rows[1]['status']=='error'
    assert jobs=={'b'*32:{'status':'running','stage':'simulate'}}
