import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_generation_trash import GenerationTrash

ID='a'*32

def job(root,status='completed'):
    folder=root/ID;folder.mkdir(parents=True)
    (folder/'job.json').write_text(json.dumps({'id':ID,'name':'example','status':status}))
    (folder/'data.bin').write_bytes(b'original dataset')
    return folder


def test_delete_and_undo_preserve_artifacts_and_remove_from_catalog(tmp_path):
    original=job(tmp_path); before=(original/'job.json').read_bytes()
    trash=GenerationTrash(tmp_path);result=trash.delete(ID)
    assert result['deleted'] and not original.exists()
    assert list(tmp_path.glob('*/job.json'))==[]
    assert (tmp_path/'.trash'/ID/'data.bin').read_bytes()==b'original dataset'
    trash.restore(ID)
    assert (original/'job.json').read_bytes()==before
    assert (original/'data.bin').read_bytes()==b'original dataset'


@pytest.mark.parametrize('status',['queued','running'])
def test_active_jobs_cannot_be_deleted(tmp_path,status):
    folder=job(tmp_path,status)
    with pytest.raises(ValueError,match='active'):GenerationTrash(tmp_path).delete(ID)
    assert folder.exists()


@pytest.mark.parametrize('identifier',['../elsewhere','../'+'a'*32,'bad','/tmp/path'])
def test_identifiers_cannot_escape_root(tmp_path,identifier):
    with pytest.raises(ValueError):GenerationTrash(tmp_path).delete(identifier)


def test_symlink_dataset_is_never_moved(tmp_path):
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir();(elsewhere/'job.json').write_text('{}')
    (tmp_path/ID).symlink_to(elsewhere,target_is_directory=True)
    with pytest.raises(ValueError):GenerationTrash(tmp_path).delete(ID)
    assert elsewhere.exists()
