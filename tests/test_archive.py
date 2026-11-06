import mujoco
from dataset_to_sim.archive import load_saved_model


def test_current_xml_archive_wins_over_obsolete_binary(tmp_path):
    (tmp_path/'scene.xml').write_text('<mujoco><worldbody><body><joint name="new_joint"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    (tmp_path/'scene.mjb').write_bytes(b'obsolete binary from an earlier run')
    model=load_saved_model(tmp_path,{'model_archive':'self-contained per-episode XML with one shared copy of source mesh assets'})
    assert model.joint('new_joint').id==0


def test_model_hash_rejects_modified_archive(tmp_path):
    import pytest
    (tmp_path/'scene.xml').write_text('<mujoco/>')
    with pytest.raises(ValueError,match='hash mismatch'):
        load_saved_model(tmp_path,{'model_archive':'self-contained per-episode XML'},'not-the-recorded-digest')
