import sys,json
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from studio_generation_view import episode_data,combine_cameras
from dataset_to_sim.strict_replay import array_hash


def fixture(tmp_path):
    folder=tmp_path/'episodes'/'00000';folder.mkdir(parents=True)
    actions=np.arange(24,dtype=float).reshape(4,6)
    states=np.arange(30,dtype=float).reshape(5,6)+1
    np.savez(folder/'plan.npz',actions=actions,timestamps=np.arange(4)/30)
    np.savez(folder/'trace.npz',native=states)
    (folder/'episode.json').write_text(json.dumps({'instruction':'Test pick','action_hash':array_hash(actions)}))
    (tmp_path/'job.json').write_text(json.dumps({'episodes':[{'status':'accepted','task_id':'red_cube_to_green_bowl'}]}))
    return folder,actions,states


def test_pre_action_state_and_native_action_alignment(tmp_path):
    folder,actions,states=fixture(tmp_path)
    result=episode_data(tmp_path,0)
    assert result['frames']==4 and result['fps']==30
    assert np.array_equal(result['state'],states[:-1])
    assert np.array_equal(result['action'],actions)
    assert result['units']==['degrees']*5+['0–100']
    with pytest.raises(ValueError):episode_data(tmp_path,-1)
    with pytest.raises(ValueError):episode_data(tmp_path,1)
    np.savez(folder/'trace.npz',native=states[:-1])
    with pytest.raises(ValueError,match='N\\+1'):episode_data(tmp_path,0)


def test_reject_failed_attempt_and_changed_commands(tmp_path):
    folder,actions,_=fixture(tmp_path)
    actions[0,0]+=1
    np.savez(folder/'plan.npz',actions=actions,timestamps=np.arange(4)/30)
    with pytest.raises(ValueError,match='hash'):episode_data(tmp_path,0)
    (tmp_path/'job.json').write_text(json.dumps({'episodes':[{'status':'rejected'}]}))
    with pytest.raises(ValueError,match='accepted'):episode_data(tmp_path,0)


def test_combined_preview_preserves_camera_order_and_frame_count(tmp_path):
    import cv2
    for view,color in [('top',(0,0,230)),('wrist',(0,220,0))]:
        writer=cv2.VideoWriter(str(tmp_path/f'{view}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
        for _ in range(4):writer.write(np.full((480,640,3),color,np.uint8))
        writer.release()
    destination=tmp_path/'paired.mp4'
    combine_cameras(tmp_path,destination,4)
    cap=cv2.VideoCapture(str(destination));assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==4
    ok,frame=cap.read();cap.release();assert ok and frame.shape==(480,1280,3)
    assert frame[:, :640].mean((0,1)).argmax()==2
    assert frame[:,640:].mean((0,1)).argmax()==1
    with pytest.raises(ValueError,match='frames'):combine_cameras(tmp_path,tmp_path/'bad.mp4',5)
