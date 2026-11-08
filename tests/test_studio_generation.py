import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from studio_generation import validate_request, GenerationManager
from studio_generation_render import sensor_transform, sensor_parameters


def test_request_is_bounded_and_no_paths():
    request = validate_request({'name': 'training-set', 'tasks': ['unstack']})
    assert request['motion_technique'] == 'natural'
    for bad in ({'name': '../escape'}, {'tasks': ['unknown']}, {'episodes_per_task': True},
                {'episodes_per_task': 1001}, {'camera_jitter': 'yes'}, {'seed': -1},
                {'speed_scale': float('nan')}, {'output': '/tmp/arbitrary'}, {'tasks': []}):
        with pytest.raises(ValueError):
            validate_request({'name': 'training-set', 'tasks': ['unstack']} | bad)


def test_sensor_transform_preserves_color_and_seed():
    frame = np.zeros((40, 60, 3), np.uint8)
    frame[:, :30] = [220, 20, 20]
    frame[:, 30:] = [20, 200, 20]
    params = sensor_parameters(np.random.default_rng(12), True)
    a = sensor_transform(frame, params, np.random.default_rng(3))
    b = sensor_transform(frame, params, np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert a[:, :20].mean((0,1)).argmax() == 0
    assert a[:, 40:].mean((0,1)).argmax() == 1
    assert np.array_equal(frame, sensor_transform(frame, sensor_parameters(np.random.default_rng(1), False), np.random.default_rng(1)))


def test_jobs_path_validation_and_interrupted_recovery(tmp_path):
    import json
    folder = tmp_path / ('a' * 32)
    folder.mkdir()
    (folder / 'job.json').write_text(json.dumps({'id': folder.name, 'status': 'running', 'errors': []}))
    manager = GenerationManager(tmp_path)
    manager.recover_interrupted()
    assert manager.get(folder.name)['status'] == 'cancelled'
    assert not manager.busy()
    with pytest.raises(ValueError):
        manager.get('../outside')


def test_process_completion_race_keeps_completed_job(tmp_path):
    import json
    from studio_generation import save
    manager = GenerationManager(tmp_path)
    folder = tmp_path / ('b' * 32); folder.mkdir()
    job = {'id': folder.name, 'status': 'running', 'errors': []}
    save(folder / 'job.json', job)
    class Process:
        def poll(self):
            save(folder / 'job.json', job | {'status': 'completed'})
            return 0
    manager.process = Process(); manager.active = folder.name
    assert manager.get(folder.name)['status'] == 'completed'


def test_home_and_torque_generation_guard_has_no_local_shadow(monkeypatch, tmp_path):
    import studio_server
    from starlette.requests import Request
    manager = GenerationManager(tmp_path)
    monkeypatch.setattr(manager, 'busy', lambda: True)
    monkeypatch.setattr(studio_server, 'generation', manager)
    request = Request({'type': 'http', 'headers': [(b'x-studio-client', b'test')], 'method': 'POST', 'path': '/'})
    for command in ('home', 'torque_on'):
        with pytest.raises(ValueError, match='generation'):
            studio_server.robot_command(command, request)


def test_visual_variants_share_physical_group():
    from copy import deepcopy
    from studio_generation import trajectory_group
    scene = {'task_id': 'red_cube_to_green_bowl', 'world': {'cube': {'color': 'red', 'position': [1,2,3]}, 'target_cube': {'color': 'green', 'position': [4,5,6]}}, 'physics_draws': {}}
    variant = deepcopy(scene)
    variant['task_id'] = 'green_cube_to_green_bowl'
    variant['world']['cube']['color'] = 'green'
    variant['world']['target_cube']['color'] = 'red'
    assert trajectory_group(scene, 'actions') == trajectory_group(variant, 'actions')
    variant['world']['cube']['position'][0] += .1
    assert trajectory_group(scene, 'actions') != trajectory_group(variant, 'actions')


def test_worker_launch_failure_is_persisted(tmp_path, monkeypatch):
    import studio_generation
    manager = GenerationManager(tmp_path)
    def fail(*args, **kwargs):
        raise OSError('No worker runtime')
    monkeypatch.setattr(studio_generation.subprocess, 'Popen', fail)
    with pytest.raises(ValueError, match='start generation worker'):
        manager.start({'tasks': ['unstack']})
    assert manager.list()[0]['status'] == 'failed'
    assert not manager.busy()


def test_generation_api_protects_mutations_and_downloads(tmp_path, monkeypatch):
    import studio_server
    from fastapi.testclient import TestClient
    manager = GenerationManager(tmp_path)
    monkeypatch.setattr(studio_server, 'generation', manager)
    monkeypatch.setattr(studio_server.hardware, 'call', lambda command, **kw: {'running': False})
    client = TestClient(studio_server.app)
    assert client.post('/api/generation', json={}).status_code == 403
    headers = {'x-studio-token': studio_server.TOKEN}
    assert client.post('/api/generation', headers=headers, json={'tasks':['unknown']}).status_code == 409
    assert client.post('/api/generation', headers=headers | {'origin':'http://evil.example'}, json={}).status_code == 403
    assert client.get('/api/generation/not-a-job').status_code == 409
    from studio_generation import save
    folder = tmp_path / ('c' * 32);folder.mkdir()
    save(folder / 'job.json', {'id':folder.name,'name':'failed-job','status':'failed','errors':[]})
    assert client.get(f'/api/generation/{folder.name}/download').status_code == 409
    monkeypatch.setattr(studio_server.hardware, 'call', lambda command, **kw: {'running': True})
    response = client.post('/api/generation', headers=headers, json={})
    assert response.status_code == 409 and 'robot' in response.json()['error']


def test_summary_uses_export_and_labels_legacy_pace(tmp_path):
    import json
    folder = tmp_path / ('c' * 32); folder.mkdir()
    (folder / 'job.json').write_text(json.dumps({'id': folder.name, 'status': 'completed',
        'export': {'total_frames': 300}, 'episodes': [{'status': 'accepted'}]}))
    summary = GenerationManager(tmp_path).get(folder.name)['summary']
    assert summary['total_duration_s'] == 10
    assert summary['accepted_episodes'] == 1
    assert summary['timing_profile'] == 'conservative'
    assert summary['resolution'] == [640, 480]


def test_recorded_timing_default_and_invalid_option():
    assert validate_request({})['timing_profile'] == 'recorded'
    assert validate_request({'timing_profile': 'conservative'})['timing_profile'] == 'conservative'
    with pytest.raises(ValueError, match='timing'):
        validate_request({'timing_profile': 'unsafe'})
