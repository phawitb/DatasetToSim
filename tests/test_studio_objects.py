"""The library must describe current generation geometry, never placeholder assets."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import studio_objects as objects


def test_catalog_reports_current_geometry_and_task_roles():
    catalog = objects.catalog()
    entries = {item['id']: item for item in catalog['objects']}
    assert set(entries) == {'cube', 'green-bowl', 'so101', 'work-mat', 'table', 'floor'}
    config = json.loads((objects.ROOT / 'calibration/native-action-current.json').read_text())
    edge = config['base_calibration']['world']['cube']['size']
    assert entries['cube']['dimensions_m'] == pytest.approx([edge]*3)
    assert entries['green-bowl']['dimensions_m'] == pytest.approx([.118, .118, .05], abs=1e-6)
    assert len([t for t in entries['green-bowl']['tasks'] if t['id'] in objects.TASKS]) == 2
    assert len([t for t in entries['cube']['tasks'] if t['id'] in objects.TASKS]) == 5
    variants = {variant['id']: variant for variant in entries['cube']['variants']}
    assert set(variants) == {'red-cube', 'green-cube', 'real-red-cube', 'tan-cube'}
    roles = {task['id']: task['role'] for task in variants['red-cube']['tasks']}
    assert roles['green_cube_to_green_bowl'] == 'Present in scene'
    assert roles['red_cube_to_green_bowl'] == 'Manipulated object'
    assert roles['green_cube_to_red_cube'] == 'Stack support'
    assert roles['unstack'] == 'Top or support cube'
    assert entries['so101']['triangle_count'] > 1000
    assert '/Users/' not in json.dumps(catalog)


@pytest.mark.parametrize('identifier', ['../secret', '/tmp/model', 'unknown', 'green-cube/../../'])
def test_model_endpoint_rejects_non_catalog_ids(identifier):
    with pytest.raises(ValueError, match='Object not found'):
        objects.object_mesh(identifier)


@pytest.mark.parametrize('identifier', ['green-cube', 'green-bowl', 'so101', 'floor'])
def test_preview_exports_finite_real_triangles_with_correct_dimensions(identifier):
    mesh = objects.object_mesh(identifier)
    assert mesh['id'] == identifier
    all_vertices = []
    for part in mesh['parts']:
        vertices = np.asarray(part['vertices']).reshape(-1, 3)
        faces = np.asarray(part['faces']).reshape(-1, 3)
        assert np.isfinite(vertices).all()
        assert faces.min() >= 0 and faces.max() < len(vertices)
        assert len(part['rgba']) == 4
        all_vertices.append(vertices)
    vertices = np.concatenate(all_vertices)
    assert np.ptp(vertices, axis=0) == pytest.approx(mesh['dimensions_m'], abs=1e-6)
    assert mesh['units'] == 'meters'


def test_catalog_colors_match_exported_simulation_geometry():
    catalog = objects.catalog()
    for item in catalog['objects']:
        for variant in item['variants']:
            mesh = objects.object_mesh(variant['id'])
            assert variant['rgba'] == mesh['parts'][0]['rgba']
    cube = next(item for item in catalog['objects'] if item['id'] == 'cube')
    assert cube['variants'][0]['dimensions_m'] == cube['variants'][1]['dimensions_m']


def test_bowl_default_and_future_generation_match_frozen_real_simulation(tmp_path):
    import mujoco
    from studio_generation_scene import initialize_generation_scene, load_generation_scene
    archive = mujoco.MjModel.from_xml_path(str(objects.ROOT / 'output/shared-calibration-v10/episode-00/scene.xml'))
    bowl_ids = [i for i in range(archive.ngeom) if archive.body(archive.geom_bodyid[i]).name == 'bowl' and archive.geom_group[i] <= 2]
    expected = archive.geom_rgba[bowl_ids[0]]
    initialize_generation_scene(tmp_path, 'green_cube_to_green_bowl', 0, {'randomize_layout': False})
    model = load_generation_scene(tmp_path)[-1]
    actual = [model.geom_rgba[i] for i in range(model.ngeom) if model.body(model.geom_bodyid[i]).name == 'bowl' and model.geom_group[i] <= 2]
    assert actual[0] == pytest.approx(expected)
    bowl = next(item for item in objects.catalog()['objects'] if item['id'] == 'green-bowl')
    assert bowl['variants'][0]['rgba'] == pytest.approx(expected)
    assert any(source['id'] == 'real-replay' for source in bowl['variants'][0]['sources'])


def test_catalog_includes_real_replay_and_new_run_materials_without_inventing_variants():
    cube = next(item for item in objects.catalog()['objects'] if item['id'] == 'cube')
    colors = [variant['rgba'] for variant in cube['variants']]
    for expected in ([.85,.045,.035,1], [.045,.65,.09,1], [.82,.07,.07,1], [.65,.48,.27,1]):
        assert any(np.allclose(color, expected) for color in colors)
    tan = next(v for v in cube['variants'] if np.allclose(v['rgba'], [.65,.48,.27,1]))
    assert [task['role'] for task in tan['tasks']] == ['Stack support']
    assert [source['id'] for source in tan['sources']] == ['new-run']
    assert all(v['sources'] and v['materials'] for v in cube['variants'])


def test_visual_colors_follow_mujoco_geometry_override_precedence():
    import mujoco
    model = mujoco.MjModel.from_xml_string('''<mujoco><asset><material name="green" rgba="0 1 0 1"/></asset>
      <worldbody><geom name="material" type="box" size="1 1 1" material="green"/>
      <geom name="override" type="box" size="1 1 1" material="green" rgba="1 0 0 1"/></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    colors = objects._visual_rgba(model, data)
    assert colors[0] == pytest.approx([0,1,0,1])
    assert colors[1] == pytest.approx([1,0,0,1])


def test_thumbnails_are_small_actual_geometry_images():
    import cv2
    images = []
    for item in objects.catalog()['objects']:
        assert item['thumbnail_url'].startswith('/api/objects/')
        payload = objects.object_thumbnail(item['variants'][0]['id'])
        assert payload.startswith(b'\x89PNG') and len(payload) < 100_000
        pixels = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert pixels.shape == (160,160,4)
        assert (pixels[:,:,3] > 0).sum() > 100
        images.append(payload)
    assert len(set(images)) == len(images)
