from pathlib import Path
import sys
import mujoco
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from studio_generation_visuals import VISUAL_DEFAULTS, validate_visual_options, configure_visuals


def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody><light pos="0 0 2" diffuse=".7 .7 .7"/><geom name="work_mat" type="plane" size="1 1 .1"/><body pos="0 0 .1"><freejoint/><geom size=".1" mass="1"/></body></worldbody></mujoco>''')


def test_validation_is_complete_strict_and_nonmutating():
    original = {'name': 'kept', 'light_intensity': 2}
    result = validate_visual_options(original)
    assert result == VISUAL_DEFAULTS | original
    assert result['light_intensity'] == 2.
    assert 'light_elevation' not in original
    for key, value in [('light_intensity', True), ('light_direction', '90'), ('light_spread', -1),
                       ('light_elevation', 86), ('shadow_strength', float('nan')),
                       ('lighting_randomization', 1), ('background_id', '../image'), ('background_id', '')]:
        with pytest.raises(ValueError):
            validate_visual_options({key: value})


def test_seeded_lights_are_coherent_and_leave_physics_unchanged():
    a, b, c = model(), model(), model()
    physics = (a.body_mass.copy(), a.geom_friction.copy(), a.qpos0.copy())
    options = VISUAL_DEFAULTS | {'lighting_randomization': True}
    da = configure_visuals(a, 42, options)
    db = configure_visuals(b, 42, options)
    dc = configure_visuals(c, 43, options)
    assert da == db
    assert da['lighting'] != dc['lighting']
    np.testing.assert_array_equal(a.light_pos, b.light_pos)
    for before, after in zip(physics, (a.body_mass, a.geom_friction, a.qpos0)):
        np.testing.assert_array_equal(before, after)


def test_fixed_controls_work_without_randomization():
    a, b = model(), model()
    configure_visuals(a, 1, VISUAL_DEFAULTS | {'light_intensity': .4, 'light_direction': -90, 'shadow_strength': 0})
    configure_visuals(b, 1, VISUAL_DEFAULTS | {'light_intensity': 2., 'light_direction': 90, 'shadow_strength': 1})
    assert b.light_diffuse.mean() > a.light_diffuse.mean()
    assert not np.allclose(a.light_pos, b.light_pos)
    assert not a.light_castshadow.any()
    assert b.light_castshadow.all()


def test_post_matching_lighting_keeps_exposure_and_shadow_difference():
    from studio_real_appearance import preserve_lighting
    matched = np.full((10, 10, 3), 120, np.uint8)
    baseline = np.full_like(matched, 100)
    bright = np.full_like(matched, 150); bright[:, :5] = 50
    result = preserve_lighting(matched, bright, baseline)
    assert result[:, :5].mean() < matched.mean() < result[:, 5:].mean()
    np.testing.assert_array_equal(preserve_lighting(matched, baseline, baseline), matched)


def test_reference_lighting_restores_model_and_never_advances_state():
    from studio_generation_visuals import reference_lighting
    m = model(); d = mujoco.MjData(m); d.time = 7.
    qpos = d.qpos.copy()
    draws = configure_visuals(m, 4, VISUAL_DEFAULTS | {'light_intensity': 2.})
    actual = m.light_diffuse.copy()
    with pytest.raises(RuntimeError):
        with reference_lighting(m, d, draws['lighting']):
            np.testing.assert_allclose(m.light_diffuse, draws['lighting']['reference']['diffuse'])
            raise RuntimeError('render failed')
    np.testing.assert_array_equal(m.light_diffuse, actual)
    np.testing.assert_array_equal(d.qpos, qpos)
    assert d.time == 7.


def test_custom_background_is_frozen_and_labeled_uncalibrated(tmp_path, monkeypatch):
    import hashlib
    import json
    import cv2
    from studio_real_appearance import RealAppearance
    source = tmp_path / 'source.png'
    cv2.imwrite(str(source), np.full((480, 640, 3), (40, 80, 180), np.uint8))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    asset = {'id': 'bg-' + digest, 'name': 'Uploaded test', 'path': str(source),
             'mode': 'uploaded', 'sha256': digest}
    monkeypatch.setattr('studio_generation_assets.resolve_background', lambda *_: pytest.fail('Must use frozen snapshot'))
    options = VISUAL_DEFAULTS | {'real_background': True, 'background_id': asset['id'], '_background_asset': asset}
    output = tmp_path / 'episode'
    appearance = RealAppearance(output, model(), options)
    assert (output / 'real-background.png').read_bytes() == source.read_bytes()
    profile = json.loads((output / 'real-appearance.json').read_text())
    assert profile['background_asset']['sha256'] == digest
    assert 'not camera-calibrated' in profile['background_mapping']
    assert appearance.provenance['background_asset']['name'] == 'Uploaded test'
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        RealAppearance(tmp_path / 'other', model(), options)


def test_unseen_preview_cube_gets_estimate_without_changing_scene(tmp_path, monkeypatch):
    from studio_real_appearance import RealAppearance
    m = model(); d = mujoco.MjData(m)
    appearance = RealAppearance(tmp_path, m, {'real_image_processing': True})
    # No cube appears in this tiny scene, which must not prevent layout previews.
    class Renderer:
        def update_scene(self, *args, **kwargs): pass
        def render(self): return np.full((8, 8, 3), 100, np.uint8)
    monkeypatch.setattr('studio_real_appearance.segmentation_ids', lambda _: np.zeros((8, 8), int))
    original = d.qpos.copy()
    appearance.calibrate_frame(Renderer(), d, None)
    np.testing.assert_array_equal(d.qpos, original)
    assert len(appearance.provenance['unseen_preview_color_estimates']) == 4
    assert all(appearance.profile['appearance'][view][kind]['sim'] for view in ('top', 'wrist') for kind in ('red', 'green'))


def test_uploaded_background_uses_estimated_wrist_response_only_when_enabled(monkeypatch):
    import cv2
    from types import SimpleNamespace
    from studio_real_appearance import RealAppearance
    a = object.__new__(RealAppearance)
    a.background = True; a.processing = True; a.background_mode = 'uploaded'
    a.classes = {0: 'work_mat', 1: 'robot'}; a.maps = {}
    a.model = SimpleNamespace(geom_size=np.zeros((2, 3)), geom_type=np.zeros(2, int))
    a.plate = np.full((480, 640, 3), 180, np.uint8); a.plate[:, 320:] = 220
    top_stats = {'mean': [150., 128., 128.], 'std': [10., 10., 10.]}
    wrist_stats = {'mean': [100., 130., 126.], 'std': [10., 10., 10.]}
    a.profile = {'surface_rgb': {'table': [180, 180, 180]}, 'appearance': {
        'top': {'work_mat': {'sim': top_stats, 'real': top_stats}},
        'wrist': {'work_mat': {'sim': wrist_stats, 'real': wrist_stats}},
    }}
    data = SimpleNamespace(geom_xpos=np.zeros((2, 3)))
    yy, xx = np.mgrid[:480, :640]
    monkeypatch.setattr('studio_real_appearance.background_maps', lambda *args: (
        xx.astype('float32'), yy.astype('float32'), np.ones((480, 640), bool)))
    rgb = np.full_like(a.plate, 100); ids = np.zeros((480, 640), int); ids[:100] = 1
    top = a.apply(rgb, ids, data, 'top')
    wrist = a.apply(rgb, ids, data, 'wrist')
    assert wrist[200:].mean() < top[200:].mean() - 20
    top_lab = cv2.cvtColor(top, cv2.COLOR_RGB2LAB).astype(float)
    wrist_lab = cv2.cvtColor(wrist, cv2.COLOR_RGB2LAB).astype(float)
    for x in (150, 450):
        np.testing.assert_allclose(wrist_lab[250, x] - top_lab[250, x], [-50, 2, -2], atol=2)
    np.testing.assert_array_equal(wrist[20:80], top[20:80])  # Foreground receives no plate response.
    a.processing = False
    np.testing.assert_array_equal(a.apply(rgb, ids, data, 'wrist'), a.apply(rgb, ids, data, 'top'))
    a.background = False; a.processing = True
    uploaded_off = a.apply(rgb, ids, data, 'wrist')
    a.background_mode = 'calibrated'
    np.testing.assert_array_equal(uploaded_off, a.apply(rgb, ids, data, 'wrist'))
