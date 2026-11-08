"""Validated, seeded render-only controls shared by previews and episodes."""
from contextlib import contextmanager
import math
import re
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

VISUAL_DEFAULTS = {
    'lighting_randomization': False, 'light_intensity': 1.,
    'light_direction': 0., 'light_elevation': 55., 'light_spread': 45.,
    'shadow_strength': .5, 'background_id': 'calibrated-real',
}
LIGHT_FIELDS = ('pos', 'dir', 'diffuse', 'ambient', 'specular', 'castshadow')


def validate_visual_options(options):
    result = VISUAL_DEFAULTS | dict(options)
    if not isinstance(result['lighting_randomization'], bool):
        raise ValueError('lighting_randomization must be boolean')
    for key, (low, high) in {
        'light_intensity': (.25, 2.5), 'light_direction': (-180, 180),
        'light_elevation': (15, 85), 'light_spread': (0, 180), 'shadow_strength': (0, 1),
    }.items():
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must be a finite number from {low} to {high}')
        result[key] = float(value)
    identifier = result['background_id']
    if not isinstance(identifier, str) or not re.fullmatch(r'calibrated-real|bg-[a-f0-9]{64}', identifier):
        raise ValueError('Invalid background_id')
    return result


def sensor_parameters(rng, enabled):
    if not enabled:
        return dict(gain=1., offset=0., gamma=1., noise=0., blur=0.)
    return dict(gain=float(rng.uniform(.92, 1.08)), offset=float(rng.uniform(-5, 5)),
                gamma=float(rng.uniform(.94, 1.06)), noise=float(rng.uniform(0, 1.5)),
                blur=float(rng.choice([0., .4, .6])))


def light_state(model):
    return {name: getattr(model, 'light_' + name).tolist() for name in LIGHT_FIELDS}


@contextmanager
def reference_lighting(model, data, lighting):
    """Temporarily render reference illumination without advancing simulation."""
    reference = (lighting or {}).get('reference')
    if reference is None:
        yield
        return
    current = light_state(model)
    try:
        for name in LIGHT_FIELDS:
            getattr(model, 'light_' + name)[:] = reference[name]
        mujoco.mj_forward(model, data)
        yield
    finally:
        for name in LIGHT_FIELDS:
            getattr(model, 'light_' + name)[:] = current[name]
        mujoco.mj_forward(model, data)


def configure_visuals(model, seed, options):
    explicit = any(key in options for key in VISUAL_DEFAULTS if key != 'background_id')
    options = validate_visual_options(options)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 917]))
    draws = {'seed': int(seed), 'camera': {}, 'materials': {}, 'lighting': {},
             'sensor': sensor_parameters(rng, options.get('sensor_augmentation', False)),
             'controls': {key: options[key] for key in VISUAL_DEFAULTS},
             'policy': 'Episode-coherent camera/material/light and photometric settings; per-frame achromatic noise. No object hue, crop, flip, geometry or action changes.'}
    reference = light_state(model)
    if options.get('visual_randomization', False):
        for name in ('work_mat', 'table', 'floor'):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name + '_geom')
            if gid >= 0:
                scale = rng.uniform(.8, 1.2, 3)
                model.geom_rgba[gid, :3] = np.clip(model.geom_rgba[gid, :3] * scale, .03, 1)
                draws['materials'][name] = model.geom_rgba[gid].tolist()
        intensity = float(rng.uniform(.82, 1.18))
        if not explicit:  # Keep old saved requests replayable.
            model.light_diffuse[:] *= intensity
            model.light_ambient[:] *= intensity
            draws['lighting']['intensity'] = intensity
    if explicit:
        lights = np.random.default_rng(np.random.SeedSequence([seed, 7219]))
        intensity, azimuth, elevation, shadow = (options[key] for key in
            ('light_intensity', 'light_direction', 'light_elevation', 'shadow_strength'))
        if options['lighting_randomization']:
            spread = options['light_spread']
            azimuth += float(lights.uniform(-spread, spread))
            elevation = float(np.clip(elevation + lights.uniform(-min(spread / 3, 20), min(spread / 3, 20)), 15, 85))
            intensity = float(np.clip(intensity * lights.uniform(.82, 1.18), .25, 2.5))
            shadow = float(np.clip(shadow + lights.uniform(-.12, .12), 0, 1))
        azimuth = (azimuth + 180) % 360 - 180
        az, el = np.deg2rad([azimuth, elevation])
        direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        # World-fixed lights illuminate the workcell. No body, geom or physics fields change.
        center = np.array([.23, 0., 0.])
        for i in range(model.nlight):
            radius = max(float(np.linalg.norm(np.asarray(reference['pos'][i]) - center)), 1.)
            model.light_pos[i] = center + radius * direction
            model.light_dir[i] = -direction
        model.light_diffuse[:] = np.asarray(reference['diffuse']) * intensity * (.5 + shadow)
        model.light_ambient[:] = np.asarray(reference['ambient']) * intensity + np.asarray(reference['diffuse']) * intensity * (1 - shadow) * .18
        model.light_specular[:] = np.asarray(reference['specular']) * intensity
        model.light_castshadow[:] = shadow > 0
        draws['lighting'] = {'intensity': intensity, 'azimuth_degrees': float(azimuth),
            'elevation_degrees': elevation, 'shadow_strength': shadow,
            'randomized': options['lighting_randomization'], 'reference': reference,
            'configured': light_state(model),
            'matching': 'Reference-light photometric matching followed by rendered luminance ratio; preserves requested exposure and shadows.'}
    if options.get('camera_jitter', False):
        for name in ('top_cam', 'wrist_cam'):
            camera = model.camera(name)
            camera.pos[:] += rng.uniform(-.003, .003, 3)
            rotation = rng.uniform(-.015, .015, 3)
            quat = camera.quat.copy()
            q = (Rotation.from_quat(quat[[1, 2, 3, 0]]) * Rotation.from_rotvec(rotation)).as_quat()
            camera.quat[:] = q[[3, 0, 1, 2]]
            camera.fovy[:] += rng.uniform(-.5, .5)
            draws['camera'][name] = {'pos': camera.pos.tolist(), 'quat': camera.quat.tolist(), 'fovy': camera.fovy.tolist()}
    return draws
