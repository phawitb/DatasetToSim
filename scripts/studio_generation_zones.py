"""Public object identities mapped onto the existing full-mat nine-zone grid."""
import math
OBJECT_ZONE_KEYS = ('red_cube', 'green_cube', 'bowl', 'unstack_destination')
ALL_ZONES = list(range(9))
SCENE_OBJECTS = ('red_cube', 'green_cube', 'bowl')
# Requests saved before scene-object selection always contained both cubes.
LEGACY_EXTRA_OBJECTS = ['red_cube', 'green_cube']


SPREAD_OBJECTS = ('red_cube', 'green_cube', 'bowl', 'unstack_destination')
# Every physical scene object can be rotated (the unstack release area is not an object).
ROTATION_OBJECTS = ('red_cube', 'green_cube', 'bowl')


def validate_zone_spread(value):
    """Per object: None (whole zone) or a 0–100 mm radius around the zone center.
    A single value applies to every object (requests from before per-object settings)."""
    if value is None or (type(value) in (int, float)):
        value = {key: value for key in SPREAD_OBJECTS}
    if not isinstance(value, dict) or set(value) - set(SPREAD_OBJECTS):
        raise ValueError('zone_spread_mm must map objects to empty (whole zone) or 0–100 mm')
    result = {}
    for key in SPREAD_OBJECTS:
        spread = value.get(key)
        if spread is not None and (type(spread) not in (int, float) or not math.isfinite(spread) or not 0 <= spread <= 100):
            raise ValueError(f'zone_spread_mm {key}: use whole zone or 0–100 mm')
        result[key] = None if spread is None else float(spread)
    return result


def validate_rotation(value):
    """Per cube rotation range in degrees, 0–360 (default 360: any direction)."""
    if type(value) in (int, float):
        value = {key: value for key in ROTATION_OBJECTS}
    if not isinstance(value, dict) or set(value) - set(ROTATION_OBJECTS):
        raise ValueError('object_yaw_range_deg must map cubes to 0–360 degrees')
    result = {}
    for key in ROTATION_OBJECTS:
        yaw = value.get(key, 360.)
        if type(yaw) not in (int, float) or not math.isfinite(yaw) or not 0 <= yaw <= 360:
            raise ValueError(f'object_yaw_range_deg {key}: use 0–360 degrees')
        result[key] = float(yaw)
    return result


def layout_role_settings(task_id, options):
    """Spread (m, None = whole zone) and cube yaw range for each sampled layout role."""
    from studio_generation_layout import LEGACY_YAW_RANGE_DEG
    color = options.get('unstack_top', 'red') if task_id == 'unstack' else task_id.split('_')[0]
    other = 'green' if color == 'red' else 'red'
    spread = validate_zone_spread(options.get('zone_spread_mm'))
    legacy = 'object_yaw_range_deg' not in options
    # Legacy requests: cubes within ±0.10 rad, bowl unrotated.
    yaw = {'red_cube': LEGACY_YAW_RANGE_DEG, 'green_cube': LEGACY_YAW_RANGE_DEG, 'bowl': 0.} if legacy else validate_rotation(options['object_yaw_range_deg'])
    other_role = 'unstack_destination' if task_id == 'unstack' else other + '_cube'
    meters = lambda v: None if v is None else v / 1000
    return {'spread': {'source_cube': meters(spread[color + '_cube']), 'other_cube': meters(spread[other_role]), 'bowl': meters(spread['bowl'])},
            # Unstack: the bottom cube copies the top cube's pose; its draw only reserves space.
            'yaw': {'source_cube': yaw[color + '_cube'], 'other_cube': yaw[color + '_cube'] if task_id == 'unstack' else yaw[other + '_cube'],
                    'bowl': yaw['bowl']}}


def validate_extra_objects(value):
    if (not isinstance(value, list) or any(not isinstance(v, str) or v not in SCENE_OBJECTS for v in value)
            or len(set(value)) != len(value)):
        raise ValueError('extra_objects must list distinct objects from red_cube, green_cube and bowl')
    return [key for key in SCENE_OBJECTS if key in value]


def task_objects(task_id, options):
    """Objects a task needs (source cube, its target) and the other cube's color."""
    color = options.get('unstack_top', 'red') if task_id == 'unstack' else task_id.split('_')[0]
    other = 'green' if color == 'red' else 'red'
    required = [color + '_cube'] + (['bowl'] if task_id.endswith('_bowl') else [other + '_cube'])
    return required, other


def scene_presence(task_id, options):
    """Which optional bodies a generated scene contains.

    By default only the objects named by the task instruction are present; any
    of the others can be added. Saved requests without ``extra_objects`` keep
    their original scenes (both cubes always, bowl only for bowl tasks).
    """
    required, other = task_objects(task_id, options)
    extras = options.get('extra_objects', LEGACY_EXTRA_OBJECTS)
    present = set(required) | set(extras)
    return {'other_cube': other + '_cube' in present, 'bowl': 'bowl' in present}


def validate_object_zones(value):
    if not isinstance(value, dict) or set(value) - set(OBJECT_ZONE_KEYS):
        raise ValueError('Object zones must map supported objects to zone lists')
    result = {}
    for key in OBJECT_ZONE_KEYS:
        zones = value.get(key, ALL_ZONES)
        if (not isinstance(zones, list) or not zones
                or any(type(z) is not int or not 0 <= z < 9 for z in zones)
                or len(set(zones)) != len(zones)):
            raise ValueError(f'{key}: select one or more distinct zones Z1–Z9')
        result[key] = sorted(zones)
    return result


def layout_zone_roles(task_id, options):
    zones = validate_object_zones(options.get('object_zones', {}))
    color = options.get('unstack_top', 'red') if task_id == 'unstack' else task_id.split('_')[0]
    other = 'green' if color == 'red' else 'red'
    source, target = zones[color + '_cube'], zones[other + '_cube']
    if task_id == 'unstack':
        source = sorted(set(source) & set(target))
        if not source:
            raise ValueError('Unstack requires at least one shared zone for both cubes')
        target = zones['unstack_destination']
    present = scene_presence(task_id, options)
    roles = {'source_cube': source}
    if present['other_cube'] or task_id == 'unstack': roles['other_cube'] = target
    if present['bowl']: roles['bowl'] = zones['bowl']
    return roles
