"""Public object identities mapped onto the existing full-mat nine-zone grid."""
OBJECT_ZONE_KEYS = ('red_cube', 'green_cube', 'bowl', 'unstack_destination')
ALL_ZONES = list(range(9))
SCENE_OBJECTS = ('red_cube', 'green_cube', 'bowl')
# Requests saved before scene-object selection always contained both cubes.
LEGACY_EXTRA_OBJECTS = ['red_cube', 'green_cube']


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
