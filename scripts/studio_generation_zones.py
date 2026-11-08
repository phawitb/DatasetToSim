"""Public object identities mapped onto the existing full-mat nine-zone grid."""
OBJECT_ZONE_KEYS = ('red_cube', 'green_cube', 'bowl', 'unstack_destination')
ALL_ZONES = list(range(9))


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
    return {'source_cube': source, 'other_cube': target, 'bowl': zones['bowl']}
