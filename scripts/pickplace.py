"""One pick-and-place algorithm for every New Run object.

Every object in the scene is a free physical body. The user picks one object
(From) and a destination (To): a place surface of another object, from the
contact-area registry (assets/objects/affordances.json), or an empty mat spot.
Stacking, unstacking, putting into or taking out of the bowl, and moving the
bowl by its rim are all the same plan: grasp at a registry grasp, carry above
every other object, set down on the destination surface, release, go Home.
A new object kind needs only a shape and contact areas in the registry.
"""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'assets/objects/affordances.json'
SHAPES = ('box', 'ring', 'bowl')
SURFACE_TYPES = ('top', 'inside')
RELEASES = ('lower', 'above_rim')
PADS = ('between', 'fixed_in_hole')
# Commanded pad height above a supporting object's top: jaw tips reach ~16 mm below
# the pad centre and the loaded arm sags ~15 mm, plus 2 mm clearance.
SUPPORT_CLEARANCE_M = .0325
PLACE_CLEARANCE_M = .005
CARRY_CLEARANCE_M = .025
MOVED_LIMIT_M = .004
COLORS = {'red': '.85 .045 .035 1', 'green': '.045 .65 .09 1', 'blue': '.1 .3 .85 1', 'yellow': '.9 .8 .1 1', None: '.65 .48 .27 1'}


# ---------------------------------------------------------------- registry
def load_registry(path=REGISTRY):
    data = json.loads(Path(path).read_text())
    validate_registry(data)
    return data


def _vector(value, n, label):
    v = np.asarray(value, float)
    if v.shape != (n,) or not np.isfinite(v).all():
        raise ValueError(f'{label} must be {n} finite numbers')
    return v


def validate_registry(data):
    if not isinstance(data, dict) or not isinstance(data.get('objects'), dict) or not data['objects']:
        raise ValueError('Contact registry needs objects')
    for kind, item in data['objects'].items():
        shape = item.get('shape', {})
        if shape.get('type') not in SHAPES:
            raise ValueError(f'{kind}: shape must be one of {SHAPES}')
        if shape['type'] == 'box':
            size = _vector(shape.get('size_m'), 3, f'{kind} size')
            if np.any(size <= 0) or np.any(size > .2):
                raise ValueError(f'{kind}: box size must be 0–200 mm')
        if not 0 < float(shape.get('mass_kg', 0)) < 1:
            raise ValueError(f'{kind}: mass must be between 0 and 1 kg')
        if not item.get('grasps'):
            raise ValueError(f'{kind}: at least one grasp is required')
        for grasp in item['grasps']:
            _vector(grasp.get('centre_m'), 3, f'{kind} grasp centre')
            if grasp.get('pads') not in PADS:
                raise ValueError(f'{kind}: grasp pads must be one of {PADS}')
            if not 0 < float(grasp.get('width_m', 0)) <= .06:
                raise ValueError(f'{kind}: grasp width must be 0–60 mm (the jaws open about 60 mm)')
            if not -.05 <= float(grasp.get('pad_height_m', 0)) <= .08:
                raise ValueError(f'{kind}: pad height must be between -50 and 80 mm')
            axis = grasp.get('jaw_axis_deg')
            if axis != 'radial' and (not isinstance(axis, list) or not axis or not all(np.isfinite(a) for a in axis)):
                raise ValueError(f'{kind}: jaw axis must be a list of angles or "radial"')
            native = grasp.get('gripper_native')
            if native != 'from_width' and not (isinstance(native, dict) and all(k in native for k in ('open', 'close', 'release'))):
                raise ValueError(f'{kind}: gripper must be "from_width" or open/close/release native values')
        for surface in item.get('place_surfaces', []):
            _vector(surface.get('centre_m'), 3, f'{kind} surface centre')
            if surface.get('type') not in SURFACE_TYPES or surface.get('release') not in RELEASES:
                raise ValueError(f'{kind}: surface type must be {SURFACE_TYPES} and release {RELEASES}')
            if 'half_extents_m' not in surface and 'radius_m' not in surface:
                raise ValueError(f'{kind}: a surface needs half extents or a radius')
    return data


def save_registry(data, path=REGISTRY):
    validate_registry(data)
    Path(path).write_text(json.dumps(data, indent=2))
    return data


# ---------------------------------------------------------------- workspace calibration
CORRECTION = ROOT / 'calibration/workspace-correction.json'
MAX_CORRECTION_M = .02


def load_correction(path=CORRECTION):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def correction_at(correction, xy):
    """Arm placement error expected at `xy` (where the jaws land minus where they were
    sent), interpolated from measured samples by inverse-distance weighting."""
    if not correction or not correction.get('samples'):
        return np.zeros(2)
    points = np.array([s['xy'] for s in correction['samples']]); errors = np.array([s['error'] for s in correction['samples']])
    d2 = np.sum((points - np.asarray(xy)[:2]) ** 2, axis=1)
    w = 1 / (d2 + correction.get('smoothing_m', .03) ** 2)
    e = (w[:, None] * errors).sum(0) / w.sum()
    n = np.linalg.norm(e)
    return e * (MAX_CORRECTION_M / n) if n > MAX_CORRECTION_M else e


def apply_correction(scene, correction, keep=()):
    """Shift the scene into the arm's own frame: an object seen at P is planned at
    P - e(P), so the real arm, which lands e(P) off, reaches P. Simulation stays
    consistent because everything (objects and empty spots) shifts together."""
    if not correction:
        return scene
    applied = {}
    for o in scene['objects']:
        if o['id'] in keep:
            continue
        e = correction_at(correction, o['position'])
        o['position'] = [o['position'][0] - e[0], o['position'][1] - e[1], *o['position'][2:]]
        applied[o['id']] = e.tolist()
    place = scene.get('place') or {}
    if place.get('area') is not None:
        e = correction_at(correction, place['area'])
        place['area'] = [place['area'][0] - e[0], place['area'][1] - e[1]]
        applied['area'] = e.tolist()
    scene['workspace_correction'] = applied
    return scene


# ---------------------------------------------------------------- geometry
def _yaw_quaternion(yaw):
    return [float(np.cos(yaw / 2)), 0., 0., float(np.sin(yaw / 2))]


def _rotate(yaw, local):
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.asarray(local, float)
    return np.r_[c * local[0] - s * local[1], s * local[0] + c * local[1], local[2:]]


def height(shape):
    return shape['size_m'][2] if shape['type'] == 'box' else shape['height_m']


def bottom_offset(shape):
    """Object origin minus its lowest point."""
    return 0. if shape['type'] == 'bowl' else height(shape) / 2


def footprint_radius(shape):
    if shape['type'] == 'box':
        return float(np.hypot(*shape['size_m'][:2]) / 2)
    return shape['outer_diameter_m'] / 2 if shape['type'] == 'ring' else shape['outer_radius_m']


def min_width(shape):
    if shape['type'] == 'box':
        return float(min(shape['size_m'][:2]))
    return shape['outer_diameter_m'] if shape['type'] == 'ring' else 2 * shape['outer_radius_m']


def mat_top(config):
    mat = config['base_calibration']['world']['mat']
    return mat['position'][2] + mat['size'][2] / 2


def surface(registry, obj, name=None):
    surfaces = registry['objects'][obj['kind']].get('place_surfaces', [])
    if not surfaces:
        raise ValueError(f'Nothing can be placed on the {obj["kind"]}')
    if name is None:
        return surfaces[0]
    for item in surfaces:
        if item['name'] == name:
            return item
    raise ValueError(f'{obj["kind"]} has no surface named {name}')


def surface_world(obj, item):
    return np.asarray(obj['position'], float) + _rotate(obj.get('yaw', 0.), item['centre_m'])


def settle(scene, registry, config):
    """Origin heights from supports: the mat, a top surface, or a bowl floor."""
    objects = {o['id']: o for o in scene['objects']}
    done = set()

    def place(o, trail=()):
        if o['id'] in done:
            return
        if o['id'] in trail:
            raise ValueError('Objects cannot support each other in a loop')
        shape = registry['objects'][o['kind']]['shape']
        support = o.get('support', 'mat')
        if support == 'mat':
            base = mat_top(config)
        else:
            if support not in objects:
                raise ValueError(f'{o["id"]} rests on an unknown object')
            below = objects[support]
            place(below, trail + (o['id'],))
            base = surface_world(below, surface(registry, below, o.get('support_surface')))[2]
        o['position'] = [float(o['position'][0]), float(o['position'][1]), float(base + bottom_offset(shape) + .0002)]
        done.add(o['id'])
    for o in scene['objects']:
        place(o)
    return scene


def carried_by(scene, root_id):
    """Objects resting (directly or not) on `root_id`: they travel with it."""
    result, changed = set(), True
    while changed:
        changed = False
        for o in scene['objects']:
            if o['id'] not in result and o.get('support') in ({root_id} | result):
                result.add(o['id']); changed = True
    return result


def grasp_candidates(registry, obj, base_xy):
    """World grasp targets for one object: pad-centre xyz (commanded), jaw yaw, spec."""
    item = registry['objects'][obj['kind']]
    position = np.asarray(obj['position'], float); yaw = float(obj.get('yaw', 0.))
    out = []
    for grasp in item['grasps']:
        centre = np.asarray(grasp['centre_m'], float)
        if grasp['pads'] == 'fixed_in_hole':
            # Fixed jaw in a round hole: aim at the hole axis; any jaw direction works.
            out.append({'grasp': grasp['name'], 'xyz': position.tolist(), 'jaw_yaw': None, 'spec': grasp})
            continue
        if grasp['jaw_axis_deg'] == 'radial':
            radius = float(np.hypot(*centre[:2]))
            toward = np.arctan2(*(np.asarray(base_xy) - position[:2])[::-1])
            # The rim point nearest the robot first, then around the ring.
            for offset in np.deg2rad([0, 30, -30, 60, -60, 90, -90]):
                angle = toward + offset
                local = np.r_[radius * np.cos(angle - yaw), radius * np.sin(angle - yaw), centre[2]]
                world = position + _rotate(yaw, local)
                out.append({'grasp': grasp['name'], 'xyz': world.tolist(), 'jaw_yaw': float(angle), 'spec': grasp})
        else:
            world = position + _rotate(yaw, centre)
            for axis in grasp['jaw_axis_deg']:
                out.append({'grasp': grasp['name'], 'xyz': world.tolist(), 'jaw_yaw': float(yaw + np.deg2rad(axis)), 'spec': grasp})
    for c in out:
        c['z_cmd'] = c['xyz'][2] + c['spec']['pad_height_m']
    return out


def plan_spec(scene, registry, config, candidate=0):
    """Grasp, place and clearance numbers the planner needs, from the registry."""
    objects = {o['id']: o for o in scene['objects']}
    if scene.get('pick') not in objects:
        raise ValueError('Choose From: the object to pick')
    picked = objects[scene['pick']]
    place = scene.get('place') or {}
    carried = carried_by(scene, picked['id'])
    if place.get('object') == picked['id'] or place.get('object') in carried:
        raise ValueError('To cannot be the picked object or something resting on it')
    base = np.asarray(config['base_calibration']['world']['robot_base']['position'][:2])
    candidates = grasp_candidates(registry, picked, base)
    if not 0 <= candidate < len(candidates):
        raise ValueError('No more grasp candidates')
    grasp = candidates[candidate]
    shape = registry['objects'][picked['kind']]['shape']
    z_cmd = grasp['z_cmd']
    support = picked.get('support', 'mat')
    if support != 'mat':
        below = objects[support]
        top = surface_world(below, surface(registry, below, picked.get('support_surface')))[2]
        # Keep the jaw tips off the object underneath (a stack, or the bowl floor).
        z_cmd = max(z_cmd, top + SUPPORT_CLEARANCE_M) if surface(registry, below, picked.get('support_surface'))['type'] == 'top' else z_cmd
    picked_bottom = picked['position'][2] - bottom_offset(shape)
    hold_depth = z_cmd - picked_bottom
    if place.get('object'):
        destination = objects[place['object']]
        item = surface(registry, destination, place.get('surface'))
        centre = surface_world(destination, item)
        if item['type'] == 'inside':
            bottom = centre[2]
        else:
            bottom = centre[2]
        if item.get('inner_radius_m') and min_width(shape) < 2 * item['inner_radius_m'] + .004:
            # Too small to bridge the hole of a ring top: set it on the ring wall instead,
            # its centre over the middle of the wall on the side facing the robot.
            middle = (item['inner_radius_m'] + item['radius_m']) / 2
            toward = base - centre[:2]
            centre = np.r_[centre[:2] + middle * toward / np.linalg.norm(toward), centre[2]]
        target = {'xy': centre[:2].tolist(), 'bottom_z': float(bottom), 'type': item['type'], 'surface': item['name'],
                  'object': destination['id'], 'release': item['release'], 'destination_yaw': float(destination.get('yaw', 0.)),
                  'destination_kind': destination['kind']}
    elif place.get('area') is not None:
        xy = _vector(place['area'], 2, 'Empty area')
        mat = config['base_calibration']['world']['mat']
        if np.any(np.abs(xy - mat['position'][:2]) > np.asarray(mat['size'][:2]) / 2 - .03):
            raise ValueError('To must be on the mat, away from its edge')
        target = {'xy': xy.tolist(), 'bottom_z': mat_top(config), 'type': 'area', 'release': 'lower'}
    else:
        raise ValueError('Choose To: another object or an empty spot')
    # Empty spots must be clear of everything that stays where it is.
    if target['type'] == 'area':
        for o in scene['objects']:
            if o['id'] == picked['id'] or o['id'] in carried:
                continue
            gap = np.linalg.norm(np.asarray(o['position'][:2]) - target['xy']) - footprint_radius(registry['objects'][o['kind']]['shape']) - footprint_radius(shape)
            if gap < .01:
                raise ValueError(f'The empty spot overlaps the {o["kind"]}; choose a clear spot')
    # Travel above every object that is not carried, holding the picked one.
    tops = [o['position'][2] - bottom_offset(registry['objects'][o['kind']]['shape']) + height(registry['objects'][o['kind']]['shape'])
            for o in scene['objects'] if o['id'] != picked['id'] and o['id'] not in carried]
    hover = max([.12, z_cmd + .105] + [t + hold_depth + CARRY_CLEARANCE_M for t in tops] +
                [target['bottom_z'] + hold_depth + CARRY_CLEARANCE_M])
    place_z = target['bottom_z'] + hold_depth + PLACE_CLEARANCE_M
    place_yaw = None;delta = 0.
    destination_shape = registry['objects'][target['destination_kind']]['shape'] if target.get('destination_kind') else None
    if shape['type'] == 'box' and destination_shape and destination_shape['type'] == 'box' and grasp['jaw_yaw'] is not None:
        # Square the held box to the box below: turn by the yaw difference within ±45°.
        delta = (target['destination_yaw'] - picked.get('yaw', 0.) + np.pi / 4) % (np.pi / 2) - np.pi / 4
        place_yaw = grasp['jaw_yaw'] + delta
    offset = np.asarray(grasp['xyz'][:2]) - np.asarray(picked['position'][:2])
    if np.linalg.norm(offset) > .003 and grasp['jaw_yaw'] is not None:
        # Held off its centre (a bowl by the rim): keep the jaw turn while carrying so
        # the object lands centred on the destination, not on the grasp point.
        place_yaw = grasp['jaw_yaw'] + delta
    pad_xy = np.asarray(target['xy']) + _rotate(delta, np.r_[offset, 0.])[:2]
    target['pad_xy'] = pad_xy.tolist()
    return {'pick': picked['id'], 'kind': picked['kind'], 'grasp': {**{k: v for k, v in grasp.items() if k != 'spec'}, 'z_cmd': float(z_cmd),
            'pads': grasp['spec']['pads'], 'width_m': grasp['spec']['width_m'], 'native': grasp['spec']['gripper_native']},
            'candidate': candidate, 'candidate_count': len(candidates), 'place': {**target, 'pad_z': float(place_z), 'yaw': place_yaw},
            'hover_m': float(hover), 'hold_depth_m': float(hold_depth), 'carried': sorted(carried),
            'picked_bottom_offset_m': bottom_offset(shape)}


# ---------------------------------------------------------------- physics scene
def _box_geom(template, name, size, rgba, mass, friction=None):
    geom = copy.deepcopy(template)
    geom.set('name', name); geom.set('type', 'box'); geom.set('size', ' '.join(str(v / 2) for v in size))
    geom.set('rgba', rgba); geom.set('mass', str(mass))
    for key in ('mesh', 'material'):
        geom.attrib.pop(key, None)
    if friction is not None:
        geom.set('friction', ' '.join(map(str, friction)))
    return geom


def _numbers(values):
    return ' '.join(str(float(v)) for v in np.ravel(values))


def build_model(cal, config, scene, registry=None):
    """All scene objects as free bodies. The picked object is the body `cube`
    (so grasp audits apply unchanged); an object destination is `target_cube`."""
    import mujoco
    from strict_action_scene import build_model as base_model
    from tape_object import add_tape_geometry
    registry = registry or load_registry()
    root = ET.fromstring(base_model(cal, config, return_xml=True))
    world = root.find('worldbody')
    cube = root.find(".//body[@name='cube']"); cube_geom = cube.find("geom[@name='cube_geom']")
    template = copy.deepcopy(cube_geom)
    bowl = root.find(".//body[@name='bowl']")
    objects = scene['objects']
    bowls = [o for o in objects if o['kind'] == 'bowl']
    if len(bowls) > 1:
        raise ValueError('Only one bowl can be modelled at a time')
    picked = next(o for o in objects if o['id'] == scene['pick'])
    destination = (scene.get('place') or {}).get('object')

    def set_pose(body, o):
        body.set('pos', _numbers(o['position'])); body.set('quat', _numbers(_yaw_quaternion(o.get('yaw', 0.))))

    def shape_geometry(body, geom, o, prefix):
        shape = registry['objects'][o['kind']]['shape']
        if shape['type'] == 'box':
            friction = None
            if o['kind'] == 'eraser':
                friction = json.loads((ROOT / 'assets/models/eraser/eraser.json').read_text()).get('friction')
            new = _box_geom(template, geom.get('name'), shape['size_m'], COLORS.get(o.get('color'), COLORS[None]) if o['kind'] == 'cube' else '.95 .95 .93 1', shape['mass_kg'], friction)
            body.remove(geom); body.append(new)
        elif shape['type'] == 'ring':
            add_tape_geometry(root, body, geom, shape['mass_kg'], prefix=prefix)
        else:
            raise ValueError('A bowl is handled separately')

    # The bowl keeps its calibrated wall/floor geometry; it is free only when picked.
    if bowls:
        set_pose(bowl, {**bowls[0], 'position': bowls[0]['position']})
    else:
        world.remove(bowl)
    if picked['kind'] == 'bowl':
        for child in list(bowl):
            bowl.remove(child); cube.append(child)
        world.remove(bowl)
        for key in ('contype', 'conaffinity'):
            cube_geom.set(key, '0')
        cube_geom.set('mass', '0'); cube_geom.set('group', '3'); cube_geom.set('size', '.001 .001 .001')
        shape = registry['objects']['bowl']['shape']
        r, h, m = shape['outer_radius_m'], shape['height_m'], shape['mass_kg']
        for geom in cube.findall('geom'):
            if geom is not cube_geom:
                geom.set('mass', '0')
        ET.SubElement(cube, 'inertial', pos=f'0 0 {h * .35}', mass=str(m), diaginertia=_numbers([m * r * r / 2.5, m * r * r / 2.5, m * r * r / 1.6]))
        set_pose(cube, picked)
    else:
        shape_geometry(cube, cube_geom, picked, '')
        set_pose(cube, picked)
    index = 0
    for o in objects:
        if o['id'] == picked['id'] or o['kind'] == 'bowl':
            continue
        name = 'target_cube' if o['id'] == destination else f'object_{index}'
        if name != 'target_cube':
            index += 1
        body = ET.SubElement(world, 'body', name=name); set_pose(body, o)
        ET.SubElement(body, 'freejoint', name=name + '_free')
        geom = copy.deepcopy(template); geom.set('name', name + '_geom'); body.append(geom)
        shape_geometry(body, geom, o, name + '_')
        o['body'] = name
    picked['body'] = 'cube'
    if bowls and picked['kind'] != 'bowl':
        bowls[0]['body'] = 'bowl'
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def body_names(scene):
    return {o['id']: o.get('body') for o in scene['objects']}


# ---------------------------------------------------------------- verdict
def _vertices(model, data, body):
    import itertools
    import mujoco
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]; start = model.mesh_vertadr[mesh]
            local = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        else:
            local = np.array(list(itertools.product([-1, 1], repeat=3))) * model.geom_size[g]
        out.append(local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    return np.vstack(out)


def verdict(model, trace, scene, spec, registry, config):
    """One success test for every pick-and-place: placed where asked, released,
    settled, and nothing else moved (objects resting on the picked one excepted)."""
    import mujoco
    from scipy.spatial.transform import Rotation
    audit = trace['audit']; pose = np.asarray(trace['cube_pose']); grasp = audit.get('grasp_support', {})
    data = mujoco.MjData(model); data.qpos[:6] = trace['qpos'][-1]
    address = model.joint('cube_free').qposadr[0]; data.qpos[address:address + 7] = pose[-1]
    poses = dict(trace.get('object_poses', {}))
    if 'target_pose' in trace:
        poses['target_cube_free'] = trace['target_pose']
    for joint, values in poses.items():
        a = model.joint(joint).qposadr[0]; data.qpos[a:a + 7] = values[-1]
    mujoco.mj_forward(model, data)
    cube = model.body('cube').id
    vertices = _vertices(model, data, cube)
    up = Rotation.from_quat(pose[-1, [4, 5, 6, 3]]).as_matrix()[2, 2]
    place = spec['place']; shape = registry['objects'][spec['kind']]['shape']
    bottom = float(vertices[:, 2].min())
    lifted = bool(pose[:, 2].max() >= pose[0, 2] + .03)
    detail = {'lifted': lifted, 'upright_cos': float(up), 'final_bottom_z_m': bottom, 'target_bottom_z_m': place['bottom_z'],
              'xy_error_m': float(np.linalg.norm(pose[-1, :2] - np.asarray(place['xy'])))}
    if place['type'] == 'inside':
        bowl = model.body('bowl').id if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bowl') >= 0 else None
        bshape = registry['objects']['bowl']['shape']; centre = data.xpos[bowl]
        local = vertices - centre; radial = np.linalg.norm(local[:, :2], axis=1)
        allowed = bshape['bottom_outer_radius_m'] - bshape['wall_m'] + (bshape['outer_radius_m'] - bshape['bottom_outer_radius_m']) * np.clip(local[:, 2], 0, None) / bshape['height_m']
        placed = bool(np.all(radial <= allowed + .0005) and local[:, 2].min() >= bshape['floor_m'] - .0015 and local[:, 2].max() <= bshape['height_m'] + .0005)
        detail['inside_bowl'] = placed
    else:
        on_level = abs(bottom - place['bottom_z']) <= .0025
        if place['type'] == 'area':
            in_region = detail['xy_error_m'] <= .012
        else:
            item = surface(registry, next(o for o in scene['objects'] if o['id'] == place['object']), place['surface'])
            dest_body = next(o['body'] for o in scene['objects'] if o['id'] == place['object'])
            frame = data.body(dest_body)
            offset = (pose[-1, :3] - frame.xpos) @ frame.xmat.reshape(3, 3)
            centre = np.asarray(item['centre_m'])[:2]
            if 'half_extents_m' in item:
                in_region = bool(np.all(np.abs(offset[:2] - centre) <= np.asarray(item['half_extents_m']) - .003))
            else:
                # Centre of mass over the supporting ring (or disc), 2 mm inside its edges.
                r = float(np.linalg.norm(offset[:2] - centre))
                in_region = bool(item.get('inner_radius_m', -1.) + .002 <= r <= item['radius_m'] - .002 and detail['xy_error_m'] <= .008)
        placed = bool(on_level and in_region and up > np.cos(np.deg2rad(8)))
        detail.update(on_level=bool(on_level), in_region=bool(in_region))
    moved = {}
    exempt = set(spec['carried'])
    names = body_names(scene)
    for o in scene['objects']:
        if o['id'] == spec['pick'] or o['id'] in exempt or names.get(o['id']) in (None, 'bowl'):
            continue
        values = poses[names[o['id']] + '_free']
        shift = float(np.max(np.linalg.norm(values[:, :3] - values[0, :3], axis=1)))
        if shift > MOVED_LIMIT_M:
            moved[o['id']] = shift
    detail['moved_objects_m'] = moved
    speed = audit.get('final_cube_linear_speed_m_s') or 0.; spin = audit.get('final_cube_angular_speed_rad_s') or 0.
    stable = bool(np.max(np.linalg.norm(pose[-30:, :3] - pose[-1, :3], axis=1)) < .0015 and speed < .01 and spin < .2)
    detail['settled'] = stable
    outcome = 'success'
    if not lifted:outcome = 'not_lifted'
    elif grasp.get('longest_secure_grasp_s', 0) < .08 or grasp.get('longest_secure_lift_s', 0) < .03:outcome = 'no_bilateral_grasp'
    elif audit['final_finger_contact']:outcome = 'not_released'
    elif moved:outcome = 'other_object_moved'
    elif not placed:outcome = 'not_placed'
    elif not stable:outcome = 'not_settled'
    return outcome, detail
