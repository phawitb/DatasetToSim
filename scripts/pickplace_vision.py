"""Pick-and-place scene from the top camera: every modelled object, with what it rests on.

After Capture, SAM segments every candidate on the mat and the measured object
models decide what each one is. Nothing is assigned a role here: the user picks
From (an object) and To (an object or an empty spot). Objects inside the bowl or
on another object's top are recognised from their masks and given that support,
so the simulation starts with them where they really are.
"""
import hashlib
import json
import uuid
from pathlib import Path

import cv2
import numpy as np

from pickplace import load_registry, bottom_offset, footprint_radius, surface, surface_world, mat_top, settle

IOU_DUPLICATE = .5
CONTAINED = .6


def _contour(mask):
    contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('Empty mask')
    return max(contours, key=cv2.contourArea)


def _hull(mask):
    hull = np.zeros((480, 640), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(np.asarray(mask, np.uint8))), 1)
    return hull > 0


def box_pose(config, mask, size, support_z, label='Object'):
    """Upright box (any footprint) fitted to a SAM silhouette, resting at `support_z`."""
    from strict_action_scene import episode_scene
    from studio_sam_geometry import fit_mask_pose
    contour = _contour(mask)
    corners = cv2.boxPoints(cv2.minAreaRect(contour)); centre = corners.mean(axis=0)
    cal = episode_scene(config, {'cube_pixel': centre.tolist(), 'cube_pixel_corners': corners.tolist(), 'bowl_pixel': [320., 240.]})
    c = cal.world['cube']; q = c['quaternion']
    initial = [*c['position'][:2], 2 * np.arctan2(q[3], q[0])]
    best = None
    # A long box can lie either way round in the rectangle SAM saw: try both.
    for turn in (0., np.pi / 2):
        try:
            pose, fit = fit_mask_pose(mask, cal.camera('top'), np.r_[initial[:2], initial[2] + turn], size[0], support_z, size[2], size[1])
        except ValueError:
            continue
        if best is None or fit['boundary_rmse_px'] < best[1]['boundary_rmse_px']:
            best = (pose, fit)
    if best is None:
        raise ValueError(f'{label} does not match the measured {size[0]*1000:.0f} × {size[1]*1000:.0f} mm box')
    pose, fit = best
    return {'xy': [float(pose[0]), float(pose[1])], 'yaw': float(pose[2]), 'fit_rmse_px': fit['boundary_rmse_px']}


def object_pose(config, registry, kind, mask, rgb, support_z):
    """Origin XY and yaw of one object whose bottom rests at `support_z`."""
    from studio_sam_geometry import tape_pose_from_mask, bowl_pose_from_mask
    shape = registry['objects'][kind]['shape']
    if shape['type'] == 'box':
        return box_pose(config, mask, shape['size_m'], support_z, kind)
    if shape['type'] == 'ring':
        world, fit, _ = tape_pose_from_mask(config, mask, label=kind)
        return {'xy': world['position'][:2], 'yaw': 0., 'fit_rmse_px': fit['boundary_rmse_px']}
    world = bowl_pose_from_mask(config, mask)
    return {'xy': world['position'][:2], 'yaw': 0.}


def classify(config, registry, rgb, mask):
    """Which registry object a mask is, or None. Cubes also report a stack."""
    from studio_sam_auto import classify as classify_known, cube_color
    info = classify_known(config, rgb, mask)
    if info and info['kind'] == 'cube' and info.get('color') is None:
        # Cubes are painted red or green; a colourless "cube" may be the clear roll
        # seen from above (e.g. inside the bowl). Prefer the roll when it fits.
        from studio_sam_geometry import tape_pose_from_mask
        try:
            _, fit, _ = tape_pose_from_mask(config, np.asarray(mask, bool), label='Object')
            return {'kind': 'tape', 'fit_rmse_px': fit['boundary_rmse_px']}
        except ValueError:
            return None  # neither a painted cube nor the roll: not a known object
    if info:
        return info
    # Any other registry box (eraser and future objects): fitted at mat height.
    m = np.asarray(mask, bool); hsv = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2HSV)
    (_, _), (rw, rh), _ = cv2.minAreaRect(_contour(m)); fill = m.sum() / max(1., rw * rh)
    for kind, entry in registry['objects'].items():
        shape = entry['shape']
        if kind in ('cube', 'tape', 'bowl') or shape['type'] != 'box' or fill < .65:
            continue
        aspect = max(rw, rh) / max(1., min(rw, rh)); expected = max(shape['size_m'][:2]) / min(shape['size_m'][:2])
        if abs(aspect - expected) > .45 * expected:
            continue
        try:
            pose = box_pose(config, m, shape['size_m'], mat_top(config), kind)
        except ValueError:
            continue
        if pose['fit_rmse_px'] <= 3.5:
            return {'kind': kind, 'color': cube_color(rgb, m), 'fit_rmse_px': pose['fit_rmse_px'], 'brightness': float(hsv[m][:, 2].mean())}
    return None


def detect(client, folder, config, decode, registry=None):
    """SAM every candidate, classify against the registry, keep supports. Writes masks."""
    from studio_sam_auto import candidate_points, robot_pixels
    from studio_sam_client import save_role
    registry = registry or load_registry()
    folder = Path(folder); image = folder / 'top.jpg'
    rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
    robot = robot_pixels(rgb); found = []; rejected = []
    for point in candidate_points(rgb):
        if any(m[point[1], point[0]] for m in [f['mask'] for f in found] + rejected):
            continue
        try:
            result = client.predict(image, point)
        except ValueError:
            continue
        mask = decode(result)
        info = None if np.count_nonzero(mask & robot) > .05 * mask.sum() else classify(config, registry, rgb, mask)
        if not info:
            rejected.append(mask); continue
        if any(np.count_nonzero(mask & f['mask']) / max(1, np.count_nonzero(mask | f['mask'])) > IOU_DUPLICATE for f in found):
            continue
        found.append({'point': point, 'result': result, 'mask': mask, 'info': info})
    if not found:
        raise ValueError('SAM found no known object on the mat (cube, tape roll, bowl, eraser)')
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    if any(f['result']['image_sha256'] != digest for f in found):
        raise ValueError('Capture changed during segmentation')
    revision = uuid.uuid4().hex
    objects = []
    for f in found:
        kind = f['info']['kind']
        if kind == 'stack':
            from studio_sam_geometry import stack_poses_from_mask
            top, bottom, fit, _ = stack_poses_from_mask(config, f['mask'], rgb)
            colours = {'red', 'green'}; top_colour = fit.get('top_color')
            bottom_colour = (colours - {top_colour}).pop() if top_colour in colours else None
            record = save_role(folder, f'object-{len(objects)}', f['result'], revision)
            record['stack'] = True
            lower = {'id': f'o{len(objects) + 1}', 'kind': 'cube', 'color': bottom_colour, **record,
                     'xy': bottom['position'][:2], 'yaw': 2 * float(np.arctan2(bottom['quaternion'][3], bottom['quaternion'][0])), 'support': 'mat'}
            upper = {'id': f'o{len(objects) + 2}', 'kind': 'cube', 'color': top_colour, **record,
                     'xy': top['position'][:2], 'yaw': 2 * float(np.arctan2(top['quaternion'][3], top['quaternion'][0])), 'support': lower['id']}
            objects += [lower, upper]
            continue
        record = save_role(folder, f'object-{len(objects)}', f['result'], revision)
        objects.append({'id': f'o{len(objects) + 1}', 'kind': kind, 'color': f['info'].get('color'), **record, 'support': 'mat'})
    _assign_supports(objects, folder, registry)
    _fit_poses(objects, folder, config, registry, rgb)
    return {'revision': revision, 'image_sha256': digest, 'model': found[0]['result']['model'], 'objects': objects,
            'detected': [{'kind': f['info']['kind'], 'point': f['point'], 'score': f['result']['score']} for f in found]}


def _mask(folder, record):
    return cv2.imread(str(Path(folder) / record['mask_file']), cv2.IMREAD_GRAYSCALE) > 0


def _assign_supports(objects, folder, registry):
    """An object whose mask lies inside a bigger object's outline rests on (or in) it."""
    by_area = sorted(objects, key=lambda o: -_mask(folder, o).sum())
    for o in objects:
        if o['support'] != 'mat':
            continue
        mask = _mask(folder, o)
        for other in by_area:
            if other is o or other['support'] == o['id'] or other.get('mask_file') == o.get('mask_file'):
                continue
            if not registry['objects'][other['kind']].get('place_surfaces'):
                continue
            hull = _hull(_mask(folder, other))
            if np.count_nonzero(mask & hull) > CONTAINED * mask.sum() and hull.sum() > 1.5 * mask.sum():
                o['support'] = other['id']; break


def _fit_poses(objects, folder, config, registry, rgb):
    """Poses at the right height: the mat, a top surface, or the bowl floor."""
    known = {o['id']: o for o in objects}
    for o in objects:
        if 'xy' in o:
            o['position'] = [*o['xy'], 0.]
            continue
        if o['support'] == 'mat':
            support_z = mat_top(config)
        else:
            below = known[o['support']]
            if 'position' not in below:
                pose = object_pose(config, registry, below['kind'], _mask(folder, below), rgb, mat_top(config))
                below['position'] = [*pose['xy'], 0.]; below['yaw'] = pose['yaw']
            below_pos = np.r_[below['position'][:2], mat_top(config) + bottom_offset(registry['objects'][below['kind']]['shape'])]
            support_z = surface_world({'position': below_pos, 'yaw': below.get('yaw', 0.)}, surface(registry, below))[2]
        if 'position' not in o:
            pose = object_pose(config, registry, o['kind'], _mask(folder, o), rgb, support_z)
            o['position'] = [*pose['xy'], 0.]; o['yaw'] = pose['yaw']
            if 'fit_rmse_px' in pose:
                o['fit_rmse_px'] = pose['fit_rmse_px']
    for o in objects:
        o.pop('xy', None)


def object_at(selection, folder, point):
    """The object whose mask holds the clicked pixel; the topmost one wins."""
    x, y = int(point[0]), int(point[1])
    hits = [o for o in selection['objects'] if _mask(folder, o)[y, x]]
    if not hits:
        return None
    supported = {o.get('support') for o in selection['objects']}
    # A stack shares one mask: prefer the object that nothing rests on (the top).
    hits.sort(key=lambda o: o['id'] in supported)
    return hits[0]


def scene_file(folder, selection, config):
    """pickplace-scene.json for the planner: objects (world), pick and place."""
    objects = [{k: o[k] for k in ('id', 'kind', 'color', 'position', 'yaw', 'support') if k in o} for o in selection['objects']]
    scene = {'objects': objects, 'pick': selection.get('from'), 'place': selection.get('to'),
             'selection_revision': selection['revision']}
    (Path(folder) / 'pickplace-scene.json').write_text(json.dumps(scene, indent=2))
    return scene


def describe(o):
    if o is None:
        return '—'
    return ((o.get('color') + ' ') if o.get('color') else '') + o['kind']


def refresh(client, folder, selection, image, config, decode, tag, registry=None):
    """Segment every selected object again on a fresh image and fit the same poses.

    Used right before hardware execution: the robot only moves if nothing moved.
    """
    from studio_sam_client import save_role
    from studio_sam_geometry import stack_poses_from_mask
    registry = registry or load_registry()
    folder = Path(folder); rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
    revision = uuid.uuid4().hex; masks = {}; fresh = []
    for o in selection['objects']:
        if o['mask_file'] not in masks:
            result = client.predict(image, o['point'])
            masks[o['mask_file']] = save_role(folder, f'{tag}-{len(masks)}', result, revision)
        fresh.append({**{k: o[k] for k in ('id', 'kind', 'color', 'support', 'point', 'stack') if k in o}, **masks[o['mask_file']]})
    done = set()
    for o in fresh:
        if o.get('stack') and o['mask_file'] not in done:
            pair = [p for p in fresh if p['mask_file'] == o['mask_file']]
            top, bottom, _, _ = stack_poses_from_mask(config, _mask(folder, o), rgb)
            for p in pair:
                pose = top if p['support'] != 'mat' else bottom
                p['xy'] = pose['position'][:2]; p['yaw'] = 2 * float(np.arctan2(pose['quaternion'][3], pose['quaternion'][0]))
            done.add(o['mask_file'])
    _fit_poses(fresh, folder, config, registry, rgb)
    return fresh


def compare(planned, observed, picked):
    """Objects must not have moved since simulation (picked: 4 mm, others: 8 mm)."""
    seen = {o['id']: o for o in observed}
    for o in planned:
        if o['id'] not in seen:
            raise ValueError('An object is missing; capture and simulate again')
        limit = .004 if o['id'] == picked else .008
        shift = float(np.linalg.norm(np.asarray(seen[o['id']]['position'][:2]) - o['position'][:2]))
        if shift > limit:
            raise ValueError(f'The {describe(o)} moved {shift*1000:.0f} mm; capture and simulate again')
        if o['kind'] in ('cube', 'eraser'):
            turn = (seen[o['id']].get('yaw', 0.) - o.get('yaw', 0.) + np.pi / 4) % (np.pi / 2) - np.pi / 4 if o['kind'] == 'cube' else \
                   (seen[o['id']].get('yaw', 0.) - o.get('yaw', 0.) + np.pi / 2) % np.pi - np.pi / 2
            if abs(turn) > np.deg2rad(10):
                raise ValueError(f'The {describe(o)} turned; capture and simulate again')


def choose(selection, folder, role, point, config, registry=None):
    """From: the clicked object. To: the clicked object's place surface, or the clicked
    empty mat spot. The full pick-and-place check runs at once so mistakes show early."""
    from studio_sam_geometry import validate_point, pixel_to_mat
    from pickplace import plan_spec
    registry = registry or load_registry()
    point = validate_point(point)
    hit = object_at(selection, folder, point)
    if role == 'from':
        if hit is None:
            raise ValueError('Click a detected object (cube, tape roll, bowl or eraser) as From')
        selection['from'] = hit['id']
        if (selection.get('to') or {}).get('object') == hit['id']:
            selection.pop('to')
    else:
        if hit is not None and hit['id'] != selection.get('from'):
            if not registry['objects'][hit['kind']].get('place_surfaces'):
                raise ValueError(f'Nothing can be placed on the {hit["kind"]}')
            selection['to'] = {'object': hit['id']}
        else:
            xy = pixel_to_mat(config, point, 0.)
            selection['to'] = {'area': [float(xy[0]), float(xy[1])], 'point': point}
    if selection.get('from') and selection.get('to'):
        scene = {'objects': [{k: o[k] for k in ('id', 'kind', 'position', 'yaw', 'support') if k in o} for o in selection['objects']],
                 'pick': selection['from'], 'place': {k: v for k, v in selection['to'].items() if k != 'point'}}
        settle(scene, registry, config)
        plan_spec(scene, registry, config)  # raises a clear reason when the choice cannot work
    return selection


def remap(selection, previous, config):
    """Carry From/To from an earlier capture onto the same objects (same kind and
    colour, within 30 mm), so Run all can capture again without losing the choice."""
    if not previous:
        return selection
    def match(item):
        best = None
        for o in selection['objects']:
            if o['kind'] != item.get('kind') or o.get('color') != item.get('color'):
                continue
            distance = float(np.linalg.norm(np.asarray(o['position'][:2]) - item['position'][:2]))
            if distance <= .03 and (best is None or distance < best[0]):
                best = (distance, o)
        return best[1]['id'] if best else None
    if previous.get('from'):
        selection['from'] = match(previous['from'])
    to = previous.get('to') or {}
    if to.get('object'):
        found = match(to['object'])
        if found:
            selection['to'] = {'object': found}
    elif to.get('area'):
        selection['to'] = {k: to[k] for k in ('area', 'point') if k in to}
    if not selection.get('from'):
        selection.pop('from', None); selection.pop('to', None)
    return selection
