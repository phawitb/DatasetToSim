"""Pick-and-place scene from the top camera: every modelled object, with what it rests on.

After Capture, SAM segments every candidate on the mat and the measured object
models decide what each one is. Nothing is assigned a role here: the user picks
From (an object) and To (an object or an empty spot). Objects inside the bowl or
on another object's top are recognised from their masks and given that support,
so the simulation starts with them where they really are.
"""
import base64
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
            # A big box (the NUC: rounded corners, trim) is ~100 px across: allow a looser
            # boundary than the 19.5 mm cube's 3 px.
            pose, fit = fit_mask_pose(mask, cal.camera('top'), np.r_[initial[:2], initial[2] + turn], size[0], support_z, size[2], size[1],
                                      tolerance_px=3. if max(size[:2]) < .06 else 5.)
        except ValueError:
            continue
        if best is None or fit['boundary_rmse_px'] < best[1]['boundary_rmse_px']:
            best = (pose, fit)
    if best is None:
        raise ValueError(f'{label} does not match the measured {size[0]*1000:.0f} × {size[1]*1000:.0f} mm box')
    pose, fit = best
    return {'xy': [float(pose[0]), float(pose[1])], 'yaw': float(pose[2]), 'fit_rmse_px': fit['boundary_rmse_px']}


def model_outline(shape):
    """Outline points of a pen or cup in its own frame, z from its bottom (metres)."""
    a = np.arange(24) * 2 * np.pi / 24
    if shape['type'] == 'cylinder':
        r, half = shape['diameter_m'] / 2 + .0005, shape['length_m'] / 2
        return np.array([[x, r * np.cos(t), r + r * np.sin(t)] for x in (-half, half) for t in a])
    top, bottom = shape['outer_radius_m'] + .001, shape['bottom_outer_radius_m']
    return np.array([[r * np.cos(t), r * np.sin(t), z] for r, z in ((bottom, 0.), (top, shape['height_m'])) for t in a])


def fit_outline(config, mask, outline, support_z, fit_yaw):
    """XY (and yaw) whose projected outline hull matches the SAM mask hull."""
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from dataset_to_sim.scene_estimation import _sample, _distance
    from studio_sam_geometry import _top_camera, pixel_to_mat
    K, R, position = _top_camera(config)
    observed = cv2.convexHull(_contour(mask))[:, 0, :].astype(float); points = _sample(observed)
    (cx, cy), (w, h), angle = cv2.minAreaRect(_contour(mask))
    height = float(outline[:, 2].max())
    xy0 = pixel_to_mat(config, [cx, cy], support_z - mat_top(config) + height / 2)
    # Long side of the rectangle, measured on the mat: the pen's axis.
    d = np.deg2rad(angle + (0 if w >= h else 90)); ends = [[cx - np.cos(d) * 20, cy - np.sin(d) * 20], [cx + np.cos(d) * 20, cy + np.sin(d) * 20]]
    lift = support_z - mat_top(config) + height / 2; e = pixel_to_mat(config, ends[1], lift) - pixel_to_mat(config, ends[0], lift); yaw0 = float(np.arctan2(e[1], e[0]))

    def project(x):
        yaw = x[2] if fit_yaw else 0.
        c, s_ = np.cos(yaw), np.sin(yaw)
        world = outline @ np.array([[c, s_, 0], [-s_, c, 0], [0, 0, 1]]) + [x[0], x[1], support_z]
        uv = (K @ (R @ (world - position).T)).T
        px = uv[:, :2] / uv[:, 2:]
        return px[ConvexHull(px).vertices]

    def residual(x):
        predicted = project(x)
        return np.r_[_distance(_sample(predicted), observed), _distance(points, predicted)]
    x0 = np.r_[xy0, yaw0] if fit_yaw else np.asarray(xy0, float)
    span = np.r_[.03, .03, .35][:len(x0)]
    fit = least_squares(residual, x0, bounds=(x0 - span, x0 + span), diff_step=1e-4, loss='soft_l1', f_scale=1., max_nfev=120)
    rmse = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
    return {'xy': [float(fit.x[0]), float(fit.x[1])], 'yaw': float(fit.x[2]) if fit_yaw else 0., 'fit_rmse_px': rmse}


def _library_shape(config, registry, mask, support_z=None):
    """The pen (long and thin) or the cup (round, tall) when their measured outline fits."""
    m = np.asarray(mask, bool)
    (_, _), (w, h), _ = cv2.minAreaRect(_contour(m))
    aspect = max(w, h) / max(1., min(w, h))
    # Also try the heights of every top surface: a cup set on the NUC is 52 mm up and
    # its outline only fits at that height.
    heights = [mat_top(config)] if support_z is not None else sorted({mat_top(config)} | {
        mat_top(config) + (e['shape']['size_m'][2] if e['shape']['type'] == 'box' else e['shape'].get('height_m', 0.))
        for e in registry['objects'].values() if any(x['type'] == 'top' for x in e.get('place_surfaces', []))})
    if support_z is not None:
        heights = [support_z]
    best = None
    for kind, entry in registry['objects'].items():
        shape = entry['shape']
        for z in heights:
            if shape['type'] == 'cylinder' and aspect > 4:
                pose = fit_outline(config, m, model_outline(shape), z, True)
            elif shape['type'] == 'cup' and aspect < 1.5:
                pose = fit_outline(config, m, model_outline(shape), z, False)
            else:
                continue
            if pose['fit_rmse_px'] <= 4. and (best is None or pose['fit_rmse_px'] < best['fit_rmse_px']):
                best = {'kind': kind, 'color': None, **pose, 'support_z': z}
    return best


def object_pose(config, registry, kind, mask, rgb, support_z, mask_feature=None):
    """Origin XY and yaw of one object whose bottom rests at `support_z`."""
    from studio_sam_geometry import tape_pose_from_mask, bowl_pose_from_mask
    shape = registry['objects'][kind]['shape']
    if mask_feature is not None:
        # Explicitly annotated identity only: a core-sized mask is ambiguous with a
        # cube, so automatic classification must not silently expand it into a roll.
        if kind != 'tape' or mask_feature != 'core_outer_rim':
            raise ValueError('Unsupported observed object feature')
        from tape_object import dimensions
        size = dimensions(); a = np.arange(48) * 2 * np.pi / 48
        r = size['core_outer_diameter_m'] / 2
        outline = np.c_[r * np.cos(a), r * np.sin(a), np.full(48, size['width_m'])]
        pose = fit_outline(config, np.asarray(mask, bool), outline, support_z, False)
        if pose['fit_rmse_px'] > 2.:
            raise ValueError('Visible core does not match the measured tape rim; capture again')
        return pose
    if shape['type'] == 'box':
        return box_pose(config, mask, shape['size_m'], support_z, kind)
    if shape['type'] in ('cylinder', 'cup'):
        pose = fit_outline(config, np.asarray(mask, bool), model_outline(shape), support_z, shape['type'] == 'cylinder')
        if pose['fit_rmse_px'] > 6:
            raise ValueError(f'The {kind} outline does not match its measured model; capture again')
        return pose
    if shape['type'] == 'ring':
        # Fit at the real support height: a roll on top of the NUC is 52 mm above the mat,
        # and fitting it at mat height put it ~1 cm off in the simulation.
        from tape_object import fit_tape_mask
        from studio_sam_geometry import pixel_to_mat
        m = np.asarray(mask, bool)
        (cx, cy), _, _ = cv2.minAreaRect(_contour(m))
        initial = pixel_to_mat(config, [cx, cy], support_z - mat_top(config) + shape['height_m'] / 2)
        camera = config['base_calibration']['cameras']['top']
        try:
            xy, fit = fit_tape_mask(m, camera, initial, support_z)
        except ValueError:
            world, fit, _ = tape_pose_from_mask(config, mask, label=kind)
            if abs(support_z - mat_top(config)) > .003:
                raise
            return {'xy': world['position'][:2], 'yaw': 0., 'fit_rmse_px': fit['boundary_rmse_px']}
        return {'xy': [float(xy[0]), float(xy[1])], 'yaw': 0., 'fit_rmse_px': fit['boundary_rmse_px']}
    world = bowl_pose_from_mask(config, mask)
    return {'xy': world['position'][:2], 'yaw': 0.}


def classify(config, registry, rgb, mask):
    """Which registry object a mask is, or None. Cubes also report a stack."""
    from studio_sam_auto import classify as classify_known, cube_color
    # Pen and cup first: a bright pen or cup otherwise passes the bowl's size test.
    library = _library_shape(config, registry, mask)
    if library:
        return library
    # A big box (the NUC's glossy top reflects the room) also passes the bowl's
    # brightness test: a square box outline that fits wins over the round bowl.
    (_, _), (rw, rh), _ = cv2.minAreaRect(_contour(np.asarray(mask, bool)))
    if max(rw, rh) >= 90:
        big = _other_box(config, registry, rgb, mask)
        if big and big['fit_rmse_px'] <= 3.:
            return big
    info = classify_known(config, rgb, mask)
    if info and info['kind'] in ('cube', 'stack'):
        # A printed box (the eraser) can also pass as a cube or a stack: keep whichever
        # measured model explains the silhouette better.
        other = _other_box(config, registry, rgb, mask)
        if other and other['fit_rmse_px'] < info.get('fit_rmse_px', np.inf):
            return other
    if info and info['kind'] == 'stack' and info.get('color') is None:
        info = None  # stacks are painted cubes; a plain box (the eraser) is checked below
    if info and info['kind'] == 'cube' and info.get('color') is None:
        # Cubes are painted red or green; a colourless "cube" may be the clear roll
        # seen from above (e.g. inside the bowl). Prefer the roll when it fits.
        from studio_sam_geometry import tape_pose_from_mask
        try:
            _, fit, _ = tape_pose_from_mask(config, np.asarray(mask, bool), label='Object')
            return {'kind': 'tape', 'fit_rmse_px': fit['boundary_rmse_px']}
        except ValueError:
            # Lighting can shift a cube's hue between colour names; its shape still fits.
            return info if info.get('fit_rmse_px', 9) <= 2. else None
    if info:
        return info
    return _other_box(config, registry, rgb, mask)


def _other_box(config, registry, rgb, mask, max_rmse=3.5):
    """Any registry box other than the cube (eraser and future objects), at mat height."""
    from studio_sam_auto import cube_color
    m = np.asarray(mask, bool); hsv = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2HSV)
    (_, _), (rw, rh), _ = cv2.minAreaRect(_contour(m)); fill = m.sum() / max(1., rw * rh)
    best = None
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
        if pose['fit_rmse_px'] <= max_rmse and (best is None or pose['fit_rmse_px'] < best['fit_rmse_px']):
            best = {'kind': kind, 'color': cube_color(rgb, m), 'fit_rmse_px': pose['fit_rmse_px'], 'brightness': float(hsv[m][:, 2].mean())}
    return best


def detect(client, folder, config, decode, registry=None):
    """SAM every candidate, classify against the registry, keep supports. Writes masks."""
    from studio_sam_auto import candidate_points, robot_pixels
    from studio_sam_client import save_role
    registry = registry or load_registry()
    folder = Path(folder); image = folder / 'top.jpg'
    rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
    robot = robot_pixels(rgb); found = []; rejected = []
    from studio_sam_geometry import mat_to_pixel
    mat = config['base_calibration']['world']['mat']; half = np.asarray(mat['size'][:2]) / 2
    corners = [mat_to_pixel(config, np.asarray(mat['position'][:2]) + half * [sx, sy]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    region = np.zeros((480, 640), np.uint8); cv2.fillConvexPoly(region, np.round(corners).astype(np.int32), 1)
    # Blob prompts first; then a coarse grid over the mat for low-contrast objects (a clear
    # tape roll on the dark mat barely stands out). Covered points are skipped, and a
    # rejected mat-sized mask does not block the grid.
    inner = cv2.erode(region, np.ones((25, 25), np.uint8)) & (1 - robot.astype(np.uint8))
    grid = [[int(x), int(y)] for y in range(12, 480, 34) for x in range(12, 640, 34) if inner[y, x]]
    for point in candidate_points(rgb, region) + grid:
        if any(m[point[1], point[0]] for m in [f['mask'] for f in found] + [r for r in rejected if r.sum() < 15000]):
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
    # Objects standing on a big object's top (a tape roll on the NUC) fall inside its mask,
    # so their prompt points were skipped: prompt where the top looks different from itself.
    hsv = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2HSV).astype(np.int16)
    for f in list(found):
        entry = registry['objects'].get(f['info']['kind'], {})
        if f['info']['kind'] in ('bowl',) or not any(x['type'] == 'top' for x in entry.get('place_surfaces', [])) or f['mask'].sum() < 6000:
            continue
        inner = cv2.erode(f['mask'].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        median = np.median(hsv[inner][:, 2])
        odd = (inner & (np.abs(hsv[:, :, 2] - median) > 45)).astype(np.uint8)
        odd = cv2.morphologyEx(odd, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(odd)
        for i in sorted(range(1, n), key=lambda i: -stats[i, 4])[:3]:
            if stats[i, 4] < 150:
                continue
            component = (labels == i).astype(np.uint8); distance = cv2.distanceTransform(component, cv2.DIST_L2, 3)
            y, x = np.unravel_index(np.argmax(distance), distance.shape); point = [int(x), int(y)]
            if any(g is not f and g['mask'][y, x] for g in found):
                continue
            try:
                result = client.predict(image, point)
            except ValueError:
                continue
            mask = decode(result)
            if mask.sum() > .6 * f['mask'].sum() or np.count_nonzero(mask & f['mask']) < .8 * mask.sum():
                continue
            info = classify(config, registry, rgb, mask)
            if info:
                found.append({'point': point, 'result': result, 'mask': mask, 'info': info})
    # The other way round: a dark box (the NUC) under a bright roll may get no prompt of
    # its own; prompt a ring just outside each small object and keep a box that fits.
    for f in list(found):
        if f['mask'].sum() >= 6000 and f['info']['kind'] not in ('cup', 'pen'):
            continue
        ys, xs = np.nonzero(f['mask']); cx, cy = xs.mean(), ys.mean(); r = np.sqrt(f['mask'].sum() / np.pi) + 14
        for a in np.arange(8) * np.pi / 4:
            x, y = int(cx + r * np.cos(a)), int(cy + r * np.sin(a))
            if not (0 <= x < 640 and 0 <= y < 480) or any(g['mask'][y, x] for g in found) or robot[y, x]:
                continue
            try:
                result = client.predict(image, [x, y])
            except ValueError:
                continue
            mask = decode(result)
            if np.count_nonzero(mask & robot) > .05 * mask.sum() or mask.sum() < 1500:
                continue
            info = classify(config, registry, rgb, mask) if mask.sum() >= 1.5 * f['mask'].sum() else None
            if not info and np.count_nonzero(mask & f['mask']) < .1 * mask.sum():
                # The small object hides part of the box's top (a cup on the NUC): fill the
                # part of its mask inside the box's own bounding rectangle and fit again.
                x, y, w, h = cv2.boundingRect(_contour(mask))
                box = np.zeros_like(mask); box[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2] = True
                filled = mask | (f['mask'] & box)
                # An occluded, filled-in outline fits less exactly: only a big box, within 5 px.
                info = _other_box(config, registry, rgb, filled, max_rmse=5.)
                if info and max(registry['objects'][info['kind']]['shape']['size_m'][:2]) < .06:
                    info = None
                if info:
                    info = {**info, 'filled': True}
                    mask = filled; result = {**result, 'png': base64.b64encode(cv2.imencode('.png', mask.astype(np.uint8) * 255)[1]).decode()}
            if info and info['kind'] not in ('cube', 'stack') and not any(np.count_nonzero(mask & g['mask']) / max(1, np.count_nonzero(mask | g['mask'])) > IOU_DUPLICATE for g in found):
                found.append({'point': [x, y], 'result': result, 'mask': mask, 'info': info})
                break
    if not found:
        raise ValueError('SAM found no known object on the mat (cube, tape roll, bowl, eraser, pen, cup, NUC)')
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    if any(f['result']['image_sha256'] != digest for f in found):
        raise ValueError('Capture changed during segmentation')
    revision = uuid.uuid4().hex
    objects = []
    for f in found:
        kind = f['info']['kind']
        if kind == 'stack':
            from studio_sam_geometry import stack_poses_from_mask
            try:
                top, bottom, fit, _ = stack_poses_from_mask(config, f['mask'], rgb)
            except ValueError:
                # Colours do not separate the cubes: fit the stack shape (same yaw for both).
                from studio_sam_geometry import cube_pose_from_mask
                top, fit, _ = cube_pose_from_mask(config, f['mask'], elevated=True)
                bottom = {**top, 'position': [*top['position'][:2], top['position'][2] - top['size'] - .0002]}
                fit['top_color'] = f['info'].get('color')
            from studio_sam_geometry import _color_pixels
            top_colour = fit.get('top_color')
            counts = {k: int(v.sum()) for k, v in _color_pixels(rgb, f['mask']).items() if k != top_colour}
            bottom_colour = max(counts, key=counts.get) if counts and max(counts.values()) > 20 else None
            record = save_role(folder, f'object-{len(objects)}', f['result'], revision)
            record['stack'] = True
            lower = {'id': f'o{len(objects) + 1}', 'kind': 'cube', 'color': bottom_colour, **record,
                     'xy': bottom['position'][:2], 'yaw': 2 * float(np.arctan2(bottom['quaternion'][3], bottom['quaternion'][0])), 'support': 'mat'}
            upper = {'id': f'o{len(objects) + 2}', 'kind': 'cube', 'color': top_colour, **record,
                     'xy': top['position'][:2], 'yaw': 2 * float(np.arctan2(top['quaternion'][3], top['quaternion'][0])), 'support': lower['id']}
            objects += [lower, upper]
            continue
        record = save_role(folder, f'object-{len(objects)}', f['result'], revision)
        objects.append({'id': f'o{len(objects) + 1}', 'kind': kind, 'color': f['info'].get('color'), **record, 'support': 'mat',
                        **({'filled': True} if f['info'].get('filled') else {})})
    _stack_pairs(objects, folder, config)
    _assign_supports(objects, folder, registry)
    _fit_poses(objects, folder, config, registry, rgb)
    # Only objects lying on the mat are part of the workspace (not the robot base or table).
    mat = config['base_calibration']['world']['mat']
    inside = lambda o: np.all(np.abs(np.asarray(o['position'][:2]) - mat['position'][:2]) <= np.asarray(mat['size'][:2]) / 2 + .005)
    dropped = {o['id'] for o in objects if not inside(o)}
    objects[:] = [o for o in objects if o['id'] not in dropped and o.get('support') not in dropped]
    return {'revision': revision, 'image_sha256': digest, 'model': found[0]['result']['model'], 'objects': objects,
            'detected': [{'kind': f['info']['kind'], 'point': f['point'], 'score': f['result']['score']} for f in found]}


def _mask(folder, record):
    return cv2.imread(str(Path(folder) / record['mask_file']), cv2.IMREAD_GRAYSCALE) > 0


def _stack_pairs(objects, folder, config):
    """Two cube masks at the same place are a stack seen as two masks (top face and
    the lower cube's sides): the bigger mask is the top cube, refitted one cube higher."""
    from studio_sam_geometry import cube_pose_from_mask
    from strict_action_scene import episode_scene
    cubes = [o for o in objects if o['kind'] == 'cube' and o['support'] == 'mat' and 'xy' not in o]
    for o in cubes:
        mask = _mask(folder, o)
        if 'position' not in o:
            world, _, _ = cube_pose_from_mask(config, mask); o['_xy'] = world['position'][:2]; o['_area'] = int(mask.sum())
    for i, a in enumerate(cubes):
        for b in cubes[i + 1:]:
            if a['support'] != 'mat' or b['support'] != 'mat':
                continue
            touching = np.any(cv2.dilate(_mask(folder, a).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & _mask(folder, b))
            if touching and np.linalg.norm(np.asarray(a['_xy']) - b['_xy']) < .03:
                top, bottom = (a, b) if a['_area'] >= b['_area'] else (b, a)
                try:
                    world, _, _ = cube_pose_from_mask(config, _mask(folder, top), elevated=True)
                except ValueError:
                    continue
                # A real stack: the raised top cube sits right over the lower one.
                if np.linalg.norm(np.asarray(world['position'][:2]) - bottom['_xy']) > .01:
                    continue
                top['support'] = bottom['id']
                top['xy'] = world['position'][:2]; top['yaw'] = 2 * float(np.arctan2(world['quaternion'][3], world['quaternion'][0]))
    for o in cubes:
        o.pop('_xy', None); o.pop('_area', None)


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
    """Poses at the right height: the mat, a top surface, or the bowl floor. Supports are
    fitted first (recursively), so a pen in a cup on the NUC uses the cup's real height."""
    known = {o['id']: o for o in objects}

    def fit(o, trail=()):
        if 'position' in o:
            return
        if 'xy' in o:
            o['position'] = [*o['xy'], 0.]
            return
        if o['support'] == 'mat' or o['support'] in trail or o['support'] not in known:
            support_z = mat_top(config)
        else:
            below = known[o['support']]
            fit(below, trail + (o['id'],))
            below_shape = registry['objects'][below['kind']]['shape']
            below_base = mat_top(config) if below['support'] == 'mat' else below.get('_bottom_z', mat_top(config))
            below_pos = np.r_[below['position'][:2], below_base + bottom_offset(below_shape)]
            support_z = surface_world({'position': below_pos, 'yaw': below.get('yaw', 0.)}, surface(registry, below))[2]
        pose = object_pose(config, registry, o['kind'], _mask(folder, o), rgb, support_z, o.get('mask_feature'))
        o['position'] = [*pose['xy'], 0.]; o['yaw'] = pose['yaw']; o['_bottom_z'] = support_z
        if 'fit_rmse_px' in pose:
            o['fit_rmse_px'] = pose['fit_rmse_px']
    for o in objects:
        fit(o)
    for o in objects:
        o.pop('xy', None); o.pop('_bottom_z', None)


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


def _predict_inside(client, image, point, mask):
    """SAM at the original click, else at the deepest interior points of the earlier mask."""
    points = [list(point)]
    distance = cv2.distanceTransform(np.asarray(mask, np.uint8), cv2.DIST_L2, 3)
    order = np.argsort(distance.ravel())[::-1][:400:80]
    points += [[float(i % 640), float(i // 640)] for i in order]
    from studio_sam_client import decode_mask
    error = 'no matching mask'; old = np.asarray(mask, bool)
    for candidate in points:
        try:
            result = client.predict(image, candidate)
        except ValueError as exc:
            error = exc; continue
        new = decode_mask(result)
        # Same object only: the fresh mask must cover most of the earlier one.
        if np.count_nonzero(new & old) / max(1, np.count_nonzero(new | old)) >= .6:
            return result
        # The earlier mask may have been filled in (a box under a cup): the fresh one then
        # lies inside it and covers most of it.
        if np.count_nonzero(new & old) >= .9 * new.sum() and new.sum() >= .5 * old.sum():
            return result
    raise ValueError(f'Could not segment an object again ({error}); capture again')


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
            result = _predict_inside(client, image, o['point'], _mask(folder, o))
            masks[o['mask_file']] = save_role(folder, f'{tag}-{len(masks)}', result, revision)
        fresh.append({**{k: o[k] for k in ('id', 'kind', 'color', 'support', 'point', 'stack', 'filled', 'mask_feature', 'identity_source') if k in o}, **masks[o['mask_file']]})
    # A box whose top was hidden by what rests on it was fitted with that part filled in:
    # fill the fresh mask the same way.
    for o in fresh:
        if not o.get('filled'):
            continue
        mask = _mask(folder, o); x, y, w, h = cv2.boundingRect(_contour(mask))
        box = np.zeros_like(mask); box[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2] = True
        for other in fresh:
            if other.get('support') == o['id']:
                mask = mask | (_mask(folder, other) & box)
        cv2.imwrite(str(folder / o['mask_file']), mask.astype(np.uint8) * 255)
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
            raise ValueError('Click a detected object (cube, tape roll, bowl, eraser, pen or cup) as From')
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
