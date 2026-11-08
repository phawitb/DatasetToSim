"""Clear tape roll as the manipulated object in place-in-bowl scenes.

The roll reuses the manipulated ``cube`` body so traces, grasp audits and
containment stay one code path. Only its geometry, mass and grip opening
change. Visual meshes come from the Blender model in assets/models/tape.

The roll is grasped through its hole: the fixed jaw goes into the core and
the moving jaw closes on the outside, pinching the wall. Collision is
therefore a hollow ring of convex wedges; ``cube_geom`` keeps the outer
cylinder only as a non-colliding outline for bowl containment.
"""
import json
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'assets/models/tape'
TAPE = 'tape'
VISUAL_PARTS = [('tape_roll', '.93 .91 .84 .55'), ('tape_core', '.55 .40 .26 1'), ('tape_liner', '.88 .90 .95 1')]
# Pads are 6 mm thick boxes; the occupied pad-centre span is object width + 6 mm.
PAD_THICKNESS_M = .006
# Open moving jaw clears the outer wall by this much before closing.
OPEN_SIDE_CLEARANCE_M = .013
# Fixed pad centre sits this far from the roll axis, towards the moving jaw,
# when it enters the hole (keeps its 6 x 16 mm pad clear of the 26 mm core).
FIXED_PAD_RADIUS_M = .0055
# The open moving jaw reaches ~55 mm from the roll axis on one side.
LAYOUT_JAW_CLEARANCE_M = .03
WALL_SEGMENTS = 24
# The opened moving jaw backs off the outer wall by this much on release.
RELEASE_CLEARANCE_M = .008
# Close this many native gripper units past first contact.
SQUEEZE_NATIVE = 12.


@lru_cache(maxsize=1)
def dimensions():
    return json.loads((MODEL_DIR / 'tape.json').read_text())


def is_tape(world):
    return world.get('cube', {}).get('object') == TAPE


def tape_pose(xy, yaw, support, cube_edge):
    """World entry for a roll lying flat on the support surface.

    ``size`` stays the cube edge: the scene builder still makes the cube box
    (and copies it for the second cube) before the body becomes the roll.
    """
    width = dimensions()['width_m']
    return {'object': TAPE, 'position': [*map(float, xy), support + width / 2 + .0002],
            'quaternion': [float(np.cos(yaw / 2)), 0., 0., float(np.sin(yaw / 2))],
            'size': cube_edge, 'width': width, 'outer_diameter': dimensions()['outer_diameter_m']}


def footprint_radius():
    return dimensions()['outer_diameter_m'] / 2 + LAYOUT_JAW_CLEARANCE_M


def grasp_width():
    """Width the jaws close across: the roll wall, core hole to outer surface."""
    size = dimensions()
    return (size['outer_diameter_m'] - size['core_inner_diameter_m']) / 2


def _obj(path):
    vertices, faces = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith('v '):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            faces.append([int(token.split('/')[0]) - 1 for token in line.split()[1:4]])
    return np.asarray(vertices), np.asarray(faces)


def _numbers(values):
    return ' '.join(str(float(x)) for x in np.ravel(values))


def apply_tape(root, world):
    """Turn the manipulated cube body of an MJCF tree into the tape roll in place."""
    if not is_tape(world):
        return root
    size = dimensions()
    width, radius = size['width_m'], size['outer_diameter_m'] / 2
    asset = root.find('asset')
    body = root.find(".//body[@name='cube']")
    geom = body.find("geom[@name='cube_geom']")
    inner = size['core_inner_diameter_m'] / 2
    angles = np.arange(48) * 2 * np.pi / 48
    outline = [[radius * np.cos(a), radius * np.sin(a), z] for z in (-width / 2, width / 2) for a in angles]
    ET.SubElement(asset, 'mesh', name='tape_collision_mesh', vertex=_numbers(outline))
    physics = {key: geom.get(key) for key in ('contype', 'conaffinity', 'friction', 'condim', 'solref', 'solimp') if geom.get(key)}
    for key in ('size', 'type', 'rgba', 'material'):
        geom.attrib.pop(key, None)
    # Outline only: bowl containment reads its vertices; it never collides or weighs.
    geom.set('type', 'mesh'); geom.set('mesh', 'tape_collision_mesh'); geom.set('group', '3')
    geom.set('rgba', '.93 .91 .84 .3'); geom.set('mass', '0'); geom.set('contype', '0'); geom.set('conaffinity', '0')
    # Hollow wall: convex wedges between the core hole and the outer surface,
    # with the cube's contact physics and the roll's mass shared between them.
    mass = world['cube'].get('mass_kg', size['mass_kg']) / WALL_SEGMENTS
    for k in range(WALL_SEGMENTS):
        edges = [k * 2 * np.pi / WALL_SEGMENTS, (k + 1) * 2 * np.pi / WALL_SEGMENTS]
        wedge = [[r * np.cos(a), r * np.sin(a), z] for z in (-width / 2, width / 2) for r in (inner, radius) for a in edges]
        ET.SubElement(asset, 'mesh', name=f'tape_wall_{k}_mesh', vertex=_numbers(wedge))
        ET.SubElement(body, 'geom', name=f'tape_wall_{k}', type='mesh', mesh=f'tape_wall_{k}_mesh', group='3',
                      rgba='.93 .91 .84 .3', mass=str(mass), **physics)
    for name, rgba in VISUAL_PARTS:
        vertices, faces = _obj(MODEL_DIR / f'{name}.obj')
        vertices = vertices - [0, 0, width / 2]  # Blender parts sit on z=0; centre them on the body
        ET.SubElement(asset, 'mesh', name=f'{name}_visual', vertex=_numbers(vertices), face=' '.join(map(str, faces.ravel())))
        ET.SubElement(body, 'geom', name=f'{name}_visual', type='mesh', mesh=f'{name}_visual', rgba=rgba,
                      contype='0', conaffinity='0', group='2', mass='0')
    return root


def jaw_openings(model, data, adapter, qpos, width):
    """Held model gripper angle for ``width``, and native open, squeeze and release commands."""
    import mujoco
    def span(q5):
        data.qpos[:6] = qpos; data.qpos[5] = q5; mujoco.mj_forward(model, data)
        return float(np.linalg.norm(data.geom('fixed_pad').xpos - data.geom('moving_pad').xpos))
    def solve(target):
        low, high = model.jnt_range[5]
        for _ in range(50):
            mid = (low + high) / 2
            low, high = (mid, high) if span(mid) < target else (low, mid)
        return (low + high) / 2
    held = solve(width + PAD_THICKNESS_M)  # after the wall slides onto the fixed pad
    opened = solve(width + PAD_THICKNESS_M + 2 * OPEN_SIDE_CLEARANCE_M)
    native = lambda q5: float(adapter.to_native(np.r_[np.zeros(5), q5])[5])
    # Squeeze command: a little inside the held width, so the servo presses at
    # its torque limit while the tracking error stays a bounded, honest number.
    # Release only opens a little: the moving jaw stays clear of the bowl wall,
    # and the fixed jaw then slides up out of the hole.
    released = solve(width + PAD_THICKNESS_M + RELEASE_CLEARANCE_M)
    return (held, float(np.ceil(native(opened))), max(0., float(np.floor(native(held) - SQUEEZE_NATIVE))),
            float(np.ceil(native(released))))


def fit_tape_mask(mask, camera, initial_xy, support):
    """Fit the roll's XY to a SAM silhouette: projected top and bottom rims vs the mask hull.

    The roll is round, so yaw is not observable and is fixed at zero.
    """
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    from dataset_to_sim.scene_estimation import _sample, _distance
    size = dimensions()
    radius, width = size['outer_diameter_m'] / 2, size['width_m']
    angles = np.arange(48) * 2 * np.pi / 48
    rim = np.array([[radius * np.cos(a), radius * np.sin(a), z] for z in (0., width) for a in angles])
    rotation = Rotation.from_quat(np.array(camera['quaternion'])[[1, 2, 3, 0]]).as_matrix()
    focal = 240 / np.tan(np.deg2rad(camera['fovy'] / 2))

    def project(xy):
        points = (rim + [xy[0], xy[1], support] - np.array(camera['position'])) @ rotation @ np.diag([1, -1, -1])
        pixels = points[:, :2] / points[:, 2:] * focal + [320, 240]
        return pixels[ConvexHull(pixels).vertices]

    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    observed = cv2.convexHull(max(contours, key=cv2.contourArea))[:, 0, :].astype(float)
    points = _sample(observed)

    def residual(xy):
        predicted = project(xy)
        return np.r_[_distance(_sample(predicted), observed), _distance(points, predicted)]
    initial = np.asarray(initial_xy, float)
    fit = least_squares(residual, initial, bounds=(initial - .03, initial + .03), diff_step=1e-4,
                        loss='soft_l1', f_scale=1., max_nfev=100)
    error = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
    # A clear roll segments less crisply than a painted cube: allow a looser boundary.
    if not fit.success or error > 5:
        raise ValueError('From mask does not match the 53 mm tape roll; click the tape again')
    return fit.x, {'method': 'measured tape roll (53 mm × 24 mm) fit to SAM mask only', 'boundary_rmse_px': error,
                   'observed_hull': observed.tolist(), 'projected_hull': project(fit.x).tolist()}
