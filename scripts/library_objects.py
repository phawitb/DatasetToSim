"""Pen, cup and Intel NUC: library objects built from dimensions, no Blender.

One set of numpy meshes serves both the 3D-Objects page (MJCF + OBJ files in
assets/models/<kind>, written by ``python scripts/library_objects.py``) and
New Run physics (``add_library_geometry`` turns a free body into the object).
Every mesh is in the object's own frame with its origin at the bounding-box
centre (z up), the frame the contact-area registry uses.

Dimensions: the NUC follows Intel's NUC7i3BNH spec (115 x 111 x 51 mm). The pen
and cup were not measured yet; their numbers below are estimates from photos
of a clear-barrel ballpoint pen with an orange cap and a ribbed plastic cup.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'assets/models'

PEN = {'length_m': .149, 'across_flats_m': .008, 'cap_diameter_m': .0098, 'cap_length_m': .032,
       'plug_length_m': .006, 'mass_kg': .006, 'friction': [.6, .005, .0001]}
CUP = {'height_m': .078, 'top_outer_diameter_m': .070, 'bottom_outer_diameter_m': .046, 'wall_m': .001,
       'floor_m': .0012, 'lip_m': .0012, 'ribs': 24, 'mass_kg': .005, 'friction': [.7, .005, .0001]}
NUC = {'size_m': [.115, .111, .051], 'top_m': .004, 'mass_kg': .7, 'friction': [.8, .005, .0001]}
KINDS = ('pen', 'cup', 'nuc')
NOTES = {
    'pen': 'Ballpoint pen with a clear hexagonal barrel, orange cap (with clip) and orange end plug. '
           'Estimated, not measured: 149 mm long, 8 mm across the barrel flats, 9.8 mm cap. Mass is an estimate.',
    'cup': 'Tapered white plastic cup with a ribbed side and a rolled rim. Estimated, not measured: 78 mm tall, '
           '70 mm across the rim, 46 mm across the base, 1 mm wall. Mass is an estimate.',
    'nuc': 'Intel NUC7i3BNH mini PC (tall chassis) from Intel\'s spec: 115 x 111 x 51 mm, glossy black top. '
           'Mass about 0.7 kg is an estimate. Too wide for the SO-101 jaws (about 60 mm), so it is a place-only object.',
}
WALL_SEGMENTS = 32


# ---------------------------------------------------------------- mesh primitives
def _prism_x(profile, x0, x1):
    """Closed prism along x from a convex (y, z) profile."""
    profile = np.asarray(profile, float); n = len(profile)
    vertices = np.vstack([np.c_[np.full(n, x), profile] for x in (x0, x1)])
    vertices = np.vstack([vertices, [[x0, *profile.mean(0)], [x1, *profile.mean(0)]]])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return vertices, np.asarray(faces)


def _circle(radius, n=32, centre=(0., 0.)):
    a = np.arange(n) * 2 * np.pi / n
    return np.c_[centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)]


def _hexagon(across_flats):
    """Hexagon in (y, z) with flats on top and bottom (the pen lies on a flat)."""
    r = across_flats / 2 / np.cos(np.pi / 6)
    return _circle(r, 6)


def _box(centre, size):
    half = np.asarray(size, float) / 2
    corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) * half + centre
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return corners, np.asarray(faces)


def _lathe(rings, n=64, closed_bottom=False, closed_top=False, ribs=0, rib_depth=0.):
    """Surface of revolution about z through rings [(radius, z, rib_weight)]."""
    a = np.arange(n) * 2 * np.pi / n
    vertices, faces = [], []
    for r, z, w in rings:
        rr = r * (1 + w * rib_depth * np.cos(ribs * a)) if ribs else np.full(n, r)
        vertices += [[x, y, z] for x, y in zip(rr * np.cos(a), rr * np.sin(a))]
    for k in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            faces += [[k * n + i, k * n + j, (k + 1) * n + j], [k * n + i, (k + 1) * n + j, (k + 1) * n + i]]
    for flag, k in ((closed_bottom, 0), (closed_top, len(rings) - 1)):
        if flag:
            vertices.append([0, 0, rings[k][1]]); c = len(vertices) - 1
            faces += [[c, k * n + (i + 1) % n, k * n + i] for i in range(n)]
    return np.asarray(vertices, float), np.asarray(faces)


# ---------------------------------------------------------------- objects
def cup_radius(z):
    """Outer wall radius at height z in the cup frame (origin at mid-height)."""
    h = CUP['height_m']; top, bottom = CUP['top_outer_diameter_m'] / 2, CUP['bottom_outer_diameter_m'] / 2
    return bottom + (top - bottom) * (z + h / 2) / h


def pen_parts():
    L, flats = PEN['length_m'], PEN['across_flats_m']
    x0, x1 = -L / 2, L / 2
    cap_r = PEN['cap_diameter_m'] / 2
    lift = cap_r - flats / 2  # the cap rests on the mat too, so its axis sits higher than the barrel's
    cap0 = x1 - PEN['cap_length_m']
    parts = [('pen_barrel', [.90, .93, .96, .45], *_prism_x(_hexagon(flats), x0 + PEN['plug_length_m'], cap0 + .004)),
             ('pen_ink', [.95, .45, .08, 1], *_prism_x(_circle(.0015, 16), x0 + .010, cap0 - .004)),
             ('pen_tip', [.75, .76, .78, 1], *_prism_x(_circle(.0012, 12), cap0 - .004, cap0 + .006)),
             ('pen_plug', [.98, .42, .05, 1], *_prism_x(_circle(.0031, 24), x0, x0 + PEN['plug_length_m'])),
             ('pen_cap', [.98, .42, .05, 1], *_prism_x(_circle(cap_r, 32, (0, lift)), cap0, x1))]
    clip_v, clip_f = _box([x1 - .016, 0, lift + cap_r + .0006], [.026, .003, .0012])
    parts.append(('pen_clip', [.98, .42, .05, 1], clip_v, clip_f))
    return parts


def cup_parts():
    h, wall, floor, lip = CUP['height_m'], CUP['wall_m'], CUP['floor_m'], CUP['lip_m']
    zs = np.linspace(-h / 2, h / 2 - lip, 12)
    # Ribs on the lower two thirds of the side, fading out towards the rim.
    weight = np.clip((h / 2 - lip - .02 - zs) / .01, 0, 1)
    outer = _lathe([(cup_radius(z), z, w) for z, w in zip(zs, weight)], ribs=CUP['ribs'], rib_depth=.025, closed_bottom=True)
    inner = _lathe([(cup_radius(z) - wall, z, 0) for z in np.linspace(-h / 2 + floor, h / 2 - lip, 6)], closed_bottom=True)
    r = cup_radius(h / 2)
    # Rolled rim: a closed loop of rings around the top edge.
    rim = _lathe([(a, b, 0) for a, b in [(r - wall, h / 2 - lip), (r + lip * .6, h / 2 - lip), (r + lip, h / 2 - lip / 2),
                                         (r + lip * .6, h / 2), (r - wall, h / 2), (r - wall, h / 2 - lip)]])
    white = [.96, .96, .95, 1]
    return [('cup_outer', white, *outer), ('cup_inner', [.93, .93, .92, 1], *inner), ('cup_rim', white, *rim)]


def nuc_parts():
    sx, sy, sz = NUC['size_m']; top = NUC['top_m']
    front = sx / 2
    parts = [('nuc_chassis', [.17, .17, .18, 1], *_box([0, 0, -top / 2], [sx, sy, sz - top])),
             ('nuc_top', [.03, .03, .04, 1], *_box([0, 0, sz / 2 - top / 2], [sx - .001, sy - .001, top]))]
    ports = [('nuc_power', [.80, .82, .85, 1], [front, -.042, .012], [.0006, .007, .007]),
             ('nuc_usb_front_a', [.05, .05, .06, 1], [front, .006, -.006], [.0006, .013, .0055]),
             ('nuc_usb_front_b', [.05, .05, .06, 1], [front, .024, -.006], [.0006, .013, .0055]),
             ('nuc_audio', [.05, .05, .06, 1], [front, .040, -.006], [.0006, .004, .004]),
             ('nuc_ir', [.25, .05, .06, 1], [front, -.020, .012], [.0006, .010, .004]),
             ('nuc_vent', [.08, .08, .09, 1], [-front, 0, .010], [.0006, .090, .012])]
    for name, rgba, centre, size in ports:
        parts.append((name, rgba, *_box(centre, size)))
    return parts


PARTS = {'pen': pen_parts, 'cup': cup_parts, 'nuc': nuc_parts}


def mass(kind):
    return {'pen': PEN, 'cup': CUP, 'nuc': NUC}[kind]['mass_kg']


def friction(kind):
    return {'pen': PEN, 'cup': CUP, 'nuc': NUC}[kind]['friction']


def collision_hulls(kind):
    """Convex collision pieces (vertex arrays) in the object frame."""
    if kind == 'pen':
        L = PEN['length_m']
        return [_prism_x(_hexagon(PEN['across_flats_m']), -L / 2, L / 2)[0]]
    if kind == 'nuc':
        return [_box([0, 0, 0], NUC['size_m'])[0]]
    h, wall, floor = CUP['height_m'], CUP['wall_m'], CUP['floor_m']
    hulls = []
    for k in range(WALL_SEGMENTS):
        edges = np.array([k, k + 1]) * 2 * np.pi / WALL_SEGMENTS
        hulls.append(np.array([[(cup_radius(z) - d) * np.cos(a), (cup_radius(z) - d) * np.sin(a), z]
                               for z in (-h / 2, h / 2) for d in (0, wall) for a in edges]))
    disc = _circle(cup_radius(-h / 2) - wall / 2, 24)
    hulls.append(np.vstack([np.c_[disc, np.full(len(disc), z)] for z in (-h / 2, -h / 2 + floor)]))
    return hulls


# ---------------------------------------------------------------- MuJoCo
def _numbers(values):
    return ' '.join(f'{float(v):.7g}' for v in np.ravel(values))


def add_library_geometry(root, body, geom, kind, mass_kg=None, prefix=''):
    """Give `body` the object's collision hulls and visual meshes; `geom` (with the
    scene's contact settings) becomes a hidden, non-colliding placeholder."""
    asset = root.find('asset')
    physics = {key: geom.get(key) for key in ('contype', 'conaffinity', 'condim', 'solref', 'solimp') if geom.get(key)}
    physics['friction'] = _numbers(friction(kind))
    for key in ('size', 'type', 'rgba', 'material', 'mesh'):
        geom.attrib.pop(key, None)
    geom.set('type', 'sphere'); geom.set('size', '.001'); geom.set('group', '3'); geom.set('mass', '0')
    geom.set('contype', '0'); geom.set('conaffinity', '0')
    hulls = collision_hulls(kind)
    total = mass(kind) if mass_kg is None else mass_kg
    for k, vertices in enumerate(hulls):
        ET.SubElement(asset, 'mesh', name=f'{prefix}{kind}_hull_{k}', vertex=_numbers(vertices))
        ET.SubElement(body, 'geom', name=f'{prefix}{kind}_hull_{k}', type='mesh', mesh=f'{prefix}{kind}_hull_{k}', group='3',
                      rgba='.9 .9 .9 .3', mass=str(total / len(hulls)), **physics)
    for name, rgba, vertices, faces in PARTS[kind]():
        ET.SubElement(asset, 'mesh', name=f'{prefix}{name}_visual', vertex=_numbers(vertices), face=' '.join(map(str, faces.ravel())))
        ET.SubElement(body, 'geom', name=f'{prefix}{name}_visual', type='mesh', mesh=f'{prefix}{name}_visual', rgba=_numbers(rgba),
                      contype='0', conaffinity='0', group='2', mass='0')


def write_models():
    """Write assets/models/<kind>/<kind>.xml, its OBJ parts and a dimensions file."""
    for kind in KINDS:
        folder = MODELS / kind; folder.mkdir(parents=True, exist_ok=True)
        root = ET.Element('mujoco', model=kind)
        ET.SubElement(root, 'compiler', angle='radian', meshdir='.')
        asset = ET.SubElement(root, 'asset')
        world = ET.SubElement(root, 'worldbody')
        parts = PARTS[kind]()
        low = min(v[:, 2].min() for _, _, v, _ in parts)
        body = ET.SubElement(world, 'body', name=kind, pos=f'0 0 {-low:.6g}')
        for name, rgba, vertices, faces in parts:
            lines = [f'v {x:.7g} {y:.7g} {z:.7g}' for x, y, z in vertices] + [f'f {a + 1} {b + 1} {c + 1}' for a, b, c in faces]
            (folder / f'{name}.obj').write_text('\n'.join(lines) + '\n')
            ET.SubElement(asset, 'mesh', name=name, file=f'{name}.obj')
            ET.SubElement(body, 'geom', name=name, type='mesh', mesh=name, rgba=_numbers(rgba), group='2', contype='0', conaffinity='0', mass='0')
        for k, vertices in enumerate(collision_hulls(kind)):
            ET.SubElement(asset, 'mesh', name=f'{kind}_hull_{k}', vertex=_numbers(vertices))
            ET.SubElement(body, 'geom', name=f'{kind}_hull_{k}', type='mesh', mesh=f'{kind}_hull_{k}', group='3', rgba='.9 .9 .9 .3',
                          mass=str(mass(kind) / len(collision_hulls(kind))), friction=_numbers(friction(kind)))
        ET.indent(root)
        header = f'<!-- Generated by scripts/library_objects.py; do not edit. {NOTES[kind]} -->\n'
        (folder / f'{kind}.xml').write_text(header + ET.tostring(root, encoding='unicode') + '\n')
        spec = {'pen': PEN, 'cup': CUP, 'nuc': NUC}[kind]
        (folder / f'{kind}.json').write_text(json.dumps({'note': NOTES[kind], **spec}, indent=2) + '\n')


if __name__ == '__main__':
    write_models()
