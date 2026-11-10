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

# Measured by the user 2026-11-10: pen 12 mm diameter x 135 mm; cup 75 mm rim and 55 mm
# base (outside), 2 mm rim; box 115 x 112 x 48 mm on four 10 mm round feet, 4 mm tall.
# Cup height: 85 mm along the slanted side -> sqrt(85² - 10²) = 84.4 mm vertical.
# Not measured: pen barrel (the cap is the 12 mm part), masses.
PEN = {'length_m': .135, 'across_flats_m': .011, 'cap_diameter_m': .012, 'cap_length_m': .030,
       'plug_length_m': .006, 'mass_kg': .006, 'friction': [.6, .005, .0001]}
CUP = {'height_m': .0844, 'top_outer_diameter_m': .075, 'bottom_outer_diameter_m': .055, 'wall_m': .002,
       'floor_m': .002, 'lip_m': .002, 'ribs': 24, 'mass_kg': .035, 'friction': [.7, .005, .0001]}
NUC = {'size_m': [.115, .112, .052], 'body_height_m': .048, 'foot_height_m': .004, 'foot_diameter_m': .010,
       'foot_inset_m': .011, 'top_m': .004, 'mass_kg': .7, 'friction': [.8, .005, .0001]}
KINDS = ('pen', 'cup', 'nuc')
NOTES = {
    'pen': 'Ballpoint pen (MALA) with a clear barrel, orange ink tube, plain orange cap and rounded orange end. '
           'Measured: 135 mm long, 12 mm across the cap; the barrel (11 mm) is estimated. Collision is a hexagonal prism so it does not roll. Mass is an estimate.',
    'cup': 'Tapered off-white plastic cup with faint horizontal ribs and a thin rim. Measured: 75 mm across the rim, '
           '55 mm across the base (outside), 2 mm rim. 85 mm along the slanted side, so 84.4 mm tall. Mass (35 g, from a 2 mm polypropylene wall) is an estimate.',
    'nuc': 'Intel NUC7i3BNH mini PC, measured: 115 x 112 x 48 mm body on four 10 mm round feet 4 mm tall (52 mm overall), glossy black top with silver trim, front USB/audio/power panel. '
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


def _lathe_x(rings, n=32, z_offset=0., **kw):
    """Surface of revolution about the x axis through rings [(radius, x)]."""
    v, f = _lathe([(r, x, 0) for r, x in rings], n=n, **kw)
    return np.c_[v[:, 2], v[:, 0], v[:, 1] + z_offset], f


def _dome(r, x_flat, x_end, steps=5):
    """Rings from a flat section at x_flat to a rounded end at x_end."""
    depth = x_end - x_flat
    return [(r * np.cos(t), x_flat + depth * np.sin(t)) for t in np.linspace(0, np.pi / 2 * .97, steps)]


def pen_parts():
    """Photos: round clear barrel, orange ink tube inside, plain orange cap with a
    rounded end (no clip), rounded orange end plug in a short clear collar."""
    L, flats = PEN['length_m'], PEN['across_flats_m']
    x0, x1 = -L / 2, L / 2
    cap_r, r = PEN['cap_diameter_m'] / 2, flats / 2
    lift = cap_r - r  # the cap rests on the mat too, so its axis sits higher than the barrel's
    cap0 = x1 - PEN['cap_length_m']
    plug = PEN['plug_length_m']
    orange = [.98, .38, .10, 1]
    cap = _lathe_x([(cap_r * .97, cap0)] + _dome(cap_r, x1 - .003, x1), z_offset=lift, closed_bottom=True, closed_top=True)
    end = _lathe_x(list(reversed(_dome(r * .95, x0 + plug * .5, x0))) + [(r * .95, x0 + plug)], closed_bottom=True, closed_top=True)
    return [('pen_barrel', [.90, .93, .96, .40], *_lathe_x([(r, x0 + plug), (r, cap0 + .004)], closed_bottom=True, closed_top=True)),
            ('pen_ink', [.86, .30, .16, 1], *_lathe_x([(.0018, x0 + .012), (.0018, cap0 - .002)], n=16, closed_bottom=True, closed_top=True)),
            ('pen_ink_plug', orange, *_lathe_x([(r * .9, x0 + plug), (r * .9, x0 + plug + .007)], closed_bottom=True, closed_top=True)),
            ('pen_end', orange, *end),
            ('pen_cap', orange, *cap)]


def cup_parts():
    """Photos: plain thin rim, faint horizontal grooves around the side, recessed foot."""
    h, wall, floor, lip = CUP['height_m'], CUP['wall_m'], CUP['floor_m'], CUP['lip_m']
    zs = np.linspace(-h / 2 + .002, h / 2 - lip, 60)
    # Shallow horizontal ribs (about 4 mm pitch) from the foot to 12 mm below the rim.
    fade = np.clip((h / 2 - .012 - zs) / .006, 0, 1)
    groove = .0002 * fade * (np.cos(2 * np.pi * (zs + h / 2) / .004) > 0.3)
    rings = [(cup_radius(-h / 2) - .0015, -h / 2, 0), (cup_radius(-h / 2 + .002), -h / 2 + .002, 0)]
    rings += [(cup_radius(z) - g, z, 0) for z, g in zip(zs[1:], groove[1:])]
    outer = _lathe(rings, closed_bottom=True)
    inner = _lathe([(cup_radius(z) - wall, z, 0) for z in np.linspace(-h / 2 + floor, h / 2 - lip, 6)], closed_bottom=True)
    r = cup_radius(h / 2)
    rim = _lathe([(a, b, 0) for a, b in [(r - wall, h / 2 - lip), (r, h / 2 - lip), (r, h / 2 - lip * .3), (r - wall * .3, h / 2),
                                         (r - wall * .7, h / 2), (r - wall, h / 2 - lip * .3), (r - wall, h / 2 - lip)]])
    white = [.95, .94, .89, 1]
    return [('cup_outer', white, *outer), ('cup_inner', [.93, .91, .84, 1], *inner), ('cup_rim', white, *rim)]


def nuc_parts():
    """Photos: black body, glossy black top with a silver trim line, front I/O panel
    (blue and yellow USB, audio, square power button) above the Intel badge, vent
    slots on both sides, rear HDMI, LAN, USB, USB-C and 19 V power."""
    sx, sy, sz = NUC['size_m']; top = NUC['top_m']; foot = NUC['foot_height_m']
    front, side, trim = sx / 2, sy / 2, .0012
    z_trim = sz / 2 - top - trim / 2
    low = -sz / 2 + foot
    parts = [('nuc_chassis', [.08, .08, .09, 1], *_box([0, 0, (low + z_trim - trim / 2) / 2], [sx - .001, sy - .001, z_trim - trim / 2 - low])),
             ('nuc_trim', [.72, .73, .76, 1], *_box([0, 0, z_trim], [sx, sy, trim])),
             ('nuc_top', [.02, .02, .025, 1], *_box([0, 0, sz / 2 - top / 2], [sx - .0006, sy - .0006, top]))]
    out = .0003
    features = [('nuc_front_panel', [.03, .03, .035, 1], [front, -.001, .006], [out, .072, .012]),
                ('nuc_usb_blue', [.10, .30, .80, 1], [front + out, -.024, .006], [out, .012, .0045]),
                ('nuc_usb_yellow', [.85, .62, .10, 1], [front + out, -.005, .006], [out, .012, .0045]),
                ('nuc_audio', [.35, .35, .38, 1], [front + out, .012, .006], [out, .0035, .0035]),
                ('nuc_power', [.15, .15, .17, 1], [front + out, .027, .006], [out, .007, .007]),
                ('nuc_badge', [.12, .35, .75, 1], [front, -.034, -.010], [out, .010, .011])]
    for name, y in (('nuc_vent_left', -side), ('nuc_vent_right', side)):
        features.append((name, [.01, .01, .012, 1], [.004, y, -.006], [.075, out, .012]))
    back = -front
    features += [('nuc_rear_vents', [.01, .01, .012, 1], [back, -.010, .011], [out, .060, .006]),
                 ('nuc_hdmi', [.35, .35, .38, 1], [back, .022, -.002], [out, .015, .005]),
                 ('nuc_lan', [.35, .35, .38, 1], [back, .002, -.002], [out, .013, .010]),
                 ('nuc_usb_rear', [.10, .30, .80, 1], [back, -.018, -.002], [out, .013, .012]),
                 ('nuc_usb_c', [.35, .35, .38, 1], [back, -.036, .002], [out, .009, .003]),
                 ('nuc_dc_in', [.35, .35, .38, 1], [back, .042, -.004], [out, .006, .006])]
    for name, rgba, centre, size in features:
        parts.append((name, rgba, *_box(centre, size)))
    for k, (x, y) in enumerate(_nuc_feet()):
        parts.append((f'nuc_foot_{k}', [.05, .05, .05, 1], *_foot(x, y)))
    return parts


def _nuc_feet():
    sx, sy, _ = NUC['size_m']; inset = NUC['foot_inset_m']
    return [(sx2 * (sx / 2 - inset), sy2 * (sy / 2 - inset)) for sx2 in (-1, 1) for sy2 in (-1, 1)]


def _foot(x, y):
    sz, h = NUC['size_m'][2], NUC['foot_height_m']
    ring = _circle(NUC['foot_diameter_m'] / 2, 16, (x, y))
    v, f = _prism_x(ring, -sz / 2, -sz / 2 + h)  # prism along its first axis, then turn it upright
    return v[:, [1, 2, 0]], f


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
        sx, sy, sz = NUC['size_m']; foot = NUC['foot_height_m']
        return [_box([0, 0, foot / 2], [sx, sy, sz - foot])[0]] + [_foot(x, y)[0] for x, y in _nuc_feet()]
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
