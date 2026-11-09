"""Read-only geometry and materials from the three actual Studio workflows.

The frozen real-replay archive and current Generate Data/New Run compilers
supply every vertex and color. No graphics context or hardware is involved.
"""
from functools import lru_cache
import copy
import cv2
import hashlib
from pathlib import Path
import tempfile
import threading

import mujoco
import numpy as np
from studio_generation_scene import TASKS, initialize_generation_scene, load_generation_scene
from dataset_to_sim.strict_replay import NativeJointAdapter
from episode0 import ROBOT
from studio_stack import build_scene_model

ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.RLock()
REPLAY_SCENE = ROOT / 'output/shared-calibration-v10/episode-00/scene.xml'
# Standalone library models that no workflow places in a scene yet.
TAPE_MODEL = ROOT / 'assets/models/tape/tape.xml'
ERASER_MODEL = ROOT / 'assets/models/eraser/eraser.xml'
_LIBRARY_BODIES = ('tape', 'eraser')
_DESCRIPTORS = {
    'red-cube': ('Red cube', 'Task object', 'Calibrated cube primitive, red generation material.'),
    'green-cube': ('Green cube', 'Task object', 'Calibrated cube primitive, green generation material.'),
    'green-bowl': ('Green bowl', 'Task object', 'Open bowl mesh with the calibrated tapered wall and floor.'),
    'tape': ('Tape roll', 'Library object', 'Clear 1-inch packing-tape roll modelled in Blender from measurements: '
             '53 mm outer diameter, 24 mm wide, 26 mm core hole, 30 mm cardboard core. '
             'Grasped through its hole (fixed jaw inside the core) in Put the tape in the green bowl.'),
    'eraser': ('Eraser', 'Library object', 'Plastic eraser measured by the user: 35 × 18 × 11 mm, with a paper sleeve band. '
               'Mass (9 g) and friction are estimates. Located automatically when a dataset is imported.'),
    'so101': ('SO-101 robot', 'Robot', 'Current robot visual meshes, camera mount, and wrist camera in the saved Home pose.'),
    'work-mat': ('Work mat', 'Environment', 'Calibrated work surface used by every generated task.'),
    'table': ('Table', 'Environment', 'The actual tabletop box used by the simulation.'),
    'floor': ('Ground plane', 'Environment', 'Infinite simulation plane; preview shows its configured 6 × 6 m display extent.'),
}


def _signature():
    files = [ROOT / 'calibration/native-action-current.json', ROOT / 'calibration/studio-home.json',
             ROOT / 'scripts/studio_generation_scene.py', ROOT / 'scripts/strict_action_scene.py',
             ROOT / 'scripts/episode0.py', ROOT / 'scripts/episode0_physics.py',
             ROOT / 'src/dataset_to_sim/scene.py', ROOT / 'scripts/studio_stack.py',
             ROOT / 'scripts/studio_objects.py', REPLAY_SCENE, ROBOT]
    files += sorted(TAPE_MODEL.parent.glob('*')) + sorted(ERASER_MODEL.parent.glob('*'))
    files += sorted((ROBOT.parent / 'assets').glob('*'))
    files += sorted((REPLAY_SCENE.parent.parent / 'assets').glob('*'))
    fingerprint = [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files if path.is_file()]
    return hashlib.sha256(repr(fingerprint).encode()).hexdigest()[:20]


def _triangles(model, index):
    kind = int(model.geom_type[index])
    size = model.geom_size[index]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[index])
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        face_start, face_count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        return model.mesh_vert[start:start + count].astype(float), model.mesh_face[face_start:face_start + face_count].copy()
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        vertices = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                             [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=float) * size
        faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                          [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
        return vertices, faces
    if kind == mujoco.mjtGeom.mjGEOM_PLANE:
        return np.array([[-size[0],-size[1],0],[size[0],-size[1],0],
                         [size[0],size[1],0],[-size[0],size[1],0]]), np.array([[0,1,2],[0,2,3]])
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        # Triangulation only: radius/half-height come from the compiled primitive.
        n = 48
        angles = np.arange(n) * 2 * np.pi / n
        vertices = [[size[0]*np.cos(a), size[0]*np.sin(a), z] for z in [-size[1], size[1]] for a in angles]
        vertices += [[0,0,-size[1]], [0,0,size[1]]]
        faces = []
        for i in range(n):
            j = (i+1) % n
            faces.extend([[i,j,n+i],[j,n+j,n+i],[2*n,j,i],[2*n+1,n+i,n+j]])
        return np.asarray(vertices), np.asarray(faces)
    raise ValueError(f'Unsupported current visual geometry type: {kind}')


def _tasks(identifier):
    result = []
    for task_id, (kind, color, instruction) in TASKS.items():
        if identifier == 'green-bowl' and kind != 'bowl':
            continue
        role = 'Work surface'
        if identifier == 'so101': role = 'Manipulator'
        elif identifier == 'green-bowl': role = 'Placement target'
        elif identifier == 'floor': role = 'Background plane'
        elif identifier in ('red-cube', 'green-cube'):
            cube_color = identifier.split('-')[0]
            if kind == 'unstack': role = 'Top or support cube'
            elif cube_color == color: role = 'Manipulated object'
            elif kind == 'stack': role = 'Stack support'
            else: role = 'Present in scene'
        result.append({'id': task_id, 'instruction': instruction, 'role': role})
    return result


def _visual_rgba(model, data):
    """Ask MuJoCo to resolve material vs explicit geom RGBA, without OpenGL."""
    scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 100)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, mujoco.MjvCamera(),
                          mujoco.mjtCatBit.mjCAT_ALL, scene)
    return {int(geom.objid): geom.rgba.copy() for geom in scene.geoms[:scene.ngeom]
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM}


def _model_parts(model, home):
    data = mujoco.MjData(model)
    if model.nq >= 6:
        data.qpos[:6] = home
    mujoco.mj_forward(model, data)
    colors = _visual_rgba(model, data)
    parts = {}
    for i, rgba in colors.items():
        if model.geom_group[i] > 2 or rgba[3] <= 0:
            continue
        name, body = model.geom(i).name, model.body(model.geom_bodyid[i]).name
        if name in ('table', 'floor', 'work_mat'):
            identifier = name.replace('_', '-')
        elif body in ('cube', 'target_cube', 'bowl', *_LIBRARY_BODIES):
            identifier = body
        else:
            identifier = 'so101'
        vertices, faces = _triangles(model, i)
        vertices = vertices @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i]
        if body in ('cube', 'target_cube', 'bowl', *_LIBRARY_BODIES):
            # Object dimensions/variants describe local geometry, not the
            # arbitrary recorded placement yaw of an episode.
            bid = int(model.geom_bodyid[i])
            vertices = (vertices - data.xpos[bid]) @ data.xmat[bid].reshape(3, 3)
        parts.setdefault(identifier, []).append({'vertices': vertices, 'faces': faces, 'rgba': rgba.tolist()})
    return parts


def _source_tasks(identifier, source_id):
    if source_id == 'generation':
        return _tasks(identifier)
    if source_id == 'library' and identifier == 'eraser':
        return [{'id': 'imported-eraser', 'instruction': 'Imported dataset · Put the eraser in the bowl', 'role': 'Manipulated object'}]
    if source_id == 'library':
        # The roll replaces the manipulated cube in its tasks.
        return [{'id': task_id, 'instruction': instruction, 'role': 'Manipulated object'}
                for task_id, (kind, color, instruction) in TASKS.items() if color == identifier] + \
               [{'id': 'new-run-tape', 'instruction': 'New Run · Place tape in bowl', 'role': 'Manipulated object'}]
    if source_id == 'real-replay':
        role = {'real-red-cube': 'Manipulated object', 'green-bowl': 'Placement target',
                'so101': 'Manipulator', 'floor': 'Background plane'}.get(identifier, 'Work surface')
        return [{'id': 'real-replay-bowl', 'instruction': 'Real dataset · Put the red cube in the green bowl.', 'role': role}]
    rows = [('new-run-bowl', 'New Run · Place cube in bowl'), ('new-run-stack', 'New Run · Stack on cube')]
    if identifier == 'tan-cube': rows = rows[1:]
    if not identifier.endswith('-cube'): rows.append(('new-run-tape', 'New Run · Place tape in bowl'))
    result = []
    for task_id, instruction in rows:
        role = {'real-red-cube': 'Manipulated object', 'tan-cube': 'Stack support',
                'so101': 'Manipulator', 'floor': 'Background plane'}.get(identifier, 'Work surface')
        if identifier == 'green-bowl': role = 'Optional obstacle' if task_id == 'new-run-stack' else 'Placement target'
        result.append(dict(id=task_id, instruction=instruction, role=role))
    return result


@lru_cache(maxsize=1)
def _snapshot(signature):
    with tempfile.TemporaryDirectory(prefix='studio-objects-') as folder:
        initialize_generation_scene(folder, 'green_cube_to_green_bowl', 0, {'randomize_layout': False})
        capture, config, _, cal, generation = load_generation_scene(folder)
    home = NativeJointAdapter(**config['adapter']).to_model(np.asarray(capture['initial_native']))
    # New Run's builder supplies its actual source-red and tan support materials.
    # Reuse measured geometry only; no image capture, planning or hardware access.
    new_run = build_scene_model(copy.deepcopy(cal), config)
    sources = [
        ('real-replay', 'Real dataset Simulation', 'output/shared-calibration-v10/episode-00/scene.xml',
         mujoco.MjModel.from_xml_path(str(REPLAY_SCENE))),
        ('generation', 'Generate Data', 'scripts/studio_generation_scene.py', generation),
        ('new-run', 'New Run', 'scripts/studio_stack.py', new_run),
        ('library', 'Model library', 'assets/models/tape/tape.xml',
         mujoco.MjModel.from_xml_path(str(TAPE_MODEL))),
        ('library', 'Model library', 'assets/models/eraser/eraser.xml',
         mujoco.MjModel.from_xml_path(str(ERASER_MODEL))),
    ]
    grouped = {key: [] for key in ('cube', 'green-bowl', 'tape', 'eraser', 'so101', 'work-mat', 'table', 'floor')}
    parts_by_id = {}
    labels = {'red-cube': 'Red · Generate Data', 'green-cube': 'Green · Generate Data',
              'real-red-cube': 'Red · Real dataset / New Run', 'tan-cube': 'Tan · New Run',
              'green-bowl': 'Sage green', 'tape': 'Clear', 'eraser': 'White', 'so101': 'Original materials', 'work-mat': 'Charcoal',
              'table': 'Off-white', 'floor': 'Light gray'}
    for source_id, source_label, source_path, model in sources:
        source = {'id': source_id, 'label': source_label, 'path': source_path}
        for body, parts in _model_parts(model, home).items():
            identifier = body
            if body == 'cube': identifier = 'green-cube' if source_id == 'generation' else 'real-red-cube'
            elif body == 'target_cube': identifier = 'red-cube' if source_id == 'generation' else 'tan-cube'
            elif body == 'bowl': identifier = 'green-bowl'
            group = 'cube' if identifier.endswith('-cube') else identifier
            vertices = np.concatenate([part['vertices'] for part in parts])
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            materials = []
            for part in parts:
                if not any(np.allclose(part['rgba'], color) for color in materials): materials.append(part['rgba'])
            # Merge identical appearances across workflows; keep source-specific
            # roles attached to that exact material variant.
            existing = next((v for v in grouped[group] if len(v['materials']) == len(materials)
                             and np.allclose(v['materials'], materials)
                             and np.allclose(v['dimensions_m'], high-low, atol=1e-6)), None)
            tasks = _source_tasks(identifier, source_id)
            if existing:
                existing['sources'].append(source)
                existing['tasks'].extend(tasks)
                continue
            descriptor = 'red-cube' if group == 'cube' else identifier
            name, category, description = _DESCRIPTORS[descriptor]
            if group == 'cube': description = 'Calibrated cube geometry with the selected workflow’s actual material.'
            if identifier == 'green-bowl': description = 'Open bowl mesh using the sage material from the real-dataset Simulation.'
            for part in parts: part['vertices'] = np.round(part['vertices'] - (low+high)/2, 7)
            # A changed source may introduce a real geometry/material variation.
            # Keep its identity distinct instead of silently replacing a model.
            if identifier in parts_by_id: identifier += '-' + source_id
            item = dict(id=identifier, name='Cube' if group == 'cube' else 'Bowl' if group == 'green-bowl' else name,
                        category=category, description=description, label=labels.get(identifier, source_label),
                        dimensions_m=np.round(high-low, 7).tolist(), tasks=tasks, sources=[source],
                        rgba=parts[0]['rgba'], materials=materials,
                        triangle_count=sum(len(part['faces']) for part in parts), part_count=len(parts),
                        mesh_url=f'/api/objects/{identifier}/mesh',
                        thumbnail_url=f'/api/objects/{identifier}/thumbnail?revision={signature}',
                        dimension_note='Home-pose bounds' if identifier.startswith('so101') else
                        'Display extent; plane is infinite' if identifier == 'floor' else 'Model bounds')
            grouped[group].append(item)
            parts_by_id[identifier] = parts
    # The current measured bowl (Generate Data / New Run) is shown first; the
    # real-dataset Simulation keeps its archived bowl for frozen benchmark runs.
    archived = lambda item: [source['id'] for source in item['sources']] == ['real-replay']
    if len(grouped['green-bowl']) > 1:
        grouped['green-bowl'].sort(key=archived)
        for item in grouped['green-bowl']:
            if archived(item):
                item.update(label='Archived · Real dataset', description='Original bowl frozen in the recorded real-dataset Simulation runs (benchmark physics). Realistic dataset views draw the current measured bowl.')
            else:
                item.update(label='Measured · current', description='Current measured bowl: 2 mm wall and rounded inner/outer bottom, used by Generate Data and New Run.')
    order = {'red-cube': 0, 'green-cube': 1, 'real-red-cube': 2, 'tan-cube': 3}
    grouped['cube'].sort(key=lambda item: order.get(item['id'], 9))
    entries = []
    for identifier, variants in grouped.items():
        entry = dict(variants[0], id=identifier, variants=variants)
        entry['tasks'] = list({task['id']: task for variant in variants for task in variant['tasks']}.values())
        entries.append(entry)
    return {'schema_version': 3, 'revision': signature, 'objects': entries,
            'task_count': len(TASKS), 'source_count': len(sources), 'units': 'meters',
            'note': 'Actual model materials from Generate Data, real-dataset Simulation and New Run. '
                    'Matching appearances share one swatch. Dataset lighting and camera color processing can change their rendered appearance.'}, parts_by_id


def catalog():
    with _LOCK:
        return _snapshot(_signature())[0]


def _lookup(identifier, signature):
    index, parts = _snapshot(signature)
    if identifier not in parts:
        raise ValueError('Object not found')
    item = next(variant for group in index['objects'] for variant in group['variants'] if variant['id'] == identifier)
    return item, parts[identifier]


def object_mesh(identifier):
    with _LOCK:
        signature = _signature()
        item, parts = _lookup(identifier, signature)
        return {'id': identifier, 'revision': signature, 'units': 'meters',
                'dimensions_m': item['dimensions_m'],
                'parts': [{'vertices': part['vertices'].ravel().tolist(), 'faces': part['faces'].ravel().tolist(),
                           'rgba': part['rgba']} for part in parts]}


@lru_cache(maxsize=32)
def _thumbnail(identifier, signature):
    """Small CPU-projected image of actual triangles; no renderer or mesh transfer."""
    _, parts = _lookup(identifier, signature)
    yaw, pitch = -.9, .5
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    basis = np.array([[-sy,cy,0], [-sp*cy,-sp*sy,cp], [cp*cy,cp*sy,sp]])
    projected = [part['vertices'] @ basis.T for part in parts]
    vertices = np.concatenate(projected)
    span = max(np.ptp(vertices[:,0]), np.ptp(vertices[:,1]), 1e-8)
    center = (vertices.max(axis=0)+vertices.min(axis=0))/2
    polygons, depths, colors = [], [], []
    light = np.array([-.4,-.6,1.]); light /= np.linalg.norm(light)
    for part, xyz in zip(parts, projected):
        triangles = part['vertices'][part['faces']]
        normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        shading = .55 + .45*np.abs(normals @ light)
        rgb = np.clip(np.asarray(part['rgba'][:3])*255*shading[:,None], 0, 255).astype(np.uint8)
        points = (xyz[:,:2]-center[:2])*132/span
        points[:,1] *= -1
        polygons.extend(np.rint(points[part['faces']]+80).astype(np.int32))
        depths.extend(xyz[part['faces'],2].mean(axis=1))
        colors.extend(rgb)
    pixels = np.zeros((160,160,4), dtype=np.uint8)
    for i in np.argsort(depths):
        bgr = [int(x) for x in colors[i][::-1]]
        cv2.fillConvexPoly(pixels, polygons[i], (*bgr,255), lineType=cv2.LINE_AA)
    ok, encoded = cv2.imencode('.png', pixels)
    if not ok: raise ValueError('Thumbnail could not be encoded')
    return encoded.tobytes()


def object_thumbnail(identifier):
    with _LOCK:
        return _thumbnail(identifier, _signature())
